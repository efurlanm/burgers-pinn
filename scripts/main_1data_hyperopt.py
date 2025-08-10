"""A Physics-Informed Neural Network (PINN) for the discovery of kinematic viscosity
in the 2D Burgers' equation, with integrated Hyperopt-based hyperparameter optimization."""

import tensorflow as tf
import numpy as np
import time
import scipy.optimize
import os
import argparse
import hyperopt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pickle
import tensorflow.keras.callbacks as callbacks

# Configure GPU memory growth for TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

def swish_activation(x):
    return tf.keras.activations.swish(x)

class PINN_Burgers2D(tf.keras.Model):
    def __init__(self, layers_config, x_data, y_data, t_data, u_data, v_data,
                 x_pde, y_pde, t_pde, nu_pde, x_min, x_max, y_min, y_max, t_min, t_max,
                 nu_min_train, nu_max_train, nu_data, annealing_rate=0.01, learning_rate=0.001, batch_size=1024):
        super().__init__()
        self.network_layers = layers_config
        self.annealing_rate = annealing_rate
        self.sharpness_factor = 5.0
        self.batch_size = batch_size
        self.pde_batch_size = 4096

        self.weight_data = tf.Variable(1.0, dtype=tf.float32, trainable=False)
        self.weight_pde = tf.Variable(1.0, dtype=tf.float32, trainable=False)

        self.x_data, self.y_data, self.t_data = x_data, y_data, t_data
        self.u_data, self.v_data = u_data, v_data
        self.x_pde, self.y_pde, self.t_pde, self.nu_pde = None, None, None, None

        self.x_min, self.x_max, self.y_min, self.y_max, self.t_min, self.t_max = x_min, x_max, y_min, y_max, t_min, t_max
        self.nu_min_train, self.nu_max_train = nu_min_train, nu_max_train
        self.nu_data = tf.Variable(nu_data, dtype=tf.float32, trainable=False)
        self.nu_regularization_weight = 1e-3

        self.dense_layers = [tf.keras.layers.Dense(
            self.network_layers[i+1], activation=swish_activation if i < len(self.network_layers) - 2 else None,
            kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-5))
            for i in range(len(self.network_layers) - 1)]

        self.optimizer_adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.optimizer_adam_data_only = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.log_nu_inverse = tf.Variable(tf.math.log(0.02), dtype=tf.float32)
        self.optimizer_adam_inverse = tf.keras.optimizers.Adam(learning_rate=0.001)



    def call(self, X, nu_min, nu_max):
        x_scaled = 2.0 * (X[:, 0:1] - self.x_min) / (self.x_max - self.x_min) - 1.0
        y_scaled = 2.0 * (X[:, 1:2] - self.y_min) / (self.y_max - self.y_min) - 1.0
        t_scaled = 2.0 * (X[:, 2:3] - self.t_min) / (self.t_max - self.t_min) - 1.0
        nu_scaled = 2.0 * (X[:, 3:4] - nu_min) / (nu_max - nu_min) - 1.0
        H = tf.concat([x_scaled, y_scaled, t_scaled, nu_scaled], axis=1)
        for layer in self.dense_layers:
            H = layer(H)
        return H

    def predict_velocity(self, x, y, t, nu):
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        t = tf.cast(t, tf.float32)
        nu = tf.cast(nu, tf.float32)
        
        X_input = tf.stack([tf.reshape(x, [-1]), tf.reshape(y, [-1]), tf.reshape(t, [-1]), tf.reshape(nu, [-1])], axis=1)
        uv = self.call(X_input, self.nu_min_train, self.nu_max_train)
        return uv[:, 0:1], uv[:, 1:2]

    def predict_velocity_inverse(self, x, y, t, nu_val):
        X_input = tf.concat([x, y, t, nu_val * tf.ones_like(x)], axis=1)
        uv = self.call(X_input, self.nu_min_train, self.nu_max_train)
        return uv[:, 0:1], uv[:, 1:2]

    def compute_pde_residual(self, x, y, t, nu):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y, t])
            u, v = self.predict_velocity(x, y, t, nu)
            
            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)
            u_t = tape.gradient(u, t)
            v_x = tape.gradient(v, x)
            v_y = tape.gradient(v, y)
            v_t = tape.gradient(v, t)

        u_xx = tape.gradient(u_x, x)
        u_yy = tape.gradient(u_y, y)
        v_xx = tape.gradient(v_x, x)
        v_yy = tape.gradient(v_y, y)
        
        del tape

        f_u = u_t + u * u_x + v * u_y - nu * (u_xx + u_yy)
        f_v = v_t + u * v_x + v * v_y - nu * (v_xx + v_yy)
        return f_u, f_v

    @tf.function
    def train_step_adam(self, x_data, y_data, t_data, u_data, v_data, nu_data, num_pde_points):
        with tf.GradientTape() as tape:
            # Data loss is computed once
            u_pred_data, v_pred_data = self.predict_velocity(x_data, y_data, t_data, nu_data)
            loss_u_data = tf.reduce_mean(tf.square(u_data - u_pred_data))
            loss_v_data = tf.reduce_mean(tf.square(v_data - v_pred_data))
            data_loss = loss_u_data + loss_v_data

            # PDE loss is accumulated over smaller batches
            total_pde_loss = tf.constant(0.0, dtype=tf.float32)
            num_batches = tf.cast(tf.math.ceil(num_pde_points / self.pde_batch_size), dtype=tf.int32)
            
            for _ in tf.range(num_batches):
                x_pde_batch = tf.random.uniform((self.pde_batch_size, 1), self.x_min, self.x_max)
                y_pde_batch = tf.random.uniform((self.pde_batch_size, 1), self.y_min, self.y_max)
                t_pde_batch = tf.random.uniform((self.pde_batch_size, 1), self.t_min, self.t_max)
                nu_pde_batch = tf.random.uniform((self.pde_batch_size, 1), self.nu_min_train, self.nu_max_train)
                
                f_u_pred, f_v_pred = self.compute_pde_residual(x_pde_batch, y_pde_batch, t_pde_batch, nu_pde_batch)
                
                nu_weights = tf.exp(-self.sharpness_factor * (nu_pde_batch - self.nu_min_train) / (self.nu_max_train - self.nu_min_train))
                loss_f_u_pde = tf.reduce_mean(nu_weights * tf.square(f_u_pred))
                loss_f_v_pde = tf.reduce_mean(nu_weights * tf.square(f_v_pred))
                
                total_pde_loss += (loss_f_u_pde + loss_f_v_pde)
                
            avg_pde_loss = total_pde_loss / tf.cast(num_batches, tf.float32)
            
            total_loss = data_loss + avg_pde_loss

        self.optimizer_adam.apply_gradients(zip(tape.gradient(total_loss, self.trainable_variables), self.trainable_variables))
        return total_loss

    def compute_data_only_loss(self, x_data, y_data, t_data, u_data, v_data, nu_data):
        u_pred, v_pred = self.predict_velocity(x_data, y_data, t_data, nu_data)
        return tf.reduce_mean(tf.square(u_data - u_pred)) + tf.reduce_mean(tf.square(v_data - v_pred))

    def train_data_only(self, dataset, epochs):
        for epoch in range(epochs):
            for (x_batch, y_batch, t_batch, u_batch, v_batch, nu_batch) in dataset:
                with tf.GradientTape() as tape:
                    loss = self.compute_data_only_loss(x_batch, y_batch, t_batch, u_batch, v_batch, nu_batch)
                self.optimizer_adam_data_only.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))

    def fit(self, data_dataset, epochs_adam, epochs_data_only, num_pde_points):
        # self.train_data_only(data_dataset, epochs_data_only)

        for epoch in range(epochs_adam):
            last_batch_loss = 0.0
            for (x_batch, y_batch, t_batch, u_batch, v_batch, nu_batch) in data_dataset:
                self.train_step_adam(x_batch, y_batch, t_batch, u_batch, v_batch, nu_batch, num_pde_points)
            
            if epoch % 500 == 0:
                # Log only the data loss of the last batch for performance
                last_batch_data_loss = self.compute_data_only_loss(x_batch, y_batch, t_batch, u_batch, v_batch, nu_batch)
                print(f"Epoch {epoch}, Last Batch Data Loss: {last_batch_data_loss.numpy():.4f}")
    
    def compute_inverse_loss(self, x, y, t, u, v, nu):
        u_pred, v_pred = self.predict_velocity_inverse(x, y, t, nu)
        return tf.reduce_mean(tf.square(u - u_pred)) + tf.reduce_mean(tf.square(v - v_pred))

    def train_inverse_problem(self, x, y, t, u, v, epochs):
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                loss = self.compute_inverse_loss(x, y, t, u, v, tf.exp(self.log_nu_inverse))
            self.optimizer_adam_inverse.apply_gradients([(tape.gradient(loss, self.log_nu_inverse), self.log_nu_inverse)])
        return tf.exp(self.log_nu_inverse).numpy()

def generate_ground_truth_data(nx, ny, nt, dx, dy, dt, nu_val, u_initial, v_initial):
    u, v = tf.identity(u_initial), tf.identity(v_initial)
    u_snapshots, v_snapshots, t_snapshots = [], [], []
    for n in range(nt + 1):
        if n in [nt // 4, nt // 2, 3 * nt // 4, nt]:
            u_snapshots.append(u)
            v_snapshots.append(v)
            t_snapshots.append(tf.constant(n * dt, dtype=tf.float32))
        un, vn = tf.identity(u), tf.identity(v)
        u_int, v_int = un[1:-1, 1:-1], vn[1:-1, 1:-1]
        u_x, u_y = (un[1:-1, 2:] - un[1:-1, :-2]) / (2 * dx), (un[2:, 1:-1] - un[:-2, 1:-1]) / (2 * dy)
        v_x, v_y = (vn[1:-1, 2:] - vn[1:-1, :-2]) / (2 * dx), (vn[2:, 1:-1] - vn[:-2, 1:-1]) / (2 * dy)
        u_xx, u_yy = (un[1:-1, 2:] - 2 * u_int + un[1:-1, :-2]) / dx**2, (un[2:, 1:-1] - 2 * u_int + un[:-2, 1:-1]) / dy**2
        v_xx, v_yy = (vn[1:-1, 2:] - 2 * v_int + vn[1:-1, :-2]) / dx**2, (vn[2:, 1:-1] - 2 * v_int + vn[:-2, 1:-1]) / dy**2
        u_next = u_int - dt * (u_int * u_x + v_int * u_y) + dt * nu_val * (u_xx + u_yy)
        v_next = v_int - dt * (u_int * v_x + v_int * v_y) + dt * nu_val * (v_xx + v_yy)
        u = tf.tensor_scatter_nd_update(un, [[j, i] for j in range(1, ny-1) for i in range(1, nx-1)], tf.reshape(u_next, [-1]))
        v = tf.tensor_scatter_nd_update(vn, [[j, i] for j in range(1, ny-1) for i in range(1, nx-1)], tf.reshape(v_next, [-1]))
    return u_snapshots, v_snapshots, t_snapshots

def run_experiment(params):
    tf.random.set_seed(params['seed'])
    np.random.seed(params['seed'])
    
    nx, ny, nt = 41, 41, 50
    x_min, x_max, y_min, y_max = 0.0, 2.0, 0.0, 2.0
    t_min, t_max = 0.0, nt * 0.001

    # --- Load Unified Dataset ---
    print("Loading unified dataset...")
    try:
        data = np.load('results/1data/unified_dataset.npz')
        t_data_tf = tf.constant(data['t'], dtype=tf.float32)
        x_data_tf = tf.constant(data['x'], dtype=tf.float32)
        y_data_tf = tf.constant(data['y'], dtype=tf.float32)
        u_data_tf = tf.constant(data['u'], dtype=tf.float32)
        v_data_tf = tf.constant(data['v'], dtype=tf.float32)
        nu_data_tf = tf.constant(data['nu'], dtype=tf.float32)
    except FileNotFoundError:
        print("[ERROR] Unified dataset not found. Please run generate_unified_dataset.py.")
        return 100.0 # Return a high error

    pinn = PINN_Burgers2D([4] + [params['neurons']] * params['layers'] + [2],
                         x_data_tf, y_data_tf, t_data_tf, u_data_tf, v_data_tf,
                         None, None, None, None, x_min, x_max, y_min, y_max, t_min, t_max,
                         0.01, 0.1, nu_data_tf, learning_rate=params['learning_rate'], batch_size=1024)
    
    data_dataset = tf.data.Dataset.from_tensor_slices(
        (x_data_tf, y_data_tf, t_data_tf, u_data_tf, v_data_tf, nu_data_tf)
    ).shuffle(buffer_size=x_data_tf.shape[0]).batch(pinn.batch_size)

    pinn.fit(data_dataset, params['adam_epochs_stage1'], params.get('epochs_data_only_stage1', 100), params['num_pde_points_stage1'])
    
    # --- Inverse Problem Data Generation ---
    nu_true_inverse = params['nu_true']
    dx, dy, dt = (x_max - x_min) / (nx - 1), (y_max - y_min) / (ny - 1), 0.001
    x_np, y_np = np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny)
    X_np, Y_np = np.meshgrid(x_np, y_np)
    u_initial_np = np.exp(-((X_np - 1)**2 / 0.5**2 + (Y_np - 1)**2 / 0.5**2)).astype(np.float32)
    v_initial_np = np.exp(-((X_np - 1)**2 / 0.5**2 + (Y_np - 1)**2 / 0.5**2)).astype(np.float32)

    u_snaps_inv, v_snaps_inv, t_snaps_inv = generate_ground_truth_data(
        nx, ny, nt, dx, dy, dt, nu_true_inverse, u_initial_np, v_initial_np)
    
    x_inv = tf.constant(X_np.flatten()[:, None].astype(np.float32))
    y_inv = tf.constant(Y_np.flatten()[:, None].astype(np.float32))
    t_inv = tf.constant(np.full_like(x_inv, t_snaps_inv[-1].numpy()).astype(np.float32))
    u_inv = tf.constant(u_snaps_inv[-1].numpy().flatten()[:, None].astype(np.float32))
    v_inv = tf.constant(v_snaps_inv[-1].numpy().flatten()[:, None].astype(np.float32))
    
    nu_discovered = pinn.train_inverse_problem(x_inv, y_inv, t_inv, u_inv, v_inv, params.get('epochs_inverse_adam_pretrain', 100))
    
    error = np.abs(nu_discovered - nu_true_inverse) / nu_true_inverse * 100
    print(f"Discovered nu: {nu_discovered:.6f}, True nu: {nu_true_inverse:.6f}, Error: {error:.4f}%")
    return error

trials = Trials()

def objective(params):
    for key in ['adam_epochs_stage1', 'epochs_data_only_stage1', 'num_pde_points_stage1', 'epochs_inverse_adam_pretrain', 'num_datasets_gene', 'neurons', 'layers']:
        if key in params:
            params[key] = int(params[key])
    error = run_experiment(params)
    return {'loss': error, 'status': STATUS_OK, 'params': params}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified PINN script for 2D Burgers' equation with optional Hyperopt.")
    parser.add_argument('--optimize', action='store_true', help='Run hyperparameter optimization.')
    parser.add_argument('--trials_file', type=str, default='results/hopt/hyperopt_trials.pkl', help='Path to save/load the hyperopt trials object.')
    # Add other arguments for single run
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--nu_true', type=float, default=0.05)
    # ... add all other relevant parameters ...
    args, _ = parser.parse_known_args()

    if args.optimize:
        print(f"--- STARTING HYPERPARAMETER OPTIMIZATION (Saving to {args.trials_file}) ---")
        
        # Load trials object if it exists
        try:
            with open(args.trials_file, 'rb') as f:
                trials = pickle.load(f)
            print(f"Loaded {len(trials.trials)} existing trials from {args.trials_file}")
        except FileNotFoundError:
            trials = Trials()
            print(f"No existing trials file found at {args.trials_file}. Creating new.")

        space = {
            'seed': hp.choice('seed', [7]), # Matching V2 Best Seed
            'nu_true': hp.choice('nu_true', [0.05]),
            'noise_level': hp.choice('noise_level', [0.0]),
            'adam_epochs_stage1': hp.choice('adam_epochs_stage1', [1000]), # Reduced for speed
            'epochs_data_only_stage1': hp.choice('epochs_data_only_stage1', [100]),
            'num_pde_points_stage1': hp.choice('num_pde_points_stage1', [10000]),
            'epochs_inverse_adam_pretrain': hp.choice('epochs_inverse_adam_pretrain', [500]),
            'num_datasets_gene': hp.choice('num_datasets_gene', [3]), # Reduced for memory safety
            'neurons': hp.choice('neurons', [50]), # Match V2
            'layers': hp.choice('layers', [4]),    # Match V2
            'learning_rate': hp.choice('learning_rate', [0.000229]) # Match V2 Champion LR
        }
        
        max_evals_per_run = 1

        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=len(trials.trials) + max_evals_per_run, trials=trials)

        print("\n" + "="*50)
        print("Hyperparameter optimization run finished.")
        
        # Save the updated trials object
        with open(args.trials_file, 'wb') as f:
            pickle.dump(trials, f)
        print(f"Saved trials to {args.trials_file}")

    else:
        # Placeholder for single run logic
        print("--- STARTING SINGLE RUN ---")
        # This part needs to be fleshed out if you want to run single experiments from this script
        pass