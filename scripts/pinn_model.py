
"""
This module defines the core components of the Physics-Informed Neural Network (PINN)
for solving the 2D Burgers' equation inverse problem. It includes the model
architecture, loss functions, and the data generation logic.
"""

import tensorflow as tf
import numpy as np
import time
import scipy.optimize

# --- Swish Activation Function ---
def swish_activation(x):
    return tf.keras.activations.swish(x)

# --- PINN for 2D Burgers' Equation ---

class PINN_Burgers2D(tf.keras.Model):
    def __init__(self, layers_config, x_data, y_data, t_data, u_data, v_data,
                 x_pde, y_pde, t_pde, nu_pde, x_min, x_max, y_min, y_max, t_min, t_max,
                 nu_min_train, nu_max_train, true_kinematic_viscosity, annealing_rate=0.01, learning_rate=0.001, pde_batch_size=4096):
        super().__init__()
        self.network_layers = layers_config
        self.annealing_rate = annealing_rate
        self.sharpness_factor = 5.0
        self.pde_batch_size = pde_batch_size
        self.weight_data = tf.Variable(1.0, dtype=tf.float32, trainable=False, name="weight_data")
        self.weight_pde = tf.Variable(1.0, dtype=tf.float32, trainable=False, name="weight_pde")
        self.x_data, self.y_data, self.t_data = x_data, y_data, t_data
        self.u_data, self.v_data = u_data, v_data
        self.x_pde, self.y_pde, self.t_pde = None, None, None
        self.nu_pde = None
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.t_min, self.t_max = t_min, t_max
        self.nu_min_train, self.nu_max_train = nu_min_train, nu_max_train
        self.true_kinematic_viscosity = tf.Variable(true_kinematic_viscosity, dtype=tf.float32, trainable=False, name="true_kinematic_viscosity")
        self.nu_regularization_weight = 1e-3
        self.dense_layers = []
        for i in range(len(self.network_layers) - 1):
            self.dense_layers.append(tf.keras.layers.Dense(
                self.network_layers[i+1], activation=swish_activation if i < len(self.network_layers) - 2 else None,
                kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-5)))
        self.initial_learning_rate = learning_rate
        self.optimizer_adam = tf.keras.optimizers.Adam(learning_rate=self.initial_learning_rate)
        self.optimizer = self.optimizer_adam
        self.lbfgs_iter = 0
        self.optimizer_adam_data_only = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.log_nu_inverse = tf.Variable(tf.math.log(0.02), dtype=tf.float32, name="log_nu_inverse")
        self.nu_inverse = tf.exp(self.log_nu_inverse)
        self.trainable_variables_inverse = [self.log_nu_inverse]
        self.optimizer_adam_inverse = tf.keras.optimizers.Adam(learning_rate=0.001)

    def set_training_data(self, u_data, v_data, nu_true):
        self.u_data = u_data
        self.v_data = v_data
        self.true_kinematic_viscosity.assign(nu_true)

    def call(self, X, nu_min, nu_max):
        x_scaled = 2.0 * (X[:, 0:1] - self.x_min) / (self.x_max - self.x_min) - 1.0
        y_scaled = 2.0 * (X[:, 1:2] - self.y_min) / (self.y_max - self.y_min) - 1.0
        t_scaled = 2.0 * (X[:, 2:3] - self.t_min) / (self.t_max - self.t_min) - 1.0
        nu_scaled = 2.0 * (X[:, 3:4] - nu_min) / (nu_max - nu_min) - 1.0
        H = tf.concat([x_scaled, y_scaled, t_scaled, nu_scaled], axis=1)
        for layer in self.dense_layers: H = layer(H)
        return H

    def predict_velocity(self, x, y, t, nu):
        x, y, t, nu = tf.cast(x, tf.float32), tf.cast(y, tf.float32), tf.cast(t, tf.float32), tf.cast(nu, tf.float32)
        x_reshaped, y_reshaped, t_reshaped, nu_reshaped = tf.reshape(x, [-1]), tf.reshape(y, [-1]), tf.reshape(t, [-1]), tf.reshape(nu, [-1])
        X_input = tf.stack([x_reshaped, y_reshaped, t_reshaped, nu_reshaped], axis=1)
        uv = self.call(X_input, self.nu_min_train, self.nu_max_train)
        return uv[:, 0:1], uv[:, 1:2]

    def predict_velocity_inverse(self, x, y, t, nu_val):
        X_input = tf.concat([x, y, t, nu_val * tf.ones_like(x)], axis=1)
        uv = self.call(X_input, self.nu_min_train, self.nu_max_train)
        return uv[:, 0:1], uv[:, 1:2]

    @tf.function
    def compute_pde_residual(self, x, y, t, nu):
        # --- PARTE 1: Derivadas de U (Memória Limpa) ---
        with tf.GradientTape() as tape_uxx:
            tape_uxx.watch(x)
            with tf.GradientTape() as tape_uyy:
                tape_uyy.watch(y)
                with tf.GradientTape() as tape_1st:
                    tape_1st.watch([x, y, t])
                    u, _ = self.predict_velocity(x, y, t, nu) # Ignora v aqui
                
                # 1ª ordem (destroi tape_1st)
                grads_u = tape_1st.gradient(u, [x, y, t])
                u_x, u_y, u_t = grads_u[0], grads_u[1], grads_u[2]
            
            # 2ª ordem Y (destroi tape_uyy)
            u_yy = tape_uyy.gradient(u_y, y)
        
        # 2ª ordem X (destroi tape_uxx)
        u_xx = tape_uxx.gradient(u_x, x)

        # --- PARTE 2: Derivadas de V (Recomputação Segura) ---
        with tf.GradientTape() as tape_vxx:
            tape_vxx.watch(x)
            with tf.GradientTape() as tape_vyy:
                tape_vyy.watch(y)
                with tf.GradientTape() as tape_1st_v:
                    tape_1st_v.watch([x, y, t])
                    _, v = self.predict_velocity(x, y, t, nu) # Ignora u aqui
                
                # 1ª ordem (destroi tape_1st_v)
                grads_v = tape_1st_v.gradient(v, [x, y, t])
                v_x, v_y, v_t = grads_v[0], grads_v[1], grads_v[2]

            # 2ª ordem Y (destroi tape_vyy)
            v_yy = tape_vyy.gradient(v_y, y)
        
        # 2ª ordem X (destroi tape_vxx)
        v_xx = tape_vxx.gradient(v_x, x)

        # --- Equações de Burgers ---
        f_u = u_t + u * u_x + v * u_y - nu * (u_xx + u_yy)
        f_v = v_t + u * v_x + v * v_y - nu * (v_xx + v_yy)

        return f_u, f_v

    @tf.function
    def compute_loss(self, x_pde, y_pde, t_pde, nu_pde):
        u_pred_data, v_pred_data = self.predict_velocity(
            self.x_data, self.y_data, self.t_data, tf.fill(tf.shape(self.x_data), self.true_kinematic_viscosity))
        loss_u_data = tf.reduce_mean(tf.square(self.u_data - u_pred_data))
        loss_v_data = tf.reduce_mean(tf.square(self.v_data - v_pred_data))
        f_u_pred, f_v_pred = self.compute_pde_residual(x_pde, y_pde, t_pde, nu_pde)
        nu_weights = tf.exp(-self.sharpness_factor * (nu_pde - self.nu_min_train) / (self.nu_max_train - self.nu_min_train))
        loss_f_u_pde = tf.reduce_mean(nu_weights * tf.square(f_u_pred))
        loss_f_v_pde = tf.reduce_mean(nu_weights * tf.square(f_v_pred))
        combined_data_loss = loss_u_data + loss_v_data
        combined_pde_loss = loss_f_u_pde + loss_f_v_pde
        total_loss = self.weight_data * combined_data_loss + self.weight_pde * combined_pde_loss
        return total_loss, combined_data_loss, combined_pde_loss, loss_f_u_pde, loss_f_v_pde

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float32), tf.TensorSpec(shape=[None, 1], dtype=tf.float32), tf.TensorSpec(shape=[None, 1], dtype=tf.float32), tf.TensorSpec(shape=[None, 1], dtype=tf.float32)])
    def train_step_adam(self, x_pde_batch, y_pde_batch, t_pde_batch, nu_pde_batch):
        with tf.GradientTape() as tape:
            total_loss, _, _, _, _ = self.compute_loss(x_pde_batch, y_pde_batch, t_pde_batch, nu_pde_batch)
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer_adam.apply_gradients(zip(gradients, self.trainable_variables))
        return total_loss

    def compute_data_only_loss(self):
        u_pred_data, v_pred_data = self.predict_velocity(
            self.x_data, self.y_data, self.t_data, tf.fill(tf.shape(self.x_data), self.true_kinematic_viscosity))
        return tf.reduce_mean(tf.square(self.u_data - u_pred_data)) + tf.reduce_mean(tf.square(self.v_data - v_pred_data))

    def train_data_only(self, epochs_data_only, training_datasets):
        print(f"Starting Data-Only Pre-training for {epochs_data_only} epochs...")
        start_time_data_only = time.time()
        for epoch in range(epochs_data_only):
            selected_dataset = training_datasets[np.random.randint(0, len(training_datasets))]
            u_data_epoch, v_data_epoch, nu_true_epoch = selected_dataset
            self.set_training_data(u_data_epoch, v_data_epoch, nu_true_epoch)
            with tf.GradientTape() as tape:
                current_loss_data = self.compute_data_only_loss()
            gradients = tape.gradient(current_loss_data, self.trainable_variables)
            self.optimizer_adam_data_only.apply_gradients(zip(gradients, self.trainable_variables))
            if epoch % 1000 == 0: print(f"Data-Only Epoch {epoch}: Data Loss = {current_loss_data:.6f} (using nu_true={nu_true_epoch:.4f})")
            if tf.math.is_nan(current_loss_data):
                print(f"[ERROR] Data-Only Loss is NaN at epoch {epoch}. Stopping training.")
                break
        duration_data_only = time.time() - start_time_data_only
        print(f"Data-Only Pre-training finished in {duration_data_only:.2f} seconds.")
        return duration_data_only

    def fit(self, epochs_adam, epochs_data_only, num_pde_points, x_min, x_max, y_min, y_max, t_min, t_max, training_datasets, callbacks=None, pde_batch_size=4096):
        duration_data_only = self.train_data_only(epochs_data_only, training_datasets)
        print("Starting Adam training (Full Loss with Curriculum and Data Sampling)...")
        start_time_adam = time.time()
        if callbacks:
            for callback in callbacks:
                callback.set_model(self)
                callback.on_train_begin()
        nu_start_range = 0.05
        total_nu_range = self.nu_max_train - self.nu_min_train
        for epoch in range(epochs_adam):
            selected_dataset = training_datasets[np.random.randint(0, len(training_datasets))]
            u_data_epoch, v_data_epoch, nu_true_epoch = selected_dataset
            self.set_training_data(u_data_epoch, v_data_epoch, nu_true_epoch)
            progress = epoch / epochs_adam
            current_nu_max = self.nu_min_train + (total_nu_range * progress)
            current_nu_max = max(current_nu_max, self.nu_min_train + nu_start_range)
            x_pde_full = tf.constant(np.random.uniform(x_min, x_max, (num_pde_points, 1)).astype(np.float32))
            y_pde_full = tf.constant(np.random.uniform(y_min, y_max, (num_pde_points, 1)).astype(np.float32))
            t_pde_full = tf.constant(np.random.uniform(t_min, t_max, (num_pde_points, 1)).astype(np.float32))
            nu_pde_full = tf.constant(np.random.uniform(self.nu_min_train, current_nu_max, (num_pde_points, 1)).astype(np.float32))
            pde_dataset = tf.data.Dataset.from_tensor_slices((x_pde_full, y_pde_full, t_pde_full, nu_pde_full))
            pde_dataset_batched = pde_dataset.batch(pde_batch_size)
            for (x_pde_batch, y_pde_batch, t_pde_batch, nu_pde_batch) in pde_dataset_batched:
                self.train_step_adam(x_pde_batch, y_pde_batch, t_pde_batch, nu_pde_batch)
            if epoch % 500 == 0:
                current_total_loss, current_combined_data_loss, current_combined_pde_loss, _, _ = self.compute_loss(
                    x_pde_full, y_pde_full, t_pde_full, nu_pde_full)
                print(f"Adam Epoch {epoch}: Loss = {current_total_loss:.6f}, Data Loss = {current_combined_data_loss:.6f} (nu_true={nu_true_epoch:.4f}), PDE Loss = {current_combined_pde_loss:.6f}")
                if tf.math.is_nan(current_total_loss):
                    print(f"[ERROR] Adam Total Loss is NaN at epoch {epoch}. Stopping training.")
                    break
                if callbacks: 
                    logs = {'loss': current_total_loss.numpy(), 'lr': self.optimizer_adam.learning_rate.numpy()}
                    for callback in callbacks: callback.on_epoch_end(epoch, logs=logs)
        adam_duration = time.time() - start_time_adam
        if callbacks:
            for callback in callbacks: callback.on_train_end()
        print(f"\nAdam training finished in {adam_duration:.2f} seconds.")
        return duration_data_only, adam_duration, 0.0 # No LBFGS in this simplified fit
    
    def compute_inverse_loss(self, x_data_inv, y_data_inv, t_data_inv, u_data_inv, v_data_inv, nu_val_for_loss):
        u_pred_inverse, v_pred_inverse = self.predict_velocity_inverse(
            x_data_inv, y_data_inv, t_data_inv, nu_val_for_loss)
        loss_inverse = tf.reduce_mean(tf.square(u_data_inv - u_pred_inverse)) + tf.reduce_mean(tf.square(v_data_inv - v_pred_inverse))
        nu_min_physical, nu_max_physical = 0.001, 0.1
        penalty_lower = tf.maximum(0.0, nu_min_physical - nu_val_for_loss)
        penalty_upper = tf.maximum(0.0, nu_val_for_loss - nu_max_physical)
        nu_reg_loss = self.nu_regularization_weight * (tf.square(penalty_lower) + tf.square(penalty_upper))
        return loss_inverse + nu_reg_loss

    def loss_and_grads_scipy_inverse(self, flat_log_nu, x_data_inv_np, y_data_inv_np, t_data_inv_np, u_data_inv_np, v_data_inv_np):
        self.log_nu_inverse.assign(tf.constant(flat_log_nu[0], dtype=tf.float32))
        x_data_inv, y_data_inv, t_data_inv, u_data_inv, v_data_inv = tf.constant(x_data_inv_np, dtype=tf.float32), tf.constant(y_data_inv_np, dtype=tf.float32), tf.constant(t_data_inv_np, dtype=tf.float32), tf.constant(u_data_inv_np, dtype=tf.float32), tf.constant(v_data_inv_np, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(self.log_nu_inverse)
            nu_inverse_traced = tf.exp(self.log_nu_inverse)
            loss_inverse = self.compute_inverse_loss(
                x_data_inv, y_data_inv, t_data_inv, u_data_inv, v_data_inv, nu_inverse_traced)
        grad = tape.gradient(loss_inverse, self.log_nu_inverse)
        return loss_inverse.numpy().astype(np.float64), np.array([grad.numpy()]).astype(np.float64)

    def train_inverse_problem(self, x_data_inv, y_data_inv, t_data_inv, u_data_inv, v_data_inv, epochs_inverse_adam, epochs_inverse_adam_pretrain):
        print("\n" + "-" * 50)
        print("Starting Stage 2: Inverse Problem (Discovering nu with Hybrid Optimization)...")
        print(f"Starting Adam pre-training for nu_inverse for {epochs_inverse_adam_pretrain} epochs...")
        start_time_adam_inverse_pretrain = time.time()
        for epoch in range(epochs_inverse_adam_pretrain):
            with tf.GradientTape() as tape:
                tape.watch(self.log_nu_inverse)
                nu_inverse_traced = tf.exp(self.log_nu_inverse)
                loss_inverse_adam = self.compute_inverse_loss(
                    x_data_inv, y_data_inv, t_data_inv, u_data_inv, v_data_inv, nu_inverse_traced)
            gradients_inverse_adam = tape.gradient(loss_inverse_adam, self.log_nu_inverse)
            self.optimizer_adam_inverse.apply_gradients([(gradients_inverse_adam, self.log_nu_inverse)])
            if epoch % 100 == 0: print(f"  Adam Inverse Pre-train Epoch {epoch}: Loss = {loss_inverse_adam:.6f}, Discovered nu = {tf.exp(self.log_nu_inverse).numpy():.6f}")
            if tf.math.is_nan(loss_inverse_adam):
                print(f"[ERROR] Adam Inverse Pre-train Loss is NaN at epoch {epoch}. Stopping pre-training.")
                break
        adam_inverse_pretrain_duration = time.time() - start_time_adam_inverse_pretrain
        print(f"Adam pre-training for nu_inverse finished in {adam_inverse_pretrain_duration:.2f} seconds.")
        print("Starting L-BFGS-B optimization for nu_inverse...")
        start_time_inverse_lbfgs = time.time()
        initial_log_nu = [self.log_nu_inverse.numpy()]
        scipy_results_inverse = scipy.optimize.minimize(
            fun=self.loss_and_grads_scipy_inverse, x0=initial_log_nu, method='L-BFGS-B', jac=True,
            args=(x_data_inv, y_data_inv, t_data_inv, u_data_inv, v_data_inv),
            options={'maxiter': 500000, 'maxfun': 500000, 'maxcor': 50, 'maxls': 50, 'ftol': 1e-10})
        inverse_lbfgs_duration = time.time() - start_time_inverse_lbfgs
        self.log_nu_inverse.assign(tf.constant(scipy_results_inverse.x[0], dtype=tf.float32))
        print(f"Stage 2 (Inverse Problem) L-BFGS-B training finished in {inverse_lbfgs_duration:.2f} seconds.")
        print(f"L-BFGS-B converged: {scipy_results_inverse.success}")
        print(f"L-BFGS-B message: {scipy_results_inverse.message}")
        print(f"L-BFGS-B iterations: {scipy_results_inverse.nit}")
        return tf.exp(self.log_nu_inverse).numpy(), inverse_lbfgs_duration, adam_inverse_pretrain_duration


def generate_ground_truth_data(nx, ny, nt, dx, dy, dt, nu_val, u_initial, v_initial):
    u, v = tf.identity(u_initial), tf.identity(v_initial)
    u_snapshots, v_snapshots, t_snapshots = [], [], []
    for n in range(nt + 1):
        if n in [int(nt / 4), int(nt / 2), int(3 * nt / 4), nt]:
            u_snapshots.append(u)
            v_snapshots.append(v)
            t_snapshots.append(tf.constant(n * dt, dtype=tf.float32))
        un, vn = tf.identity(u), tf.identity(v)
        u_int, v_int = un[1:-1, 1:-1], vn[1:-1, 1:-1]
        u_x = (un[1:-1, 2:] - un[1:-1, :-2]) / (2 * dx)
        u_y = (un[2:, 1:-1] - un[:-2, 1:-1]) / (2 * dy)
        v_x = (vn[1:-1, 2:] - vn[1:-1, :-2]) / (2 * dx)
        v_y = (vn[2:, 1:-1] - vn[:-2, 1:-1]) / (2 * dy)
        u_xx = (un[1:-1, 2:] - 2 * u_int + un[1:-1, :-2]) / dx**2
        u_yy = (un[2:, 1:-1] - 2 * u_int + un[:-2, 1:-1]) / dy**2
        v_xx = (vn[1:-1, 2:] - 2 * v_int + vn[1:-1, :-2]) / dx**2
        v_yy = (vn[2:, 1:-1] - 2 * v_int + vn[:-2, 1:-1]) / dy**2
        u_next = u_int - dt * (u_int * u_x + v_int * u_y) + dt * nu_val * (u_xx + u_yy)
        v_next = v_int - dt * (u_int * v_x + v_int * v_y) + dt * nu_val * (v_xx + v_yy)
        u = tf.tensor_scatter_nd_update(un, [[j, i] for j in range(1, ny-1) for i in range(1, nx-1)], tf.reshape(u_next, [-1]))
        v = tf.tensor_scatter_nd_update(vn, [[j, i] for j in range(1, ny-1) for i in range(1, nx-1)], tf.reshape(v_next, [-1]))
    return u_snapshots, v_snapshots, t_snapshots
