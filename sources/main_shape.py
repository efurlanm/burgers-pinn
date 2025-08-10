# -*- coding: utf-8 -*-
"""
@author: Gemini

A Physics-Informed Neural Network (PINN) for the discovery of kinematic viscosity
in the 2D Burgers' equation.

This script implements a PINN using a hybrid optimization approach. It starts with
the Adam optimizer for robust initial convergence and switches to SciPy's L-BFGS-B
optimizer for fine-tuning. The primary goal is to accurately discover the
kinematic viscosity parameter (`nu`) of the 2D Burgers' equation.

This script is structured as follows:
1.  **Parameters**: Defines the physical domain, grid properties, and neural network architecture.
2.  **PINN_Burgers2D Class**: Implements the core PINN model.
    -   Initializes the neural network and the discoverable parameter `nu`.
    -   Defines the loss function, combining data fidelity and PDE residual losses.
    -   Implements the two-stage training process (Adam and L-BFGS-B).
3.  **Data Generation**:
    -   Uses an internal Finite Difference Method (FDM) solver (using TensorFlow ops)
      to generate the ground truth data for training.
    -   Prepares the data by flattening and selecting points for training.
4.  **Main Execution**:
    -   Initializes and trains the PINN model.
    -   Prints the final discovered `nu` value.
    -   Saves the final results to a `.npz` file in the `results/` directory.
"""

import tensorflow as tf
import numpy as np
import time
import scipy.optimize
import tensorflow.compat.v1 as tf_v1
import os
import argparse

# Disable eager execution for TensorFlow 1.x compatibility
tf_v1.disable_eager_execution()

# --- PINN for 2D Burgers' Equation ---

class PINN_Burgers2D:
    """
    A Physics-Informed Neural Network for the 2D Burgers' Equation.
    """
    def __init__(self, layers, true_nu, x_data, y_data, t_data, u_data, v_data,
                 x_pde, y_pde, t_pde, x_min, x_max, y_min, y_max, t_min, t_max,
                 lambda_data, lambda_pde):
        """
        Initializes the PINN model.
        """
        self.layers = layers
        self.true_nu = true_nu
        self.lambda_data = lambda_data
        self.lambda_pde = lambda_pde

        # Store training (data) and collocation (pde) points
        self.x_data, self.y_data, self.t_data = x_data, y_data, t_data
        self.u_data, self.v_data = u_data, v_data
        self.x_pde, self.y_pde, self.t_pde = x_pde, y_pde, t_pde

        # Store domain bounds for input scaling (normalization)
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.t_min, self.t_max = t_min, t_max

        # Initialize the neural network weights and biases
        self.weights, self.biases = self.initialize_neural_network(self.layers)

        # Discoverable parameter: log of kinematic viscosity for stability
        self.log_nu_pinn = tf.Variable(tf.math.log(0.06), dtype=tf.float32,
                                       name="log_nu_pinn")

        # Define the list of all trainable variables
        self.trainable_variables = self.weights + self.biases + [self.log_nu_pinn]

        # Define loss, gradients, and the Adam optimizer training operation
        self.loss_op = self.compute_loss()
        self.gradient_op = tf_v1.gradients(self.loss_op, self.trainable_variables)

        # TensorFlow session setup
        self.session = tf_v1.Session(config=tf_v1.ConfigProto(
            allow_soft_placement=True, log_device_placement=False))

        # Define loss, gradients, and the Adam optimizer training operation
        self.train_op_adam = tf_v1.train.AdamOptimizer(learning_rate=0.001).minimize(
            self.loss_op, var_list=self.trainable_variables)

        # Define data-only loss and optimizer for pre-training (DG-PINN Phase 1)
        u_pred_data, v_pred_data = self.predict_velocity(
            self.x_data, self.y_data, self.t_data)
        self.loss_data_only_op = (tf.reduce_mean(tf.square(self.u_data - u_pred_data)) +
                                  tf.reduce_mean(tf.square(self.v_data - v_pred_data)))
        self.train_op_adam_data_only = tf_v1.train.AdamOptimizer(learning_rate=0.001).minimize(
            self.loss_data_only_op, var_list=self.trainable_variables)

        # Initialize all TensorFlow variables
        self.session.run(tf_v1.global_variables_initializer())

    def initialize_neural_network(self, layers):
        """
        Initializes weights and biases using Xavier initialization.
        """
        weights, biases = [], []
        for l in range(len(layers) - 1):
            weight = self.xavier_initializer(size=[layers[l], layers[l + 1]])
            bias = tf.Variable(tf.zeros([1, layers[l + 1]]), dtype=tf.float32)
            weights.append(weight)
            biases.append(bias)
        return weights, biases

    def xavier_initializer(self, size):
        """
        Implements Xavier (or Glorot) initialization for network weights.
        """
        in_dim, out_dim = size[0], size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal(
            [in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_network_model(self, X):
        """
        Defines the forward pass of the neural network.
        """
        # Input scaling to the range [-1, 1] for better training performance
        x_scaled = 2.0 * (X[:, 0:1] - self.x_min) / (self.x_max - self.x_min) - 1.0
        y_scaled = 2.0 * (X[:, 1:2] - self.y_min) / (self.y_max - self.y_min) - 1.0
        t_scaled = 2.0 * (X[:, 2:3] - self.t_min) / (self.t_max - self.t_min) - 1.0
        H = tf.concat([x_scaled, y_scaled, t_scaled], axis=1)

        # Forward pass through the network layers
        for l in range(len(self.weights) - 1):
            W, b = self.weights[l], self.biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W, b = self.weights[-1], self.biases[-1]
        Y = tf.add(tf.matmul(H, W), b) # Final layer is linear
        return Y

    def predict_velocity(self, x, y, t):
        """
        Predicts the velocity components (u, v) at given (x, y, t) coordinates.
        """
        X_input = tf.concat([x, y, t], axis=1)
        uv = self.neural_network_model(X_input)
        return uv[:, 0:1], uv[:, 1:2]

    def compute_pde_residual(self, x, y, t):
        """
        Computes the residual of the 2D Burgers' equation using automatic differentiation.
        """
        u, v = self.predict_velocity(x, y, t)

        # First-order derivatives
        u_t = tf_v1.gradients(u, t)[0]
        u_x = tf_v1.gradients(u, x)[0]
        u_y = tf_v1.gradients(u, y)[0]
        v_t = tf_v1.gradients(v, t)[0]
        v_x = tf_v1.gradients(v, x)[0]
        v_y = tf_v1.gradients(v, y)[0]

        # Second-order derivatives
        u_xx = tf_v1.gradients(u_x, x)[0]
        u_yy = tf_v1.gradients(u_y, y)[0]
        v_xx = tf_v1.gradients(v_x, x)[0]
        v_yy = tf_v1.gradients(v_y, y)[0]

        # PDE residuals (f_u and f_v should be zero for a perfect solution)
        nu = tf.exp(self.log_nu_pinn)
        f_u = u_t + u * u_x + v * u_y - nu * (u_xx + u_yy)
        f_v = v_t + u * v_x + v * v_y - nu * (v_xx + v_yy)
        return f_u, f_v

    def compute_loss(self):
        """
        Computes the total loss, which is a weighted sum of data and PDE losses.
        """
        # Data loss (mean squared error between predicted and training data)
        u_pred_data, v_pred_data = self.predict_velocity(
            self.x_data, self.y_data, self.t_data)
        loss_data = (tf.reduce_mean(tf.square(self.u_data - u_pred_data)) +
                     tf.reduce_mean(tf.square(self.v_data - v_pred_data)))

        # PDE loss (mean squared error of the PDE residuals at collocation points)
        f_u_pred, f_v_pred = self.compute_pde_residual(
            self.x_pde, self.y_pde, self.t_pde)
        loss_pde = (tf.reduce_mean(tf.square(f_u_pred)) +
                    tf.reduce_mean(tf.square(f_v_pred)))

        # Total loss with weighting (hyperparameters to tune)
        return self.lambda_data * loss_data + self.lambda_pde * loss_pde

    def train_step_adam(self):
        """
        Performs a single training step using the Adam optimizer.
        """
        self.session.run(self.train_op_adam)

    def train_data_only(self, epochs_data_only):
        """
        Performs training using only the data loss for a specified number of epochs.
        """
        print(f"Starting Data-Only Pre-training for {epochs_data_only} epochs...")
        start_time_data_only = time.time()
        for epoch in range(epochs_data_only):
            self.session.run(self.train_op_adam_data_only)
            if epoch % 1000 == 0:
                current_loss_data = self.session.run(self.loss_data_only_op)
                print(f"Data-Only Epoch {epoch}: Data Loss = {current_loss_data:.6f}")
        end_time_data_only = time.time()
        duration_data_only = end_time_data_only - start_time_data_only
        print(f"Data-Only Pre-training finished in {duration_data_only:.2f} seconds.")
        return duration_data_only

    def get_flat_variables(self):
        """
        Returns all trainable variables as a single flattened NumPy array.
        """
        return self.session.run(tf.concat([tf.reshape(var, [-1])
                                         for var in self.trainable_variables], axis=0))

    def set_flat_variables(self, flat_variables):
        """
        Assigns new values to the trainable variables from a flattened array.
        """
        assign_ops = []
        idx = 0
        for var in self.trainable_variables:
            shape = var.get_shape().as_list()
            size = int(np.prod(shape))
            assign_ops.append(tf_v1.assign(
                var, tf.reshape(flat_variables[idx:idx + size], shape)))
            idx += size
        return tf.group(*assign_ops)

    def loss_and_grads_scipy(self, flat_weights):
        """
        A wrapper function to compute loss and gradients for the SciPy L-BFGS-B optimizer.
        """
        # Assign the new weights from the optimizer
        self.session.run(self.set_flat_variables(
            tf.constant(flat_weights, dtype=tf.float32)))

        # Compute loss and gradients
        loss_value, grads_value = self.session.run([self.loss_op, self.gradient_op])

        # Flatten the gradients to a 1D array
        flat_grads = np.concatenate([grad.flatten() for grad in grads_value])

        # Print progress (can be verbose)
        # print(f"  L-BFGS-B: Loss = {loss_value:.6e}, "
        #       f"Grad Norm = {np.linalg.norm(flat_grads):.6e}, "
        #       f"nu_pinn_grad = {grads_value[-1]:.6e}")
        return loss_value.astype(np.float64), flat_grads.astype(np.float64)

    def train(self, epochs_adam, epochs_data_only):
        """
        Trains the PINN using a two-stage optimization: Data-Only Adam, then Full Adam, then L-BFGS-B.
        """
        # --- Data-Only Adam Optimization (Phase 1) ---
        duration_data_only = self.train_data_only(epochs_data_only)

        # --- Adam Optimization (Phase 2) ---
        print("Starting Adam training (Full Loss)...")
        start_time_adam = time.time()
        for epoch in range(epochs_adam):
            self.train_step_adam()
            if epoch % 100 == 0:
                current_loss, current_nu = self.session.run(
                    [self.loss_op, tf.exp(self.log_nu_pinn)])
                print(f"Adam Epoch {epoch}: Loss = {current_loss:.6f}, "
                      f"Discovered nu = {current_nu:.6f}")
        end_time_adam = time.time()
        adam_duration = end_time_adam - start_time_adam
        print(f"Adam training finished in {adam_duration:.2f} seconds.")

        # --- L-BFGS-B Optimization ---
        print("Starting L-BFGS-B training with SciPy...")
        start_time_lbfgs = time.time()
        initial_weights = self.get_flat_variables()

        scipy_results = scipy.optimize.minimize(
            fun=self.loss_and_grads_scipy,
            x0=initial_weights,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 100000, 'maxfun': 100000, 'maxcor': 100,
                     'maxls': 50, 'ftol': 1e-20}
        )
        end_time_lbfgs = time.time()
        lbfgs_duration = end_time_lbfgs - start_time_lbfgs

        self.session.run(self.set_flat_variables(
            tf.constant(scipy_results.x, dtype=tf.float32)))

        print(f"L-BFGS-B training finished in {lbfgs_duration:.2f} seconds.")
        print(f"L-BFGS-B converged: {scipy_results.success}")
        print(f"L-BFGS-B message: {scipy_results.message}")
        print(f"L-BFGS-B iterations: {scipy_results.nit}")
        
        return duration_data_only, adam_duration, lbfgs_duration


# --- Data Generation ---

def generate_ground_truth_data(nx, ny, nt, dx, dy, dt,
                               nu_val,
                               u_initial, v_initial):
    """
    Generates ground truth data using a TensorFlow-based Finite Difference Method.
    """
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

        u_next = (u_int - dt * (u_int * u_x + v_int * u_y) +
                  dt * nu_val * (u_xx + u_yy))
        v_next = (v_int - dt * (u_int * v_x + v_int * v_y) +
                  dt * nu_val * (v_xx + v_yy))

        u = tf.tensor_scatter_nd_update(
            un, [[j, i] for j in range(1, ny-1) for i in range(1, nx-1)],
            tf.reshape(u_next, [-1]))
        v = tf.tensor_scatter_nd_update(
            vn, [[j, i] for j in range(1, ny-1) for i in range(1, nx-1)],
            tf.reshape(v_next, [-1]))

    return u_snapshots, v_snapshots, t_snapshots

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Run PINN for Burgers Equation')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    seed_value = args.seed
    tf_v1.set_random_seed(seed_value)
    np.random.seed(seed_value)
    print(f"Running with seed: {seed_value}")
    # ---

    # --- Main Parameters ---

    # Grid and Time Configuration
    grid_points_x = 41
    grid_points_y = 41
    time_steps = 50
    true_kinematic_viscosity = 0.05  # Ground truth for 'nu'

    # Domain Boundaries
    x_min, x_max = 0.0, 2.0
    y_min, y_max = 0.0, 2.0
    t_min, t_max = 0.0, time_steps * 0.001

    # Neural Network Architecture
    # Input: (x, y, t), Output: (u, v)
    layers = [3, 60, 60, 60, 60, 2]
    
    # --- Experiment Configuration ---
    adam_epochs = 1000
    epochs_data_only = 10000 # New parameter for DG-PINN pre-training
    lambda_data_weight = 100.0
    lambda_pde_weight = 1.0
    num_pde_points = 60000
    # ---
    
    overall_start_time = time.time()

    # --- Data Preparation ---
    data_prep_start_time = time.time()
    x_np = np.linspace(x_min, x_max, grid_points_x)
    y_np = np.linspace(y_min, y_max, grid_points_y)
    X_np, Y_np = np.meshgrid(x_np, y_np)
    dx, dy, dt = x_np[1] - x_np[0], y_np[1] - y_np[0], 0.001

    center_x, center_y = 1.0, 1.0
    sigma_x, sigma_y = 0.25, 0.25
    u_initial_np = np.exp(-((X_np - center_x)**2 / (2 * sigma_x**2) +
                            (Y_np - center_y)**2 / (2 * sigma_y**2)))
    v_initial_np = np.exp(-((X_np - center_x)**2 / (2 * sigma_x**2) +
                            (Y_np - center_y)**2 / (2 * sigma_y**2)))
    u_initial_tf = tf.constant(u_initial_np, dtype=tf.float32)
    v_initial_tf = tf.constant(v_initial_np, dtype=tf.float32)

    true_nu_tf = tf.constant(true_kinematic_viscosity, dtype=tf.float32)
    u_true_tf, v_true_tf, t_true_tf = generate_ground_truth_data(
        grid_points_x, grid_points_y, time_steps, dx, dy, dt,
        true_nu_tf, u_initial_tf, v_initial_tf)

    with tf_v1.Session() as sess_data_gen:
        u_true_list, v_true_list, t_true_list = sess_data_gen.run(
            [u_true_tf, v_true_tf, t_true_tf])

    X_data_list, Y_data_list, T_data_list = [], [], []
    U_data_list, V_data_list = [], []

    for i in range(len(t_true_list)):
        X_data_list.append(X_np.flatten()[:, None])
        Y_data_list.append(Y_np.flatten()[:, None])
        T_data_list.append(np.full_like(X_np.flatten()[:, None], t_true_list[i]))
        U_data_list.append(u_true_list[i].flatten()[:, None])
        V_data_list.append(v_true_list[i].flatten()[:, None])

    X_data_flat = np.concatenate(X_data_list)
    Y_data_flat = np.concatenate(Y_data_list)
    T_data_flat = np.concatenate(T_data_list)
    U_data_flat = np.concatenate(U_data_list)
    V_data_flat = np.concatenate(V_data_list)

    x_pde = tf.constant(np.random.uniform(
        x_min, x_max, (num_pde_points, 1)), dtype=tf.float32)
    y_pde = tf.constant(np.random.uniform(
        y_min, y_max, (num_pde_points, 1)), dtype=tf.float32)
    t_pde = tf.constant(np.random.uniform(
        t_min, t_max, (num_pde_points, 1)), dtype=tf.float32)

    x_data_tf = tf.constant(X_data_flat, dtype=tf.float32)
    y_data_tf = tf.constant(Y_data_flat, dtype=tf.float32)
    t_data_tf = tf.constant(T_data_flat, dtype=tf.float32)
    u_data_tf = tf.constant(U_data_flat, dtype=tf.float32)
    v_data_tf = tf.constant(V_data_flat, dtype=tf.float32)
    data_prep_duration = time.time() - data_prep_start_time
    # ---

    # --- Model Initialization ---
    model_init_start_time = time.time()
    pinn = PINN_Burgers2D(
        layers, true_kinematic_viscosity, x_data_tf, y_data_tf, t_data_tf,
        u_data_tf, v_data_tf, x_pde, y_pde, t_pde,
        x_min, x_max, y_min, y_max, t_min, t_max,
        lambda_data_weight, lambda_pde_weight)
    model_init_duration = time.time() - model_init_start_time
    # ---

    print(f"Ground truth nu: {true_kinematic_viscosity}")
    initial_nu_guess = np.exp(pinn.session.run(pinn.log_nu_pinn))
    print(f"Initial guess for nu: {initial_nu_guess:.6f}")
    print(f"Training with: Adam Epochs = {adam_epochs}, Data-Only Epochs = {epochs_data_only}, Lambda Data = {lambda_data_weight}, Lambda PDE = {lambda_pde_weight}, PDE Points = {num_pde_points}")

    # --- Model Training ---
    duration_data_only, adam_duration, lbfgs_duration = pinn.train(epochs_adam=adam_epochs, epochs_data_only=epochs_data_only)
    # ---

    # --- Results and Saving ---
    final_nu = np.exp(pinn.session.run(pinn.log_nu_pinn))
    print("-" * 50)
    print(f"Final Discovered nu: {final_nu:.6f}")
    print(f"Ground Truth nu: {true_kinematic_viscosity}")

    X_plot_flat = X_np.flatten()[:, None]
    Y_plot_flat = Y_np.flatten()[:, None]
    T_plot_flat = np.full_like(X_plot_flat, t_max)

    u_pinn_pred_flat, v_pinn_pred_flat = pinn.session.run(
        pinn.predict_velocity(
            tf.constant(X_plot_flat, dtype=tf.float32),
            tf.constant(Y_plot_flat, dtype=tf.float32),
            tf.constant(T_plot_flat, dtype=tf.float32)
        )
    )

    u_pinn_pred = u_pinn_pred_flat.reshape((grid_points_y, grid_points_x))
    v_pinn_pred = v_pinn_pred_flat.reshape((grid_points_y, grid_points_x))

    # Calculate MSE for u and v predictions
    mse_u = np.mean((u_true_list[-1].flatten() - u_pinn_pred_flat)**2)
    mse_v = np.mean((v_true_list[-1].flatten() - v_pinn_pred_flat)**2)
    total_mse = mse_u + mse_v

    print(f"Prediction MSE (u): {mse_u:.6e}")
    print(f"Prediction MSE (v): {mse_v:.6e}")
    print(f"Total Prediction MSE (u+v): {total_mse:.6e}")

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    output_filename = f"shape_results_data_{lambda_data_weight}_pde_{lambda_pde_weight}_epochs_{adam_epochs}_pdepoints_{num_pde_points}_seed_{seed_value}.npz"
    output_file = os.path.join(results_dir, output_filename)
    
    overall_duration = time.time() - overall_start_time
    
    np.savez(
        output_file,
        X=X_np,
        Y=Y_np,
        u_true=u_true_list[-1],
        u_pinn_pred=u_pinn_pred,
        nu_pinn=final_nu,
        true_nu=true_kinematic_viscosity,
        duration_total=overall_duration,
        duration_data_prep=data_prep_duration,
        duration_model_init=model_init_duration,
        duration_adam=adam_duration,
        duration_lbfgs=lbfgs_duration,
        mse_u=mse_u,
        mse_v=mse_v,
        total_mse=total_mse
    )

    print(f"Results saved to {output_file}")
    print("--- HPC Performance Metrics ---")
    print(f"Data Preparation Duration: {data_prep_duration:.2f} seconds")
    print(f"Model Initialization Duration: {model_init_duration:.2f} seconds")
    print(f"Data-Only Pre-training Duration: {duration_data_only:.2f} seconds")
    print(f"Adam Training Duration: {adam_duration:.2f} seconds")
    print(f"L-BFGS-B Training Duration: {lbfgs_duration:.2f} seconds")
    print(f"Total Execution Duration: {overall_duration:.2f} seconds")
