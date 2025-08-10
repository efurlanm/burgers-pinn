#
"""
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
                 lambda_data, lambda_pde, learning_rate=0.001):
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
        self.total_loss_op, self.loss_data_op, self.loss_pde_op = self.compute_individual_losses()
        self.loss_op = self.total_loss_op
        self.gradient_op = tf_v1.gradients(self.loss_op, self.trainable_variables)

        # TensorFlow session setup
        self.session = tf_v1.Session(config=tf_v1.ConfigProto(
            allow_soft_placement=True, log_device_placement=False))

        # Define loss, gradients, and the Adam optimizer training operation
        self.train_op_adam = tf_v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            self.loss_op, var_list=self.trainable_variables)

        # Define data-only loss and optimizer for pre-training (DG-PINN Phase 1)
        u_pred_data, v_pred_data = self.predict_velocity(
            self.x_data, self.y_data, self.t_data)
        self.loss_data_only_op = (tf.reduce_mean(tf.square(self.u_data - u_pred_data)) +
                                  tf.reduce_mean(tf.square(self.v_data - v_pred_data)))
        self.train_op_adam_data_only = tf_v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            self.loss_data_only_op, var_list=self.trainable_variables)

        # Initialize all TensorFlow variables
        self.session.run(tf_v1.global_variables_initializer())

        # Counter for L-BFGS-B iterations for printing
        self.lbfgs_iter = 0

        

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
        total_loss = self.lambda_data * loss_data + self.lambda_pde * loss_pde
        return total_loss, loss_data, loss_pde

    def compute_loss(self):
        """
        Computes the total loss, which is a weighted sum of data and PDE losses.
        """
        total_loss, _, _ = self.compute_individual_losses()
        return total_loss

    def compute_individual_losses(self):
        """
        Computes the individual data and PDE losses.
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

        return self.lambda_data * loss_data + self.lambda_pde * loss_pde, loss_data, loss_pde

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
                print(f"Data-Only Iter {epoch}: Data Loss = {current_loss_data:.6f}")
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
        loss_value, loss_data_value, loss_pde_value, grads_value, current_nu = self.session.run(
            [self.loss_op, self.loss_data_op, self.loss_pde_op, self.gradient_op, tf.exp(self.log_nu_pinn)])

        # Flatten the gradients to a 1D array
        flat_grads = np.concatenate([grad.flatten() for grad in grads_value])

        # Increment L-BFGS-B iteration counter
        self.lbfgs_iter += 1

        # Print progress every 100 iterations
        if self.lbfgs_iter % 100 == 0:
            print(f"L-BFGS-B Iter {self.lbfgs_iter}: Loss = {loss_value:.6f}, "
                  f"Discovered nu = {current_nu:.6f}, "
                  f"loss_data = {loss_data_value:.6f}, "
                  f"loss_pde = {loss_pde_value:.6f}")
        return loss_value.astype(np.float64), flat_grads.astype(np.float64)

    def train(self, epochs_adam, epochs_data_only):
        """
        Trains the PINN using a two-stage optimization: Data-Only Adam, then Full Adam, then L-BFGS-B.
        """
        # --- Explicitly set initial nu to 0.02 for robustness test ---
        initial_nu_guess = 0.02
        print(f"Explicitly setting initial nu to: {initial_nu_guess}")
        self.session.run(tf_v1.assign(self.log_nu_pinn, tf.math.log(initial_nu_guess)))
        # --- Data-Only Adam Optimization (Phase 1) ---
        duration_data_only = self.train_data_only(epochs_data_only)

        # --- Adam Optimization (Phase 2) ---
        print("Starting Adam training (Full Loss)...")
        start_time_adam = time.time()
        for epoch in range(epochs_adam):
            self.train_step_adam()
            if epoch % 100 == 0 or epoch == epochs_adam - 1:
                current_loss, current_nu, current_loss_data, current_loss_pde = self.session.run(
                    [self.loss_op, tf.exp(self.log_nu_pinn), self.loss_data_op, self.loss_pde_op])
                print(f"Adam Iter {epoch}: Loss = {current_loss:.6f}, "
                      f"Discovered nu = {current_nu:.6f}, "
                      f"loss_data = {current_loss_data:.6f}, "
                      f"loss_pde = {current_loss_pde:.6f}")
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

        final_lbfgs_loss, final_lbfgs_loss_data, final_lbfgs_loss_pde, final_lbfgs_nu = self.session.run(
            [self.loss_op, self.loss_data_op, self.loss_pde_op, tf.exp(self.log_nu_pinn)])

        print(f"L-BFGS-B Final Iter {scipy_results.nit}: Loss = {final_lbfgs_loss:.6f}, "
              f"Discovered nu = {final_lbfgs_nu:.6f}, "
              f"loss_data = {final_lbfgs_loss_data:.6f}, "
              f"loss_pde = {final_lbfgs_loss_pde:.6f}")

        print(f"L-BFGS-B training finished in {lbfgs_duration:.2f} seconds.")
        print(f"L-BFGS-B converged: {scipy_results.success}")
        print(f"L-BFGS-B message: {scipy_results.message}")
        print(f"L-BFGS-B iterations: {scipy_results.nit}")
        
        return duration_data_only, adam_duration, lbfgs_duration

    def find_best_initial_nu(self, nu_candidates):
        """
        Finds the best initial nu from a list of candidates.
        """
        print("Searching for the best initial nu...")
        best_nu = None
        lowest_loss = float('inf')

        for nu in nu_candidates:
            # Reset the model
            self.session.run(tf_v1.global_variables_initializer())
            # Set the nu value
            self.session.run(tf_v1.assign(self.log_nu_pinn, tf.math.log(nu)))

            # Train for a few epochs
            for epoch in range(500):
                self.train_step_adam()
            
            # Get the loss
            loss = self.session.run(self.loss_op)
            print(f"  Candidate nu: {nu}, Loss: {loss}")

            if loss < lowest_loss:
                lowest_loss = loss
                best_nu = nu
        
        return best_nu


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
    # --- Experiment Setup ---
    # --- Experiment Setup (Single Run) ---
    learning_rate = 1.0e-05 # Fixed learning rate for single run
    num_runs = 1 # Single run
    
    # --- Epochs for Single Run ---
    adam_epochs = 2000
    epochs_data_only = 10000

    # --- Main Parameters ---
    grid_points_x = 41
    grid_points_y = 41
    time_steps = 50
    true_kinematic_viscosity = 0.05
    x_min, x_max = 0.0, 2.0
    y_min, y_max = 0.0, 2.0
    t_min, t_max = 0.0, time_steps * 0.001
    layers = [3, 60, 60, 60, 60, 2]
    adam_epochs = 2000
    epochs_data_only = 10000
    lambda_data_weight = 1.0
    lambda_pde_weight = 1000.0
    num_pde_points = 6724

    # --- Data Preparation (do this once) ---
    x_np = np.linspace(x_min, x_max, grid_points_x)
    y_np = np.linspace(y_min, y_max, grid_points_y)
    X_np, Y_np = np.meshgrid(x_np, y_np)
    dx, dy, dt = x_np[1] - x_np[0], y_np[1] - y_np[0], 0.001
    center_x, center_y = 1.0, 1.0
    sigma_x, sigma_y = 0.25, 0.25
    u_initial_np = np.exp(-((X_np - center_x)**2 / (2 * sigma_x**2) + (Y_np - center_y)**2 / (2 * sigma_y**2)))
    v_initial_np = np.exp(-((X_np - center_x)**2 / (2 * sigma_x**2) + (Y_np - center_y)**2 / (2 * sigma_y**2)))
    u_initial_tf = tf.constant(u_initial_np, dtype=tf.float32)
    v_initial_tf = tf.constant(v_initial_np, dtype=tf.float32)
    true_nu_tf = tf.constant(true_kinematic_viscosity, dtype=tf.float32)
    u_true_tf, v_true_tf, t_true_tf = generate_ground_truth_data(
        grid_points_x, grid_points_y, time_steps, dx, dy, dt, true_nu_tf, u_initial_tf, v_initial_tf)
    with tf_v1.Session() as sess_data_gen:
        u_true_list, v_true_list, t_true_list = sess_data_gen.run([u_true_tf, v_true_tf, t_true_tf])
    X_data_list, Y_data_list, T_data_list, U_data_list, V_data_list = [], [], [], [], []
    for i in range(len(t_true_list)):
        X_data_list.append(X_np.flatten()[:, None])
        Y_data_list.append(Y_np.flatten()[:, None])
        T_data_list.append(np.full_like(X_np.flatten()[:, None], t_true_list[i]))
        U_data_list.append(u_true_list[i].flatten()[:, None])
        V_data_list.append(v_true_list[i].flatten()[:, None])
    X_data_flat, Y_data_flat, T_data_flat, U_data_flat, V_data_flat = (
        np.concatenate(X_data_list), np.concatenate(Y_data_list), np.concatenate(T_data_list),
        np.concatenate(U_data_list), np.concatenate(V_data_list))
    print(f"--- Starting Run 1/1 ---")
    
    # Reset TensorFlow graph for a clean run
    tf_v1.reset_default_graph()
    
    # Set random seeds for reproducibility
    seed_value = 1
    tf_v1.set_random_seed(seed_value)
    np.random.seed(seed_value)
    
    # Use predefined learning rate
    # learning_rate is already defined above
    print(f"Running with Seed: {seed_value}, Learning Rate: {learning_rate:.6e}")

        # --- Data and Model Setup for this run ---
    x_data_tf = tf.constant(X_data_flat, dtype=tf.float32)
    y_data_tf = tf.constant(Y_data_flat, dtype=tf.float32)
    t_data_tf = tf.constant(T_data_flat, dtype=tf.float32)
    u_data_tf = tf.constant(U_data_flat, dtype=tf.float32)
    v_data_tf = tf.constant(V_data_flat, dtype=tf.float32)
    x_pde = x_data_tf
    y_pde = y_data_tf
    t_pde = t_data_tf

    pinn = PINN_Burgers2D(
        layers, true_kinematic_viscosity, x_data_tf, y_data_tf, t_data_tf,
        u_data_tf, v_data_tf, x_pde, y_pde, t_pde,
        x_min, x_max, y_min, y_max, t_min, t_max,
        lambda_data_weight, lambda_pde_weight, learning_rate)

    # --- Model Training ---
    duration_data_only, adam_duration, lbfgs_duration = pinn.train(epochs_adam=adam_epochs, epochs_data_only=epochs_data_only)
        
        # --- Store and Print Results for parsing by shell script ---
    final_nu = np.exp(pinn.session.run(pinn.log_nu_pinn))
    relative_error = np.abs(final_nu - true_kinematic_viscosity) / true_kinematic_viscosity
    final_loss = pinn.session.run(pinn.loss_op) # Get the final total loss (MSE)

    # Print results in a parsable format (CSV-like line)
    print(f"RESULT_COLO1000_RUN,1,{learning_rate:.6e},{final_nu:.6f},{relative_error:.4f},{final_loss:.6e},{duration_data_only:.2f},{adam_duration:.2f},{lbfgs_duration:.2f}")
    
    print(f"Run 1 Finished. Discovered nu: {final_nu:.6f}, Relative Error: {relative_error:.4f}")
    pinn.session.close()