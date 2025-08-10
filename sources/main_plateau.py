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

# Custom Swish activation function
def swish_activation(x):
    return x * tf.sigmoid(x)

# Disable eager execution for TensorFlow 1.x compatibility
tf_v1.disable_eager_execution()

# --- PINN for 2D Burgers' Equation ---

class PINN_Burgers2D:
    """
    A Physics-Informed Neural Network for the 2D Burgers' Equation.
    """
    def __init__(self, layers, x_data, y_data, t_data, u_data, v_data,
                 x_pde, y_pde, t_pde, nu_pde, x_min, x_max, y_min, y_max, t_min, t_max,
                 nu_min_train, nu_max_train, lambda_data_init, lambda_pde_init, annealing_rate=0.9):
        """
        Initializes the PINN model for parametric nu generalization.
        """
        self.layers = layers
        self.annealing_rate = annealing_rate

        # Initialize lambda_data and lambda_pde as trainable variables
        self.lambda_data = tf.Variable(lambda_data_init, dtype=tf.float32, name="lambda_data")
        self.lambda_pde = tf.Variable(lambda_pde_init, dtype=tf.float32, name="lambda_pde")

        # Store training (data) and collocation (pde) points
        self.x_data, self.y_data, self.t_data = x_data, y_data, t_data
        self.u_data, self.v_data = u_data, v_data
        self.x_pde, self.y_pde, self.t_pde = x_pde, y_pde, t_pde
        self.nu_pde = nu_pde # New: nu input for PDE points

        # Store domain bounds for input scaling (normalization)
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.t_min, self.t_max = t_min, t_max
        self.nu_min_train, self.nu_max_train = nu_min_train, nu_max_train # New: nu bounds for scaling

        # Initialize the neural network weights and biases
        self.weights, self.biases = self.initialize_neural_network(self.layers)

        # The trainable variables are now only the network's weights and biases
        self.trainable_variables = self.weights + self.biases + [self.lambda_data, self.lambda_pde]

        # Define loss, gradients, and the Adam optimizer training operation for Stage 1
        total_loss_tensor, data_loss_tensor, pde_loss_tensor = self.compute_loss()
        self.loss_op = total_loss_tensor # This is the main loss for optimization
        self.loss_data_op = data_loss_tensor # For monitoring and annealing
        self.loss_pde_op = pde_loss_tensor # For monitoring and annealing
        self.gradient_op = tf_v1.gradients(self.loss_op, self.trainable_variables)

        # TensorFlow session setup
        self.session = tf_v1.Session(config=tf_v1.ConfigProto(
            allow_soft_placement=True, log_device_placement=False))

        # Define Adam optimizer training operation for Stage 1
        self.learning_rate = tf_v1.Variable(0.001, dtype=tf.float32, name="learning_rate") # Initial learning rate
        self.train_op_adam = tf_v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.loss_op, var_list=self.trainable_variables)

        # Variables for ReduceLROnPlateau
        self.best_loss = tf_v1.Variable(np.inf, dtype=tf.float32, name="best_loss")
        self.patience_counter = tf_v1.Variable(0, dtype=tf.int32, name="patience_counter")
        self.reduce_lr_factor = 0.5 # Factor by which the learning rate will be reduced
        self.reduce_lr_patience = 500 # Number of epochs with no improvement after which learning rate will be reduced
        self.reduce_lr_min_lr = 1e-6 # Minimum learning rate

        # Define data-only loss and optimizer for pre-training (DG-PINN Phase 1)
        u_pred_data, v_pred_data = self.predict_velocity(
            self.x_data, self.y_data, self.t_data, tf.zeros_like(self.x_data)) # nu is dummy here
        self.loss_data_only_op = (tf.reduce_mean(tf.square(self.u_data - u_pred_data)) +
                                  tf.reduce_mean(tf.square(self.v_data - v_pred_data)))
        self.train_op_adam_data_only = tf_v1.train.AdamOptimizer(learning_rate=0.001).minimize(
            self.loss_data_only_op, var_list=self.trainable_variables)

        # --- Stage 2: Inverse Problem (Discovering nu) ---
        # Placeholders for observed data for the inverse problem
        self.x_data_inverse = tf_v1.placeholder(tf.float32, shape=[None, 1], name='x_data_inverse')
        self.y_data_inverse = tf_v1.placeholder(tf.float32, shape=[None, 1], name='y_data_inverse')
        self.t_data_inverse = tf_v1.placeholder(tf.float32, shape=[None, 1], name='t_data_inverse')
        self.u_data_inverse = tf_v1.placeholder(tf.float32, shape=[None, 1], name='u_data_inverse')
        self.v_data_inverse = tf_v1.placeholder(tf.float32, shape=[None, 1], name='v_data_inverse')

        # Discoverable parameter for Stage 2 (nu_inverse)
        self.log_nu_inverse = tf.Variable(tf.math.log(0.05), dtype=tf.float32, name="log_nu_inverse")
        self.nu_inverse = tf.exp(self.log_nu_inverse)

        # Trainable variables for Stage 2: only nu_inverse
        self.trainable_variables_inverse = [self.log_nu_inverse]

        # Predict velocities using the *frozen* network and the discoverable nu_inverse
        u_pred_inverse, v_pred_inverse = self.predict_velocity_inverse(
            self.x_data_inverse, self.y_data_inverse, self.t_data_inverse, self.nu_inverse)

        # Loss for Stage 2: MSE between predicted and observed data
        self.loss_inverse_op = (tf.reduce_mean(tf.square(self.u_data_inverse - u_pred_inverse)) +
                                tf.reduce_mean(tf.square(self.v_data_inverse - v_pred_inverse)))

        # Adam optimizer for Stage 2
        self.train_op_adam_inverse = tf_v1.train.AdamOptimizer(learning_rate=0.001).minimize(
            self.loss_inverse_op, var_list=self.trainable_variables_inverse)

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

    def neural_network_model(self, X, nu_min, nu_max):
        """
        Defines the forward pass of the parametric neural network.
        """
        # Input scaling to the range [-1, 1]
        x_scaled = 2.0 * (X[:, 0:1] - self.x_min) / (self.x_max - self.x_min) - 1.0
        y_scaled = 2.0 * (X[:, 1:2] - self.y_min) / (self.y_max - self.y_min) - 1.0
        t_scaled = 2.0 * (X[:, 2:3] - self.t_min) / (self.t_max - self.t_min) - 1.0
        # Scale the new nu input
        nu_scaled = 2.0 * (X[:, 3:4] - nu_min) / (nu_max - nu_min) - 1.0

        H = tf.concat([x_scaled, y_scaled, t_scaled, nu_scaled], axis=1)

        # Forward pass through the network layers
        for l in range(len(self.weights) - 1):
            W, b = self.weights[l], self.biases[l]
            H = swish_activation(tf.add(tf.matmul(H, W), b))
        W, b = self.weights[-1], self.biases[-1]
        Y = tf.add(tf.matmul(H, W), b) # Final layer is linear
        return Y

    def predict_velocity(self, x, y, t, nu):
        """
        Predicts the velocity components (u, v) at given (x, y, t) coordinates and nu.
        """
        X_input = tf.concat([x, y, t, nu], axis=1)
        uv = self.neural_network_model(X_input, self.nu_min_train, self.nu_max_train)
        return uv[:, 0:1], uv[:, 1:2]

    def predict_velocity_inverse(self, x, y, t, nu_val):
        """
        Predicts the velocity components (u, v) for the inverse problem, using a specific nu_val.
        This method uses the *trained* network weights and biases.
        """
        # Ensure nu_val has the same number of rows as x, y, t
        nu_val_reshaped = tf.cast(tf.fill(tf.shape(x), nu_val), dtype=tf.float32)
        X_input = tf.concat([x, y, t, nu_val_reshaped], axis=1)
        uv = self.neural_network_model(X_input, self.nu_min_train, self.nu_max_train)
        return uv[:, 0:1], uv[:, 1:2]

    def compute_pde_residual(self, x, y, t, nu):
        """
        Computes the residual of the 2D Burgers' equation using automatic differentiation.
        """
        u, v = self.predict_velocity(x, y, t, nu)

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
        f_u = u_t + u * u_x + v * u_y - nu * (u_xx + u_yy)
        f_v = v_t + u * v_x + v * v_y - nu * (v_xx + v_yy)
        return f_u, f_v

    def compute_loss(self):
        """
        Computes the total loss, which is a weighted sum of data and PDE losses.
        """
        # Data loss (mean squared error between predicted and training data)
        u_pred_data, v_pred_data = self.predict_velocity(
            self.x_data, self.y_data, self.t_data, tf.zeros_like(self.x_data)) # nu is dummy here
        loss_data = (tf.reduce_mean(tf.square(self.u_data - u_pred_data)) +
                     tf.reduce_mean(tf.square(self.v_data - v_pred_data)))

        # PDE loss (mean squared error of the PDE residuals at collocation points)
        f_u_pred, f_v_pred = self.compute_pde_residual(
            self.x_pde, self.y_pde, self.t_pde, self.nu_pde)
        loss_pde = (tf.reduce_mean(tf.square(f_u_pred)) +
                    tf.reduce_mean(tf.square(f_v_pred)))

        # Total loss with weighting (hyperparameters to tune)
        total_loss = self.lambda_data * loss_data + self.lambda_pde * loss_pde
        return total_loss, loss_data, loss_pde

    def train_step_adam(self, x_pde_batch, y_pde_batch, t_pde_batch, nu_pde_batch):
        """
        Performs a single training step using the Adam optimizer.
        """
        # Create a feed_dict for the placeholders
        feed_dict = {
            self.x_pde: x_pde_batch,
            self.y_pde: y_pde_batch,
            self.t_pde: t_pde_batch,
            self.nu_pde: nu_pde_batch,
        }
        self.session.run(self.train_op_adam, feed_dict=feed_dict)


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

    def loss_and_grads_scipy(self, flat_weights, x_pde_batch, y_pde_batch, t_pde_batch, nu_pde_batch):
        """
        A wrapper function to compute loss and gradients for the SciPy L-BFGS-B optimizer.
        """
        # Assign the new weights from the optimizer
        self.session.run(self.set_flat_variables(
            tf.constant(flat_weights, dtype=tf.float32)))

        # Create a feed_dict for the placeholders
        feed_dict = {
            self.x_pde: x_pde_batch,
            self.y_pde: y_pde_batch,
            self.t_pde: t_pde_batch,
            self.nu_pde: nu_pde_batch,
        }

        # Compute loss and gradients
        loss_value, grads_value = self.session.run([self.loss_op, self.gradient_op], feed_dict=feed_dict)

        # Flatten the gradients to a 1D array
        flat_grads = np.concatenate([grad.flatten() for grad in grads_value])

        return loss_value.astype(np.float64), flat_grads.astype(np.float64)

    def train(self, epochs_adam, epochs_data_only, num_pde_points, x_min, x_max, y_min, y_max, t_min, t_max):
        """
        Trains the PINN using a two-stage optimization: Data-Only Adam, then Full Adam, then L-BFGS-B.
        """
        # --- Data-Only Adam Optimization (Phase 1) ---
        duration_data_only = self.train_data_only(epochs_data_only)

        # --- Adam Optimization (Phase 2) ---
        print("Starting Adam training (Full Loss)...")
        start_time_adam = time.time()
        for epoch in range(epochs_adam):
            # Generate new PDE points for each epoch (or batch)
            x_pde_batch = np.random.uniform(x_min, x_max, (num_pde_points, 1)).astype(np.float32)
            y_pde_batch = np.random.uniform(y_min, y_max, (num_pde_points, 1)).astype(np.float32)
            t_pde_batch = np.random.uniform(t_min, t_max, (num_pde_points, 1)).astype(np.float32)
            nu_pde_batch = np.random.uniform(self.nu_min_train, self.nu_max_train, (num_pde_points, 1)).astype(np.float32)

            self.train_step_adam(x_pde_batch, y_pde_batch, t_pde_batch, nu_pde_batch)
            if epoch % 100 == 0:
                current_total_loss, _, _ = self.session.run([self.loss_op, self.loss_data_op, self.loss_pde_op], feed_dict={
                    self.x_pde: x_pde_batch,
                    self.y_pde: y_pde_batch,
                    self.t_pde: t_pde_batch,
                    self.nu_pde: nu_pde_batch,
                })
                print(f"Adam Epoch {epoch}: Loss = {current_total_loss:.6f}, Current LR = {self.session.run(self.learning_rate):.6e}")
                current_lambda_data, current_lambda_pde = self.session.run([self.lambda_data, self.lambda_pde])
                print(f"  Lambda Data = {current_lambda_data:.6f}, Lambda PDE = {current_lambda_pde:.6f}")

                # ReduceLROnPlateau logic
                # Fetch current values of best_loss, patience_counter, and learning_rate
                current_best_loss, current_patience_counter, current_lr = self.session.run(
                    [self.best_loss, self.patience_counter, self.learning_rate]
                )

                if current_total_loss < current_best_loss:
                    # Update best_loss and reset patience_counter
                    self.session.run(tf_v1.assign(self.best_loss, current_total_loss))
                    self.session.run(tf_v1.assign(self.patience_counter, 0))
                else:
                    # Increment patience_counter
                    self.session.run(tf_v1.assign_add(self.patience_counter, 1))

                # Check if patience is exhausted and learning rate needs to be reduced
                if current_patience_counter >= self.reduce_lr_patience:
                    new_lr = max(current_lr * self.reduce_lr_factor, self.reduce_lr_min_lr)
                    if new_lr < current_lr:
                        self.session.run(tf_v1.assign(self.learning_rate, new_lr))
                        self.session.run(tf_v1.assign(self.patience_counter, 0)) # Reset patience
                        print(f"  Reducing learning rate to {new_lr:.6e} due to plateau.")
                print() # Added newline for better readability
        end_time_adam = time.time()
        adam_duration = end_time_adam - start_time_adam
        print()
        print(f"Adam training finished in {adam_duration:.2f} seconds.")

        # --- L-BFGS-B Optimization ---
        print("Starting L-BFGS-B training with SciPy...")
        start_time_lbfgs = time.time()
        initial_weights = self.get_flat_variables()

        # Generate a fixed set of PDE points for L-BFGS-B
        x_pde_lbfgs = np.random.uniform(x_min, x_max, (num_pde_points, 1)).astype(np.float32)
        y_pde_lbfgs = np.random.uniform(y_min, y_max, (num_pde_points, 1)).astype(np.float32)
        t_pde_lbfgs = np.random.uniform(t_min, t_max, (num_pde_points, 1)).astype(np.float32)
        nu_pde_lbfgs = np.random.uniform(self.nu_min_train, self.nu_max_train, (num_pde_points, 1)).astype(np.float32)

        scipy_results = scipy.optimize.minimize(
            fun=self.loss_and_grads_scipy,
            x0=initial_weights,
            method='L-BFGS-B',
            jac=True,
            args=(x_pde_lbfgs, y_pde_lbfgs, t_pde_lbfgs, nu_pde_lbfgs), # Pass PDE points as args
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

    def train_inverse_problem(self, x_data_inv, y_data_inv, t_data_inv, u_data_inv, v_data_inv, epochs_inverse_adam):
        """
        Trains only the nu_inverse parameter using Adam optimizer for Stage 2.
        """
        print("\n" + "-" * 50)
        print("Starting Stage 2: Inverse Problem (Discovering nu)...")
        print(f"Training nu_inverse for {epochs_inverse_adam} epochs...")
        
        # Initialize only the log_nu_inverse variable
        self.session.run(tf_v1.variables_initializer([self.log_nu_inverse]))

        feed_dict_inverse = {
            self.x_data_inverse: x_data_inv,
            self.y_data_inverse: y_data_inv,
            self.t_data_inverse: t_data_inv,
            self.u_data_inverse: u_data_inv,
            self.v_data_inverse: v_data_inv,
        }

        start_time_inverse_adam = time.time()
        for epoch in range(epochs_inverse_adam):
            self.session.run(self.train_op_adam_inverse, feed_dict=feed_dict_inverse)
            if epoch % 100 == 0:
                current_loss_inverse, current_nu_inverse = self.session.run(
                    [self.loss_inverse_op, self.nu_inverse], feed_dict=feed_dict_inverse)
                print(f"Inverse Epoch {epoch}: Loss = {current_loss_inverse:.6f}, Discovered nu = {current_nu_inverse:.6f}")
        end_time_inverse_adam = time.time()
        inverse_adam_duration = end_time_inverse_adam - start_time_inverse_adam
        print(f"Stage 2 (Inverse Problem) training finished in {inverse_adam_duration:.2f} seconds.")
        
        final_nu_inverse = self.session.run(self.nu_inverse, feed_dict=feed_dict_inverse)
        return final_nu_inverse, inverse_adam_duration

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
    true_kinematic_viscosity = 0.05  # Ground truth for 'nu' (for data generation)
    nu_min_train, nu_max_train = 0.01, 0.1 # Range for nu during Stage 1 training

    # Domain Boundaries
    x_min, x_max = 0.0, 2.0
    y_min, y_max = 0.0, 2.0
    t_min, t_max = 0.0, time_steps * 0.001

    # Neural Network Architecture
    # Input: (x, y, t, nu), Output: (u, v)
    layers = [4, 60, 60, 60, 60, 60, 2] # Added one hidden layer
    
    # --- Experiment Configuration (Stage 1) ---
    adam_epochs_stage1 = 5000
    epochs_data_only_stage1 = 10000 # New parameter for DG-PINN pre-training
    lambda_data_weight_stage1 = 1.0
    lambda_pde_weight_stage1 = 1.0
    num_pde_points_stage1 = 40000

    # --- Experiment Configuration (Stage 2) ---
    epochs_inverse_adam_stage2 = 5000 # Epochs for optimizing nu in Stage 2
    true_nu_for_inverse_problem = 0.05 # The 'unknown' nu we want to discover in Stage 2
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

    # Generate ground truth data for Stage 1 training (for a range of nu values)
    # and also for Stage 2 inverse problem (for a specific true_nu_for_inverse_problem)
    true_nu_tf_stage1 = tf.constant(true_kinematic_viscosity, dtype=tf.float32) # This is just for data generation, not for training nu
    u_true_tf_stage1, v_true_tf_stage1, t_true_tf_stage1 = generate_ground_truth_data(
        grid_points_x, grid_points_y, time_steps, dx, dy, dt,
        true_nu_tf_stage1, u_initial_tf, v_initial_tf)

    # Generate data for the inverse problem (Stage 2)
    true_nu_tf_stage2 = tf.constant(true_nu_for_inverse_problem, dtype=tf.float32)
    u_true_tf_stage2, v_true_tf_stage2, t_true_tf_stage2 = generate_ground_truth_data(
        grid_points_x, grid_points_y, time_steps, dx, dy, dt,
        true_nu_tf_stage2, u_initial_tf, v_initial_tf)


    with tf_v1.Session() as sess_data_gen:
        u_true_list_stage1, v_true_list_stage1, t_true_list_stage1 = sess_data_gen.run(
            [u_true_tf_stage1, v_true_tf_stage1, t_true_tf_stage1])
        u_true_list_stage2, v_true_list_stage2, t_true_list_stage2 = sess_data_gen.run(
            [u_true_tf_stage2, v_true_tf_stage2, t_true_tf_stage2])


    X_data_list, Y_data_list, T_data_list = [], [], []
    U_data_list, V_data_list = [], []

    for i in range(len(t_true_list_stage1)):
        X_data_list.append(X_np.flatten()[:, None])
        Y_data_list.append(Y_np.flatten()[:, None])
        T_data_list.append(np.full_like(X_np.flatten()[:, None], t_true_list_stage1[i]))
        U_data_list.append(u_true_list_stage1[i].flatten()[:, None])
        V_data_list.append(v_true_list_stage1[i].flatten()[:, None])

    X_data_flat = np.concatenate(X_data_list)
    Y_data_flat = np.concatenate(Y_data_list)
    T_data_flat = np.concatenate(T_data_list)
    U_data_flat = np.concatenate(U_data_list)
    V_data_flat = np.concatenate(V_data_list)

    # PDE points are now placeholders, will be fed in batches during training
    x_pde_placeholder = tf_v1.placeholder(tf.float32, shape=[None, 1], name='x_pde')
    y_pde_placeholder = tf_v1.placeholder(tf.float32, shape=[None, 1], name='y_pde')
    t_pde_placeholder = tf_v1.placeholder(tf.float32, shape=[None, 1], name='t_pde')
    nu_pde_placeholder = tf_v1.placeholder(tf.float32, shape=[None, 1], name='nu_pde')


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
        layers, x_data_tf, y_data_tf, t_data_tf,
        u_data_tf, v_data_tf, x_pde_placeholder, y_pde_placeholder, t_pde_placeholder,
        nu_pde_placeholder, # Pass nu_pde_placeholder
        x_min, x_max, y_min, y_max, t_min, t_max,
        nu_min_train, nu_max_train, # Pass nu_min_train, nu_max_train
        lambda_data_weight_stage1, lambda_pde_weight_stage1, annealing_rate=0.9)
    model_init_duration = time.time() - model_init_start_time
    # ---

    print(f"--- Stage 1: Parametric PINN Training ---")
    print(f"Training with: Adam Epochs = {adam_epochs_stage1}, Data-Only Epochs = {epochs_data_only_stage1}, Lambda Data = {lambda_data_weight_stage1}, Lambda PDE = {lambda_pde_weight_stage1}, PDE Points = {num_pde_points_stage1}, Nu Range = [{nu_min_train}, {nu_max_train}])")

    # --- Model Training (Stage 1) ---
    duration_data_only_stage1, adam_duration_stage1, lbfgs_duration_stage1 = pinn.train(
        epochs_adam=adam_epochs_stage1, epochs_data_only=epochs_data_only_stage1,
        num_pde_points=num_pde_points_stage1, x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max, t_min=t_min, t_max=t_max)
    # ---

    # --- Stage 2: Inverse Problem (Discovering a specific nu) ---
    # Prepare data for the inverse problem (using the last snapshot of true data for true_nu_for_inverse_problem)
    X_data_inverse_flat = X_np.flatten()[:, None]
    Y_data_inverse_flat = Y_np.flatten()[:, None]
    T_data_inverse_flat = np.full_like(X_data_inverse_flat, t_true_list_stage2[-1]).astype(np.float32)
    U_data_inverse_flat = u_true_list_stage2[-1].flatten()[:, None].astype(np.float32)
    V_data_inverse_flat = v_true_list_stage2[-1].flatten()[:, None].astype(np.float32)

    final_nu_inverse, inverse_adam_duration = pinn.train_inverse_problem(
        X_data_inverse_flat, Y_data_inverse_flat, T_data_inverse_flat,
        U_data_inverse_flat, V_data_inverse_flat, epochs_inverse_adam_stage2)

    print("\n" + "-" * 50)
    print(f"Stage 2: Discovered nu (Inverse Problem): {final_nu_inverse:.6f}")
    print(f"Stage 2: Ground Truth nu for Inverse Problem: {true_nu_for_inverse_problem}")

    # --- Results and Saving ---
    print("-" * 50)
    print("Generating predictions for various nu values (using Stage 1 trained model)...")

    X_plot_flat = X_np.flatten()[:, None]
    Y_plot_flat = Y_np.flatten()[:, None]
    T_plot_flat = np.full_like(X_plot_flat, t_max)

    nu_values_for_plotting = [0.01, 0.05, 0.1] # Example nu values for plotting
    results_for_plotting = {}

    for nu_val in nu_values_for_plotting:
        Nu_plot_flat = np.full_like(X_plot_flat, nu_val).astype(np.float32)
        u_pinn_pred_flat, v_pinn_pred_flat = pinn.session.run(
            pinn.predict_velocity(
                tf.constant(X_plot_flat, dtype=tf.float32),
                tf.constant(Y_plot_flat, dtype=tf.float32),
                tf.constant(T_plot_flat, dtype=tf.float32),
                tf.constant(Nu_plot_flat, dtype=tf.float32)
            )
        )
        results_for_plotting[f'u_pred_nu_{nu_val}'] = u_pinn_pred_flat.reshape((grid_points_y, grid_points_x))
        results_for_plotting[f'v_pred_nu_{nu_val}'] = v_pinn_pred_flat.reshape((grid_points_y, grid_points_x))

        # Calculate MSE for u and v predictions for this nu_val (against true_kinematic_viscosity data)
        # Note: This MSE is against the data generated with true_kinematic_viscosity,
        # not necessarily the nu_val being plotted.
        if nu_val == true_kinematic_viscosity:
            mse_u_stage1_plot = np.mean((u_true_list_stage1[-1].flatten() - u_pinn_pred_flat)**2)
            mse_v_stage1_plot = np.mean((v_true_list_stage1[-1].flatten() - v_pinn_pred_flat)**2)
            total_mse_stage1_plot = mse_u_stage1_plot + mse_v_stage1_plot
            print(f"Prediction MSE (u) for nu={nu_val} (Stage 1 model): {mse_u_stage1_plot:.6e}")
            print(f"Prediction MSE (v) for nu={nu_val} (Stage 1 model): {mse_v_stage1_plot:.6e}")
            print(f"Total Prediction MSE (u+v) for nu={nu_val} (Stage 1 model): {total_mse_stage1_plot:.6e}")
            results_for_plotting['mse_u_stage1_plot'] = mse_u_stage1_plot
            results_for_plotting['mse_v_stage1_plot'] = mse_v_stage1_plot
            results_for_plotting['total_mse_stage1_plot'] = total_mse_stage1_plot

    # Calculate MSE for Stage 2 discovered nu against its true data
    u_pred_inverse_final, v_pred_inverse_final = pinn.session.run(
        pinn.predict_velocity_inverse(
            tf.constant(X_data_inverse_flat, dtype=tf.float32),
            tf.constant(Y_data_inverse_flat, dtype=tf.float32),
            tf.constant(T_data_inverse_flat, dtype=tf.float32),
            tf.constant(final_nu_inverse, dtype=tf.float32) # Use the discovered nu
        )
    )
    mse_u_stage2_final = np.mean((U_data_inverse_flat - u_pred_inverse_final)**2)
    mse_v_stage2_final = np.mean((V_data_inverse_flat - v_pred_inverse_final)**2)
    total_mse_stage2_final = mse_u_stage2_final + mse_v_stage2_final
    print(f"Prediction MSE (u) for Discovered nu={final_nu_inverse:.6f} (Stage 2): {mse_u_stage2_final:.6e}")
    print(f"Prediction MSE (v) for Discovered nu={final_nu_inverse:.6f} (Stage 2): {mse_v_stage2_final:.6e}")
    print(f"Total Prediction MSE (u+v) for Discovered nu={final_nu_inverse:.6f} (Stage 2): {total_mse_stage2_final:.6e}")

    results_for_plotting['mse_u_stage2_final'] = mse_u_stage2_final
    results_for_plotting['mse_v_stage2_final'] = mse_v_stage2_final
    results_for_plotting['total_mse_stage2_final'] = total_mse_stage2_final

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    output_filename = f"parametric_inverse_results_nu_{true_nu_for_inverse_problem}_s1_epochs_{adam_epochs_stage1}_s2_epochs_{epochs_inverse_adam_stage2}_seed_{seed_value}.npz"
    output_file = os.path.join(results_dir, output_filename)
    
    overall_duration = time.time() - overall_start_time
    
    np.savez(
        output_file,
        X=X_np,
        Y=Y_np,
        u_true_stage1=u_true_list_stage1[-1],
        v_true_stage1=v_true_list_stage1[-1],
        u_true_stage2=u_true_list_stage2[-1],
        v_true_stage2=v_true_list_stage2[-1],
        true_nu_stage1=true_kinematic_viscosity,
        true_nu_stage2_inverse=true_nu_for_inverse_problem,
        discovered_nu_stage2=final_nu_inverse,
        nu_values_for_plotting=nu_values_for_plotting,
        **results_for_plotting, # Unpack the dictionary of predictions and MSEs
        duration_total=overall_duration,
        duration_data_prep=data_prep_duration,
        duration_model_init=model_init_duration,
        duration_adam_stage1=adam_duration_stage1,
        duration_lbfgs_stage1=lbfgs_duration_stage1,
        duration_inverse_adam_stage2=inverse_adam_duration,
    )

    print(f"Results saved to {output_file}")
    print("---" * 10 + " HPC Performance Metrics " + "---" * 10)
    print(f"Data Preparation Duration: {data_prep_duration:.2f} seconds")
    print(f"Model Initialization Duration: {model_init_duration:.2f} seconds")
    print(f"Stage 1 Data-Only Pre-training Duration: {duration_data_only_stage1:.2f} seconds")
    print(f"Stage 1 Adam Training Duration: {adam_duration_stage1:.2f} seconds")
    print(f"Stage 1 L-BFGS-B Training Duration: {lbfgs_duration_stage1:.2f} seconds")
    print(f"Stage 2 Inverse Problem Training Duration: {inverse_adam_duration:.2f} seconds")
    print(f"Total Execution Duration: {overall_duration:.2f} seconds")
    
    # Close the TensorFlow session
    pinn.session.close()
    print("TensorFlow session closed.")
