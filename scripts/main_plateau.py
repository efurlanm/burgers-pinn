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
    -   Implement the two-stage training process (Adam and L-BFGS-B).
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
import os
import argparse
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

# --- Swish Activation Function ---
def swish_activation(x):
    return tf.keras.activations.swish(x)

# --- PINN for 2D Burgers' Equation ---

class PINN_Burgers2D(tf.keras.Model):
    """
    A Physics-Informed Neural Network for the 2D Burgers' Equation.
    """
    def __init__(self, layers_config, x_data, y_data, t_data, u_data, v_data,
                 x_pde, y_pde, t_pde, nu_pde, x_min, x_max, y_min, y_max, t_min, t_max,
                 nu_min_train, nu_max_train, true_kinematic_viscosity, annealing_rate=0.01):
        """
        Initializes the PINN model for parametric nu generalization.
        """
        super().__init__()
        self.network_layers = layers_config
        self.annealing_rate = annealing_rate
        self.sharpness_factor = 5.0 # Factor to control the sharpness of nu-based weighting (reverted)

        # Initialize individual loss weights for adaptive weighting
        self.weight_data = tf.Variable(1.0, dtype=tf.float32, trainable=False, name="weight_data")
        self.weight_pde = tf.Variable(1.0, dtype=tf.float32, trainable=False, name="weight_pde")

        # Store training (data) and collocation (pde) points
        self.x_data, self.y_data, self.t_data = x_data, y_data, t_data
        self.u_data, self.v_data = u_data, v_data
        # PDE points will be passed directly in TF2, no placeholders needed
        self.x_pde, self.y_pde, self.t_pde = None, None, None # Will be set during training
        self.nu_pde = None # Will be set during training

        # Store domain bounds for input scaling (normalization)
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.t_min, self.t_max = t_min, t_max
        self.nu_min_train, self.nu_max_train = nu_min_train, nu_max_train # New: nu bounds for scaling
        self.true_kinematic_viscosity = true_kinematic_viscosity # Store true nu for regularization
        self.nu_regularization_weight = 1e-3 # Increased weight for nu regularization

        # Initialize the neural network using Keras layers
        self.dense_layers = []
        for i in range(len(self.network_layers) - 1):
            self.dense_layers.append(tf.keras.layers.Dense(
                self.network_layers[i+1], activation=swish_activation if i < len(self.network_layers) - 2 else None,
                kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-5)))

        # The trainable variables are now only the network's weights and biases
        # Keras automatically tracks trainable variables of layers added to the model.

        # Define Adam optimizer for Stage 1
        self.initial_learning_rate = 0.001 # Initial learning rate as a float
        self.optimizer_adam = tf.keras.optimizers.Adam(learning_rate=self.initial_learning_rate)
        self.optimizer = self.optimizer_adam # Set the main optimizer for Keras callbacks

        # Counter for L-BFGS-B iterations for printing
        self.lbfgs_iter = 0

        # Define data-only loss and optimizer for pre-training (DG-PINN Phase 1)
        self.optimizer_adam_data_only = tf.keras.optimizers.Adam(learning_rate=0.001)

        # --- Stage 2: Inverse Problem (Discovering nu) ---
        # Discoverable parameter for Stage 2 (nu_inverse)
        self.log_nu_inverse = tf.Variable(tf.math.log(0.02), dtype=tf.float32, name="log_nu_inverse")
        self.nu_inverse = tf.exp(self.log_nu_inverse)

        # Trainable variables for Stage 2: only log_nu_inverse
        self.trainable_variables_inverse = [self.log_nu_inverse]

        # Adam optimizer for Stage 2
        self.optimizer_adam_inverse = tf.keras.optimizers.Adam(learning_rate=0.001)

    def update_individual_weights(self, x_pde, y_pde, t_pde, nu_pde):
        """
        Calculates gradient statistics and updates individual loss weights
        based on the annealing algorithm from Wang et al. (2021).
        """
        trainable_vars = self.trainable_variables # All trainable variables of the model

        with tf.GradientTape(persistent=True) as tape:
            # Re-calculate individual loss terms within this tape context
            total_loss_dummy, combined_data_loss, combined_pde_loss, _, _ = self.compute_loss(
                x_pde, y_pde, t_pde, nu_pde)

        # Compute gradients for each individual loss term
        grads_data = tape.gradient(combined_data_loss, trainable_vars)
        grads_pde = tape.gradient(combined_pde_loss, trainable_vars)
        del tape

        # Flatten gradients and compute mean/max absolute values
        all_weights = [
            self.weight_data, self.weight_pde
        ]
        all_mean_grads = []
        all_max_grads = []

        # Helper to process gradients
        def process_grads(grads):
            # Filter out None gradients and cast to float32
            filtered_grads = [tf.cast(g, tf.float32) for g in grads if g is not None]
            if not filtered_grads: # Handle case where all gradients are None
                return tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)
            flat_grads = tf.concat([tf.reshape(g, [-1]) for g in filtered_grads], axis=0)
            if tf.size(flat_grads) == 0: # Handle empty flattened gradients
                return tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)
            return tf.reduce_mean(tf.abs(flat_grads)), tf.reduce_max(tf.abs(flat_grads))

        mean_grad_data, max_grad_data = process_grads(grads_data)
        mean_grad_pde, max_grad_pde = process_grads(grads_pde)

        all_mean_grads = [mean_grad_data, mean_grad_pde]
        all_max_grads = [max_grad_data, max_grad_pde]

        # Find the global maximum gradient reference (maxGradRef)
        non_zero_max_grads = [g for g in all_max_grads if g > 1e-8]
        if non_zero_max_grads:
            max_grad_ref = tf.reduce_max(tf.stack(non_zero_max_grads))
        else:
            max_grad_ref = tf.constant(1.0, dtype=tf.float32) # Default if all grads are zero

        # Debug prints
        tf.print("Epoch:", tf.cast(self.optimizer_adam.iterations, tf.float32), "Max Grad Ref:", max_grad_ref)

        # Update each individual weight
        for i, weight_var in enumerate(all_weights):
            current_mean_grad = all_mean_grads[i]
            # Only update if mean_grad is not negligible and max_grad_ref is meaningful
            if current_mean_grad > 1e-8 and max_grad_ref > 1e-8:
                lambda_hat = max_grad_ref / (current_mean_grad + 1e-8)
                new_weight = (1.0 - self.annealing_rate) * weight_var + self.annealing_rate * lambda_hat
                weight_var.assign(tf.clip_by_value(new_weight, 1e-1, 1e6))
            # If gradients are too small, keep weight at a minimum or previous value
            elif current_mean_grad <= 1e-8:
                weight_var.assign(tf.constant(1e-1, dtype=tf.float32)) # Set to min clip value

    def call(self, X, nu_min, nu_max):
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
        for layer in self.dense_layers:
            H = layer(H)
        return H

    def predict_velocity(self, x, y, t, nu):
        """
        Predicts the velocity components (u, v) at given (x, y, t) coordinates and nu.
        """
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        t = tf.cast(t, tf.float32)
        nu = tf.cast(nu, tf.float32)

        # Reshape to (N,) and then stack to get (N, 4)
        x_reshaped = tf.reshape(x, [-1])
        y_reshaped = tf.reshape(y, [-1])
        t_reshaped = tf.reshape(t, [-1])
        nu_reshaped = tf.reshape(nu, [-1])

        X_input = tf.stack([x_reshaped, y_reshaped, t_reshaped, nu_reshaped], axis=1)
        
        uv = self.call(X_input, self.nu_min_train, self.nu_max_train)
        return uv[:, 0:1], uv[:, 1:2]

    def predict_velocity_inverse(self, x, y, t, nu_val):
        """
        Predicts the velocity components (u, v) for the inverse problem, using a specific nu_val.
        This method uses the *trained* network weights and biases.
        """
        # nu_val is expected to be a scalar tensor (self.nu_inverse or nu_inverse_traced)
        # TensorFlow will broadcast it to match the batch dimension of x, y, t
        X_input = tf.concat([x, y, t, nu_val * tf.ones_like(x)], axis=1)
        uv = self.call(X_input, self.nu_min_train, self.nu_max_train)
        return uv[:, 0:1], uv[:, 1:2]

    def compute_pde_residual(self, x, y, t, nu):
        """
        Computes the residual of the 2D Burgers' equation using automatic differentiation.
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y, t])
            u, v = self.predict_velocity(x, y, t, nu)

            # First-order derivatives
            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)
            u_t = tape.gradient(u, t)
            v_x = tape.gradient(v, x)
            v_y = tape.gradient(v, y)
            v_t = tape.gradient(v, t)

        # Second-order derivatives
        u_xx = tape.gradient(u_x, x)
        u_yy = tape.gradient(u_y, y)
        v_xx = tape.gradient(v_x, x)
        v_yy = tape.gradient(v_y, y)

        del tape # Release the tape

        # PDE residuals (f_u and f_v should be zero for a perfect solution)
        f_u = u_t + u * u_x + v * u_y - nu * (u_xx + u_yy)
        f_v = v_t + u * v_x + v * v_y - nu * (v_xx + v_yy)
        return f_u, f_v

    def compute_loss(self, x_pde, y_pde, t_pde, nu_pde):
        """
        Computes the total loss with nu-based weighting for the PDE component.
        """
        # Data loss (mean squared error between predicted and training data)
        u_pred_data, v_pred_data = self.predict_velocity(
            self.x_data, self.y_data, self.t_data, tf.fill(tf.shape(self.x_data), self.true_kinematic_viscosity)) # Use true_kinematic_viscosity for data loss
        loss_u_data = tf.reduce_mean(tf.square(self.u_data - u_pred_data))
        loss_v_data = tf.reduce_mean(tf.square(self.v_data - v_pred_data))

        # PDE loss with nu-based weighting
        f_u_pred, f_v_pred = self.compute_pde_residual(
            x_pde, y_pde, t_pde, nu_pde)

        # --- Nu-based Weighting Logic for PDE residuals ---
        # The goal is to give more weight to low 'nu' values.
        # We will use a function that increases sharply as nu -> nu_min.
        # Example: an inverted exponential decay.
        nu_weights = tf.exp(-self.sharpness_factor * (nu_pde - self.nu_min_train) / (self.nu_max_train - self.nu_min_train))

        # Apply nu_weights to the squared PDE residuals before averaging
        loss_f_u_pde = tf.reduce_mean(nu_weights * tf.square(f_u_pred))
        loss_f_v_pde = tf.reduce_mean(nu_weights * tf.square(f_v_pred))

        # Total loss with individual adaptive weights
        combined_data_loss = loss_u_data + loss_v_data
        combined_pde_loss = loss_f_u_pde + loss_f_v_pde

        # Total loss with individual adaptive weights
        total_loss = (
            self.weight_data * combined_data_loss +
            self.weight_pde * combined_pde_loss)
        return total_loss, combined_data_loss, combined_pde_loss, loss_f_u_pde, loss_f_v_pde

    @tf.function
    def train_step_adam(self, x_pde_batch, y_pde_batch, t_pde_batch, nu_pde_batch):
        """
        Performs a single training step using the Adam optimizer.
        """
        with tf.GradientTape() as tape:
            total_loss, _, _, _, _ = self.compute_loss(
                x_pde_batch, y_pde_batch, t_pde_batch, nu_pde_batch)

        # Get gradients of the total loss with respect to the trainable variables
        gradients = tape.gradient(total_loss, self.trainable_variables)

        # Apply gradients using the Adam optimizer
        self.optimizer_adam.apply_gradients(zip(gradients, self.trainable_variables))

        return total_loss

    def compute_data_only_loss(self):
        u_pred_data, v_pred_data = self.predict_velocity(
            self.x_data, self.y_data, self.t_data, tf.fill(tf.shape(self.x_data), self.true_kinematic_viscosity)) # Use true_kinematic_viscosity for data loss
        loss_data = (tf.reduce_mean(tf.square(self.u_data - u_pred_data)) +
                     tf.reduce_mean(tf.square(self.v_data - v_pred_data)))
        return loss_data

    def train_data_only(self, epochs_data_only):
        """
        Performs training using only the data loss for a specified number of epochs.
        """
        print(f"Starting Data-Only Pre-training for {epochs_data_only} epochs...")
        start_time_data_only = time.time()
        for epoch in range(epochs_data_only):
            with tf.GradientTape() as tape:
                current_loss_data = self.compute_data_only_loss()
            gradients = tape.gradient(current_loss_data, self.trainable_variables)
            self.optimizer_adam_data_only.apply_gradients(zip(gradients, self.trainable_variables))

            if epoch % 1000 == 0:
                print(f"Data-Only Epoch {epoch}: Data Loss = {current_loss_data:.6f}")
            if tf.math.is_nan(current_loss_data):
                print(f"[ERROR] Data-Only Loss is NaN at epoch {epoch}. Stopping training.")
                break
        end_time_data_only = time.time()
        duration_data_only = end_time_data_only - start_time_data_only
        print(f"Data-Only Pre-training finished in {duration_data_only:.2f} seconds.")
        return duration_data_only

    def fit(self, epochs_adam, epochs_data_only, num_pde_points, x_min, x_max, y_min, y_max, t_min, t_max, callbacks=None):
        """
        Trains the PINN using a two-stage optimization: Data-Only Adam, then Full Adam.
        """
        # --- Data-Only Adam Optimization (Phase 1) ---
        duration_data_only = self.train_data_only(epochs_data_only)

        # --- Adam Optimization (Phase 2) ---
        print("Starting Adam training (Full Loss with Curriculum)...")
        start_time_adam = time.time()

        # Initialize callbacks
        if callbacks:
            for callback in callbacks:
                callback.set_model(self)
                callback.on_train_begin()

        # Curriculum Parameters
        nu_start_range = 0.05 # Start with a smaller nu range, e.g.: [0.01, 0.05]
        total_nu_range = self.nu_max_train - self.nu_min_train

        for epoch in range(epochs_adam):
            # --- Curriculum Logic ---
            # Linearly increases the 'nu' range to be trained over epochs
            progress = epoch / epochs_adam
            current_nu_max = self.nu_min_train + (total_nu_range * progress)
            # Ensures we start with a minimum range and do not exceed the maximum
            current_nu_max = max(current_nu_max, self.nu_min_train + nu_start_range)

            # Generate new PDE points for each epoch (or batch)
            x_pde_batch = tf.constant(np.random.uniform(x_min, x_max, (num_pde_points, 1)).astype(np.float32))
            y_pde_batch = tf.constant(np.random.uniform(y_min, y_max, (num_pde_points, 1)).astype(np.float32))
            t_pde_batch = tf.constant(np.random.uniform(t_min, t_max, (num_pde_points, 1)).astype(np.float32))
            # Sample nu from the range currently allowed by the curriculum
            nu_pde_batch = tf.constant(np.random.uniform(self.nu_min_train, current_nu_max, (num_pde_points, 1)).astype(np.float32))

            self.train_step_adam(x_pde_batch, y_pde_batch, t_pde_batch, nu_pde_batch)

            # Update individual loss weights periodically
            if epoch % 500 == 0:
                self.update_individual_weights(x_pde_batch, y_pde_batch, t_pde_batch, nu_pde_batch)

            if epoch % 500 == 0:
                current_total_loss, current_combined_data_loss, current_combined_pde_loss, _, _ = self.compute_loss(
                    x_pde_batch, y_pde_batch, t_pde_batch, nu_pde_batch)
                print(f"Adam Epoch {epoch}: Loss = {current_total_loss:.6f}, Data Loss = {current_combined_data_loss:.6f}, PDE Loss = {current_combined_pde_loss:.6f}")

                if tf.math.is_nan(current_total_loss):
                    print(f"[ERROR] Adam Total Loss is NaN at epoch {epoch}. Stopping training.")
                    break

                # Call on_epoch_end for callbacks
                if callbacks:
                    logs = {'loss': current_total_loss.numpy(), 'lr': self.optimizer_adam.learning_rate.numpy()}
                    for callback in callbacks:
                        callback.on_epoch_end(epoch, logs=logs)
                
        end_time_adam = time.time()
        adam_duration = end_time_adam - start_time_adam

        # Call on_train_end for callbacks
        if callbacks:
            for callback in callbacks:
                callback.on_train_end()
        print()
        print(f"Adam training finished in {adam_duration:.2f} seconds.")
        print(f"Final weight_data: {self.weight_data.numpy():.6f}")
        print(f"Final weight_pde: {self.weight_pde.numpy():.6f}")

        return duration_data_only, adam_duration, 0.0 # L-BFGS-B duration is 0.0 as it's removed

    def compute_inverse_loss(self, x_data_inv, y_data_inv, t_data_inv, u_data_inv, v_data_inv, nu_val_for_loss):
        u_pred_inverse, v_pred_inverse = self.predict_velocity_inverse(
            x_data_inv, y_data_inv, t_data_inv, nu_val_for_loss)
        loss_inverse = (tf.reduce_mean(tf.square(u_data_inv - u_pred_inverse)) +
                                tf.reduce_mean(tf.square(v_data_inv - v_pred_inverse)))
        # Add nu regularization term (Range-Based Regularization for real-world scenario)
        nu_min_physical = 0.001 # Define based on physical knowledge
        nu_max_physical = 0.1   # Define based on physical knowledge

        penalty_lower = tf.maximum(0.0, nu_min_physical - nu_val_for_loss)
        penalty_upper = tf.maximum(0.0, nu_val_for_loss - nu_max_physical)

        nu_reg_loss = self.nu_regularization_weight * (tf.square(penalty_lower) + tf.square(penalty_upper))
        return loss_inverse + nu_reg_loss

    def loss_and_grads_scipy_inverse(self, flat_log_nu, x_data_inv_np, y_data_inv_np, t_data_inv_np, u_data_inv_np, v_data_inv_np):
        """
        A wrapper function to compute loss and gradients for the SciPy L-BFGS-B optimizer for the inverse problem.
        """
        # Assign the new log_nu value from the optimizer
        self.log_nu_inverse.assign(tf.constant(flat_log_nu[0], dtype=tf.float32))

        # Convert NumPy arrays to TensorFlow tensors
        x_data_inv = tf.constant(x_data_inv_np, dtype=tf.float32)
        y_data_inv = tf.constant(y_data_inv_np, dtype=tf.float32)
        t_data_inv = tf.constant(t_data_inv_np, dtype=tf.float32)
        u_data_inv = tf.constant(u_data_inv_np, dtype=tf.float32)
        v_data_inv = tf.constant(v_data_inv_np, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(self.log_nu_inverse)
            nu_inverse_traced = tf.exp(self.log_nu_inverse)
            loss_inverse = self.compute_inverse_loss(
                x_data_inv, y_data_inv, t_data_inv, u_data_inv, v_data_inv, nu_inverse_traced
            )

        # Compute gradients with respect to log_nu_inverse
        grad = tape.gradient(loss_inverse, self.log_nu_inverse)
        
        # Return loss and a flattened gradient array
        return loss_inverse.numpy().astype(np.float64), np.array([grad.numpy()]).astype(np.float64)

    def train_inverse_problem(self, x_data_inv, y_data_inv, t_data_inv, u_data_inv, v_data_inv, epochs_inverse_adam, epochs_inverse_adam_pretrain):
        """
        Trains only the nu_inverse parameter using L-BFGS-B optimizer for Stage 2.
        """
        print("\n" + "-" * 50)
        print("Starting Stage 2: Inverse Problem (Discovering nu with Hybrid Optimization)...")
        
        # --- Adam Pre-training for log_nu_inverse ---
        print(f"Starting Adam pre-training for nu_inverse for {epochs_inverse_adam_pretrain} epochs...")
        start_time_adam_inverse_pretrain = time.time()
        for epoch in range(epochs_inverse_adam_pretrain):
            with tf.GradientTape() as tape:
                tape.watch(self.log_nu_inverse)
                nu_inverse_traced = tf.exp(self.log_nu_inverse)
                loss_inverse_adam = self.compute_inverse_loss(
                    x_data_inv, y_data_inv, t_data_inv, u_data_inv, v_data_inv, nu_inverse_traced
                )
            gradients_inverse_adam = tape.gradient(loss_inverse_adam, self.log_nu_inverse)
            self.optimizer_adam_inverse.apply_gradients([(gradients_inverse_adam, self.log_nu_inverse)])
            
            if epoch % 100 == 0:
                print(f"  Adam Inverse Pre-train Epoch {epoch}: Loss = {loss_inverse_adam:.6f}, Discovered nu = {tf.exp(self.log_nu_inverse).numpy():.6f}")
            if tf.math.is_nan(loss_inverse_adam):
                print(f"[ERROR] Adam Inverse Pre-train Loss is NaN at epoch {epoch}. Stopping pre-training.")
                break
        end_time_adam_inverse_pretrain = time.time()
        adam_inverse_pretrain_duration = end_time_adam_inverse_pretrain - start_time_adam_inverse_pretrain
        print(f"Adam pre-training for nu_inverse finished in {adam_inverse_pretrain_duration:.2f} seconds.")

        # --- L-BFGS-B Optimization for log_nu_inverse ---
        print("Starting L-BFGS-B optimization for nu_inverse...")
        start_time_inverse_lbfgs = time.time()

        initial_log_nu = [self.log_nu_inverse.numpy()]

        scipy_results_inverse = scipy.optimize.minimize(
            fun=self.loss_and_grads_scipy_inverse,
            x0=initial_log_nu,
            method='L-BFGS-B',
            jac=True,
            args=(x_data_inv, y_data_inv, t_data_inv, u_data_inv, v_data_inv),
            options={'maxiter': 500000, 'maxfun': 500000, 'maxcor': 50,
                     'maxls': 50, 'ftol': 1e-10} # Relaxed ftol for inverse problem (increased iterations)
        )
        
        end_time_inverse_lbfgs = time.time()
        inverse_lbfgs_duration = end_time_inverse_lbfgs - start_time_inverse_lbfgs

        # Update log_nu_inverse with the optimized value
        self.log_nu_inverse.assign(tf.constant(scipy_results_inverse.x[0], dtype=tf.float32))
        
        print(f"Stage 2 (Inverse Problem) L-BFGS-B training finished in {inverse_lbfgs_duration:.2f} seconds.")
        print(f"L-BFGS-B converged: {scipy_results_inverse.success}")
        print(f"L-BFGS-B message: {scipy_results_inverse.message}")
        print(f"L-BFGS-B iterations: {scipy_results_inverse.nit}")

        final_nu_inverse = tf.exp(self.log_nu_inverse).numpy()
        return final_nu_inverse, inverse_lbfgs_duration, adam_inverse_pretrain_duration

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
    parser = argparse.ArgumentParser(description="PINN for 2D Burgers' equation with nu discovery.")
    parser.add_argument('--nu_initial', type=float, default=0.01,
                        help='Initial kinematic viscosity for the curriculum training range (nu_min_train).')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for reproducibility.')
    parser.add_argument('--nu_true', type=float, default=0.05,
                        help='True kinematic viscosity for data generation and inverse problem.')
    parser.add_argument('--noise_level', type=float, default=0.0, help='Percentage of Gaussian noise to add to the data (e.g., 0.01 for 1%).')
    parser.add_argument('--adam_epochs_stage1', type=int, default=5000,
                        help='Number of Adam epochs for Stage 1 (parametric PINN training).')
    parser.add_argument('--epochs_inverse_adam_stage2', type=int, default=5000,
                        help='Number of Adam epochs for Stage 2 (inverse problem, nu discovery).')
    parser.add_argument('--epochs_inverse_adam_pretrain', type=int, default=1000,
                        help='Number of Adam pre-training epochs for Stage 2 inverse problem.')
    args = parser.parse_args()

    # --- Seed Configuration ---
    # Set random seeds for reproducibility
    seed_value = args.seed
    tf.random.set_seed(seed_value)
    np.random.seed(seed_value)
    print(f"Running with seed: {seed_value}")
    # ---

    # --- Main Parameters ---

    # Grid and Time Configuration
    grid_points_x = 41
    grid_points_y = 41
    time_steps = 50
    true_kinematic_viscosity = args.nu_true  # Ground truth for 'nu' (for data generation)
    nu_min_train = args.nu_initial # Range for nu during Stage 1 training
    nu_max_train = 0.1 # Max nu for curriculum training range

    # Domain Boundaries
    x_min, x_max = 0.0, 2.0
    y_min, y_max = 0.0, 2.0
    t_min, t_max = 0.0, time_steps * 0.001

    # Neural Network Architecture
    # Input: (x, y, t, nu), Output: (u, v)
    layers_config = [4, 60, 60, 60, 60, 60, 2] # Added one hidden layer
    
    # --- Experiment Configuration (Stage 1) ---
    adam_epochs_stage1 = args.adam_epochs_stage1
    epochs_data_only_stage1 = 500 # Increased for full training
    lambda_data_weight_stage1 = 1.0
    lambda_pde_weight_stage1 = 1.0
    num_pde_points_stage1 = 10000

    # Initialize ReduceLROnPlateau callback
    reduce_lr_on_plateau = callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.2, # Adjusted factor
        patience=50, # Adjusted patience
        min_lr=1e-7, # Adjusted min_lr
        verbose=1
    )

    # --- Experiment Configuration (Stage 2) ---
    epochs_inverse_adam_stage2 = args.epochs_inverse_adam_stage2
    epochs_inverse_adam_pretrain = args.epochs_inverse_adam_pretrain
    true_nu_for_inverse_problem = args.nu_true # The 'unknown' nu we want to discover in Stage 2
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

    u_true_list_stage1, v_true_list_stage1, t_true_list_stage1 = (
        [u.numpy() for u in u_true_tf_stage1], [v.numpy() for v in v_true_tf_stage1], [t.numpy() for t in t_true_tf_stage1])
    u_true_list_stage2, v_true_list_stage2, t_true_list_stage2 = (
        [u.numpy() for u in u_true_tf_stage2], [v.numpy() for v in v_true_tf_stage2], [t.numpy() for t in t_true_tf_stage2])

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

    x_data_tf = tf.constant(X_data_flat, dtype=tf.float32)
    y_data_tf = tf.constant(Y_data_flat, dtype=tf.float32)
    t_data_tf = tf.constant(T_data_flat, dtype=tf.float32)
    # Add Gaussian noise to the data if noise_level > 0
    if args.noise_level > 0:
        print(f"Adding {args.noise_level*100:.2f}% Gaussian noise to data.")
        # Calculate standard deviation of the data
        std_u = np.std(U_data_flat)
        std_v = np.std(V_data_flat)
        
        # Generate Gaussian noise
        noise_u = np.random.normal(0, std_u * args.noise_level, U_data_flat.shape)
        noise_v = np.random.normal(0, std_v * args.noise_level, V_data_flat.shape)
        
        # Add noise to the data
        U_data_flat = U_data_flat + noise_u.astype(np.float32)
        V_data_flat = V_data_flat + noise_v.astype(np.float32)

    u_data_tf = tf.constant(U_data_flat, dtype=tf.float32)
    v_data_tf = tf.constant(V_data_flat, dtype=tf.float32)
    data_prep_duration = time.time() - data_prep_start_time
    # ---

    # --- Model Initialization ---
    model_init_start_time = time.time()
    pinn = PINN_Burgers2D(
        layers_config, x_data_tf, y_data_tf, t_data_tf,
        u_data_tf, v_data_tf,
        None, None, None, # x_pde, y_pde, t_pde are not placeholders anymore
        None, # nu_pde is not a placeholder anymore
        x_min, x_max, y_min, y_max, t_min, t_max,
        nu_min_train, nu_max_train, # Pass nu_min_train, nu_max_train
        true_kinematic_viscosity, # Pass true_kinematic_viscosity for regularization
        annealing_rate=0.1) # Adjusted annealing rate
    model_init_duration = time.time() - model_init_start_time
    # ---

    print(f"--- Stage 1: Parametric PINN Training ---")
    print(f"Training with: Adam Epochs = {adam_epochs_stage1}, Data-Only Epochs = {epochs_data_only_stage1}, Lambda Data = {lambda_data_weight_stage1}, Lambda PDE = {lambda_pde_weight_stage1}, PDE Points = {num_pde_points_stage1}, Nu Range = [{nu_min_train}, {nu_max_train}])")

    # --- Model Training (Stage 1) ---
    duration_data_only_stage1, adam_duration_stage1, _ = pinn.fit(
        epochs_adam=adam_epochs_stage1, epochs_data_only=epochs_data_only_stage1,
        num_pde_points=num_pde_points_stage1, x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max, t_min=t_min, t_max=t_max,
        callbacks=[reduce_lr_on_plateau])
    # ---

    # --- Stage 2: Inverse Problem (Discovering a specific nu) ---
    # Prepare data for the inverse problem (using the last snapshot of true data for true_nu_for_inverse_problem)
    X_data_inverse_flat = X_np.flatten()[:, None].astype(np.float32)
    Y_data_inverse_flat = Y_np.flatten()[:, None].astype(np.float32)
    T_data_inverse_flat = np.full_like(X_data_inverse_flat, t_true_list_stage2[-1]).astype(np.float32)
    U_data_inverse_flat = u_true_list_stage2[-1].flatten()[:, None].astype(np.float32)
    V_data_inverse_flat = v_true_list_stage2[-1].flatten()[:, None].astype(np.float32)

    final_nu_inverse, inverse_lbfgs_duration, adam_inverse_pretrain_duration = pinn.train_inverse_problem(
        X_data_inverse_flat, Y_data_inverse_flat, T_data_inverse_flat,
        U_data_inverse_flat, V_data_inverse_flat, epochs_inverse_adam_stage2, epochs_inverse_adam_pretrain)

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
        u_pinn_pred_flat, v_pinn_pred_flat = pinn.predict_velocity(
            tf.constant(X_plot_flat, dtype=tf.float32),
            tf.constant(Y_plot_flat, dtype=tf.float32),
            tf.constant(T_plot_flat, dtype=tf.float32),
            tf.constant(Nu_plot_flat, dtype=tf.float32)
        )
        results_for_plotting[f'u_pred_nu_{nu_val}'] = u_pinn_pred_flat.numpy().reshape((grid_points_y, grid_points_x))
        results_for_plotting[f'v_pred_nu_{nu_val}'] = v_pinn_pred_flat.numpy().reshape((grid_points_y, grid_points_x))

        # Calculate MSE for u and v predictions for this nu_val (against true_kinematic_viscosity data)
        # Note: This MSE is against the data generated with true_kinematic_viscosity,
        # not necessarily the nu_val being plotted.
        if nu_val == true_kinematic_viscosity:
            mse_u_stage1_plot = np.mean((u_true_list_stage1[-1].flatten() - u_pinn_pred_flat.numpy())**2)
            mse_v_stage1_plot = np.mean((v_true_list_stage1[-1].flatten() - v_pinn_pred_flat.numpy())**2)
            total_mse_stage1_plot = mse_u_stage1_plot + mse_v_stage1_plot
            print(f"Prediction MSE (u) for nu={nu_val} (Stage 1 model): {mse_u_stage1_plot:.6e}")
            print(f"Prediction MSE (v) for nu={nu_val} (Stage 1 model): {mse_v_stage1_plot:.6e}")
            print(f"Total Prediction MSE (u+v) for nu={nu_val} (Stage 1 model): {total_mse_stage1_plot:.6e}")
            results_for_plotting['mse_u_stage1_plot'] = mse_u_stage1_plot
            results_for_plotting['mse_v_stage1_plot'] = mse_v_stage1_plot
            results_for_plotting['total_mse_stage1_plot'] = total_mse_stage1_plot

    # Calculate MSE for Stage 2 discovered nu against its true data
    u_pred_inverse_final, v_pred_inverse_final = pinn.predict_velocity_inverse(
        tf.constant(X_data_inverse_flat, dtype=tf.float32),
        tf.constant(Y_data_inverse_flat, dtype=tf.float32),
        tf.constant(T_data_inverse_flat, dtype=tf.float32),
        tf.constant(final_nu_inverse, dtype=tf.float32) # Use the discovered nu
    )
    mse_u_stage2_final = np.mean((U_data_inverse_flat - u_pred_inverse_final.numpy())**2)
    mse_v_stage2_final = np.mean((V_data_inverse_flat - v_pred_inverse_final.numpy())**2)
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
        duration_lbfgs_stage1=0.0, # L-BFGS-B removed from Stage 1
        duration_inverse_adam_stage2=inverse_lbfgs_duration,
        duration_inverse_adam_pretrain=adam_inverse_pretrain_duration, # New duration
    )

    print(f"Results saved to {output_file}")
    print("--- " * 10 + " HPC Performance Metrics " + " ---" * 10)
    print(f"Data Preparation Duration: {data_prep_duration:.2f} seconds")
    print(f"Model Initialization Duration: {model_init_duration:.2f} seconds")
    print(f"Stage 1 Data-Only Pre-training Duration: {duration_data_only_stage1:.2f} seconds")
    print(f"Stage 1 Adam Training Duration: {adam_duration_stage1:.2f} seconds")
    print(f"Stage 1 L-BFGS-B Training Duration: 0.00 seconds (Removed)")
    print(f"Stage 2 Inverse Problem Adam Pre-training Duration: {adam_inverse_pretrain_duration:.2f} seconds")
    print(f"Stage 2 Inverse Problem L-BFGS-B Training Duration: {inverse_lbfgs_duration:.2f} seconds")
    print(f"Total Execution Duration: {overall_duration:.2f} seconds")
