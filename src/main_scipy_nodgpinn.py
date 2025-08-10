import tensorflow as tf
import numpy as np
import time
import scipy.optimize # Import scipy for L-BFGS-B
import tensorflow.compat.v1 as tf_v1 # Use TF1.x compatibility for session

# Disable TF2.x behaviors for TF1.x compatibility
tf_v1.disable_eager_execution()

# Set random seeds for reproducibility
seed_value = 2 # You can change this for different runs
tf_v1.set_random_seed(seed_value)
np.random.seed(seed_value)


# --- Parameters ---
nx = 41
ny = 41
nt = 50 # Number of time steps for data generation and PINN training
nu_real = 0.05 # True viscosity for generating "measured" data

# Domain bounds
x_min, x_max = 0.0, 2.0
y_min, y_max = 0.0, 2.0
t_min, t_max = 0.0, nt * 0.001

# Neural Network Architecture
layers = [3, 60, 60, 60, 60, 2] # Input (x,y,t), 4 hidden layers, Output (u,v)

# --- PINN Setup ---
class PINN_Burgers2D:
    def __init__(self, layers, nu_real, x_data, y_data, t_data, u_data, v_data, x_pde, y_pde, t_pde, x_min, x_max, y_min, y_max, t_min, t_max):
        self.layers = layers
        self.nu_real = nu_real

        # Store data
        self.x_data = x_data
        self.y_data = y_data
        self.t_data = t_data
        self.u_data = u_data
        self.v_data = v_data
        self.x_pde = x_pde
        self.y_pde = y_pde
        self.t_pde = t_pde

        # Store domain bounds for scaling
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.t_min, self.t_max = t_min, t_max

        # Initialize neural network weights and biases
        self.weights, self.biases = self.initialize_neural_network(layers)

        # Discoverable parameter (log of kinematic viscosity)
        self.log_nu_pinn = tf.Variable(tf.math.log(0.06), dtype=tf.float32, name="log_nu_pinn")

        # Adam Optimizer
        self.optimizer_adam = tf_v1.train.AdamOptimizer(learning_rate=0.001)

        # Trainable variables
        self.trainable_variables = self.weights + self.biases + [self.log_nu_pinn] # Ensure log_nu_pinn is last

        # TensorFlow session for SciPy optimizer
        self.sess = tf_v1.Session(config=tf_v1.ConfigProto(allow_soft_placement=True, log_device_placement=False))

        # Define loss and gradients for SciPy optimizer
        self.loss_tf = self.loss_function()
        self.grads_tf = tf_v1.gradients(self.loss_tf, self.trainable_variables)

        # Define Adam training operation (creates optimizer variables)
        self.train_op_adam = self.optimizer_adam.minimize(self.loss_tf, var_list=self.trainable_variables)

        # Initialize all TensorFlow variables (including optimizer's)
        init = tf_v1.global_variables_initializer()
        self.sess.run(init)


    def initialize_neural_network(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(num_layers - 1):
            weight = self.xavier_initializer(size=[layers[l], layers[l + 1]])
            bias = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(weight)
            biases.append(bias)
        return weights, biases

    def xavier_initializer(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_network_model(self, X):
        num_layers = len(self.weights) + 1
        
        # Apply input scaling
        x_scaled = 2.0 * (X[:, 0:1] - self.x_min) / (self.x_max - self.x_min) - 1.0
        y_scaled = 2.0 * (X[:, 1:2] - self.y_min) / (self.y_max - self.y_min) - 1.0
        t_scaled = 2.0 * (X[:, 2:3] - self.t_min) / (self.t_max - self.t_min) - 1.0
        H = tf.concat([x_scaled, y_scaled, t_scaled], axis=1)
        for l in range(num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = self.weights[-1]
        b = self.biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def predict_u_v(self, x, y, t):
        X_input = tf.concat([x, y, t], axis=1)
        uv = self.neural_network_model(X_input)
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        return u, v

    def pde_residual(self, x, y, t):
        # Note: For SciPy optimizer, we need to use TF1.x style gradient computation
        # This means we need to define placeholders for x, y, t if they are not already
        # part of the graph. However, since they are passed as tf.constant, we can
        # directly use them in the graph definition.
        # The original code used tf.GradientTape, which is TF2.x eager execution style.
        # For TF1.x graph mode, we define the operations and then run them in a session.

        # Create symbolic tensors for inputs if not already part of the graph
        # For this setup, x, y, t are already tf.constant, so they are part of the graph
        # when loss_function is called during graph construction.

        u, v = self.predict_u_v(x, y, t)

        u_t = tf_v1.gradients(u, t)[0]
        u_x = tf_v1.gradients(u, x)[0]
        u_y = tf_v1.gradients(u, y)[0]

        v_t = tf_v1.gradients(v, t)[0]
        v_x = tf_v1.gradients(v, x)[0]
        v_y = tf_v1.gradients(v, y)[0]

        u_xx = tf_v1.gradients(u_x, x)[0]
        u_yy = tf_v1.gradients(u_y, y)[0]

        v_xx = tf_v1.gradients(v_x, x)[0]
        v_yy = tf_v1.gradients(v_y, y)[0]

        nu = tf.exp(self.log_nu_pinn)
        f_u = u_t + u * u_x + v * u_y - nu * (u_xx + u_yy)
        f_v = v_t + u * v_x + v * v_y - nu * (v_xx + v_yy)
        return f_u, f_v

    def loss_function(self):
        u_pred_data, v_pred_data = self.predict_u_v(self.x_data, self.y_data, self.t_data)
        loss_data = (tf.reduce_mean(tf.square(self.u_data - u_pred_data)) + tf.reduce_mean(tf.square(self.v_data - v_pred_data)))
        f_u_pred, f_v_pred = self.pde_residual(self.x_pde, self.y_pde, self.t_pde)
        loss_pde = tf.reduce_mean(tf.square(f_u_pred)) + \
                   tf.reduce_mean(tf.square(f_v_pred))
        return 10 * loss_data + 5 * loss_pde

    def train_step_adam(self):
        # Execute the pre-defined Adam training operation
        self.sess.run(self.train_op_adam)
        # Fetch the current loss value after the training step
        loss_value = self.sess.run(self.loss_tf)
        return loss_value

    def flatten_trainable_variables(self):
        return tf.concat([tf.reshape(var, [-1]) for var in self.trainable_variables], axis=0)

    def assign_trainable_variables(self, flat_variables):
        # Create a list of assignment operations
        assign_ops = []
        idx = 0
        for var in self.trainable_variables:
            shape = var.shape
            size = tf.size(var)
            assign_ops.append(tf_v1.assign(var, tf.reshape(flat_variables[idx:idx + size], shape)))
            idx += size
        return assign_ops

    def _loss_and_grads_scipy(self, flat_weights):
        # Assign flat_weights to trainable variables
        assign_ops = self.assign_trainable_variables(tf.constant(flat_weights, dtype=tf.float32))
        self.sess.run(assign_ops)
        
        # Compute loss and gradients
        loss_value, grads_value = self.sess.run([self.loss_tf, self.grads_tf])
        
        # Check for None gradients
        for i, grad in enumerate(grads_value):
            if grad is None:
                print(f"WARNING: Gradient for trainable_variables[{i}] is None.")
                # Handle None gradient, e.g., replace with zeros
                grads_value[i] = np.zeros_like(self.sess.run(self.trainable_variables[i]))

        # Flatten gradients
        flat_grads = np.concatenate([grad.flatten() for grad in grads_value])
        
        print(f"  L-BFGS-B: Loss = {loss_value:.6e}, Grad Norm = {np.linalg.norm(flat_grads):.6e}, nu_pinn_grad = {grads_value[-1]:.6e}")
        return loss_value.astype(np.float64), flat_grads.astype(np.float64)

    def train(self, epochs_adam):
        print("Starting Adam training...")
        for epoch in range(epochs_adam):
            loss_value = self.train_step_adam()
            if epoch % 10 == 0:
                current_loss, current_nu = self.sess.run([self.loss_tf, tf.exp(self.log_nu_pinn)])
                print(f"Adam Epoch {epoch}: Loss = {current_loss:.6f}, Discovered nu = {current_nu:.6f}")

        print("Adam training finished.")
        print("Starting L-BFGS-B training with SciPy...")

        initial_position = self.sess.run(self.flatten_trainable_variables()) # Convert to numpy for scipy

        # SciPy L-BFGS-B optimizer
        scipy_results = scipy.optimize.minimize(
            fun=self._loss_and_grads_scipy,
            x0=initial_position,
            method='L-BFGS-B',
            jac=True, # Indicate that fun returns both loss and Jacobian
            options={
                'maxiter': 100000,
                'maxfun': 100000,
                'maxcor': 100,
                'maxls': 50,
                'ftol': 1e-20 # Even more relaxed tolerance for SciPy
            }
        )

        # Update trainable variables with the optimized values
        assign_ops = self.assign_trainable_variables(tf.constant(scipy_results.x, dtype=tf.float32))
        self.sess.run(assign_ops)

        print("L-BFGS-B training finished.")
        print(f"L-BFGS-B converged: {scipy_results.success}") # scipy.optimize.minimize uses 'success' for convergence
        print(f"L-BFGS-B message: {scipy_results.message}")
        print(f"L-BFGS-B number of iterations: {scipy_results.nit}")
        print(f"L-BFGS-B function evaluations: {scipy_results.nfev}")

    

# --- Data Generation ---
def forward_simulation_for_data(nx, ny, nt, dx, dy, dt, nu_val, u_initial, v_initial):
    u = tf.identity(u_initial)
    v = tf.identity(v_initial)
    
    u_data = []
    v_data = []
    t_data = []

    for n in range(nt + 1):
        if n in [int(nt/4), int(nt/2), int(3*nt/4), nt]:
            u_data.append(u)
            v_data.append(v)
            t_data.append(tf.constant(n * dt, dtype=tf.float32))

        un = tf.identity(u)
        vn = tf.identity(v)
        u_interior = un[1:-1, 1:-1]
        v_interior = vn[1:-1, 1:-1]
        u_x = (un[1:-1, 2:] - un[1:-1, :-2]) / (2 * dx)
        u_y = (un[2:, 1:-1] - un[:-2, 1:-1]) / (2 * dy)
        v_x = (vn[1:-1, 2:] - vn[1:-1, :-2]) / (2 * dx)
        v_y = (vn[2:, 1:-1] - vn[:-2, 1:-1]) / (2 * dy)
        u_xx = (un[1:-1, 2:] - 2 * u_interior + un[1:-1, :-2]) / dx**2
        u_yy = (un[2:, 1:-1] - 2 * u_interior + un[:-2, 1:-1]) / dy**2
        v_xx = (vn[1:-1, 2:] - 2 * v_interior + vn[1:-1, :-2]) / dx**2
        v_yy = (vn[2:, 1:-1] - 2 * v_interior + vn[:-2, 1:-1]) / dy**2
        u_conv = u_interior * u_x + v_interior * u_y
        v_conv = u_interior * v_x + v_interior * v_y
        u_diff = nu_val * (u_xx + u_yy)
        v_diff = nu_val * (v_xx + v_yy)
        u_next = u_interior - dt * u_conv + dt * u_diff
        v_next = v_interior - dt * v_conv + dt * v_diff
        u = tf.tensor_scatter_nd_update(un, [[j, i] for j in range(1, ny-1) for i in range(1, nx-1)], tf.reshape(u_next, [-1]))
        v = tf.tensor_scatter_nd_update(vn, [[j, i] for j in range(1, ny-1) for i in range(1, nx-1)], tf.reshape(v_next, [-1]))
    return u_data, v_data, t_data

# Grid for data generation
x_np = np.linspace(x_min, x_max, nx)
y_np = np.linspace(y_min, y_max, ny)
X_np, Y_np = np.meshgrid(x_np, y_np)
dx = x_np[1] - x_np[0]
dy = y_np[1] - y_np[0]
dt = 0.001

# Initial conditions
center_x, center_y = 1.0, 1.0
sigma_x, sigma_y = 0.25, 0.25
u_initial_np = np.exp(-((X_np - center_x)**2 / (2 * sigma_x**2) + (Y_np - center_y)**2 / (2 * sigma_y**2)))
v_initial_np = np.exp(-((X_np - center_x)**2 / (2 * sigma_x**2) + (Y_np - center_y)**2 / (2 * sigma_y**2)))
u_initial_tf = tf.constant(u_initial_np, dtype=tf.float32)
v_initial_tf = tf.constant(v_initial_np, dtype=tf.float32)

# Generate "measured" data (TensorFlow symbolic tensors)
u_medido_tf, v_medido_tf, t_medido_tf = forward_simulation_for_data(nx, ny, nt, dx, dy, dt, tf.constant(nu_real, dtype=tf.float32), u_initial_tf, v_initial_tf)

# Create a global session to run the initial data generation
with tf_v1.Session(config=tf_v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess_global:
    # Initialize global variables (if any are created before PINN class)
    sess_global.run(tf_v1.global_variables_initializer())
    u_medido_list, v_medido_list, t_medido_list = sess_global.run([u_medido_tf, v_medido_tf, t_medido_tf])

# Prepare data for PINN training
X_data_flat_list = []
Y_data_flat_list = []
T_data_flat_list = []
U_data_flat_list = []
V_data_flat_list = []

for i in range(len(t_medido_list)):
    X_data_flat_list.append(X_np.flatten()[:, None])
    Y_data_flat_list.append(Y_np.flatten()[:, None])
    T_data_flat_list.append(np.full_like(X_np.flatten()[:, None], t_medido_list[i]))
    U_data_flat_list.append(u_medido_list[i].flatten()[:, None])
    V_data_flat_list.append(v_medido_list[i].flatten()[:, None])

X_data_flat = np.concatenate(X_data_flat_list)
Y_data_flat = np.concatenate(Y_data_flat_list)
T_data_flat = np.concatenate(T_data_flat_list)
U_data_flat = np.concatenate(U_data_flat_list)
V_data_flat = np.concatenate(V_data_flat_list)

num_pde_points = 80000
x_pde = tf.constant(np.random.uniform(x_min, x_max, (num_pde_points, 1)), dtype=tf.float32)
y_pde = tf.constant(np.random.uniform(y_min, y_max, (num_pde_points, 1)), dtype=tf.float32)
t_pde = tf.constant(np.random.uniform(t_min, t_max, (num_pde_points, 1)), dtype=tf.float32)

x_data_tf = tf.constant(X_data_flat, dtype=tf.float32)
y_data_tf = tf.constant(Y_data_flat, dtype=tf.float32)
t_data_tf = tf.constant(T_data_flat, dtype=tf.float32)
u_data_tf = tf.constant(U_data_flat, dtype=tf.float32)
v_data_tf = tf.constant(V_data_flat, dtype=tf.float32)

# --- Training ---
pinn = PINN_Burgers2D(layers, nu_real, x_data_tf, y_data_tf, t_data_tf, u_data_tf, v_data_tf, x_pde, y_pde, t_pde, x_min, x_max, y_min, y_max, t_min, t_max)

print(f"True nu: {nu_real}")
print(f"Initial nu_pinn guess: {np.exp(pinn.sess.run(pinn.log_nu_pinn)):.6f}") # Use sess.run for TF1.x variable

pinn.train(epochs_adam=2000) # 2000 epochs for Adam, then L-BFGS-B

print("-" * 50)
print(f"Final Discovered nu: {np.exp(pinn.sess.run(pinn.log_nu_pinn)):.6f}") # Use sess.run for TF1.x variable
print(f"True nu: {nu_real}")

# --- Save Results ---
X_plot_flat = X_np.flatten()[:, None]
Y_plot_flat = Y_np.flatten()[:, None]
T_plot_flat = np.full_like(X_plot_flat, t_max)

# Need to use TF1.x style prediction for consistency with the session
u_pinn_pred_flat, v_pinn_pred_flat = pinn.sess.run(
    pinn.predict_u_v(tf.constant(X_plot_flat, dtype=tf.float32),
                      tf.constant(Y_plot_flat, dtype=tf.float32),
                      tf.constant(T_plot_flat, dtype=tf.float32))
)

u_pinn_pred = u_pinn_pred_flat.reshape((ny, nx)) # Access the first element of the tuple and reshape

np.savez('pinn_results_03_scipy_test.npz', # Changed filename
         X=X_np,
         Y=Y_np,
         u_medido=u_medido_list[-1],
         u_pinn_pred=u_pinn_pred,
         nu_pinn=np.exp(pinn.sess.run(pinn.log_nu_pinn)))

print("Resultados salvos em pinn_results_03_scipy_test.npz")
