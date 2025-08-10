import tensorflow as tf
import numpy as np
import time
import tensorflow_probability as tfp


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
layers = [3, 40, 40, 40, 40, 2] # Input (x,y,t), 4 hidden layers, Output (u,v)

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
        self.optimizer_adam = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Trainable variables
        self.trainable_variables = self.weights + self.biases + [self.log_nu_pinn]

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
        with tf.GradientTape(persistent=True) as tape_outer:
            tape_outer.watch(x)
            tape_outer.watch(y)
            tape_outer.watch(t)
            with tf.GradientTape(persistent=True) as tape_inner:
                tape_inner.watch(x)
                tape_inner.watch(y)
                tape_inner.watch(t)
                u, v = self.predict_u_v(x, y, t)
            u_x = tape_inner.gradient(u, x)
            u_y = tape_inner.gradient(u, y)
            u_t = tape_inner.gradient(u, t)
            v_x = tape_inner.gradient(v, x)
            v_y = tape_inner.gradient(v, y)
            v_t = tape_inner.gradient(v, t)
        u_xx = tape_outer.gradient(u_x, x)
        u_yy = tape_outer.gradient(u_y, y)
        v_xx = tape_outer.gradient(v_x, x)
        v_yy = tape_outer.gradient(v_y, y)
        del tape_outer
        del tape_inner
        nu = tf.exp(self.log_nu_pinn)
        f_u = u_t + u * u_x + v * u_y - nu * (u_xx + u_yy)
        f_v = v_t + u * v_x + v * v_y - nu * (v_xx + v_yy)
        return f_u, f_v

    def loss_function(self):
        u_pred_data, v_pred_data = self.predict_u_v(self.x_data, self.y_data, self.t_data)
        loss_data = tf.reduce_mean(tf.square(self.u_data - u_pred_data)) + \
                    tf.reduce_mean(tf.square(self.v_data - v_pred_data))
        f_u_pred, f_v_pred = self.pde_residual(self.x_pde, self.y_pde, self.t_pde)
        loss_pde = tf.reduce_mean(tf.square(f_u_pred)) + \
                   tf.reduce_mean(tf.square(f_v_pred))
        return loss_data + loss_pde

    def train_step_adam(self):
        with tf.GradientTape() as tape:
            loss = self.loss_function()
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer_adam.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def flatten_trainable_variables(self):
        return tf.concat([tf.reshape(var, [-1]) for var in self.trainable_variables], axis=0)

    def assign_trainable_variables(self, flat_variables):
        idx = 0
        for var in self.trainable_variables:
            shape = var.shape
            size = tf.size(var)
            var.assign(tf.reshape(flat_variables[idx:idx + size], shape))
            idx += size

    def train(self, epochs_adam):
        print("Starting Adam training...")
        for epoch in range(epochs_adam):
            loss_value = self.train_step_adam()
            if epoch % 100 == 0:
                print(f"Adam Epoch {epoch}: Loss = {loss_value.numpy():.6f}, Discovered nu = {tf.exp(self.log_nu_pinn).numpy():.6f}")

        print("Adam training finished.")
        print("Starting L-BFGS-B training...")

        initial_position = self.flatten_trainable_variables()

        # TFP L-BFGS-B optimizer
        def lbfgs_loss_and_grads(current_weights):
            self.assign_trainable_variables(current_weights)
            with tf.GradientTape() as tape:
                loss = self.loss_function()
            grads = tape.gradient(loss, self.trainable_variables)
            # Flatten gradients
            flat_grads = tf.concat([tf.reshape(grad, [-1]) for grad in grads], axis=0)
            return loss, flat_grads

        lbfgs_results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=lbfgs_loss_and_grads,
            initial_position=initial_position,
            num_correction_pairs=100,
            max_line_search_iterations=50,
            max_iterations=100000,
            tolerance=1e-6
        )

        # Update trainable variables with the optimized values
        self.assign_trainable_variables(lbfgs_results.position)

        print("L-BFGS-B training finished.")
        print(f"L-BFGS-B converged: {lbfgs_results.converged}")
        print(f"L-BFGS-B number of iterations: {lbfgs_results.num_iterations}")
        print(f"L-BFGS-B function evaluations: {lbfgs_results.num_objective_evaluations}")

    

# --- Data Generation ---
def forward_simulation_for_data(nx, ny, nt, dx, dy, dt, nu_val, u_initial, v_initial):
    u = tf.identity(u_initial)
    v = tf.identity(v_initial)
    for n in range(nt):
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
    return u, v

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

# Generate "measured" data
u_medido, v_medido = forward_simulation_for_data(nx, ny, nt, dx, dy, dt, tf.constant(nu_real, dtype=tf.float32), u_initial_tf, v_initial_tf)

# Prepare data for PINN training
X_data_flat = X_np.flatten()[:, None]
Y_data_flat = Y_np.flatten()[:, None]
T_data_flat = np.full_like(X_data_flat, t_max)
U_data_flat = u_medido.numpy().flatten()[:, None]
V_data_flat = v_medido.numpy().flatten()[:, None]

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
print(f"Initial nu_pinn guess: {tf.exp(pinn.log_nu_pinn).numpy():.6f}")

pinn.train(epochs_adam=2000) # 2000 epochs for Adam, then L-BFGS-B

print("-" * 50)
print(f"Final Discovered nu: {tf.exp(pinn.log_nu_pinn).numpy():.6f}")
print(f"True nu: {nu_real}")

# --- Save Results ---
X_plot_flat = X_np.flatten()[:, None]
Y_plot_flat = Y_np.flatten()[:, None]
T_plot_flat = np.full_like(X_plot_flat, t_max)

u_pinn_pred_flat, v_pinn_pred_flat = pinn.predict_u_v(
    tf.constant(X_plot_flat, dtype=tf.float32),
    tf.constant(Y_plot_flat, dtype=tf.float32),
    tf.constant(T_plot_flat, dtype=tf.float32)
)

u_pinn_pred = tf.reshape(u_pinn_pred_flat, (ny, nx)).numpy()

np.savez('pinn_results_03.npz',
         X=X_np,
         Y=Y_np,
         u_medido=u_medido.numpy(),
         u_pinn_pred=u_pinn_pred,
         nu_pinn=tf.exp(pinn.log_nu_pinn).numpy())

print("Resultados salvos em pinn_results_03.npz")