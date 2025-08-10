import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# This program solves the 2D Burgers' equations using an explicit finite difference method.
# The equations are:
# u_t + u*u_x + v*u_y = nu*(u_xx + u_yy)
# v_t + u*v_x + v*v_y = nu*(v_xx + v_yy)

# --- Problem Parameters ---
nx = 41  # Number of grid points in the x-direction
ny = 41  # Number of grid points in the y-direction
nt = 100 # Number of time steps
nu = 0.01 # Kinematic viscosity coefficient
dx = 2 / (nx - 1) # Spatial step size in x
dy = 2 / (ny - 1) # Spatial step size in y
dt = 0.001 # Time step size

# CFL (Courant-Friedrichs-Lewy) stability condition check:
# For an explicit scheme like this, the time step must be sufficiently small.
# Stability is governed by:
# dt <= (dx*dy) / (2*nu*(dx+dy) + max_u*dy + max_v*dx)
# Since the maximum velocity is not known a priori, dt is adjusted manually.
# The chosen value of 0.001 is small enough for this problem.
# A more robust check could be added inside the time loop.

# --- Grid Initialization and Initial Conditions ---

# Create the spatial grid
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

# Initial condition for u and v
# A Gaussian pulse is used as the initial condition in a region of the grid.
# The rest of the grid has zero velocity.
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))

# Position of the pulse center
center_x, center_y = 1.0, 1.0
sigma_x, sigma_y = 0.25, 0.25

# Create the Gaussian pulse
u_pulse = np.exp(-((X - center_x)**2 / (2 * sigma_x**2) + (Y - center_y)**2 / (2 * sigma_y**2)))
v_pulse = np.exp(-((X - center_x)**2 / (2 * sigma_x**2) + (Y - center_y)**2 / (2 * sigma_y**2)))

# Assign the pulse to the initial velocities
u = u_pulse
v = v_pulse

# Boundary Conditions (Dirichlet): The grid boundaries have zero velocity.
# Since we initialized the grid with zeros, the boundaries are already set.
# The boundaries will not be updated in the main loop.

# --- Main Solver Loop (Time-Stepping) ---
# This uses a Forward-Time, Backward-Space (FTBS) scheme for the convection term
# and a Forward-Time, Central-Space (FTCS) scheme for the diffusion term.

for n in range(nt + 1):
    un = u.copy()
    vn = v.copy()

    # Iterate over the internal grid points to apply the finite difference formula
    # The range starts at 1 and ends at nx-1 (or ny-1) to exclude the boundaries.
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            # Convection terms (using backward difference for stability)
            u_conv = un[j, i] * (un[j, i] - un[j, i - 1]) / dx + vn[j, i] * (un[j, i] - un[j - 1, i]) / dy
            v_conv = un[j, i] * (vn[j, i] - vn[j, i - 1]) / dx + vn[j, i] * (vn[j, i] - vn[j - 1, i]) / dy

            # Diffusion terms (using central difference)
            u_diff = nu * ((un[j, i + 1] - 2 * un[j, i] + un[j, i - 1]) / dx**2 + (un[j + 1, i] - 2 * un[j, i] + un[j - 1, i]) / dy**2)
            v_diff = nu * ((vn[j, i + 1] - 2 * vn[j, i] + vn[j, i - 1]) / dx**2 + (vn[j + 1, i] - 2 * vn[j, i] + vn[j - 1, i]) / dy**2)

            # Update the values of 'u' and 'v' using the explicit forward Euler scheme
            u[j, i] = un[j, i] - dt * u_conv + dt * u_diff
            v[j, i] = vn[j, i] - dt * v_conv + dt * v_diff


# --- Results Visualization ---

# Create the figure and subplots for visualization
fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(111, projection='3d')

# Plot the surface of the 'u' velocity
surf = ax.plot_surface(X, Y, u, cmap=cm.viridis)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('u-velocity')
ax.set_title('Final Velocity Distribution u(x, y, t)')

# Save the figure to the results directory
plt.savefig('results/burgers2d_diff_results.jpg')
