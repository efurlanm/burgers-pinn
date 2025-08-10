"""
This script visualizes the results from a PINN (Physics-Informed Neural Network)
model for the 2D Burgers' equation, specifically for the "shape" focus.
It loads the ground truth and predicted data from a specified .npz file and
generates a side-by-side 3D plot for comparison.

Usage:
    python plot_results_shape.py [input_file.npz]

If no input file is provided, it defaults to using a sample 'shape' .npz file.
The output plot is saved to the 'results/' directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import os

# --- File Handling ---

# Define the results directory
results_dir = 'results'

# Set the default input file if none is provided via command line
default_input_file = 'shape_results_data_100.0_pde_1.0_epochs_1000_pdepoints_60000_seed_1.npz'
input_file = sys.argv[1] if len(sys.argv) > 1 else default_input_file

# Construct the full path to the input file
input_path = os.path.join(results_dir, input_file)

# Check if the input file exists
if not os.path.exists(input_path):
    print(f"Error: Input file '{input_path}' not found.")
    sys.exit(1)

# Output plot filename
output_filename = 'burgers2d_shape_comparison.jpg'
output_path = os.path.join(results_dir, output_filename)


# --- Load Data ---
print(f"Loading results from: {input_path}")
try:
    data = np.load(input_path)
    X_np = data['X']
    Y_np = data['Y']
    u_true = data['u_true']
    u_pinn_pred = data['u_pinn_pred']
    nu_pinn = data['nu_pinn']
    true_nu = data.get('true_nu') # Use .get() for backward compatibility
    mse_u = data.get('mse_u')
    mse_v = data.get('mse_v')
    total_mse = data.get('total_mse')
except KeyError as e:
    print(f"Error: The input file is missing a required key: {e}")
    sys.exit(1)


# --- Visualization ---
fig = plt.figure(figsize=(18, 8), dpi=120)

# First subplot: Ground Truth Solution
ax1 = fig.add_subplot(121, projection='3d')
if true_nu is not None:
    ax1.set_title(f"Ground Truth Solution (nu = {true_nu:.4f})")
else:
    ax1.set_title("Ground Truth Solution")
surf1 = ax1.plot_surface(X_np, Y_np, u_true, cmap=cm.viridis, antialiased=True)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')
ax1.set_zlabel('u-velocity')

# Second subplot: PINN Predicted Solution
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title(f"PINN Predicted Solution (nu = {nu_pinn:.6f})\nMSE (u) = {mse_u:.2e}, MSE (v) = {mse_v:.2e}\nTotal MSE = {total_mse:.2e}")
surf2 = ax2.plot_surface(X_np, Y_np, u_pinn_pred, cmap=cm.viridis, antialiased=True)
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Y-axis')
ax2.set_zlabel('u-velocity')

plt.tight_layout()
plt.savefig(output_path)

print(f"Plot saved to: {output_path}")


# --- Statistics ---
print("\n--- Solution Statistics ---")
error_l2 = np.linalg.norm(u_true - u_pinn_pred) / np.linalg.norm(u_true)
print(f"Relative L2 Error: {error_l2:.4e}")

if true_nu is not None:
    error_nu = abs(nu_pinn - true_nu) / true_nu
    print(f"Relative Error in nu: {error_nu:.4e}")

print(f"\nGround Truth: min={np.min(u_true):.4f}, max={np.max(u_true):.4f}, mean={np.mean(u_true):.4f}")
print(f"PINN Predicted: min={np.min(u_pinn_pred):.4f}, max={np.max(u_pinn_pred):.4f}, mean={np.mean(u_pinn_pred):.4f}")
