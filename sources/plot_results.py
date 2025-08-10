# -*- coding: utf-8 -*-
"""
@author: Gemini

This script visualizes the results from a PINN (Physics-Informed Neural Network)
model for the 2D Burgers' equation. It loads the ground truth and predicted data
from a specified .npz file and generates a side-by-side 3D plot for comparison.

Usage:
    python plot_results.py [input_file.npz]

If no input file is provided, it defaults to using 'pinn_results_precision.npz'
from the 'results/' directory. The output plot is saved to the same directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import os

# --- File Handling ---

# Define the results directory
results_dir = 'results'

# Define default input files for comparison
default_gt_file = 'precision_results_data_1.0_pde_1.0_epochs_1000.npz'
default_scipy_file = 'pinn_results_03_scipy_test.npz'
default_precision_file = 'precision_results_data_1.0_pde_1.0_epochs_2000_pdepoints_60000_seed_1.npz'

# Get input files from command line or use defaults
if len(sys.argv) > 3:
    gt_file = sys.argv[1]
    scipy_file = sys.argv[2]
    precision_file = sys.argv[3]
else:
    gt_file = default_gt_file
    scipy_file = default_scipy_file
    precision_file = default_precision_file

# Construct full paths
gt_path = os.path.join(results_dir, gt_file)
scipy_path = os.path.join(results_dir, scipy_file)
precision_path = os.path.join(results_dir, precision_file)

# Check if files exist
if not os.path.exists(gt_path):
    print(f"Error: Ground Truth file '{gt_path}' not found.")
    sys.exit(1)
if not os.path.exists(scipy_path):
    print(f"Error: SciPy PINN file '{scipy_path}' not found.")
    sys.exit(1)
if not os.path.exists(precision_path):
    print(f"Error: Precision PINN file '{precision_path}' not found.")
    sys.exit(1)

# Output plot filename
output_filename = 'burgers2d_precision_comparison.jpg'
output_path = os.path.join(results_dir, output_filename)


# --- Load Data ---
print(f"Loading Ground Truth from: {gt_path}")
gt_data = np.load(gt_path)
X_np = gt_data['X']
Y_np = gt_data['Y']
u_true = gt_data['u_true']
true_nu_gt = gt_data.get('true_nu')

print(f"Loading SciPy PINN results from: {scipy_path}")
scipy_data = np.load(scipy_path)
u_pinn_scipy = scipy_data['u_pinn_pred']
nu_pinn_scipy = scipy_data['nu_pinn']

print(f"Loading Precision PINN results from: {precision_path}")
precision_data = np.load(precision_path)
u_pinn_precision = precision_data['u_pinn_pred']
nu_pinn_precision = precision_data['nu_pinn']


# --- Calculate MSEs ---
mse_scipy = np.mean((u_true - u_pinn_scipy)**2)
mse_precision = np.mean((u_true - u_pinn_precision)**2)


# --- Visualization ---
fig = plt.figure(figsize=(24, 8), dpi=120) # Increased figure size for 3 subplots

# First subplot: Ground Truth Solution
ax1 = fig.add_subplot(131, projection='3d') # 1 row, 3 columns, 1st plot
ax1.set_title(f"Ground Truth (nu = {true_nu_gt:.4f})")
surf1 = ax1.plot_surface(X_np, Y_np, u_true, cmap=cm.viridis, antialiased=True)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')
ax1.set_zlabel('u-velocity')

# Second subplot: PINN SciPy Predicted Solution
ax2 = fig.add_subplot(132, projection='3d') # 1 row, 3 columns, 2nd plot
ax2.set_title(f"PINN SciPy (nu = {nu_pinn_scipy:.6f})\nMSE = {mse_scipy:.2e}")
surf2 = ax2.plot_surface(X_np, Y_np, u_pinn_scipy, cmap=cm.viridis, antialiased=True)
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Y-axis')
ax2.set_zlabel('u-velocity')

# Third subplot: PINN Precision Predicted Solution
ax3 = fig.add_subplot(133, projection='3d') # 1 row, 3 columns, 3rd plot
ax3.set_title(f"PINN Precision (nu = {nu_pinn_precision:.6f})\nMSE = {mse_precision:.2e}")
surf3 = ax3.plot_surface(X_np, Y_np, u_pinn_precision, cmap=cm.viridis, antialiased=True)
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
ax3.set_xlabel('X-axis')
ax3.set_ylabel('Y-axis')
ax3.set_zlabel('u-velocity')

plt.tight_layout()
plt.savefig(output_path)

print(f"Plot saved to: {output_path}")

# --- Statistics ---
print("\n--- Comparison Statistics ---")
print(f"Ground Truth nu: {true_nu_gt:.4f}")
print(f"PINN SciPy Discovered nu: {nu_pinn_scipy:.6f}, MSE: {mse_scipy:.2e}")
print(f"PINN Precision Discovered nu: {nu_pinn_precision:.6f}, MSE: {mse_precision:.2e}")
