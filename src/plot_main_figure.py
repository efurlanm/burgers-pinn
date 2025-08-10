import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Import for 3D plotting
from matplotlib import cm # Import colormaps
import argparse
import os

def plot_main_figure(precision_npz_path, shape_npz_path, output_filename="fig_main_comparison.jpg"):
    # Load data for Precision focus
    precision_data = np.load(precision_npz_path)
    X_precision = precision_data['X']
    Y_precision = precision_data['Y']
    u_true_precision = precision_data['u_true']
    u_pinn_pred_precision = precision_data['u_pinn_pred']
    nu_pinn_precision = precision_data['nu_pinn']
    true_nu_precision = precision_data['true_nu']
    total_mse_precision = precision_data['total_mse']

    # Load data for Shape focus
    shape_data = np.load(shape_npz_path)
    # X_shape = shape_data['X'] # Assuming X, Y, u_true are the same for both
    # Y_shape = shape_data['Y']
    # u_true_shape = shape_data['u_true']
    u_pinn_pred_shape = shape_data['u_pinn_pred']
    nu_pinn_shape = shape_data['nu_pinn']
    true_nu_shape = shape_data['true_nu']
    total_mse_shape = shape_data['total_mse']

    # Create the 3-panel figure
    fig = plt.figure(figsize=(18, 8))

    # Panel 1: Ground Truth
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X_precision, Y_precision, u_true_precision, cmap=cm.viridis, antialiased=True)
    # fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    ax1.set_title(f'(a) Ground Truth (\nu = {true_nu_precision})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u-velocity')

    # Panel 2: Precision Focus Result
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X_precision, Y_precision, u_pinn_pred_precision, cmap=cm.viridis, antialiased=True)
    # fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    ax2.set_title(f'(b) Precision (\nu = {nu_pinn_precision:.5f}, MSE = {total_mse_precision:.3e})')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u-velocity')

    # Panel 3: Shape Focus Result
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(X_precision, Y_precision, u_pinn_pred_shape, cmap=cm.viridis, antialiased=True)
    # fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
    ax3.set_title(f'(c) Shape (\nu = {nu_pinn_shape:.5f}, MSE = {total_mse_shape:.3e})')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('u-velocity')

    # plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.95, hspace=0.2, wspace=0.2)
    
    output_path = output_filename
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate main comparison figure for Burgers PINN.')
    parser.add_argument('--precision_npz', type=str, required=True,
                        help='Path to the .npz file for the Precision focus result.')
    parser.add_argument('--shape_npz', type=str, required=True,
                        help='Path to the .npz file for the Shape focus result.')
    parser.add_argument('--output_filename', type=str, default="fig_main_comparison.jpg",
                        help='Name of the output image file (e.g., fig_main_comparison.jpg).')
    
    args = parser.parse_args()

    plot_main_figure(args.precision_npz, args.shape_npz, args.output_filename)
