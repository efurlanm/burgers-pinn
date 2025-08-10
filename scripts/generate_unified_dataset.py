
import tensorflow as tf
import numpy as np
import os
import argparse
import time

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def generate_ground_truth_data(nx, ny, nt, dx, dy, dt,
                               nu_val,
                               u_initial, v_initial):
    """
    Generates ground truth data using a TensorFlow-based Finite Difference Method.
    This function is copied from main_hopt.py for consistency.
    """
    u, v = tf.identity(u_initial), tf.identity(v_initial)
    u_snapshots, v_snapshots, t_snapshots = [], [], []

    # Define snapshot times
    snapshot_timesteps = {int(nt * 0.25), int(nt * 0.5), int(nt * 0.75), nt}

    for n in range(nt + 1):
        if n in snapshot_timesteps:
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

def main(args):
    """
    Main function to generate, unify, and save the dataset.
    """
    start_time = time.time()

    # --- Set Seed for Reproducibility ---
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Using seed: {args.seed}")

    # --- FDM Parameters (consistent with main_hopt.py) ---
    grid_points_x = 41
    grid_points_y = 41
    time_steps = 50
    x_min, x_max = 0.0, 2.0
    y_min, y_max = 0.0, 2.0
    
    # --- Unified Dataset Parameters ---
    num_datasets = args.num_datasets
    nu_min = args.nu_min
    nu_max = args.nu_max
    noise_level = args.noise_level
    output_path = args.output_path

    # --- Data Generation ---
    print(f"Generating {num_datasets} datasets for nu range [{nu_min}, {nu_max}]...")

    x_np = np.linspace(x_min, x_max, grid_points_x)
    y_np = np.linspace(y_min, y_max, grid_points_y)
    X_np, Y_np = np.meshgrid(x_np, y_np)
    dx, dy, dt = x_np[1] - x_np[0], y_np[1] - y_np[0], 0.001

    # Initial condition
    center_x, center_y = 1.0, 1.0
    sigma_x, sigma_y = 0.25, 0.25
    u_initial_np = np.exp(-((X_np - center_x)**2 / (2 * sigma_x**2) + 
                            (Y_np - center_y)**2 / (2 * sigma_y**2)))
    v_initial_np = np.exp(-((X_np - center_x)**2 / (2 * sigma_x**2) + 
                            (Y_np - center_y)**2 / (2 * sigma_y**2)))
    u_initial_tf = tf.constant(u_initial_np, dtype=tf.float32)
    v_initial_tf = tf.constant(v_initial_np, dtype=tf.float32)

    all_t, all_x, all_y, all_u, all_v, all_nu = [], [], [], [], [], []

    for i in range(num_datasets):
        current_nu = np.random.uniform(nu_min, nu_max)
        print(f"  [{i+1}/{num_datasets}] Generating data for nu = {current_nu:.6f}...")
        
        u_snaps, v_snaps, t_snaps = generate_ground_truth_data(
            grid_points_x, grid_points_y, time_steps, dx, dy, dt,
            tf.constant(current_nu, dtype=tf.float32), u_initial_tf, v_initial_tf)

        # Process snapshots for the current nu
        for t_val, u_snap, v_snap in zip(t_snaps, u_snaps, v_snaps):
            t_flat = np.full_like(X_np.flatten(), t_val.numpy())
            x_flat = X_np.flatten()
            y_flat = Y_np.flatten()
            u_flat = u_snap.numpy().flatten()
            v_flat = v_snap.numpy().flatten()
            nu_flat = np.full_like(x_flat, current_nu)

            all_t.append(t_flat)
            all_x.append(x_flat)
            all_y.append(y_flat)
            all_u.append(u_flat)
            all_v.append(v_flat)
            all_nu.append(nu_flat)

    # --- Concatenate all data into large arrays ---
    t_unified = np.concatenate(all_t).astype(np.float32)[:, None]
    x_unified = np.concatenate(all_x).astype(np.float32)[:, None]
    y_unified = np.concatenate(all_y).astype(np.float32)[:, None]
    u_unified = np.concatenate(all_u).astype(np.float32)[:, None]
    v_unified = np.concatenate(all_v).astype(np.float32)[:, None]
    nu_unified = np.concatenate(all_nu).astype(np.float32)[:, None]

    # --- Add Noise (Optional) ---
    if noise_level > 0:
        print(f"Adding {noise_level*100:.2f}% Gaussian noise...")
        std_u = np.std(u_unified)
        std_v = np.std(v_unified)
        noise_u = np.random.normal(0, std_u * noise_level, u_unified.shape)
        noise_v = np.random.normal(0, std_v * noise_level, v_unified.shape)
        u_unified += noise_u.astype(np.float32)
        v_unified += noise_v.astype(np.float32)
        print(f"  Noise added. Std(u): {std_u:.4f}, Std(v): {std_v:.4f}")

    # --- Shuffle the unified dataset ---
    print("Shuffling the unified dataset...")
    num_points = t_unified.shape[0]
    shuffled_indices = np.random.permutation(num_points)

    t_unified = t_unified[shuffled_indices]
    x_unified = x_unified[shuffled_indices]
    y_unified = y_unified[shuffled_indices]
    u_unified = u_unified[shuffled_indices]
    v_unified = v_unified[shuffled_indices]
    nu_unified = nu_unified[shuffled_indices]
    
    # --- Save to .npz file ---
    print(f"Saving unified dataset to: {output_path}")
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        
    np.savez_compressed(
        output_path,
        t=t_unified,
        x=x_unified,
        y=y_unified,
        u=u_unified,
        v=v_unified,
        nu=nu_unified
    )

    end_time = time.time()
    print("-" * 50)
    print("Dataset Generation Summary:")
    print(f"  Total data points: {num_points}")
    print(f"  Number of nu samples: {num_datasets}")
    print(f"  Shape of t: {t_unified.shape}")
    print(f"  Shape of x: {x_unified.shape}")
    print(f"  Shape of y: {y_unified.shape}")
    print(f"  Shape of u: {u_unified.shape}")
    print(f"  Shape of v: {v_unified.shape}")
    print(f"  Shape of nu: {nu_unified.shape}")
    print(f"  File saved successfully.")
    print(f"  Total time: {end_time - start_time:.2f} seconds.")
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a unified dataset for PINN training.")
    parser.add_argument('--num_datasets', type=int, default=20,
                        help='Number of different nu values to sample for the dataset.')
    parser.add_argument('--nu_min', type=float, default=0.01,
                        help='Minimum nu value for the training range.')
    parser.add_argument('--nu_max', type=float, default=0.1,
                        help='Maximum nu value for the training range.')
    parser.add_argument('--noise_level', type=float, default=0.0,
                        help='Percentage of Gaussian noise to add to the data (e.g., 0.01 for 1%).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--output_path', type=str, default='results/1data/unified_dataset.npz',
                        help='Path to save the generated .npz file.')
    args = parser.parse_args()
    main(args)
