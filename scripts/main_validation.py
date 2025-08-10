
"""
Validation Experiment Script

This script performs a formal validation of a trained PINN model.
Its responsibilities are:
1.  Define the best hyperparameters found during HPO.
2.  Train the PINN model from scratch using these best parameters (Stage 1).
3.  Load the independent, fixed validation dataset.
4.  Evaluate the trained model on the validation dataset by solving the
    inverse problem (Stage 2).
5.  Report the final percentage error.

This provides a clear, single-command way to reproduce the validation result.
"""

import tensorflow as tf
import numpy as np
import os
import argparse

# Import the core model and data generation function from our new module
from pinn_model import PINN_Burgers2D, generate_ground_truth_data

# --- GPU Configuration ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- Reusable Evaluation Function (copied from HPO script for consistency) ---
def evaluate_pinn_on_dataset(pinn_model, dataset_path, params):
    print(f"--- Starting Evaluation on: {os.path.basename(dataset_path)} ---")
    try:
        data = np.load(dataset_path)
        num_eval_points = 1681 # 41x41 grid for one time step
        x_data = data['x'][:num_eval_points]
        y_data = data['y'][:num_eval_points]
        t_data = data['t'][:num_eval_points]
        u_data = data['u'][:num_eval_points]
        v_data = data['v'][:num_eval_points]
        true_nu_for_eval = data['nu'][0][0]
    except Exception as e:
        print(f"Error loading or processing dataset {dataset_path}: {e}")
        return float('inf')

    epochs_inverse_adam_pretrain = params.get('epochs_inverse_adam_pretrain', 1000)
    
    discovered_nu, _, _ = pinn_model.train_inverse_problem(
        x_data, y_data, t_data,
        u_data, v_data,
        5000, # Using a fixed number of epochs for consistency in validation
        epochs_inverse_adam_pretrain
    )

    error = abs(true_nu_for_eval - discovered_nu) / true_nu_for_eval
    percentage_error = error * 100

    print(f"  > Ground Truth nu (Validation): {true_nu_for_eval:.6f}")
    print(f"  > Discovered nu (Validation):   {discovered_nu:.6f}")
    print(f"  > Final Percentage Error: {percentage_error:.4f}%")
    print(f"--- Evaluation Finished ---")
    return percentage_error

def main(args):
    """ Main function to run the validation experiment. """
    
    # --- Best Hyperparameters (from HPO-001 and HPO-002 analysis) ---
    best_params = {
        'seed': args.seed,
        'neurons': 50,
        'layers': 4,
        'learning_rate': 0.0002299,
        'adam_epochs_stage1': 5000,
        'epochs_data_only_stage1': 500,
        'num_pde_points_stage1': 20000,
        'epochs_inverse_adam_pretrain': 2000,
        'num_datasets_gene': 15,
        'noise_level': 0.0399,
        'nu_initial': 0.01, # Standard training range start
        'pde_batch_size_stage1': 4096,
    }
    print("--- Starting Validation Experiment ---")
    print("Using best hyperparameters found:")
    for key, val in best_params.items():
        print(f"  - {key}: {val}")
    
    # --- Train the PINN model from scratch using the best params ---
    # (This logic is a simplified version of run_training_stage from the HPO script)
    
    tf.random.set_seed(best_params['seed'])
    np.random.seed(best_params['seed'])

    grid_points_x, grid_points_y, time_steps = 41, 41, 50
    nu_min_train, nu_max_train = best_params['nu_initial'], 0.1
    x_min, x_max, y_min, y_max = 0.0, 2.0, 0.0, 2.0
    t_min, t_max = 0.0, time_steps * 0.001
    layers_config = [4] + [best_params['neurons']] * best_params['layers'] + [2]

    # --- Data Generation for Training ---
    x_np, y_np = np.linspace(x_min, x_max, grid_points_x), np.linspace(y_min, y_max, grid_points_y)
    X_np, Y_np = np.meshgrid(x_np, y_np)
    dx, dy, dt = x_np[1] - x_np[0], y_np[1] - y_np[0], 0.001
    center_x, center_y, sigma_x, sigma_y = 1.0, 1.0, 0.25, 0.25
    u_initial_np = np.exp(-((X_np - center_x)**2 / (2 * sigma_x**2) + (Y_np - center_y)**2 / (2 * sigma_y**2)))
    v_initial_np = np.exp(-((X_np - center_x)**2 / (2 * sigma_x**2) + (Y_np - center_y)**2 / (2 * sigma_y**2)))
    u_initial_tf, v_initial_tf = tf.constant(u_initial_np, dtype=tf.float32), tf.constant(v_initial_np, dtype=tf.float32)

    training_datasets = []
    X_data_list_s1, Y_data_list_s1, T_data_list_s1 = [], [], []
    for i in range(best_params['num_datasets_gene']):
        current_nu_true = np.random.uniform(nu_min_train, nu_max_train)
        u_true_tf, v_true_tf, t_true_tf = generate_ground_truth_data(
            grid_points_x, grid_points_y, time_steps, dx, dy, dt,
            tf.constant(current_nu_true, dtype=tf.float32), u_initial_tf, v_initial_tf)
        u_true_list, v_true_list, t_true_list = [u.numpy() for u in u_true_tf], [v.numpy() for v in v_true_tf], [t.numpy() for t in t_true_tf]
        U_data_list, V_data_list = [], []
        for j in range(len(t_true_list)):
            if i == 0:
                X_data_list_s1.append(X_np.flatten()[:, None])
                Y_data_list_s1.append(Y_np.flatten()[:, None])
                T_data_list_s1.append(np.full_like(X_np.flatten()[:, None], t_true_list[j]))
            U_data_list.append(u_true_list[j].flatten()[:, None])
            V_data_list.append(v_true_list[j].flatten()[:, None])
        U_data_flat, V_data_flat = np.concatenate(U_data_list), np.concatenate(V_data_list)
        if best_params['noise_level'] > 0:
            std_u, std_v = np.std(U_data_flat), np.std(V_data_flat)
            U_data_flat += np.random.normal(0, std_u * best_params['noise_level'], U_data_flat.shape).astype(np.float32)
            V_data_flat += np.random.normal(0, std_v * best_params['noise_level'], V_data_flat.shape).astype(np.float32)
        u_data_tf_i, v_data_tf_i = tf.constant(U_data_flat, dtype=tf.float32), tf.constant(V_data_flat, dtype=tf.float32)
        training_datasets.append((u_data_tf_i, v_data_tf_i, current_nu_true))

    x_data_tf = tf.constant(np.concatenate(X_data_list_s1), dtype=tf.float32)
    y_data_tf = tf.constant(np.concatenate(Y_data_list_s1), dtype=tf.float32)
    t_data_tf = tf.constant(np.concatenate(T_data_list_s1), dtype=tf.float32)
    u_data_tf_initial, v_data_tf_initial, nu_true_initial = training_datasets[0]

    # --- Model Initialization and Training (Stage 1) ---
    pinn = PINN_Burgers2D(
        layers_config, x_data_tf, y_data_tf, t_data_tf,
        u_data_tf_initial, v_data_tf_initial, None, None, None, None,
        x_min, x_max, y_min, y_max, t_min, t_max,
        nu_min_train, nu_max_train, nu_true_initial,
        learning_rate=best_params['learning_rate'])
    
    pinn.fit(
        epochs_adam=best_params['adam_epochs_stage1'], 
        epochs_data_only=best_params['epochs_data_only_stage1'],
        num_pde_points=best_params['num_pde_points_stage1'],
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, t_min=t_min, t_max=t_max,
        training_datasets=training_datasets,
        pde_batch_size=best_params['pde_batch_size_stage1'])
        
    # --- Evaluation on Validation Set (Stage 2) ---
    evaluate_pinn_on_dataset(pinn, args.validation_file, best_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a validation experiment for the PINN model.")
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for the training process.')
    parser.add_argument('--validation_file', type=str,
                        default='results/validation_dataset.npz',
                        help='Path to the .npz validation dataset file.')
    args = parser.parse_args()
    main(args)
