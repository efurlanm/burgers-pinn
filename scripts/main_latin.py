import tensorflow as tf
import numpy as np
import time
import os
import argparse
import tensorflow.keras.callbacks as callbacks
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pickle
import scipy.stats.qmc # Import for Latin Hypercube

# Import the core model and data generation function from our new module
from pinn_model import PINN_Burgers2D, generate_ground_truth_data, swish_activation

# --- GPU Configuration ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- Evaluation Function ---
def evaluate_pinn_on_dataset(pinn_model, dataset_path, params):
    """
    Evaluates a trained PINN model on a given dataset (.npz file).
    This function performs Stage 2 (inverse problem) of the PINN workflow.
    """
    print(f"--- Starting Evaluation on: {os.path.basename(dataset_path)} ---")
    try:
        data = np.load(dataset_path)
        # We take a subset of the data for evaluation to speed up the process
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

    epochs_inverse_adam = params.get('epochs_inverse_adam_stage2', 5000)
    epochs_inverse_adam_pretrain = params.get('epochs_inverse_adam_pretrain', 1000)

    discovered_nu, _, _ = pinn_model.train_inverse_problem(
        x_data, y_data, t_data,
        u_data, v_data,
        epochs_inverse_adam, epochs_inverse_adam_pretrain
    )

    error = abs(true_nu_for_eval - discovered_nu) / true_nu_for_eval
    percentage_error = error * 100

    print(f"  > Ground Truth nu (Validation): {true_nu_for_eval:.6f}")
    print(f"  > Discovered nu (Validation):   {discovered_nu:.6f}")
    print(f"  > Percentage Error (Validation): {percentage_error:.4f}%")
    print(f"--- Evaluation Finished ---")

    return percentage_error

# --- Training Function ---
def run_training_stage(params):
    """
    Runs Stage 1 of the experiment: training the parametric PINN model.
    """
    seed_value = params.get('seed', 1)
    tf.random.set_seed(seed_value)
    np.random.seed(seed_value)
    print(f"Running training with seed: {seed_value}")

    # Physical and Network parameters
    grid_points_x, grid_points_y, time_steps = 41, 41, 50
    nu_min_train, nu_max_train = params.get('nu_initial', 0.01), 0.1
    x_min, x_max, y_min, y_max = 0.0, 2.0, 0.0, 2.0
    t_min, t_max = 0.0, time_steps * 0.001
    layers_config = [4] + [params.get('neurons', 60)] * params.get('layers', 5) + [2]

    # Training parameters
    adam_epochs_stage1 = params.get('adam_epochs_stage1', 5000)
    epochs_data_only_stage1 = params.get('epochs_data_only_stage1', 500)
    num_pde_points_stage1 = params.get('num_pde_points_stage1', 10000)
    pde_batch_size_stage1 = params.get('pde_batch_size_stage1', 4096)
    
    # --- Data Generation (for Training Set) ---
    x_np, y_np = np.linspace(x_min, x_max, grid_points_x), np.linspace(y_min, y_max, grid_points_y)
    X_np, Y_np = np.meshgrid(x_np, y_np)
    dx, dy, dt = x_np[1] - x_np[0], y_np[1] - y_np[0], 0.001
    
    center_x, center_y, sigma_x, sigma_y = 1.0, 1.0, 0.25, 0.25
    u_initial_np = np.exp(-((X_np - center_x)**2 / (2 * sigma_x**2) + (Y_np - center_y)**2 / (2 * sigma_y**2)))
    v_initial_np = np.exp(-((X_np - center_x)**2 / (2 * sigma_x**2) + (Y_np - center_y)**2 / (2 * sigma_y**2)))
    u_initial_tf, v_initial_tf = tf.constant(u_initial_np, dtype=tf.float32), tf.constant(v_initial_np, dtype=tf.float32)

    num_datasets_for_generalization = params.get('num_datasets_gene', 10)
    training_datasets = []
    print(f"Generating {num_datasets_for_generalization} datasets for generalization training using LATIN HYPERCUBE SAMPLING...")
    
    # --- LATIN HYPERCUBE SAMPLING IMPLEMENTATION ---
    sampler = scipy.stats.qmc.LatinHypercube(d=1, seed=seed_value)
    sample = sampler.random(n=num_datasets_for_generalization)
    # Scale samples to the desired range [nu_min_train, nu_max_train]
    nu_values_lhs = scipy.stats.qmc.scale(sample, [nu_min_train], [nu_max_train]).flatten()
    
    X_data_list_s1, Y_data_list_s1, T_data_list_s1 = [], [], []
    for i, current_nu_true in enumerate(nu_values_lhs):
        # current_nu_true is now from LHS, not random.uniform
        print(f"  > Dataset {i+1}: nu = {current_nu_true:.6f}")
        
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
        noise_level = params.get('noise_level', 0.0)
        if noise_level > 0:
            std_u, std_v = np.std(U_data_flat), np.std(V_data_flat)
            U_data_flat += np.random.normal(0, std_u * noise_level, U_data_flat.shape).astype(np.float32)
            V_data_flat += np.random.normal(0, std_v * noise_level, V_data_flat.shape).astype(np.float32)
            
        u_data_tf_i, v_data_tf_i = tf.constant(U_data_flat, dtype=tf.float32), tf.constant(V_data_flat, dtype=tf.float32)
        training_datasets.append((u_data_tf_i, v_data_tf_i, current_nu_true))

    x_data_tf = tf.constant(np.concatenate(X_data_list_s1), dtype=tf.float32)
    y_data_tf = tf.constant(np.concatenate(Y_data_list_s1), dtype=tf.float32)
    t_data_tf = tf.constant(np.concatenate(T_data_list_s1), dtype=tf.float32)
    u_data_tf_initial, v_data_tf_initial, nu_true_initial = training_datasets[0]

    # --- Model Initialization and Training ---
    pinn = PINN_Burgers2D(
        layers_config, x_data_tf, y_data_tf, t_data_tf,
        u_data_tf_initial, v_data_tf_initial, None, None, None, None,
        x_min, x_max, y_min, y_max, t_min, t_max,
        nu_min_train, nu_max_train, nu_true_initial,
        annealing_rate=0.1, learning_rate=params.get('learning_rate', 0.001))

    print(f"--- Stage 1: Parametric PINN Training ---")
    pinn.fit(
        epochs_adam=adam_epochs_stage1, epochs_data_only=epochs_data_only_stage1,
        num_pde_points=num_pde_points_stage1, x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max, t_min=t_min, t_max=t_max,
        training_datasets=training_datasets, callbacks=None,
        pde_batch_size=pde_batch_size_stage1)
    
    return pinn

# --- HPO Objective Function ---
def objective_function(params):
    """
    The main objective function for hyperopt.
    It trains the model and evaluates it on the validation set.
    """
    overall_start_time = time.time()
    
    # Stage 1: Train the parametric model
    trained_pinn = run_training_stage(params)
    
    # Stage 2: Evaluate the trained model on the fixed validation set
    validation_dataset_path = 'results/validation_dataset.npz'
    validation_error = evaluate_pinn_on_dataset(trained_pinn, validation_dataset_path, params)
    
    # Save results
    run_id = params.get('run_id', f"trial_{time.time():.0f}")
    results_dir = params.get('results_dir', 'results/latin') # Default to 'results/latin'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    output_filename = f"{run_id}.npz"
    output_file = os.path.join(results_dir, output_filename)
    
    np.savez(
        output_file,
        params=params,
        validation_error=validation_error,
        duration_total=time.time() - overall_start_time
    )
    print(f"Results for trial saved to {output_file}")
    
    return {'loss': validation_error, 'status': STATUS_OK}

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PINN HPO for 2D Burgers' equation.")
    parser.add_argument('--optimize', action='store_true', help='Flag to run hyperparameter optimization.')
    parser.add_argument('--trials_file', type=str, default='results/latin/latin_trials.pkl', help='File to save and load hyperopt trials.')
    # Arguments for single-run mode
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generators.')
    parser.add_argument('--run_id', type=str, default=f'run_{int(time.time())}', help='A unique identifier for the run.')
    parser.add_argument('--results_dir', type=str, default='results/latin', help='Directory to save results.')
    parser.add_argument('--adam_epochs_stage1', type=int, default=6000, help='Number of Adam epochs for Stage 1.')
    args = parser.parse_args()

    if args.optimize:
        space = {
            'seed': hp.choice('seed', [42]), # Testing Seed 42 first
            'neurons': hp.choice('neurons', [50]),
            'layers': hp.choice('layers', [4]),
            'learning_rate': hp.choice('learning_rate', [0.000229]), # Champion LR
            'adam_epochs_stage1': hp.choice('adam_epochs_stage1', [6000]), # Champion Epochs
            'epochs_data_only_stage1': hp.choice('epochs_data_only_stage1', [1500]),
            'num_pde_points_stage1': hp.choice('num_pde_points_stage1', [15000]),
            'epochs_inverse_adam_pretrain': hp.choice('epochs_inverse_adam_pretrain', [2000]),
            'num_datasets_gene': hp.choice('num_datasets_gene', [19]), # Full dataset count
            'noise_level': hp.choice('noise_level', [0.0399]) # Champion Noise
        }
        
        try:
            with open(args.trials_file, 'rb') as f:
                trials = pickle.load(f)
            print(f"Loaded {len(trials.trials)} existing trials from {args.trials_file}.")
        except FileNotFoundError:
            trials = Trials()
            print("No existing trials file found. Starting new optimization.")
            
        best = fmin(
            fn=objective_function,
            space=space,
            algo=tpe.suggest,
            max_evals=len(trials.trials) + 1, # Run 1 new trial
            trials=trials
        )
        
        with open(args.trials_file, 'wb') as f:
            pickle.dump(trials, f)
            
        print("\n" + "="*50)
        print("Hyperparameter optimization finished.")
        print("Best parameters found:")
        print(best)
        print("="*50)

    else:
        # This block is for running a single experiment without HPO
        print("Running a single experiment with provided parameters.")
        params = {
             'seed': args.seed,
             'run_id': args.run_id,
             'results_dir': args.results_dir,
             'neurons': 50, 'layers': 4, 'learning_rate': 0.000229,
             'adam_epochs_stage1': args.adam_epochs_stage1, 'epochs_data_only_stage1': 1500,
             'num_pde_points_stage1': 15000, 'epochs_inverse_adam_pretrain': 2000,
             'num_datasets_gene': 19, 'noise_level': 0.0399
        }
        objective_function(params)