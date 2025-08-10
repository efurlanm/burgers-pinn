"""
A standalone script for benchmarking the performance of the PINN model,
targeting a 30-minute execution time.
"""

import tensorflow as tf
import numpy as np
import time
import os

# Import the PINN class from the main script
from main_1data_hyperopt import PINN_Burgers2D

def run_benchmark():
    # --- Fixed Hyperparameters for Benchmark ---
    params = {
        'seed': 42,
        'neurons': 60,
        'layers': 5,
        'learning_rate': 0.001,
        'adam_epochs_stage1': 70, # Reduced epochs to target a 30-minute runtime
        'num_pde_points_stage1': 15000
    }

    tf.random.set_seed(params['seed'])
    np.random.seed(params['seed'])
    
    nx, ny, nt = 41, 41, 50
    x_min, x_max, y_min, y_max = 0.0, 2.0, 0.0, 2.0
    t_min, t_max = 0.0, nt * 0.001

    # --- Load Unified Dataset ---
    print("Loading unified dataset...")
    try:
        data = np.load('results/1data/unified_dataset.npz')
        t_data_tf = tf.constant(data['t'], dtype=tf.float32)
        x_data_tf = tf.constant(data['x'], dtype=tf.float32)
        y_data_tf = tf.constant(data['y'], dtype=tf.float32)
        u_data_tf = tf.constant(data['u'], dtype=tf.float32)
        v_data_tf = tf.constant(data['v'], dtype=tf.float32)
        nu_data_tf = tf.constant(data['nu'], dtype=tf.float32)
    except FileNotFoundError:
        print("[ERROR] Unified dataset not found. Please run generate_unified_dataset.py.")
        return

    pinn = PINN_Burgers2D([4] + [params['neurons']] * params['layers'] + [2],
                         x_data_tf, y_data_tf, t_data_tf, u_data_tf, v_data_tf,
                         None, None, None, None, x_min, x_max, y_min, y_max, t_min, t_max,
                         0.01, 0.1, nu_data_tf, learning_rate=params['learning_rate'], batch_size=1024)
    
    data_dataset = tf.data.Dataset.from_tensor_slices(
        (x_data_tf, y_data_tf, t_data_tf, u_data_tf, v_data_tf, nu_data_tf)
    ).shuffle(buffer_size=x_data_tf.shape[0]).batch(pinn.batch_size)

    print("\n--- Starting 30-Minute Benchmark ---")
    print(f"Training for {params['adam_epochs_stage1']} epochs...")

    start_time = time.time()
    
    pinn.fit(data_dataset, params['adam_epochs_stage1'], 0, params['num_pde_points_stage1'])
    
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_epoch = total_time / params['adam_epochs_stage1']

    print("\n" + "="*50)
    print("--- 30-Minute Benchmark Finished ---")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per epoch: {time_per_epoch:.2f} seconds")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()
