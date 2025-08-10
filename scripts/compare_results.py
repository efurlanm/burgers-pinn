import numpy as np
import os

def analyze_results(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    try:
        data = np.load(filepath)
        discovered_nu = data['discovered_nu_stage2']
        true_nu = data['true_nu_stage2_inverse']
        mse = data['total_mse_stage2_final']
        
        print(f"Analysis for: {os.path.basename(filepath)}")
        print(f"  - True nu: {true_nu:.6f}")
        print(f"  - Discovered nu: {discovered_nu:.6f}")
        print(f"  - Final MSE: {mse:.6e}")
        print(f"  - Error (%): {abs(true_nu - discovered_nu) / true_nu * 100:.4f}%")
        print("-" * 30)
        
        return {
            "file": os.path.basename(filepath),
            "true_nu": true_nu,
            "discovered_nu": discovered_nu,
            "mse": mse
        }
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

# --- Analysis ---
print("--- 'gene' Experiment (Baseline) ---")
gene_test_file = 'results/gene/gene_results_s1_epochs_100_s2_epochs_5000_seed_1.npz' # A quick run for comparison
gene_full_run_file = 'results/gene/gene_results_s1_epochs_5000_s2_epochs_5000_seed_1.npz'
analyze_results(gene_test_file)
analyze_results(gene_full_run_file)


print("\n--- 'fft' Experiment (Fine-Tuning) ---")
fft_test_file = 'results/fft/fft_results_s1_epochs_10_s2_epochs_5000_seed_1.npz'
fft_full_run_file = 'results/fft/fft_results_s1_epochs_5000_s2_epochs_5000_seed_1.npz'
analyze_results(fft_test_file)
analyze_results(fft_full_run_file)
