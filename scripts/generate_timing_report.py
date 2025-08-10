import numpy as np
import glob
import os
import re

def get_total_duration_from_npz(file_path):
    """Extracts total duration from a single .npz file."""
    try:
        data = np.load(file_path, allow_pickle=True)
        if 'duration_total' in data:
            return data['duration_total'].item()
    except Exception as e:
        print(f"Warning: Could not process file {file_path}: {e}")
    return None

def analyze_durations(directory, pattern):
    """Analyzes durations for a set of experiments."""
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return "N/A", "N/A"
    
    durations = [get_total_duration_from_npz(f) for f in files]
    durations = [d for d in durations if d is not None]
    
    if not durations:
        return "N/A", "N/A"
        
    mean_duration = np.mean(durations)
    std_duration = np.std(durations)
    
    # Handle single file case where std_dev is not applicable
    if len(durations) == 1:
        std_duration = "N/A"
    else:
        std_duration = f"{std_duration/60:.2f}"

    return f"{mean_duration/60:.2f}", std_duration


def main():
    """Main function to generate the timing report."""
    
    # --- Data Collection ---
    # Specialist Case (nu=0.05) - Taking the average of the 3 seeds
    specialist_mean, specialist_std = analyze_durations('BASE/results', 'parametric_inverse_results_nu_0.05_*.npz')
    
    # Surrogate V1 (from logs as it was a micro-experiment)
    surrogate_v1_time = "5.83" # Manually extracted from log as it was a specific run

    # Surrogate V2 (Random Sampling Ensemble)
    surrogate_v2_mean, surrogate_v2_std = analyze_durations('results/ensemble', 'ensemble_seed_*.npz')

    # Surrogate V2 + LHS Ensemble
    lhs_mean, lhs_std = analyze_durations('results/latin_ensemble', 'latin_seed_*.npz')
    
    # Surrogate V2 + LHS Extended
    lhs_extended_mean, _ = analyze_durations('results/lhs2', 'lhs2_extended_training_seed_2.npz')

    # --- Report Generation ---
    print("## Timing Data Collected ##")

    print("\n### Section 5.6 Data (Markdown Table) ###")
    print("| Estratégia                        | Tempo Médio (min) | Desvio Padrão (min) | Notas                               |")
    print("| :-------------------------------- | :---------------: | :-----------------: | :---------------------------------- |")
    print(f"| 1. Especialista ($\\nu=0.05$)        | {specialist_mean}           | {specialist_std}              | Média de 3 execuções.               |")
    print(f"| 2. Surrogate V1 (Unified)         | {surrogate_v1_time}             | N/A                 | Micro-experimento, 3 datasets.      |")
    print(f"| 3. Surrogate V2 (Random)          | {surrogate_v2_mean}           | {surrogate_v2_std}            | Ensemble de 3 execuções.            |")
    print(f"| 4. Surrogate V2 + LHS             | {lhs_mean}           | {lhs_std}            | Ensemble de 5 execuções.            |")
    print(f"| 5. Otimização Focada (LHS Ext.)   | {lhs_extended_mean}           | N/A                 | Execução única, 15.000 épocas.      |")

if __name__ == "__main__":
    main()