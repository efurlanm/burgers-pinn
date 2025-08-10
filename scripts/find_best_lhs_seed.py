import numpy as np
import glob
import os
import re

def find_best_lhs_seed():
    """
    Finds the best performing seed from the latin ensemble runs.
    """
    results_dir = "results/latin_ensemble"
    result_files = glob.glob(os.path.join(results_dir, "latin_seed_*.npz"))

    if not result_files:
        print("ERROR: No result files found.")
        return

    best_error = float('inf')
    best_seed = -1

    for f in result_files:
        try:
            data = np.load(f, allow_pickle=True)
            error = data['validation_error'].item()
            
            # Extract seed number from filename
            match = re.search(r'latin_seed_(\d+).npz', os.path.basename(f))
            if match:
                seed = int(match.group(1))
                if error < best_error:
                    best_error = error
                    best_seed = seed
        except Exception as e:
            print(f"Could not process file {f}: {e}")

    if best_seed != -1:
        print(f"Best seed found: {best_seed}")
        print(f"Validation error: {best_error:.4f}%")
    else:
        print("Could not determine the best seed.")

if __name__ == "__main__":
    find_best_lhs_seed()
