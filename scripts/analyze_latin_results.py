import numpy as np
import glob
import os

def analyze_latin_results():
    """
    Analyzes the results from the latin ensemble runs, calculates statistics,
    and prints a Markdown table.
    """
    results_dir = "results/latin_ensemble"
    result_files = glob.glob(os.path.join(results_dir, "latin_seed_*.npz"))

    if not result_files:
        print(f"No result files found in {results_dir}")
        return

    errors = []
    for f in result_files:
        try:
            data = np.load(f, allow_pickle=True)
            # The validation_error is a scalar array, so we extract it with .item()
            errors.append(data['validation_error'].item())
        except Exception as e:
            print(f"Error loading or processing file {f}: {e}")

    if not errors:
        print("No valid error data could be extracted.")
        return

    mean_error = np.mean(errors)
    std_error = np.std(errors)
    min_error = np.min(errors)
    max_error = np.max(errors)

    # Print results in a Markdown table format
    print("### Análise da Estratégia Latin Hypercube Sampling (LHS)")
    print("| Métrica Estatística | Erro Percentual (%)")
    print("|:--------------------|:--------------------|")
    print(f"| Média               | {mean_error:.4f}            |")
    print(f"| Desvio Padrão       | {std_error:.4f}             |")
    print(f"| Mínimo              | {min_error:.4f}             |")
    print(f"| Máximo              | {max_error:.4f}             |")
    print(f"\n**Observação:** Resultados baseados em {len(errors)} execuções com diferentes sementes aleatórias.")

if __name__ == "__main__":
    analyze_latin_results()
