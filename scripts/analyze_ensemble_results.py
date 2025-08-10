import numpy as np
import glob
import re

def analyze_ensemble_logs():
    log_files = glob.glob("logs/hopt_ensemble_run_seed_*.txt")
    print(f"Encontrados {len(log_files)} arquivos de log.")

    percentages = []
    times_adam = []
    times_lbfgs = []
    
    # Extrair dados de cada log
    for log_file in log_files:
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Extrair Percent Error
            perc_match = re.search(r"Percentage Error: ([0-9\.]+)%", content)
            if perc_match:
                percentages.append(float(perc_match.group(1)))
            
            # Extrair Tempos
            adam_match = re.search(r"Adam pre-training for nu_inverse finished in ([0-9\.]+) seconds", content)
            if adam_match:
                times_adam.append(float(adam_match.group(1)))
                
            lbfgs_match = re.search(r"Stage 2 \(Inverse Problem\) L-BFGS-B training finished in ([0-9\.]+) seconds", content)
            if lbfgs_match:
                times_lbfgs.append(float(lbfgs_match.group(1)))

    if not percentages:
        print("Nenhum dado de erro percentual encontrado.")
        return

    # Calcular estatísticas
    mean_error = np.mean(percentages)
    std_error = np.std(percentages)
    min_error = np.min(percentages)
    max_error = np.max(percentages)
    median_error = np.median(percentages)
    
    total_times = np.array(times_adam) + np.array(times_lbfgs)
    mean_time = np.mean(total_times)
    std_time = np.std(total_times)

    print("-" * 50)
    print("ANÁLISE ESTATÍSTICA DO ENSEMBLE (10 Seeds)")
    print("-" * 50)
    print(f"Erro de Generalização (%):")
    print(f"  Média:        {mean_error:.4f}%")
    print(f"  Desvio Padrão: {std_error:.4f}%")
    print(f"  Mediana:      {median_error:.4f}%")
    print(f"  Mínimo:       {min_error:.4f}% (Melhor Seed)")
    print(f"  Máximo:       {max_error:.4f}%")
    print("-" * 50)
    print(f"Tempo de Inferência (s):")
    print(f"  Média:        {mean_time:.2f}s")
    print(f"  Desvio Padrão: {std_time:.2f}s")
    print("-" * 50)
    print("Valores Individuais de Erro (%):")
    for p in percentages:
        print(f"  {p:.4f}%")

if __name__ == "__main__":
    analyze_ensemble_logs()