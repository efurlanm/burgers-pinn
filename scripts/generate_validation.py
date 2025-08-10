import tensorflow as tf
import numpy as np
import os
from pinn_model import generate_ground_truth_data

def main():
    # Configuração do "Hold-Out" Dataset
    # Valor inédito de nu para teste de generalização
    SEED = 42
    NU_VAL = 0.0382  # Valor citado nos relatórios como teste hold-out
    OUTPUT_PATH = "validation_dataset.npz"
    
    print(f"Gerando dataset de validação (Hold-out) com nu={NU_VAL}...")
    
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    
    # Parâmetros da Grade (Mesmos do treino)
    nx, ny, nt = 41, 41, 50
    x_min, x_max = 0.0, 2.0
    y_min, y_max = 0.0, 2.0
    dt = 0.001
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)

    # Condição Inicial
    x_np = np.linspace(x_min, x_max, nx)
    y_np = np.linspace(y_min, y_max, ny)
    X_np, Y_np = np.meshgrid(x_np, y_np)
    u_initial_np = np.exp(-((X_np - 1)**2 / 0.25**2 + (Y_np - 1)**2 / 0.25**2)).astype(np.float32)
    v_initial_np = u_initial_np.copy()

    # Gera a simulação
    u_snaps, v_snaps, t_snaps = generate_ground_truth_data(
        nx, ny, nt, dx, dy, dt, 
        tf.constant(NU_VAL, dtype=tf.float32), 
        tf.constant(u_initial_np), 
        tf.constant(v_initial_np)
    )

    # Pega o último snapshot para o problema inverso
    u_final = u_snaps[-1].numpy().flatten()[:, None]
    v_final = v_snaps[-1].numpy().flatten()[:, None]
    t_final = np.full_like(u_final, t_snaps[-1].numpy())
    x_flat = X_np.flatten()[:, None]
    y_flat = Y_np.flatten()[:, None]
    nu_flat = np.full_like(u_final, NU_VAL)

    # Salva
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez(OUTPUT_PATH, 
             x=x_flat, y=y_flat, t=t_final, 
             u=u_final, v=v_final, nu=nu_flat)
    print(f"Sucesso! Arquivo salvo em: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
