## Review 020

### Análise da Falha de Convergência do TFP L-BFGS-B

**Observação:**

Apesar de alinhar os hiperparâmetros com a versão bem-sucedida do SciPy, o otimizador `tfp.optimizer.lbfgs_minimize` em `burgers2d-03.py` falhou em convergir (`converged: False`), parando após apenas 3 iterações. O valor final de `nu` (`0.059979`) mal melhorou após a fase Adam.

**Análise da Causa Raiz:**

A principal diferença entre as implementações do L-BFGS-B do SciPy e do TensorFlow Probability (TFP) está em seus critérios de parada padrão:

*   **SciPy (`ftol`):** O critério `ftol` monitora a **mudança no valor da função de perda**. A otimização para quando a redução relativa da perda cai abaixo de um limiar, o que funcionou bem no nosso caso de teste.
*   **TFP (`tolerance`):** O critério `tolerance` monitora a **norma do gradiente projetado (PG-norm)**. A otimização para quando a norma do gradiente é menor que a tolerância. Um gradiente pequeno significa que o otimizador está em um ponto estacionário (um platô, mínimo local ou sela).

Ao definir `tolerance=1e-9`, nós inadvertidamente impomos uma condição extremamente rigorosa sobre o gradiente. O otimizador provavelmente encontrou um platô onde o gradiente era muito pequeno (menor que `1e-9`) e parou, acreditando ter convergido, embora a perda ainda pudesse ser melhorada.

### Plano de Ação Revisado

1.  **Ajustar a Tolerância do TFP:** A tolerância de `1e-9` era muito agressiva. Vou relaxar o critério de parada para `tolerance=1e-6`. Este valor ainda exige um gradiente pequeno, mas é uma condição mais razoável que pode permitir que o otimizador continue por mais iterações para refinar o valor de `nu`.

### Próximos Passos

1.  Implementar a mudança de tolerância em `burgers2d-03.py`.
2.  Executar o script novamente e analisar o `log_pinn_03.txt` para ver se o aumento da tolerância permite que o otimizador TFP L-BFGS-B itere por mais tempo e melhore a precisão do `nu` descoberto.
<br><sub>Last edited: 2025-08-05 12:39:36</sub>
