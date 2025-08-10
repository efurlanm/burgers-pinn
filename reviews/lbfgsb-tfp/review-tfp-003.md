## Review 021

### Análise da Falha na Iteração do TFP L-BFGS-B

**Observação:**

Ajustar a tolerância para `1e-6` não resolveu o problema. O log `log_pinn_03.txt` mostra que o otimizador `tfp.optimizer.lbfgs_minimize` parou após apenas **uma única iteração**, com `converged: False`. Notavelmente, o valor de `nu` degradou de `0.049942` (após o Adam) para `0.048080`.

**Análise da Causa Raiz:**

O problema não é mais o critério de parada por tolerância, mas uma falha na própria mecânica da otimização. A terminação após uma única iteração malsucedida sugere fortemente que o algoritmo de **busca de linha (line search)** do otimizador está falhando. Ele calcula uma direção de busca, dá um passo inicial, avalia o resultado e, ao constatar que o passo não melhorou a solução (ou a piorou), não consegue encontrar um tamanho de passo alternativo melhor e, portanto, desiste.

Isso pode ser causado por uma aproximação da matriz Hessiana imprecisa, que resulta em uma direção ou magnitude de passo inicial de baixa qualidade.

### Plano de Ação Revisado

Para tornar o otimizador TFP L-BFGS-B mais robusto e seu comportamento mais próximo ao da bem-sucedida implementação do SciPy, vamos ajustar parâmetros que controlam a memória do otimizador e a persistência da busca de linha:

1.  **`num_correction_pairs=100`**: Este parâmetro é análogo ao `maxcor` do SciPy. Ele aumenta a memória do otimizador, permitindo que ele construa uma aproximação mais precisa da matriz Hessiana, o que deve levar a direções de busca de melhor qualidade.
2.  **`max_line_search_iterations=50`**: Este parâmetro é análogo ao `maxls` do SciPy. Ele permite que o algoritmo de busca de linha tente mais vezes encontrar um tamanho de passo que satisfaça as condições de convergência (condições de Wolfe), em vez de desistir prematuramente.

O objetivo é dar ao otimizador as ferramentas para ser mais "inteligente" e "persistente", evitando que ele falhe na primeira dificuldade.

### Próximos Passos

1.  Implementar os parâmetros `num_correction_pairs` e `max_line_search_iterations` na chamada `tfp.optimizer.lbfgs_minimize` em `burgers2d-03.py`.
2.  Executar o script e analisar o log para verificar se essas mudanças permitem que o otimizador itere múltiplas vezes e refine a solução encontrada pelo Adam.
<br><sub>Last edited: 2025-08-05 12:52:25</sub>
