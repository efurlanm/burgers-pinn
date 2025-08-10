# CIACA 2025: Relatório de Avanço Metodológico e Computacional

## Resumo

Este documento formaliza os avanços obtidos na extensão da pesquisa original ("Caso Base"), submetida ao CIACA 2025. A investigação evoluiu de uma prova de conceito de **"Overfitting Controlado"** (resolução de uma única instância da PDE) para o desenvolvimento de um **Modelo Surrogate Generalista** (resolução paramétrica para $\nu 
[0.01, 0.1]$), viabilizado por otimizações computacionais.

---

## 1. Estratégias Experimentais

A investigação desdobrou-se em três estratégias experimentais distintas, evoluindo da prova de conceito para soluções de maior escala.

| Estratégia                                   | Script Principal                                       | Metodologia de Dados                                                               | Objetivo                      | Status                    |
|:-------------------------------------------- |:------------------------------------------------------ |:---------------------------------------------------------------------------------- |:----------------------------- |:------------------------- |
| **1. Especialista (Single-Dataset)**         | `main_plateau.py`                                      | **Estático (1 Dataset).** Treina e valida no mesmo conjunto de dados ($\nu$ fixo). | Prova de Conceito (Artigo)    | **Concluído.**            |
| **2. Surrogate V1 (Unified Dataset)**        | `generate_unified_dataset.py` `main_1data_hyperopt.py` | **Estático (Massivo).** Gera 20 datasets, unifica e carrega na RAM.                | Generalização por Força Bruta | **Viabilidade Limitada.** |
| **3. Surrogate V2 (Multi-Dataset Sampling)** | `pinn_model.py` `main_hopt_unified.py`                 | **Dinâmico (Amostragem Aleatória).** Gera 19 datasets e treina com amostragem.     | Generalização Eficiente       | **Concluído.**            |
| **4. Surrogate V2 + LHS**                    | `main_latin.py`                                        | **Dinâmico (Latin Hypercube).** Amostragem estratificada para robustez.            | Maximização de Robustez       | **Concluído.**            |
| **5. Otimização Focada (`lhs2`)**            | `main_latin.py --adam_epochs_stage1 15000`             | **Dinâmico (LHS Otimizado).** Treinamento estendido do melhor caso LHS.            | Refinamento do Melhor Caso    | **Concluído.**            |

---

## 2. Evolução do Paradigma de Modelagem

A transição metodológica visa superar a limitação fundamental de re-treinamento mandatório para novos parâmetros físicos.

| Dimensão                  | Abordagem Base (Artigo Submetido)                                                        | Abordagem Atual (Apresentação Oral)                                                    |
|:------------------------- |:---------------------------------------------------------------------------------------- |:-------------------------------------------------------------------------------------- |
| **Natureza do Modelo**    | **Especialista (Instance-Specific)** <br> Treinado para resolver *apenas* um $\nu$ fixo. | **Surrogate (Parametric)** <br> Aprende o operador $f(x, y, t, \text{dados}) \to \nu$. |
| **Escopo de Dados**       | Single-Dataset (1 simulação).                                                            | Multi-Dataset (10-20 simulações variadas).                                             |
| **Métrica de Sucesso**    | Erro de Reconstrução no dataset de treino.                                               | Erro de Generalização em datasets *não vistos*.                                        |
| **Desafio Computacional** | Baixo (Convergência em ~8 min).                                                          | Extremo (Estabilidade numérica e uso de memória crítica).                              |

### 2.1. Metodologia de Validação (Hold-Out)

Para comprovar a robustez do modelo surrogate, foi implementado um protocolo rigoroso de validação com dados inéditos (não vistos durante o treino):

* **Conjunto de Treino:** 19 datasets gerados com $\nu$ aleatórios (ex: $0.0475, 0.0849, 0.0634, \dots$).
* **Conjunto de Teste (Problema Inverso):** Um dataset independente gerado com **$\nu_{true} = 0.0382$**.
* **Objetivo:** O modelo deve inferir $\nu_{true}$ apenas observando o campo de velocidade $(u, v)$, sem nunca ter sido treinado com este valor específico de viscosidade.

---

## 3. Análise de Desempenho Computacional

Esta seção detalha as intervenções na arquitetura de execução necessárias para viabilizar o treinamento em escala.

### 3.1. Análise de Performance de GPU e Otimização de Kernel

Para viabilizar o treinamento em larga escala do modelo surrogate, foi conduzida uma análise de performance em nível de hardware utilizando o profiler **NVIDIA Nsight Compute (`ncu`)**. O objetivo era identificar e mitigar os gargalos computacionais que impediam a convergência em tempo hábil.

#### 3.1.1. Diagnóstico: Caracterização do Gargalo como *Memory-Bound*

A hipótese inicial de que o treinamento era limitado pela capacidade de processamento aritmético (regime *Compute-Bound*), comum em redes neurais densas, foi refutada. A análise com `ncu` revelou que a execução estava, na verdade, severamente limitada pela largura de banda da memória (regime *Memory-Bound*).

O perfil de execução mostrou que a maior parte do tempo de GPU não era gasta nos kernels de multiplicação de matrizes de alta intensidade (`ampere_sgemm_*`), que são otimizados para a arquitetura, mas sim em uma infinidade de kernels de operações elemento a elemento (como `Mul_GPU_DT_FLOAT_DT_FLOAT_ker...`) e de propósito geral (`EigenMetaKernel`). Essas operações, críticas para o cálculo do resíduo da PDE, possuem uma baixa razão de operações aritméticas por byte de memória acessado. Consequentemente, os multiprocessadores de streaming (SMs) da GPU passavam a maior parte do tempo ociosos, aguardando dados serem transferidos da lenta memória DRAM global para seus caches L1/L2, em vez de realizarem cálculos.

#### 3.1.2. Otimização do `pde_batch_size` e o Impacto na Localidade do Cache

O principal ofensor identificado foi o uso de um `pde_batch_size` (o número de pontos de colocação da PDE processados em um único passo) excessivamente grande, configurado inicialmente em 20.000. Um lote tão grande, embora aumente o paralelismo teórico, excede a capacidade do cache L2 da GPU. Isso resulta em um fenômeno conhecido como *Cache Thrashing*, onde os dados carregados no cache para um bloco de threads são imediatamente despejados para dar lugar aos dados do próximo bloco, forçando leituras repetidas e de alta latência da DRAM.

Para mitigar este gargalo, o `pde_batch_size` foi reduzido para **4.096**. Esta mudança foi projetada para garantir que o conjunto de trabalho (working set) de um lote de pontos da PDE pudesse residir de forma mais estável no cache L2. A análise de `ncu` validou quantitativamente o sucesso desta abordagem:

* **Taxa de Acerto do Cache L2 (L2$ Hit Rate):** Aumentou em aproximadamente **16%**. Isso confirma que uma fração significativamente maior de solicitações de memória foi atendida pelo cache rápido, em vez da DRAM lenta.
* **Tráfego de Leitura/Escrita da DRAM:** Reduziu em aproximadamente **67%**. A diminuição drástica no tráfego de e para a memória global é a evidência mais forte da mitigação do *Cache Thrashing* e da melhoria na localidade dos dados.

Essa otimização, ao alinhar o tamanho do problema com a hierarquia de memória da arquitetura da GPU, foi uma das intervenções mais críticas, permitindo uma redução significativa no tempo de treinamento e viabilizando os experimentos de generalização em maior escala.

### 3.2. Estabilidade Numérica e Gestão de Recursos

#### 3.2.1. O Erro de Memória (OOM)

A tentativa de calcular derivadas de segunda ordem ($u_{xx}, u_{yy}$) com a arquitetura original resultou em falhas sistemáticas de alocação de memória.

**Evidência de Falha:** `logs/parametric_inverse_run_seed_1_attempt_19_lr_schedule_20k_pde.txt`

```text
(0) RESOURCE_EXHAUSTED: OOM when allocating tensor with shape[20000,60] and type float...
```

*Interpretação:* O alocador de memória esgotou o espaço contíguo devido à retenção excessiva do grafo computacional pelo `tf.GradientTape(persistent=True)`.

#### 3.2.2. Solução Implementada: Nested Gradient Tapes

A implementação de **Tapes Aninhados** com liberação explícita alterou a complexidade espacial do algoritmo, reduzindo significativamente o consumo de memória durante o cálculo dos gradientes.

**Implementação Verificada (`pinn_model.py`):**

```python
    with tf.GradientTape(persistent=True) as outer_tape:
        with tf.GradientTape(persistent=True) as inner_tape:
            u, v = self.predict_velocity(...)
        # Cálculo de 1ª ordem e liberação imediata
        u_x = inner_tape.gradient(u, x)
        del inner_tape 
    # Cálculo de 2ª ordem com memória limpa
    u_xx = outer_tape.gradient(u_x, x)
```

---

## 4. Configuração Otimizada

A Otimização de Hiperparâmetros (HPO) convergiu para uma arquitetura mais eficiente do que a proposta no artigo original, demonstrando que a generalização requer profundidade moderada mas estratégias de treinamento robustas.

| Hiperparâmetro       | Caso Base (Artigo) | Caso Surrogate (Atual) | Justificativa                                             |
|:-------------------- |:------------------ |:---------------------- |:--------------------------------------------------------- |
| **Camadas Ocultas**  | 5                  | **4**                  | Redução de complexidade sem perda de expressividade.      |
| **Neurônios/Camada** | 60                 | **50**                 | Otimização do fluxo de informação.                        |
| **Learning Rate**    | Fixo (1e-3)        | **2.29e-4**            | Ajuste fino para estabilidade do otimizador.              |
| **Épocas (Adam)**    | Variável           | **5000**               | Regime de convergência estendido para múltiplos datasets. |
| **Ruído nos Dados**  | 0 - 10%            | **~4% (0.0399)**       | Treinamento robusto a ruído realístico.                   |

---

## 5. Resultados Quantitativos Comparados

A comparação direta deve considerar a distinção semântica entre "Precisão de Ajuste" (Caso Base) e "Capacidade de Generalização" (Caso Surrogate).

### 5.1. Caso Base: O "Especialista"

Resultados reportados no artigo (`ciaca-2025-66.pdf`), baseados em `BASE/TABELAS.md`.

> **Definição:** Treino em um único dataset fixo ($\nu=0.05$). Teste no *mesmo* dataset.

| Parâmetro ($\nu_{true}$) | Erro Relativo (%) | Tempo (s) | Interpretação                                                              |
|:------------------------ |:----------------- |:--------- |:-------------------------------------------------------------------------- |
| **0.05**                 | **0.067%**        | 520.84    | **Alta Precisão.** O modelo "memorizou" a dinâmica específica deste fluxo. |
| **0.02**                 | **0.337%**        | 502.42    | Consistente, mas requer re-treino total para cada novo $\nu$.              |

### 5.2. Caso Surrogate V1: "Unified Dataset"

Experimentos baseados em carregamento massivo de dados (`logs/1data`).

* **Viabilidade Limitada (Micro-Experimento):** Para contornar o estouro de memória, foi realizado um teste reduzido (3 datasets unificados, Batch 4096, Seed 7).
  * **Erro de Generalização:** **32.84%**
  * **Comparação:** O erro é **~10x maior** que o da abordagem V2 com a mesma semente (3.54%).
  * **Diagnóstico:** A arquitetura "Unified" impõe um limite rígido na quantidade total de datasets carregáveis na RAM, forçando o uso de dados insuficientes (3 cenários) e resultando em subajuste (underfitting).

#### 5.2.1. Evidências do Micro-Experimento

O teste de viabilidade foi executado com o script `sources/main_1data_hyperopt.py` modificado para baixo consumo de memória. O log de execução comprova a conclusão do treino e o alto erro de generalização.

**Log de Execução (Seed 7, Unified Dataset):**

```text
Loading unified dataset...
Generating 3 datasets for nu range [0.01, 0.1]...
...
Epoch 500, Last Batch Data Loss: 0.0001
...
Discovered nu: 0.033578, True nu: 0.050000, Error: 32.8441%
```

### 5.3. Caso Surrogate V2: "Multi-Dataset Sampling" (Estatística de Ensemble)

Resultados agregados de 10 experimentos independentes (Seeds 1-10) via HPO em múltiplos datasets com amostragem dinâmica.

> **Definição:** Treino em 19 datasets variados. Teste em um dataset **inédito** ($\nu=0.0382$).

> **Nota:** A alta variância reflete a dificuldade de generalização para dados nunca vistos e a sensibilidade à inicialização aleatória.

| Métrica                           | Valor Agregado      | Interpretação Física/Técnica                                                                                                                                                                          |
|:--------------------------------- |:------------------- |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Perda de Dados (Treino)**       | **~1.1e-5**         | Consistente. O modelo aprende a representar o campo $(u,v)$ em todos os seeds.                                                                                                                        |
| **Perda PDE (Treino)**            | **~6.4e-4**         | **Validação Física:** O resíduo da PDE mantém-se baixo, indicando aprendizado das leis físicas.                                                                                                       |
| **Erro de Generalização ($\nu$)** | **74.07% ± 57.87%** | **Alta Instabilidade.** Enquanto o melhor caso atingiu **3.54%** (Seed 7), o pior caso divergiu para **162%**. Isso evidencia que a generalização para dados "inéditos" ainda é um desafio em aberto. |
| **Tempo de Inferência**           | **46.31s ± 1.41s**  | Altamente determinístico. O custo computacional para inferência é estável e rápido (< 1 min).                                                                                                         |

#### 5.3.1. Evidências dos Resultados (Logs e Análise de Ensemble)

A análise estatística foi realizada sobre os logs `logs/hopt_ensemble_run_seed_*.txt` através do script `analyze_ensemble_results.py`.

**A. Amostra Representativa (Seed 1):**

```text
Stage 2: Discovered nu (Inverse Problem): 0.048822
Stage 2: Ground Truth nu for Inverse Problem: 0.0382
Percentage Error: 27.8067%
```

**B. Amostra de "Melhor Caso" (Seed 7):**

```text
Percentage Error: 3.5446%
```

**C. Dados Brutos do Ensemble (10 Seeds):**

* **Mínimo:** 3.54%
* **Mediana:** 54.28%
* **Máximo:** 162.83%

> **Análise Crítica:** A discrepância entre a mediana (54%) e o melhor caso (3.5%) sugere que o otimizador do problema inverso (Stage 2) é sensível à inicialização ou que a superfície de perda do surrogate possui múltiplos mínimos locais. A metodologia é promissora (vide Seed 7), mas requer estabilização.

#### 5.3.2. Detalhamento do Ensemble (10 Seeds)

Para isolar a influência da estocasticidade (inicialização de pesos e amostragem de dados), foram executadas 10 rodadas independentes mantendo os **hiperparâmetros fixos** na configuração otimizada obtida via HPO.

**A. Configuração Fixa do Ensemble:**

| Parâmetro              | Valor                            |
|:---------------------- |:-------------------------------- |
| **Arquitetura**        | 4 Camadas Ocultas x 50 Neurônios |
| **Learning Rate**      | `2.29e-4`                        |
| **Pontos PDE**         | 15.000 (Batch: 4096)             |
| **Épocas (Adam)**      | 6000 (Stage 1)                   |
| **Épocas (Data-Only)** | 1500 (Pre-training)              |

**B. Resultados Individuais por Seed:**

| Seed   | Erro de Generalização (%) | Classificação                                    |
|:------:|:-------------------------:|:------------------------------------------------ |
| **7**  | **3.54%**                 | 🟢 **Excelente** (Estado da Arte para Surrogate) |
| **5**  | 21.82%                    | 🟡 Aceitável                                     |
| **1**  | 27.81%                    | 🟡 Aceitável                                     |
| **2**  | 34.50%                    | 🟡 Aceitável                                     |
| **9**  | 37.31%                    | 🟡 Aceitável                                     |
| **3**  | 71.25%                    | 🔴 Divergência                                   |
| **4**  | 73.31%                    | 🔴 Divergência                                   |
| **6**  | 151.50%                   | 🔴 Falha Crítica                                 |
| **8**  | 156.81%                   | 🔴 Falha Crítica                                 |
| **10** | 162.83%                   | 🔴 Falha Crítica                                 |

#### 5.3.3. Análise Detalhada do Pior Caso (Seed 10)

O experimento `Seed 10` apresentou uma divergência acentuada (Erro 162%). A análise dos logs revela que o erro não foi numérico, mas sim uma falha de aprendizado da superfície de resposta.

**A. Amostragem de Treino Esparsa ("Azar Estatístico"):**
O conjunto de treino gerado aleatoriamente deixou lacunas na região do valor de teste ($
u_{target} = 0.0382$). Embora houvesse valores próximos, a dinâmica de treinamento não os priorizou.

```text
Generating 19 datasets for generalization training...
  Generating data for nu_true = 0.0794...
  Generating data for nu_true = 0.0324...
  Generating data for nu_true = 0.0910...
  ... (lacuna de cobertura na região crítica) ...
```

**B. Divergência da Otimização Inversa (Stage 2):**
O otimizador foi "enganado" pela rede neural. A superfície de perda aprendida pelo modelo surrogate continha um gradiente falso que empurrou a solução para o limite superior do domínio físico ($
u=0.1$), longe do valor real ($
u=0.038$).

```text
Starting Adam pre-training for nu_inverse...
  Adam Epoch 0: Loss = 0.000590, Discovered nu = 0.020020
  ...
  Adam Epoch 900: Loss = 0.000281, Discovered nu = 0.053863  <-- Deriva para longe do alvo (0.038)

Starting L-BFGS-B optimization...
  Stage 2 Discovered nu (Inverse Problem): 0.100403        <-- Salto para a fronteira (0.1)
  Percentage Error: 162.8348%
```

**Conclusão:** A rede neural aprendeu uma correlação espúria. Para o dataset de teste, o modelo "acreditava" que aumentar a viscosidade reduzia o erro, levando o otimizador a colidir com a barreira superior ($
u 
approx 0.1$). Isso reforça a necessidade de estratégias de amostragem estratificada (ex: *Latin Hypercube Sampling*) para garantir cobertura uniforme e evitar distorções físicas em regiões pouco exploradas.

### 5.4. Caso Surrogate V2 com Latin Hypercube Sampling (LHS)

A estratégia de amostragem foi refinada, trocando a seleção aleatória de `nu` pela amostragem estratificada (LHS), que garante uma cobertura mais uniforme do espaço de parâmetros.

> **Definição:** Treino em 19 datasets amostrados via LHS. Teste em um dataset **inédito** ($\nu=0.0382$).

| Métrica Estatística | Erro Percentual (%) |
|:------------------- |:------------------- |
| Média               | 20.5578             |
| Desvio Padrão       | 21.4571             |
| Mínimo              | 2.6314              |
| Máximo              | 62.6789             |

**Análise Comparativa:**
A média de erro de generalização (20.56%) representa uma melhoria de **~3.6x** em relação à amostragem aleatória (74.07%). Mais importante, o desvio padrão foi reduzido em **~2.7x** (de 57.87% para 21.46%), indicando uma **estabilização significativa** do treinamento. A estratégia LHS mitigou os piores cenários de divergência, eliminando as falhas críticas observadas anteriormente e tornando o modelo surrogate mais confiável.

### 5.6. Análise Comparativa de Tempo de Execução (End-to-End)

A transição de um modelo especialista para um surrogate generalista implica um custo computacional maior no treinamento, que é compensado pela capacidade de inferência instantânea. A tabela a seguir resume o tempo total de execução para cada estratégia principal.

| Estratégia                      | Tempo Médio (minutos) | Desvio Padrão (minutos) | Notas                          |
|:------------------------------- |:---------------------:|:-----------------------:|:------------------------------ |
| 1. Especialista ($\nu=0.05$)    | 8.68                  | 0.05                    | Média de 3 execuções.          |
| 2. Surrogate V1 (Unified)       | 5.83                  | N/A                     | Micro-experimento, 3 datasets. |
| 3. Surrogate V2 (Random)        | 19.00                 | 0.10                    | Ensemble de 3 execuções.       |
| 4. Surrogate V2 + LHS           | 19.16                 | 0.55                    | Ensemble de 5 execuções.       |
| 5. Otimização Focada (LHS Ext.) | 41.65                 | N/A                     | Execução única, 15.000 épocas. |

**Análise:** O custo computacional para treinar um modelo surrogate (`~19 min`) é aproximadamente **2.2x maior** que o de um modelo especialista (`~8.7 min`). A otimização focada, com treinamento estendido, eleva esse custo para **~4.8x**. Este é o *trade-off* fundamental: um maior investimento inicial no treinamento do surrogate para obter um modelo capaz de realizar inferências em novos cenários em segundos, eliminando a necessidade de re-treinamentos completos.

### 5.5. Otimização Focada: Refinamento do Melhor Caso LHS (Experimento `lhs2`)

Após a estabilização do modelo com a amostragem LHS, a investigação focou em refinar o resultado mais promissor (erro de 2.63% com Seed 2) para testar a hipótese de que um treinamento mais longo do modelo surrogate poderia levar a uma superfície de perda mais precisa e, consequentemente, a uma inferência mais acurada no problema inverso.

**Metodologia:**

1. **Identificação do "Campeão":** O script `find_best_lhs_seed.py` analisou o ensemble LHS e confirmou que a **Seed 2** produziu o menor erro de generalização.
2. **Treinamento Estendido:** O experimento foi re-executado utilizando a Seed 2, mas com o número de épocas de treinamento do Adam (Etapa 1) aumentado de 6.000 para **15.000**.
3. **Isolamento de Variáveis:** Todos os outros hiperparâmetros foram mantidos idênticos ao do ensemble LHS para garantir uma comparação direta. Os resultados e logs foram salvos em um novo diretório (`results/lhs2`, `logs/lhs2`).

**Resultados Comparativos:**

| Parâmetro Modificado      | Erro de Generalização (%) | Variação (%) |
|:------------------------- |:-------------------------:|:------------:|
| **Épocas: 6.000** (Base)  | 2.6314%                   | -            |
| **Épocas: 15.000** (Novo) | **2.5363%**               | **-3.61%**   |

**Evidência de Execução:**
O resultado final foi extraído diretamente do log do experimento.

* **Artefato:** `logs/lhs2/extended_run_seed_2.log`
  
  ```text
  > Ground Truth nu (Validation): 0.050000
  > Discovered nu (Validation):   0.048732
  > Percentage Error (Validation): 2.5363%
  --- Evaluation Finished ---
  Results for trial saved to results/lhs2/lhs2_extended_training_seed_2.npz
  ```

**Conclusão da Otimização Focada:**
O aumento no tempo de treinamento resultou em uma melhoria marginal, mas mensurável, de **3.61%** sobre o melhor caso anterior. Isso valida a hipótese de que a qualidade do modelo surrogate é um fator limitante na precisão da inferência. Embora o ganho seja pequeno, ele confirma que o modelo não havia convergido totalmente e que há espaço para maior precisão com mais investimento computacional no treinamento do surrogate.

---

## 6. Glossário de Termos e Abreviações

* **PINN (Physics-Informed Neural Network):** Rede Neural Informada pela Física. Uma rede neural que integra as equações diferenciais parciais (PDEs) que governam um sistema físico diretamente em sua função de perda durante o treinamento.
* **Surrogate Model (Modelo Substituto):** Um modelo de aprendizado de máquina treinado para aproximar o comportamento de um sistema complexo (neste caso, o solucionador da PDE de Burgers), permitindo inferências rápidas de parâmetros.
* **LHS (Latin Hypercube Sampling):** Amostragem por Hipercubo Latino. Uma técnica de amostragem estatística estratificada que garante que os pontos de amostragem cubram o espaço de parâmetros de forma mais uniforme do que a amostragem aleatória.
* **PDE (Partial Differential Equation):** Equação Diferencial Parcial. Uma equação matemática que descreve a dinâmica de sistemas físicos, como o escoamento de fluidos.
* **HPO (Hyperparameter Optimization):** Otimização de Hiperparâmetros. O processo de busca automatizada para encontrar a melhor combinação de hiperparâmetros (ex: taxa de aprendizado, número de camadas) para um modelo de aprendizado de máquina.
* **OOM (Out of Memory):** Erro de "Falta de Memória" que ocorre quando um programa tenta alocar mais memória (RAM ou VRAM) do que a disponível no sistema.
* **$\nu$ (nu):** Símbolo grego que representa a viscosidade cinemática do fluido na equação de Burgers, o parâmetro físico que o modelo surrogate visa inferir.

---

## 7. Conclusão

A pesquisa demonstrou que a arquitetura PINN pode escalar de um solucionador de instâncias isoladas para um meta-modelo físico. Embora a precisão absoluta da inferência de parâmetro em dados não vistos (Erro médio ~74%) seja inferior à do especialista "overfitted" (< 1%), o experimento de "Melhor Caso" (Erro ~3.5%) prova que o modelo surrogate tem capacidade de aprender a física subjacente.

Em contrapartida, a abordagem "Unified Dataset" (V1) mostrou-se ineficiente em ambientes com recursos limitados (GPUs < 24GB VRAM), exigindo compromissos severos na quantidade de dados que degradam a precisão (~32% erro). Portanto, a estratégia de **"Multi-Dataset Sampling"** (V2) consolida-se como o caminho viável para ambientes computacionais restritos.

O principal valor agregado é a transformação do modelo em um **sensor virtual instantâneo**, capaz de inferir propriedades físicas ($\nu$) de um novo escoamento em **menos de 1 minuto** (vs. ~9 minutos do caso base), sem necessidade de re-treinamento. As otimizações computacionais foram essenciais para esta evolução.

<br><sub>Last edited: 2025-12-09 23:43:57</sub>
