# Análise Comparativa de Desempenho (Rastreável)

Este documento apresenta a análise comparativa final entre o modelo Base e o modelo Atual, com rastreabilidade explícita para os arquivos de log que servem como fonte de evidência para cada resultado.

---

## 1. Modelo Base (Especialista): Desempenho e Evidência

Os cálculos de média (0.48%) e desvio padrão (0.51%) para o erro percentual foram baseados nos **12 experimentos** publicados na Tabela 1 do paper `ciaca-2025-66.pdf`.

**Fonte de Evidência (Arquivos de Log):**
A análise dos logs revelou o padrão de nomenclatura `plateau_nu_true_{...}_nu_initial_{...}_seed_{...}.txt`. Os 12 arquivos que correspondem aos resultados do paper são:

```
logs/plateau/plateau_nu_true_0.02_nu_initial_0.2_seed_17.txt
logs/plateau/plateau_nu_true_0.02_nu_initial_0.2_seed_53.txt
logs/plateau/plateau_nu_true_0.02_nu_initial_0.2_seed_89.txt
logs/plateau/plateau_nu_true_0.02_nu_initial_0.09_seed_17.txt
logs/plateau/plateau_nu_true_0.02_nu_initial_0.09_seed_53.txt
logs/plateau/plateau_nu_true_0.02_nu_initial_0.09_seed_89.txt
logs/plateau/plateau_nu_true_0.05_nu_initial_0.2_seed_17.txt
logs/plateau/plateau_nu_true_0.05_nu_initial_0.2_seed_53.txt
logs/plateau/plateau_nu_true_0.05_nu_initial_0.2_seed_89.txt
logs/plateau/plateau_nu_true_0.05_nu_initial_0.09_seed_17.txt
logs/plateau/plateau_nu_true_0.05_nu_initial_0.09_seed_53.txt
logs/plateau/plateau_nu_true_0.05_nu_initial_0.09_seed_89.txt
```

---

## 2. Modelo Atual (Generalista - LHS): Desempenho e Evidência

Os cálculos de média (20.56%) e desvio padrão (21.46%) foram baseados nos resultados de 5 execuções com sementes diferentes, conforme descrito em `latin_analysis_results.md`.

**Fonte de Evidência (Arquivos de Log):**
A análise do script `run_latin.sh` confirmou que os seguintes 5 arquivos de log são a fonte da verdade:

```
logs/latin/latin_run_seed_1.txt
logs/latin/latin_run_seed_2.txt
logs/latin/latin_run_seed_3.txt
logs/latin/latin_run_seed_4.txt
logs/latin/latin_run_seed_5.txt
```

---

## 3. Tabela Comparativa Final (Auditável)

| Modelo                  | Erro Percentual Médio | Desvio Padrão (Erro %) |
|:----------------------- |:--------------------- |:---------------------- |
| **Base (Especialista)** | **0.48 %**            | **0.51 %**             |
| **Atual (Generalista)** | 20.56 %               | 21.46 %                |

---

## 4. Conclusões de Desempenho

* O **Modelo Base**, validado contra os resultados publicados no paper, é altamente preciso (erro médio de 0.48%) e demonstra consistência relativamente alta (desvio padrão de 0.51%) em uma variedade de cenários de teste específicos.
* O **Modelo Atual (LHS)**, ao generalizar para uma faixa de parâmetros, exibe um erro médio **~43x maior** e um desvio padrão **~42x maior**, destacando o aumento dramático na complexidade e instabilidade da tarefa de generalização em comparação com os casos de teste específicos.

---

## 5. Análise Funcional e Casos de Uso

A diferença de desempenho está diretamente ligada ao **propósito fundamentalmente distinto** de cada modelo:

* **Modelo Base (Especialista):**
  
  * **Função:** Resolve um **problema inverso**. Seu objetivo é **descobrir** o valor da viscosidade (`nu`) a partir de dados observados de um fluxo.
  * **Dependência:** Ele **requer uma estimativa inicial de `nu`** como ponto de partida para o processo de otimização. A convergência para o valor correto depende dessa aproximação.
  * **Caso de Uso:** Ideal para cenários onde se tem dados de um sistema físico e se deseja determinar um parâmetro desconhecido desse sistema (ex: `nu`), assumindo que se tenha uma estimativa inicial razoável.

* **Modelo Atual (Generalista - LHS):**
  
  * **Função:** Resolve um **problema direto (paramétrico)**. Seu objetivo é **prever** o comportamento do fluxo para **qualquer valor de `nu`** dentro da faixa em que foi treinado.
  * **Dependência:** Ele **não precisa de uma aproximação de `nu`** porque não o está descobrindo. Em vez disso, `nu` é um **dado de entrada** para o modelo.
  * **Caso de Uso:** Ideal para cenários de simulação, onde se deseja explorar rapidamente como o sistema se comporta ao variar o parâmetro `nu`, sem a necessidade de resolver equações complexas a cada vez.
<br><sub>Last edited: 2025-12-07 13:11:34</sub>
