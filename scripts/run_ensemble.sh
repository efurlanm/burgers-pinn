#!/bin/bash

# ==============================================================================
# Script para ExecuÃ§Ã£o de Ensemble Runs
#
# Objetivo: Avaliar a estabilidade e robustez do melhor modelo encontrado
#           pela otimizaÃ§Ã£o de hiperparÃ¢metros (HPO).
#
# Metodologia: O script executa o modelo principal com os mesmos
#              hiperparÃ¢metros, mas com diferentes seeds de inicializaÃ§Ã£o,
#              permitindo a anÃ¡lise estatÃ­stica dos resultados.
# ==============================================================================

# Ativa o ambiente Conda necessÃ¡rio
echo "Ativando o ambiente Conda 'tf2'..."
source $HOME/conda/bin/activate tf2

# Verifica se a ativaÃ§Ã£o foi bem-sucedida
if [ $? -ne 0 ]; then
    echo "Erro: NÃ£o foi possÃ­vel ativar o ambiente Conda 'tf2'."
    exit 1
fi

# HiperparÃ¢metros fixos (baseado no melhor resultado de HPO-001)
ADAM_EPOCHS_STAGE1=5000
EPOCHS_DATA_ONLY_STAGE1=500
EPOCHS_INVERSE_ADAM_PRETRAIN=2000
LAYERS=4
LEARNING_RATE=0.0002299
NEURONS=50
NOISE_LEVEL=0.0399
# NU_TRUE serÃ¡ gerado aleatoriamente dentro do script principal para cada run
NUM_DATASETS_GENE=15
NUM_PDE_POINTS_STAGE1=20000

# Seeds para o ensemble run (escala reduzida)
SEEDS=(1 2 3)

# DiretÃ³rio de logs e resultados
LOG_DIR="burgers-pinn/logs/ensemble"
RESULTS_DIR="burgers-pinn/results/ensemble"
mkdir -p $LOG_DIR
mkdir -p $RESULTS_DIR

echo "Iniciando a execuÃ§Ã£o do ensemble..."
echo "Total de runs: ${#SEEDS[@]}"
echo "HiperparÃ¢metros fixos:"
echo "  - ADAM_EPOCHS_STAGE1: $ADAM_EPOCHS_STAGE1"
echo "  - LAYERS: $LAYERS"
echo "  - NEURONS: $NEURONS"
echo "  - LEARNING_RATE: $LEARNING_RATE"
echo "--------------------------------------------------"

# Loop atravÃ©s das seeds para executar os runs
for SEED in "${SEEDS[@]}"
do
    RUN_ID="ensemble_seed_${SEED}"
    LOG_FILE="${LOG_DIR}/${RUN_ID}.txt"

    echo "Iniciando run com SEED: $SEED"
    echo "  - Run ID: $RUN_ID"
    echo "  - Log serÃ¡ salvo em: $LOG_FILE"

    # Executa o script principal em background
    python main_hopt_unified.py \
        --adam_epochs_stage1 $ADAM_EPOCHS_STAGE1 \
        --epochs_data_only_stage1 $EPOCHS_DATA_ONLY_STAGE1 \
        --epochs_inverse_adam_pretrain $EPOCHS_INVERSE_ADAM_PRETRAIN \
        --layers $LAYERS \
        --learning_rate $LEARNING_RATE \
        --neurons $NEURONS \
        --noise_level $NOISE_LEVEL \
        --num_datasets_gene $NUM_DATASETS_GENE \
        --num_pde_points_stage1 $NUM_PDE_POINTS_STAGE1 \
        --seed $SEED \
        --run_id "${RUN_ID}" \
        --results_dir $RESULTS_DIR \
        > "$LOG_FILE" 2>&1 &

    # Captura o PID do processo em background
    PID=$!
    echo "  - Processo iniciado com PID: $PID. Aguardando a conclusÃ£o..."

    # Aguarda a conclusÃ£o do processo
    wait $PID

    # Verifica o cÃ³digo de saÃ­da
    if [ $? -eq 0 ]; then
        echo "  - Run com SEED $SEED concluÃ­do com sucesso."
    else
        echo "  - ERRO: Run com SEED $SEED falhou. Verifique o log: $LOG_FILE"
    fi
    echo "--------------------------------------------------"
done

echo "ExecuÃ§Ã£o do ensemble concluÃ­da."