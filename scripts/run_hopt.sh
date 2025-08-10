#!/bin/bash

# =====================================================================
# Script de Orquestração de HPO (Corrigido para Acumular Conhecimento)
# =====================================================================

# 1. Verificação de Segurança
if [ -z "$IMG" ]; then
    echo "ERRO CRÍTICO: A variável \$IMG não está definida."
    exit 1
fi

if [ ! -f "$IMG" ]; then
    echo "ERRO CRÍTICO: Arquivo de contêiner não encontrado em: $IMG"
    exit 1
fi

# 2. Configuração
N_JOBS=10
# CORREÇÃO: Usamos um ÚNICO arquivo para acumular o histórico das 50 tentativas
TRIALS_FILE="results/hopt/hyperopt_history.pkl" 

echo "=== Iniciando Orquestração (run_hopt.sh) ==="
echo "Data: $(date)"
echo "Contêiner: $IMG"
echo "Arquivo de Histórico: $TRIALS_FILE"
echo "--------------------------------------------------"

# --- LOOP DE EXECUÇÃO ---

for i in $(seq 1 $N_JOBS)
do
    # O log continua separado para facilitar sua leitura
    LOG_FILE="logs/hopt_parallel/run_$i.log"
    
    echo ">>> Iniciando Rodada $i de $N_JOBS (Acumulando +5 tentativas)"
    echo "    Log: $LOG_FILE"
    
    # Executa o Python usando SEMPRE o mesmo arquivo .pkl
    singularity exec --nv -B /scratch "$IMG" python main_hopt_unified.py \
        --optimize \
        --trials_file "$TRIALS_FILE" > "$LOG_FILE" 2>&1
        
    if [ $? -ne 0 ]; then
        echo "    [ALERTA] A Rodada $i falhou. Verifique $LOG_FILE."
        # Não damos exit para tentar salvar o que for possível nas próximas
    else
        echo "    [OK] Rodada $i concluída com sucesso."
    fi
done

echo "=== HPO Finalizado em $(date) ==="
