#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false

PYTHON_SCRIPT="main_PRADA.py"
LOG_DIR="./mylogs"
mkdir -p "$LOG_DIR"

M_VALUES=(1 3 7 9 12 15 20)

for M in "${M_VALUES[@]}"; do
    LOG_FILE="${LOG_DIR}/mulusers_M${M}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "===== Beginning running M=${M}: $(date) =====" | tee -a "$LOG_FILE"
    echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
    
    TOKENIZERS_PARALLELISM=false python3 "$PYTHON_SCRIPT" \
        --M "${M}" \
        2>&1 | tee -a "$LOG_FILE"
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "===== Success M=${M}: $(date) =====" | tee -a "$LOG_FILE"
    else
        echo "===== Failed M=${M}（Exit code: $EXIT_CODE）: $(date) =====" | tee -a "$LOG_FILE"
    fi
    sleep 2
done
echo "===== All M values completed: $(date) ====="
