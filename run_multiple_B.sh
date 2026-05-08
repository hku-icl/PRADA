#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false

PYTHON_SCRIPT="main_PRADA.py"
LOG_DIR="./mylogs"
mkdir -p "$LOG_DIR"

B_VALUES=(1e+6 5e+6 1e+7 2e+7 4e+7 6e+7)

for B in "${B_VALUES[@]}"; do
    LOG_FILE="${LOG_DIR}/mulusers_B${B}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "===== beginning running B=${B}: $(date) =====" | tee -a "$LOG_FILE"
    echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
    
    TOKENIZERS_PARALLELISM=false python3 "$PYTHON_SCRIPT" \
        --B "${B}" \
        2>&1 | tee -a "$LOG_FILE"
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "===== Success B=${B}: $(date) =====" | tee -a "$LOG_FILE"
    else
        echo "===== Failed B=${B}（Exit code: $EXIT_CODE）: $(date) =====" | tee -a "$LOG_FILE"
    fi
    sleep 2
done
echo "===== All B values completed: $(date) ====="