#!/bin/bash

PYTHON_SCRIPT="main_train_all.py"
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/newtrain_$(date +%Y%m%d_%H%M%S).log"

echo "===== beginning running: $(date) =====" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

TOKENIZERS_PARALLELISM=false python3 "$PYTHON_SCRIPT" \
    --data_names "math" \
    --data_dir "./external/qwen25_math_evaluation/data" \
    --draft_model_name_or_path "Qwen/Qwen2.5-Math-1.5B-Instruct" \
    --target_model_name_or_path "Qwen/Qwen2.5-Math-7B-Instruct" \
    --prm_name_or_path "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B" \
    --draft_model_ip_address "http://localhost:12340/v1" \
    --target_model_ip_address "http://localhost:12341/v1" \
    --prm_ip_address "http://localhost:12342/v1" \
    --prm_threshold "0.7" \
    --max_steps "100" \
    --output_dir "outputs/draft_Qwen2.5-Math-1.5B-Instruct_target_Qwen2.5-Math-7B-Instruct_prm_Skywork-o1-Open-PRM-Qwen-2.5-7B/math_eval" \
    --split "test" \
    --prompt_type "qwen25-math-cot" \
    --num_test_sample "-1" \
    --seed "0" \
    --temperature "0" \
    --n_sampling "1" \
    --top_p "1" \
    --start "0" \
    --end "4500" \
    --save_outputs \
    --overwrite \
    --max_tokens_per_call "4096" \
    --beta "2.5e-11" \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "===== Success: $(date) =====" | tee -a "$LOG_FILE"
else
    echo "===== Failed (Exit code: $EXIT_CODE): $(date) =====" | tee -a "$LOG_FILE"
    exit $EXIT_CODE
fi