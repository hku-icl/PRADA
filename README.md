# PRADA

PRADA is a resource-aware inference scheduling framework for mathematical reasoning with large language models. It coordinates a draft model, a target model, and a process reward model (PRM), while explicitly considering user count, bandwidth budget, processing delay, communication delay, and queueing delay.

Paper Title: Accelerating Heterogeneous Agent Collaboration in Dynamic Edge Networks
Paper DOI: [https://doi.org/10.20944/preprints202605.0251.v1](https://doi.org/10.20944/preprints202605.0251.v1)

## Overview

This repository contains the code, scripts, model checkpoints, evaluation utilities, and generated experiment results for PRADA. The current implementation is built around:

- Qwen2.5-Math models as draft/target reasoning models
- Skywork-o1-Open-PRM as the process reward model
- vLLM OpenAI-compatible API servers
- Qwen2.5-Math evaluation datasets and answer checking tools

## Repository Layout

```text
.
|-- main_PRADA.py              # PRADA inference and scheduling entry point
|-- main_train_all.py          # PRADA training entry point
|-- model_pai.pth              # trained policy network checkpoint
|-- model_v.pth                # trained value network checkpoint
|-- model_d1.pth               # trained delay prediction network checkpoint
|-- model_randa.pth            # trained reward/advantage network checkpoint
|-- run_train.sh               # training script
|-- run_multiple_M.sh          # batch experiment over user counts M
|-- run_multiple_B.sh          # batch experiment over bandwidth budgets B
|-- serve.sh                   # start draft, target, and PRM services in background
|-- scripts/                   # individual vLLM service scripts
|-- external/
|   |-- qwen25_math_evaluation/     # datasets, parsing, grading, and math evaluation
|   `-- skywork_o1_prm_inference/   # Skywork PRM inference utilities
|-- skywork-o1-prm-inference/       # Skywork PRM code copy
`-- myresults/                 # generated plots, npy files, and json result details
```

## Environment

Create and activate a Python environment:

```bash
conda create -n prada python=3.11 -y
conda activate prada
```

Install evaluation and runtime dependencies:

```bash
pip install -r external/qwen25_math_evaluation/requirements.txt
pip install openai matplotlib numpy
```

Install the Skywork PRM helper package:

```
git clone https://github.com/SkyworkAI/skywork-o1-prm-inference.git
cd skywork-o1-prm-inference
pip install -e .
```

## Model Services
PRADA talks to local vLLM servers through OpenAI-compatible API endpoints. The default ports are:

```text
draft model : http://localhost:12340/v1
target model: http://localhost:12341/v1
PRM         : http://localhost:12342/v1
```

Start all three services in the background:

```bash
bash serve.sh
```

`serve.sh` runs:

```bash
nohup scripts/serve_draft_model.sh > draft.log 2>&1 &
nohup scripts/serve_target_model.sh > target.log 2>&1 &
nohup scripts/serve_prm.sh > prm.log 2>&1 &
```

You can also start them one by one:

```bash
bash scripts/serve_draft_model.sh
bash scripts/serve_target_model.sh
bash scripts/serve_prm.sh
```

Before running, edit `CUDA_VISIBLE_DEVICES`, `tensor-parallel-size`, and `gpu_memory_utilization` in the service scripts according to your machine.

## PRADA Inference

The main inference entry point is:

```bash
python3 main_PRADA.py
```

Important arguments:

```text
--data_names                  dataset name, default mmlu_stem
--data_dir                    dataset directory
--draft_model_name_or_path    draft model name or path
--draft_model_ip_address      draft model API endpoint
--target_model_name_or_path   target model name or path
--target_model_ip_address     target model API endpoint
--B                           bandwidth budget, default 4e+7
--M                           number of users, default 9
--beta                        delay/resource trade-off coefficient
--FlopsLLM                    LLM FLOPs parameter
--max_tokens_per_call         maximum generated tokens per API call
```

Example:

```bash
TOKENIZERS_PARALLELISM=false python3 main_PRADA.py \
  --data_names gsm8k \
  --data_dir ./external/qwen25_math_evaluation/data \
  --draft_model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct \
  --target_model_name_or_path Qwen/Qwen2.5-Math-7B-Instruct \
  --draft_model_ip_address http://localhost:12340/v1 \
  --target_model_ip_address http://localhost:12341/v1 \
  --split test \
  --prompt_type qwen25-math-cot \
  --temperature 0 \
  --n_sampling 1 \
  --top_p 1 \
  --start 0 \
  --end -1 \
  --max_tokens_per_call 4096 \
  --B 4e+7 \
  --M 9
```

`main_PRADA.py` loads these checkpoints from the repository root:

```text
model_pai.pth
model_d1.pth
model_randa.pth
```

Generated results are written to:

```text
myresults/<data_names>/
```

## Batch Experiments

Run PRADA over multiple user counts:

```bash
bash run_multiple_M.sh
```

Current `M` values:

```text
1 3 7 9 12 15 20
```

Run PRADA over multiple bandwidth budgets:

```bash
bash run_multiple_B.sh
```

Current `B` values:

```text
1e+6 5e+6 1e+7 2e+7 4e+7 6e+7
```

Both scripts write logs to:

```text
mylogs/
```

## Training

Train PRADA models with:

```bash
bash run_train.sh
```

The training script runs `main_train_all.py`, logs to `logs/`, and saves checkpoints every 10 epochs:

```text
model_pai_ep<epoch>_<beta>.pth
model_v_ep<epoch>_<beta>.pth
model_d1_ep<epoch>_<beta>.pth
model_randa_ep<epoch>_<beta>.pth
```

## Datasets

Datasets are stored under:

```text
external/qwen25_math_evaluation/data/
```

Available datasets include:

```text
gsm8k
math
math500
mmlu_stem
gaokao2023en
gaokao2024_I
gaokao2024_II
gaokao2024_mix
gpqa
sat_math
svamp
asdiv
aqua
mawps
tabmwp
minerva_math
```

Default datasets:

```text
main_PRADA.py     mmlu_stem
main_train_all.py math500
```

## Results

`myresults/` contains generated experiment outputs, including:

```text
accs*.npy / accsVSepoch*.png
avg_processing_delay*.npy / *.png
avg_communication_delay*.npy / *.png
avg_queue_delay*.npy / *.png
task_delay_details*.json
```

The result subdirectories include:

```text
compare_by_M/              # comparison by user count M
compare_by_B/              # comparison by bandwidth B
random_binary_policy/      # random binary policy baseline
random_ternary_scheduler/  # random ternary scheduler baseline
pi_theta/                  # policy-related plots
```

## Notes

- Run `bash serve.sh` before inference or training so that the draft, target, and PRM endpoints are available.
- The default scripts assume a multi-GPU machine. Adjust GPU IDs before running on a different server.
- Some comments in older Python files may show encoding artifacts; the core entry points and argument names are still readable from the code.

## Acknowledgement

This project uses or builds on:

- Qwen2.5-Math evaluation
- Skywork-o1-Open-PRM
- vLLM OpenAI-compatible APIs

## Citation

If you find this project useful, please cite:

```bibtex
@article{202605.0251,
	doi = {10.20944/preprints202605.0251.v1},
	url = {https://doi.org/10.20944/preprints202605.0251.v1},
	year = 2026,
	month = {May},
	publisher = {Preprints},
	author = {Tianji He and Yulin Shao and Fen Hou},
	title = {Accelerating Heterogeneous Agent Collaboration in Dynamic Edge Networks},
	journal = {Preprints}
}
```
