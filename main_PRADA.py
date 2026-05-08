import random
import os
import json
import math
import argparse
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

from external.qwen25_math_evaluation.evaluate import evaluate
from external.qwen25_math_evaluation.utils import load_jsonl, construct_prompt
from external.qwen25_math_evaluation.parser import *
from external.qwen25_math_evaluation.trajectory import *
from external.qwen25_math_evaluation.data_loader import load_data
from external.qwen25_math_evaluation.python_executor import PythonExecutor

import numpy as np
import asyncio


def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def to_float(x):
    if isinstance(x, (float, int)):
        return float(x)
    if hasattr(x, "cpu"):
        x = x.cpu().detach()
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def sanitize_randa_outputs(randa_output):
    return torch.clamp(randa_output, min=1e-8)


def compute_processing_delay(num_tokens_star, num_tokens_generate, model_type):
    dmodelS = 1536
    dmodelL = 3584

    if model_type.upper() == "SLM":
        dmodel = dmodelS
    elif model_type.upper() == "LLM":
        dmodel = dmodelL
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    delay = 28*(
        num_tokens_generate * (dmodel ** 2)
        + dmodel * (2 * num_tokens_star + num_tokens_generate - 1) * num_tokens_generate / 2
    )
    return float(delay)


def compute_communication_delay(num_tokens_star, snr, bandwidth, flops_llm, beta, M):
    rate_term = max(np.log2(snr + 1.0), 1e-12)
    bandwidth = max(float(bandwidth), 1e-12)
    delay = 32.0 * flops_llm * num_tokens_star / (M * bandwidth * rate_term)
    return float(delay)


def evaluate_single(output, prompt, orig_idx, data_name, draft_tokenizer, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)

    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    example = examples[orig_idx]
    idx = example["idx"]

    example["question"] = parse_question(example, data_name)
    if example["question"] == "":
        return 0

    gt_cot, gt_ans = parse_ground_truth(example, data_name)
    example["gt_ans"] = gt_ans
    full_prompt = construct_prompt(example, data_name, args)

    if idx == args.start:
        print(full_prompt)

    sample = {
        "idx": idx,
        "question": example["question"],
        "gt_cot": gt_cot,
        "gt": gt_ans,
        "prompt": full_prompt,
    }

    for key in [
        "level", "type", "unit", "solution_type", "choices", "solution",
        "ques_type", "ans_type", "answer_type", "dataset", "subfield",
        "filed", "theorem", "answer",
    ]:
        if key in example:
            sample[key] = example[key]

    input_prompts = [sample["prompt"] for _ in range(args.n_sampling)]
    if args.apply_chat_template:
        input_prompts = [
            draft_tokenizer.apply_chat_template(
                [{"role": "user", "content": p.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in input_prompts
        ]

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    if args.prompt_type == "cot":
        stop_words.append("\n\nQuestion:")
    elif args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")

    remain_prompt = []
    end_prompt = []
    query = prompt + output

    if args.prompt_type == "pal":
        remain_prompt = [(orig_idx, query)]
        if "```python" in output:
            output = extract_program(query)
    elif args.prompt_type == "cot":
        end_prompt = [(orig_idx, query)]
    elif "boxed" not in output and output.endswith("```"):
        remain_prompt = [(orig_idx, query)]
    else:
        end_prompt = [(orig_idx, query)]

    end_prompt.extend(remain_prompt)
    end_prompt = sorted(end_prompt, key=lambda x: x[0])

    codes = []
    _, end_prompt_text = end_prompt[0]
    code = end_prompt_text.split(input_prompts[0])[-1].strip()
    for stop_word in stop_words:
        if stop_word in code:
            code = code.split(stop_word)[0].strip()
    codes.append(code)

    results = [run_execute(executor, c, args.prompt_type, data_name) for c in codes]
    result = results[0]
    preds = [result[0]]
    reports = [result[1]]

    for j in range(len(preds)):
        if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in ["A", "B", "C", "D", "E"]:
            preds[j] = choice_answer_clean(code)
        elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
            preds[j] = "".join([c for c in preds[j] if c in ["A", "B", "C", "D", "E"]])

    sample.pop("prompt")
    sample.update(
        {"code": code, "pred": preds, "report": reports, "token_counts": [], "turn_info": [], "reward": []}
    )
    samples = [sample]
    samples.extend(processed_samples)

    all_samples, result_json, _, correct_indices = evaluate(
        samples=samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    return len(correct_indices) != 0


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim=512):
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.policy_head(x)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=512):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.value_head(x)


class Delay(nn.Module):
    def __init__(self, state_dim, hidden_dim=512):
        super().__init__()
        self.film = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, state_dim * 2),
        )

        self.delay_head = nn.Sequential(
            nn.BatchNorm1d(state_dim),
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, token_num):
        film_params = self.film(token_num.unsqueeze(-1))
        if film_params.dim() == 1:
            film_params = film_params.unsqueeze(0)
        gamma, beta = film_params.chunk(2, dim=1)
        x = x * gamma + beta
        return self.delay_head(x)


class RandA(nn.Module):
    def __init__(self, state_dim, hidden_dim=512):
        super().__init__()
        self.reward_head = nn.Sequential(
            nn.BatchNorm1d(state_dim),
            nn.Linear(state_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.reward_head(x)


def get_state_from_batch(slm, slm_tokenizer, prompts, slm_name="Qwen/Qwen2.5-Math-1.5B-Instruct"):
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    if slm_tokenizer.pad_token is None:
        slm_tokenizer.pad_token = slm_tokenizer.eos_token

    batch_size = 1
    all_results = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        inputs = slm_tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=4096,
            return_attention_mask=True
        ).to(device)

        slm.eval()
        with torch.no_grad():
            outputs = slm(
                **inputs,
                use_cache=True,
                output_hidden_states=True
            )

        hidden_states = outputs.hidden_states[-1]
        batch_size_current = hidden_states.shape[0]
        last_valid_indices = inputs["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(batch_size_current, device=device)
        last_hidden_states = hidden_states[batch_indices, last_valid_indices, :]

        all_results.append(last_hidden_states.float().cpu())

        del inputs, outputs, hidden_states, last_hidden_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = torch.cat(all_results, dim=0)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="mmlu_stem", type=str)
    parser.add_argument("--data_dir", default="./external/qwen25_math_evaluation/data", type=str)
    parser.add_argument("--draft_model_name_or_path", default="Qwen/Qwen2.5-Math-1.5B-Instruct", type=str)
    parser.add_argument("--draft_model_ip_address", default="http://localhost:12340/v1", type=str)
    parser.add_argument("--target_model_name_or_path", default="Qwen/Qwen2.5-Math-7B-Instruct", type=str)
    parser.add_argument("--target_model_ip_address", default="http://localhost:12341/v1", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="qwen25-math-cot", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--step_word", type=str, default="\n\n")
    parser.add_argument("--prm_threshold", type=float, default=0.7)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--beta", type=float, default=2.5e-11)
    parser.add_argument("--B", type=float, default=4e+7)
    parser.add_argument("--M", type=int, default=9)
    parser.add_argument("--FlopsLLM", type=float, default=8e+13)
    parser.add_argument("--eta", type=float, default=1e-3)
    parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--adapt_few_shot", action="store_true")
    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(examples)

    examples = examples[args.start: len(examples) if args.end == -1 else args.end]

    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_delta{args.prm_threshold}_maxsteps{args.max_steps}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(list(load_jsonl(f"{output_dir}/{data_name}/{f}")))

    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def gain_dataset(data_name, args, draft_tokenizer):
    examples, _, _ = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        for key in [
            "level", "type", "unit", "solution_type", "choices", "solution",
            "ques_type", "ans_type", "answer_type", "dataset", "subfield",
            "filed", "theorem", "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    samples = sorted(samples, key=lambda x: x["idx"])
    input_prompts = [sample["prompt"] for sample in samples for _ in range(args.n_sampling)]
    if args.apply_chat_template:
        input_prompts = [
            draft_tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]

    remain_prompts = [prompt for _, prompt in enumerate(input_prompts)]
    return remain_prompts, samples


def generate_task_from_dataset(examples, idx):
    sample = examples[idx]
    return sample, idx


async def main():
    args = parse_args()
    set_global_seed(args.seed)
    data_name = args.data_names
    scheduler_queue = asyncio.Queue()
    llmprocessing_queue = []

    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    model_pai = Actor(1536).to(device)
    model_delay = Delay(1536).to(device)
    model_randa = RandA(1536).to(device)
    model_pai.eval()
    model_delay.eval()
    model_randa.eval()
    model_pai.load_state_dict(torch.load("model_pai.pth"))
    model_delay.load_state_dict(torch.load("model_d1.pth"))
    model_randa.load_state_dict(torch.load("model_randa.pth"))

    openai_api_key = "EMPTY"
    draft_client = OpenAI(
        api_key=openai_api_key,
        base_url=args.draft_model_ip_address,
    )
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_model_name_or_path, trust_remote_code=True)

    target_client = OpenAI(
        api_key=openai_api_key,
        base_url=args.target_model_ip_address,
    )
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model_name_or_path, trust_remote_code=True)

    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    slm_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    slm = AutoModelForCausalLM.from_pretrained(
        slm_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        low_cpu_mem_usage=True
    )
    slm_tokenizer = AutoTokenizer.from_pretrained(slm_name)

    LAMDA = 3
    await main_loop(
        scheduler_queue, llmprocessing_queue, draft_client, draft_tokenizer,
        target_client, target_tokenizer, slm, slm_tokenizer,
        model_pai, model_randa, model_delay, data_name, args, LAMDA
    )


async def main_loop(
    scheduler_queue, llmprocessing_queue, draft_client, draft_tokenizer,
    target_client, target_tokenizer, slm, slm_tokenizer,
    model_pai, model_randa, model_delay, data_name, args, LAMDA=3
):
    dataset_users, _ = gain_dataset(data_name, args, draft_tokenizer)
    total_dataset_length = len(dataset_users)

    set_global_seed(args.seed)

    remaining_indices = list(range(total_dataset_length))

    os.makedirs("myresults", exist_ok=True)

    current_queue_size = 0
    finished_indices = []
    accs = []
    total_nums = []
    avg_processing_delays = []
    avg_communication_delays = []
    avg_queue_delays = []

    all_tasks_injected = False

    while True:
        if not all_tasks_injected:
            if len(remaining_indices) > 0:
                num_tasks = np.random.poisson(LAMDA)
                num_tasks = min(num_tasks, len(remaining_indices))

                sampled_indices = remaining_indices[:num_tasks]
                remaining_indices = remaining_indices[num_tasks:]

                for idx_dataset in sampled_indices:
                    prompt, idx_test = generate_task_from_dataset(dataset_users, idx_dataset)

                    task_stats = {
                        "processing_delay": 0.0,
                        "communication_delay": 0.0,
                        "queue_delay": 0.0,
                        "queue_count": 0,
                        "step_queue_delay": 0.0,
                        "step_queue_count": 0,
                    }
                    current_prompt = [[current_queue_size, idx_test, prompt, [], task_stats]]
                    current_queue_size += 1

                    state = get_state_from_batch(
                        slm, slm_tokenizer, [prompt], slm_name="Qwen/Qwen2.5-Math-1.5B-Instruct"
                    ).to(next(model_pai.parameters()).device)

                    logits = model_pai(state)
                    probs = torch.softmax(logits, dim=-1)
                    action = torch.argmax(probs, dim=-1)

                    before_response = prompt
                    num_step = 1
                    num_unchanged = 0

                    if action.item() == 0:
                        asyncio.create_task(
                            slm_task_processing(
                                scheduler_queue, llmprocessing_queue, slm, slm_tokenizer,
                                before_response, current_prompt, draft_client, draft_tokenizer,
                                target_client, target_tokenizer, num_step,
                                finished_indices, num_unchanged,
                                model_pai, model_randa, model_delay, args
                            )
                        )
                    else:
                        num_token = len(slm_tokenizer.encode(prompt))
                        randa = sanitize_randa_outputs(model_randa(state))
                        r1 = randa[:, 0]
                        a1 = randa[:, 1]
                        delay = model_delay(
                            state,
                            torch.tensor(num_token, dtype=torch.float32).to(state.device)
                        )
                        task = [
                            current_prompt, r1, a1, delay, num_token,
                            probs[:, 0], num_step, before_response, num_unchanged
                        ]
                        await scheduler_queue.put(task)

            if len(remaining_indices) == 0:
                all_tasks_injected = True

        for i in range(len(llmprocessing_queue)):
            task = llmprocessing_queue[i]
            task[3] -= 0.001 * args.FlopsLLM * args.beta / args.M
            if task[3] < 0:
                task[3] = 0

        num_step = 0
        before_response = 0
        current_prompt = 0
        num_unchanged = 0

        await scheduler_processing(
            scheduler_queue, llmprocessing_queue, slm, slm_tokenizer,
            before_response, current_prompt, draft_client, draft_tokenizer,
            target_client, target_tokenizer, num_step, finished_indices,
            num_unchanged, model_pai, model_randa, model_delay, args
        )

        LLM_length = len(llmprocessing_queue)
        waiting_length = scheduler_queue.qsize()
        finished_length = len(finished_indices)
        correct_num = sum([1 for item in finished_indices if item["is_correct"]])
        acc = 0 if finished_length == 0 else correct_num / finished_length

        avg_processing_delay = 0.0 if finished_length == 0 else np.mean(
            [item["processing_delay"] for item in finished_indices]
        )
        avg_communication_delay = 0.0 if finished_length == 0 else np.mean(
            [item["communication_delay"] for item in finished_indices]
        )
        avg_queue_delay = 0.0 if finished_length == 0 else np.mean(
            [item["queue_delay"] for item in finished_indices]
        )

        accs.append(acc)
        total_nums.append(LLM_length + waiting_length)
        avg_processing_delays.append(avg_processing_delay)
        avg_communication_delays.append(avg_communication_delay)
        avg_queue_delays.append(avg_queue_delay)

        print(
            f"waiting_length={waiting_length}, "
            f"LLM_length={LLM_length}, "
            f"finished_length={finished_length}, "
            f"correct_num={correct_num}, "
            f"all_tasks_injected={all_tasks_injected}, "
            f"avg_processing_delay={avg_processing_delay:.6f}, "
            f"avg_communication_delay={avg_communication_delay:.6f}, "
            f"avg_queue_delay={avg_queue_delay:.6f}"
        )

        if all_tasks_injected and finished_length >= total_dataset_length:
            plt.figure(figsize=(12, 6))
            plt.plot(np.array(accs), label="accs", color="blue", marker="o", linewidth=2, markersize=4)
            plt.title("accs VS epoch")
            plt.xlabel("epoch")
            plt.ylabel("acc")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"myresults/{args.data_names}/accsVSepoch_M{args.M}_B{args.B}.png", dpi=300)
            plt.close()
            np.save(f"myresults/{args.data_names}/accs_M{args.M}_B{args.B}.npy", np.array(accs))

            plt.figure(figsize=(12, 6))
            plt.plot(np.array(avg_processing_delays), label="avg_processing_delay", color="blue", marker="o", linewidth=2, markersize=4)
            plt.title("avg_processing_delay VS epoch")
            plt.xlabel("epoch")
            plt.ylabel("avg_processing_delay")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"myresults/{args.data_names}/avg_processing_delay_M{args.M}_B{args.B}.png", dpi=300)
            plt.close()
            np.save(f"myresults/{args.data_names}/avg_processing_delay_M{args.M}_B{args.B}.npy", np.array(avg_processing_delays))

            plt.figure(figsize=(12, 6))
            plt.plot(np.array(avg_communication_delays), label="avg_communication_delay", color="blue", marker="o", linewidth=2, markersize=4)
            plt.title("avg_communication_delay VS epoch")
            plt.xlabel("epoch")
            plt.ylabel("avg_communication_delay")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"myresults/{args.data_names}/avg_communication_delay_M{args.M}_B{args.B}.png", dpi=300)
            plt.close()
            np.save(f"myresults/{args.data_names}/avg_communication_delay_M{args.M}_B{args.B}.npy", np.array(avg_communication_delays))

            plt.figure(figsize=(12, 6))
            plt.plot(np.array(avg_queue_delays), label="avg_queue_delay", color="blue", marker="o", linewidth=2, markersize=4)
            plt.title("avg_queue_delay VS epoch")
            plt.xlabel("epoch")
            plt.ylabel("avg_queue_delay")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"myresults/{args.data_names}/avg_queue_delay_M{args.M}_B{args.B}.png", dpi=300)
            plt.close()
            np.save(f"myresults/{args.data_names}/avg_queue_delay_M{args.M}_B{args.B}.npy", np.array(avg_queue_delays))

            with open(f"myresults/{args.data_names}/task_delay_details_M{args.M}_B{args.B}.json", "w", encoding="utf-8") as f:
                json.dump(finished_indices, f, ensure_ascii=False, indent=2)

            break

        await asyncio.sleep(0.001)


async def slm_task_processing(
    scheduler_queue, llmprocessing_queue, slm, slm_tokenizer, before_response, current_prompt,
    draft_client, draft_tokenizer, target_client, target_tokenizer, num_step,
    finished_indices, num_unchanged, model_pai, model_randa, model_delay, args
):
    draft_prompt = [p + "".join(r[0] for r in responses) for _, _, p, responses, _ in current_prompt]

    draft_batch_response = draft_client.completions.create(
        model=args.draft_model_name_or_path.split("/")[-1],
        prompt=draft_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens_per_call,
        stop=[args.step_word],
    ).choices
    draft_batch_response = sorted(draft_batch_response, key=lambda x: int(x.index))

    now_prompts = []
    for (orig_idx, idx_test, prompt, prev_responses, task_stats), new_response in zip(current_prompt, draft_batch_response):
        now_prompts.append((orig_idx, idx_test, prompt, prev_responses, task_stats, new_response))

    next_prompts = []
    for orig_idx, idx_test, prompt, prev_responses, task_stats, response in sorted(now_prompts, key=lambda x: x[0]):
        response_text = response.text + args.step_word

        context_text_before = prompt + "".join(r[0] for r in prev_responses)
        num_tokens_star = len(draft_tokenizer.encode(context_text_before))
        num_tokens_generate = len(draft_tokenizer.encode(response.text))
        task_stats["processing_delay"] += compute_processing_delay(
            num_tokens_star=num_tokens_star,
            num_tokens_generate=num_tokens_generate,
            model_type="SLM"
        )

        full_responses = prev_responses + [(response_text, 0)]
        full_responses_text = "".join(r[0] for r in full_responses)

        if (
            (response.stop_reason is None)
            or len(draft_tokenizer.encode(prompt + full_responses_text)) >= args.max_tokens_per_call
            or num_step >= args.max_steps - 1
            or num_unchanged >= args.patience - 1
        ):
            output = full_responses_text[:-len(args.step_word)].rstrip()
            is_correct = evaluate_single(output, prompt, idx_test, args.data_names, draft_tokenizer, args)
            finished_indices.append({
                "orig_idx": orig_idx,
                "is_correct": is_correct,
                "processing_delay": task_stats["processing_delay"],
                "communication_delay": task_stats["communication_delay"],
                "queue_delay": task_stats["queue_delay"],
                "queue_count": task_stats.get("queue_count", 0),
                "step_queue_count": task_stats.get("step_queue_count", 0),
            })
            return
        else:
            next_prompts.append((orig_idx, idx_test, prompt, full_responses, task_stats))

    current_prompts = next_prompts

    if draft_batch_response[0].text != before_response:
        num_unchanged = 0
        before_response = draft_batch_response[0].text
    elif draft_batch_response[0].text == before_response:
        num_unchanged += 1

    prompts = [p + "".join(r[0] for r in responses) for _, _, p, responses, _ in current_prompts]
    state = get_state_from_batch(slm, slm_tokenizer, prompts, slm_name="Qwen/Qwen2.5-Math-1.5B-Instruct").to(next(model_pai.parameters()).device)
    logits = model_pai(state)
    probs = torch.softmax(logits, dim=-1)
    action = torch.argmax(probs, dim=-1)

    if action.item() == 0:
        await slm_task_processing(
            scheduler_queue, llmprocessing_queue, slm, slm_tokenizer,
            before_response, current_prompts, draft_client, draft_tokenizer,
            target_client, target_tokenizer, num_step + 1,
            finished_indices, num_unchanged, model_pai, model_randa, model_delay, args
        )
    else:
        num_token = len(slm_tokenizer.encode(prompts[0]))
        randa = sanitize_randa_outputs(model_randa(state))
        r1 = randa[:, 0]
        a1 = randa[:, 1]
        delay = model_delay(state, torch.tensor(num_token, dtype=torch.float32).to(state.device))
        task = [current_prompts, r1, a1, delay, num_token, probs[:, 0], num_step, before_response, num_unchanged]
        await scheduler_queue.put(task)


async def llm_task_processing(
    llm_queue_lock, this_task, scheduler_queue, llmprocessing_queue, slm, slm_tokenizer, before_response, current_prompt,
    draft_client, draft_tokenizer, target_client, target_tokenizer, num_step,
    finished_indices, num_unchanged, model_pai, model_randa, model_delay, args
):
    target_prompt = [p + "".join(r[0] for r in responses) for _, _, p, responses, _ in current_prompt]

    target_batch_response = target_client.completions.create(
        model=args.target_model_name_or_path.split("/")[-1],
        prompt=target_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens_per_call,
        stop=[args.step_word],
    ).choices
    target_batch_response = sorted(target_batch_response, key=lambda x: int(x.index))

    now_prompts = []
    for (orig_idx, idx_test, prompt, prev_responses, task_stats), new_response in zip(current_prompt, target_batch_response):
        now_prompts.append((orig_idx, idx_test, prompt, prev_responses, task_stats, new_response))

    next_prompts = []
    for orig_idx, idx_test, prompt, prev_responses, task_stats, response in sorted(now_prompts, key=lambda x: x[0]):
        response_text = response.text + args.step_word

        context_text_before = prompt + "".join(r[0] for r in prev_responses)
        num_tokens_star = len(target_tokenizer.encode(context_text_before))
        num_tokens_generate = len(target_tokenizer.encode(response.text))
        task_stats["processing_delay"] += compute_processing_delay(
            num_tokens_star=num_tokens_star,
            num_tokens_generate=num_tokens_generate,
            model_type="LLM"
        )

        full_responses = prev_responses + [(response_text, 0)]
        full_responses_text = "".join(r[0] for r in full_responses)

        if (
            (response.stop_reason is None)
            or len(target_tokenizer.encode(prompt + full_responses_text)) >= args.max_tokens_per_call
            or num_step >= args.max_steps - 1
            or num_unchanged >= args.patience - 1
        ):
            output = full_responses_text[:-len(args.step_word)].rstrip()
            is_correct = evaluate_single(output, prompt, idx_test, args.data_names, draft_tokenizer, args)
            finished_indices.append({
                "orig_idx": orig_idx,
                "is_correct": is_correct,
                "processing_delay": task_stats["processing_delay"],
                "communication_delay": task_stats["communication_delay"],
                "queue_delay": task_stats["queue_delay"],
                "queue_count": task_stats.get("queue_count", 0),
                "step_queue_count": task_stats.get("step_queue_count", 0),
            })
            async with llm_queue_lock:
                for i, task in enumerate(llmprocessing_queue):
                    if task[0] == this_task[0]:
                        del llmprocessing_queue[i]
                        break
            return
        else:
            next_prompts.append((orig_idx, idx_test, prompt, full_responses, task_stats))

    current_prompts = next_prompts

    if target_batch_response[0].text != before_response:
        num_unchanged = 0
        before_response = target_batch_response[0].text
    elif target_batch_response[0].text == before_response:
        num_unchanged += 1

    async with llm_queue_lock:
        for i, task in enumerate(llmprocessing_queue):
            if task[0] == this_task[0]:
                del llmprocessing_queue[i]
                break

    prompts = [p + "".join(r[0] for r in responses) for _, _, p, responses, _ in current_prompts]
    numtoken = len(draft_tokenizer.encode(prompts[0]))
    state = get_state_from_batch(slm, slm_tokenizer, prompts, slm_name="Qwen/Qwen2.5-Math-1.5B-Instruct").to(next(model_pai.parameters()).device)
    logits = model_pai(state)
    probs = torch.softmax(logits, dim=-1)
    action = torch.argmax(probs, dim=-1)

    if action.item() == 0:
        await slm_task_processing(
            scheduler_queue, llmprocessing_queue, slm, slm_tokenizer,
            before_response, current_prompts, draft_client, draft_tokenizer,
            target_client, target_tokenizer, num_step + 1,
            finished_indices, num_unchanged, model_pai, model_randa, model_delay, args
        )
    else:
        randa = sanitize_randa_outputs(model_randa(state))
        r1 = randa[:, 0]
        a1 = randa[:, 1]
        delay = model_delay(state, torch.tensor(numtoken, dtype=torch.float32).to(state.device))
        task = [current_prompts, r1, a1, delay, numtoken, probs[:, 0], num_step, before_response, num_unchanged]
        await scheduler_queue.put(task)


async def scheduler_processing(
    scheduler_queue, llmprocessing_queue, slm, slm_tokenizer, before_response, current_prompt,
    draft_client, draft_tokenizer, target_client, target_tokenizer, num_step,
    finished_indices, num_unchanged, model_pai, model_randa, model_delay, args
):
    llm_queue_lock = asyncio.Lock()
    LLM_length = len(llmprocessing_queue)
    beta = args.beta
    B = args.B
    M = args.M
    FlopsLLM = args.FlopsLLM
    max_step_queue_waits = 3
    slot_queue_penalty = 0.001 * args.FlopsLLM / args.M

    current_prompts = []
    r1s = []
    a1s = []
    delays = []
    numtokens = []
    p0s = []
    snrs = []
    num_steps = []
    before_responses = []
    num_unchangeds = []

    snr_range = range(0, 41)
    snr_th = 10

    async def requeue_or_force_local(task):
        current_prompt, r1, a1, delay, numtoken, p0, num_step, before_response, num_unchanged = task
        task_stats = current_prompt[0][4]
        if task_stats.get("step_queue_count", 0) >= max_step_queue_waits:
            task_stats["step_queue_count"] = 0
            task_stats["step_queue_delay"] = 0.0
            asyncio.create_task(
                slm_task_processing(
                    scheduler_queue, llmprocessing_queue, slm, slm_tokenizer,
                    before_response, current_prompt, draft_client, draft_tokenizer,
                    target_client, target_tokenizer, num_step + 1,
                    finished_indices, num_unchanged, model_pai, model_randa, model_delay, args
                )
            )
            return

        task_stats["queue_count"] = task_stats.get("queue_count", 0) + 1
        task_stats["queue_delay"] += slot_queue_penalty
        task_stats["step_queue_count"] = task_stats.get("step_queue_count", 0) + 1
        task_stats["step_queue_delay"] = task_stats.get("step_queue_delay", 0.0) + slot_queue_penalty
        await scheduler_queue.put(task)

    delays_llm = []
    for i in range(LLM_length):
        task = llmprocessing_queue[i]
        delays_llm.append(float(task[3]))

    while not scheduler_queue.empty():
        task = await scheduler_queue.get()
        sampled_snr = random.choices(snr_range, k=1)

        if int(sampled_snr[0]) < snr_th:
            await requeue_or_force_local(task)
        else:
            sampled_snr[0] = 10 ** (sampled_snr[0] / 10)
            current_prompt, r1, a1, delay, numtoken, p0, num_step, before_response, num_unchanged = task

            current_prompts.append(current_prompt)
            r1s.append(to_float(r1))
            a1s.append(to_float(a1))
            delays.append(to_float(delay)) 
            p0s.append(to_float(p0))
            numtokens.append(float(numtoken))
            snrs.append(float(sampled_snr[0]))
            num_steps.append(num_step)
            before_responses.append(before_response)
            num_unchangeds.append(num_unchanged)

    if len(current_prompts) == 0:
        return

    numtokens_np = np.array(numtokens, dtype=np.float64)
    snrs_np = np.array(snrs, dtype=np.float64)
    delays_np = np.array(delays, dtype=np.float64)
    a1s_np = np.array(a1s, dtype=np.float64)
    p0s_np = np.array(p0s, dtype=np.float64)
    beta_eff = beta

    def compute_values(mu_value: float):
        b_val = np.sqrt(
            (32.0 * FlopsLLM * beta_eff * numtokens_np) /
            (max(mu_value, 1e-12) * M * np.log2(snrs_np + 1.0))
        )
        c_com_val = (
            32.0 * FlopsLLM * beta_eff * numtokens_np /
            (M * b_val * np.log2(snrs_np + 1.0))
        )
        c_com_ref = (
            32.0 * FlopsLLM * beta_eff * numtokens_np /
            (M * B * np.log2(snrs_np + 1.0))
        )
        w0_val = a1s_np / p0s_np - c_com_ref - mu_value * B
        w_val = a1s_np / p0s_np - c_com_val - mu_value * b_val
        overline_w_val = delays_np - mu_value * b_val
        return b_val, c_com_val, w0_val, w_val, overline_w_val

    if M == LLM_length:
        mu = 0.0
        max_inner_iterations = 100

        b, c_com, w0, w, overline_w = compute_values(mu)

        queue_indices = []
        slm_indices = []

        for i in range(len(current_prompts)):
            if w0[i] >= 0.0:
                queue_indices.append(i) 
            else:
                slm_indices.append(i)  

        for i in queue_indices:
            task = [
                current_prompts[i], r1s[i], a1s[i], delays[i], numtokens[i], p0s[i],
                num_steps[i], before_responses[i], num_unchangeds[i]
            ]
            await requeue_or_force_local(task)

        for i in slm_indices:
            task_stats = current_prompts[i][0][4]
            task_stats["step_queue_count"] = 0
            task_stats["step_queue_delay"] = 0.0
            asyncio.create_task(
                slm_task_processing(
                    scheduler_queue, llmprocessing_queue, slm, slm_tokenizer,
                    before_responses[i], current_prompts[i], draft_client, draft_tokenizer,
                    target_client, target_tokenizer, num_steps[i] + 1,
                    finished_indices, num_unchangeds[i], model_pai, model_randa, model_delay, args
                )
            )
        return

    else:
        capacity = M - LLM_length
        snr_mean = max(np.mean(snrs_np), 1e-30)
        token_mean = max(np.mean(numtokens_np), 1e-30)

        max_inner_iterations = 100

        final_A = []
        final_Q = []
        final_S = []
        final_b = np.zeros(len(current_prompts), dtype=np.float64)
        final_priority = np.zeros(len(current_prompts), dtype=np.float64)

        def solve_optimal_bandwidth_for_A(admitted_indices):
            optimal_b = np.zeros(len(current_prompts), dtype=np.float64)
            if len(admitted_indices) == 0:
                return 0.0, optimal_b

            coeff = (
                32.0 * FlopsLLM * beta_eff * numtokens_np[admitted_indices] /
                (M * np.log2(snrs_np[admitted_indices] + 1.0))
            )
            coeff = np.maximum(coeff, 1e-30)
            mu_opt = (float(np.sum(np.sqrt(coeff))) / max(B, 1e-30)) ** 2
            mu_opt = max(mu_opt, 1e-30)
            optimal_b[admitted_indices] = np.sqrt(coeff / mu_opt)

            total_bandwidth = float(np.sum(optimal_b[admitted_indices]))
            if total_bandwidth > 0.0:
                optimal_b[admitted_indices] *= B / total_bandwidth

            return mu_opt, optimal_b

        def build_schedule_for_mu(mu_value, admission_limit):
            b, c_com, w0, w, overline_w = compute_values(mu_value)
            w_effective = w

            if len(current_prompts) <= capacity:
                priority = np.array(w_effective, dtype=np.float64)
                candidates = [i for i in range(len(current_prompts)) if w_effective[i] >= 0.0]
                A = sorted(candidates, key=lambda x: priority[x], reverse=True)[:admission_limit]
                Q = []
                S = [i for i in range(len(current_prompts)) if i not in A]
                overline_effective = overline_w
            else:
                overline_w = -mu_value * b
                overline_effective = overline_w
                priority = np.minimum(w_effective, overline_effective)
                order_indices = list(np.argsort(-priority))

                for _inner in range(max_inner_iterations):
                    server_times = sorted(delays_llm) + [0.0] * max(M - len(delays_llm), 0)
                    server_times = server_times[:M]
                    tw_new = np.zeros(len(current_prompts), dtype=np.float64)

                    for idx in order_indices:
                        j = int(np.argmin(server_times))
                        tw_new[idx] = server_times[j]
                        server_times[j] = server_times[j] + delays_np[idx]

                    overline_w_new = tw_new - mu_value * b
                    overline_effective_new = overline_w_new
                    priority_new = np.minimum(w_effective, overline_effective_new)
                    new_order_indices = list(np.argsort(-priority_new))

                    overline_w = overline_w_new
                    overline_effective = overline_effective_new
                    priority = priority_new

                    if new_order_indices == order_indices:
                        break
                    order_indices = new_order_indices

                candidates = [i for i in range(len(current_prompts)) if min(w_effective[i], overline_effective[i]) >= 0.0]
                candidates = sorted(candidates, key=lambda x: priority[x], reverse=True)

                A = candidates[:admission_limit]
                if len(A) >= admission_limit and len(A) >= capacity and admission_limit > 0:
                    lambda_s = min(priority[i] for i in A)
                else:
                    lambda_s = 0.0
                Q = [
                    i for i in range(len(current_prompts))
                    if i not in A and overline_effective[i] < lambda_s and w_effective[i] >= overline_effective[i]
                ]
                S = [i for i in range(len(current_prompts)) if i not in A and i not in Q]

            bandwidth_sum = float(np.sum(b[A])) if len(A) > 0 else 0.0
            if len(A) == 0 or bandwidth_sum <= 0.0 or bandwidth_sum > 2.0 * B:
                return None

            value_scale = max(
                float(np.percentile(np.abs(np.concatenate([w_effective, overline_effective])), 75)),
                1.0,
            )
            immediate_value = float(np.sum(priority[A])) / value_scale
            bandwidth_gap = abs(bandwidth_sum - B) / max(B, 1e-30)
            score = immediate_value - 0.1 * bandwidth_gap

            return {
                "score": score,
                "A": A,
                "Q": Q,
                "S": S,
                "b": b.copy(),
                "priority": priority.copy(),
                "mu": mu_value,
                "admission_count": len(A),
                "bandwidth_gap": bandwidth_gap,
            }

        def build_no_competition_schedule_for_mu(mu_value):
            b, c_com, w0, w, overline_w = compute_values(mu_value)
            w_effective = w
            priority = np.array(w_effective, dtype=np.float64)
            positive_candidates = [i for i in range(len(current_prompts)) if w_effective[i] >= 0.0]
            A = sorted(positive_candidates, key=lambda x: priority[x], reverse=True)[:capacity]
            Q = []
            S = [i for i in range(len(current_prompts)) if i not in A]

            bandwidth_sum = float(np.sum(b[A])) if len(A) > 0 else 0.0
            if len(A) == 0 or bandwidth_sum <= 0.0 or bandwidth_sum > 2.0 * B:
                return None

            value_scale = max(
                float(np.percentile(np.abs(w_effective), 75)),
                1.0,
            )
            immediate_value = float(np.sum(priority[A])) / value_scale
            bandwidth_gap = abs(bandwidth_sum - B) / max(B, 1e-30)
            score = immediate_value - 0.1 * bandwidth_gap

            return {
                "score": score,
                "A": A,
                "Q": Q,
                "S": S,
                "b": b.copy(),
                "priority": priority.copy(),
                "mu": mu_value,
                "admission_count": len(A),
                "bandwidth_gap": bandwidth_gap,
            }

        def search_best_schedule():
            max_admission_limit = min(capacity, len(current_prompts))
            if max_admission_limit <= 0:
                return None

            def evaluate_log_mu_grid(log_mu_values, builder, current_best):
                stage_best = None
                best_local = current_best
                for log_mu in log_mu_values:
                    solution = builder(10.0 ** float(log_mu))
                    if solution is None:
                        continue
                    if stage_best is None or solution["score"] > stage_best["score"]:
                        stage_best = solution
                    if best_local is None or solution["score"] > best_local["score"]:
                        best_local = solution
                return stage_best, best_local

            stage_grid_points = [41, 41, 31, 31, 21]

            def run_grid_search(builder, admission_scale):
                best_solution = None
                base_mu = max(
                    (admission_scale ** 2) * 32.0 * FlopsLLM * beta_eff * token_mean /
                    (B ** 2 * M * np.log2(snr_mean + 1.0)),
                    1e-30,
                )
                center = math.log10(base_mu)
                half_width = 8.0

                for num_points in stage_grid_points:
                    stage_best, best_solution = evaluate_log_mu_grid(
                        np.linspace(center - half_width, center + half_width, num_points),
                        builder,
                        best_solution,
                    )
                    if stage_best is None:
                        break
                    center = math.log10(stage_best["mu"])
                    half_width = max(half_width / 3.0, 0.005)

                if best_solution is not None:
                    dense_center = math.log10(best_solution["mu"])
                    dense_half_width = 0.02
                    _, best_solution = evaluate_log_mu_grid(
                        np.linspace(dense_center - dense_half_width, dense_center + dense_half_width, 81),
                        builder,
                        best_solution,
                    )
                return best_solution

            for admission_limit in range(max_admission_limit, 0, -1):
                def competitive_builder(mu_value, target_limit=admission_limit):
                    solution = build_schedule_for_mu(mu_value, target_limit)
                    if solution is None or solution["admission_count"] != target_limit:
                        return None
                    return solution

                best_solution = run_grid_search(competitive_builder, admission_limit)
                if best_solution is not None:
                    return best_solution

            return run_grid_search(build_no_competition_schedule_for_mu, max(max_admission_limit, 1))

        best_solution = search_best_schedule()
        if best_solution is not None:
            final_A = best_solution["A"]
            final_Q = best_solution["Q"]
            final_S = best_solution["S"]
            final_b = best_solution["b"]
            final_priority = best_solution["priority"]

        if len(final_A) < capacity and len(final_Q) > 0:
            fill_slots = capacity - len(final_A)
            promoted_from_q = sorted(final_Q, key=lambda x: final_priority[x], reverse=True)[:fill_slots]
            if len(promoted_from_q) > 0:
                promoted_set = set(promoted_from_q)
                final_A = final_A + promoted_from_q
                final_Q = [i for i in final_Q if i not in promoted_set]

        if len(final_A) == 0:
            final_Q = []
            final_S = list(range(len(current_prompts)))

        bounded_final_Q = []
        overflow_to_S = []
        for i in final_Q:
            if current_prompts[i][0][4].get("step_queue_count", 0) >= max_step_queue_waits:
                overflow_to_S.append(i)
            else:
                bounded_final_Q.append(i)
        final_Q = bounded_final_Q
        final_S.extend(overflow_to_S)

        if len(final_A) > 0:
            _, final_b = solve_optimal_bandwidth_for_A(final_A)
        else:
            final_b = np.zeros(len(current_prompts), dtype=np.float64)

        bandwidth_sum = float(np.sum(final_b[final_A])) if len(final_A) > 0 else 0.0
        print(f"used_b={bandwidth_sum}, B={B}, ratio={bandwidth_sum / max(B, 1e-30)},lengthA={len(final_A)},lengthQ={len(final_Q)},lengthS={len(final_S)}")

        for i in final_A:
            task_stats = current_prompts[i][0][4]
            prompt_text = current_prompts[i][0][2]
            response_list = current_prompts[i][0][3]
            num_tokens_star = len(draft_tokenizer.encode(prompt_text + "".join(r[0] for r in response_list)))
            task_stats["step_queue_count"] = 0
            task_stats["step_queue_delay"] = 0.0

            task_stats["communication_delay"] += compute_communication_delay(
                num_tokens_star=num_tokens_star,
                snr=snrs[i],
                bandwidth=final_b[i],
                flops_llm=args.FlopsLLM,
                beta=args.beta,
                M=args.M,
            )

            task = [
                current_prompts[i], r1s[i], a1s[i], delays[i], numtokens[i], p0s[i],
                num_steps[i], before_responses[i], num_unchangeds[i]
            ]

            async with llm_queue_lock:
                llmprocessing_queue.append(task)

            asyncio.create_task(
                llm_task_processing(
                    llm_queue_lock, task, scheduler_queue, llmprocessing_queue, slm, slm_tokenizer,
                    before_responses[i], current_prompts[i], draft_client, draft_tokenizer,
                    target_client, target_tokenizer, num_steps[i] + 1,
                    finished_indices, num_unchangeds[i], model_pai, model_randa, model_delay, args
                )
            )

        for i in final_Q:
            task = [
                current_prompts[i], r1s[i], a1s[i], delays[i], numtokens[i], p0s[i],
                num_steps[i], before_responses[i], num_unchangeds[i]
            ]
            await requeue_or_force_local(task)

        for i in final_S:
            task_stats = current_prompts[i][0][4]
            task_stats["step_queue_count"] = 0
            task_stats["step_queue_delay"] = 0.0
            asyncio.create_task(
                slm_task_processing(
                    scheduler_queue, llmprocessing_queue, slm, slm_tokenizer,
                    before_responses[i], current_prompts[i], draft_client, draft_tokenizer,
                    target_client, target_tokenizer, num_steps[i] + 1,
                    finished_indices, num_unchangeds[i], model_pai, model_randa, model_delay, args
                )
            )

asyncio.run(main())
