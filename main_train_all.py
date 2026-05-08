import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
from datetime import datetime
from tqdm import tqdm
import copy

from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

from external.qwen25_math_evaluation.evaluate import evaluate
from external.qwen25_math_evaluation.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from external.qwen25_math_evaluation.parser import *
from external.qwen25_math_evaluation.trajectory import *
from external.qwen25_math_evaluation.data_loader import load_data
from external.qwen25_math_evaluation.python_executor import PythonExecutor
from external.skywork_o1_prm_inference.model_utils.io_utils import prepare_input, derive_step_rewards_vllm


# 策略+价值网络（共享特征层）
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
        )  # 策略头
    def forward(self, x):
        logits = self.policy_head(x)
        return logits
    
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
        )  # 价值头

    def forward(self, x):
        value = self.value_head(x)
        return value

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
        )  # 延迟头
    def forward(self, x, token_num):
        film_params = self.film(token_num.unsqueeze(-1))  # [B, state_dim*2]
        gamma, beta = film_params.chunk(2, dim=1)  # [B, state_dim], [B, state_dim]
        x = x * gamma + beta
        delay = self.delay_head(x)
        return delay
    
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
        )  # 奖励头
    def forward(self, x):
        RandA = self.reward_head(x)
        return RandA

def get_state_from_batch(slm, slm_tokenizer, prompts, slm_name="Qwen/Qwen2.5-Math-1.5B-Instruct"):
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    if slm_tokenizer.pad_token is None:
        slm_tokenizer.pad_token = slm_tokenizer.eos_token  # 用 eos_token 作为 pad_token

    batch_size=1

    # 存储所有批次的结果
    all_results = []
    
    # 将prompts拆分为多个批次，每批最多batch_size个（这里固定为5）
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]  # 取当前批次的5个prompt（最后一批可能不足5个）
        
        # 对当前批次进行处理
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
        
        # 提取最后一层的hidden state，并取每个序列的最后一个有效token
        hidden_states = outputs.hidden_states[-1]  # shape: [batch_size_current, seq_len, hidden_dim]
        batch_size_current = hidden_states.shape[0]
        last_valid_indices = inputs["attention_mask"].sum(dim=1) - 1  # 每个序列的最后有效token索引
        batch_indices = torch.arange(batch_size_current, device=device)
        last_hidden_states = hidden_states[batch_indices, last_valid_indices, :]  # 提取最后有效token的hidden state
        
        # 将当前批次结果存入总列表
        all_results.append(last_hidden_states.float().cpu())
        
        # 清理当前批次的中间变量和缓存，减少内存占用
        del inputs, outputs, hidden_states, last_hidden_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 将所有批次的结果拼接成一个张量（shape: [total_prompts, hidden_dim]）
    result = torch.cat(all_results, dim=0)
    return result 

def train_gainSAR(h, slm, slm_tokenizer, optimizer_pai, optimizer_v, model_pai, model_v, target_model, args, draft_client, target_client, prm_client, draft_tokenizer, target_tokenizer, prm_tokenizer,
          prompts, problems, dmodelS=1536, dmodelL=3584):
    """
    优化后的训练循环：
    - 将每一步的 values / log_probs / next_values 对齐到全局索引（len(prompts)）
    - 使用 target_model 作为 bootstrapping 的价值来源
    - 向量化处理，避免 shape mismatch
    """

    criterion = nn.MSELoss()
    N = len(prompts)  # 全部样本数（固定）
    outputs = [None] * N
    token_counts = [(0, 0, 0) for _ in range(N)]
    step_info = [[] for _ in range(N)]
    current_prompts = [(i, p, []) for i, p in enumerate(prompts)]
    all_rewards = [[] for _ in range(N)]
    all_actions = [[] for _ in range(N)]
    all_states = [[] for _ in range(N)]
    all_values = [[] for _ in range(N)]
    all_values_next = [[] for _ in range(N)]
    all_logits = [[] for _ in range(N)]
    all_d1s = [[] for _ in range(N)]
    all_numtokens = [[] for _ in range(N)]
    current_problems = problems
    num_step = 0
    pre_num_finished = 0
    num_unchanged = 0

    device = next(model_pai.parameters()).device if any(p.requires_grad for p in model_pai.parameters()) else torch.device("cpu")

    while current_prompts:
        # ------- 1) 保存 prev_indices（对应下面得到的 values） -------
        prev_indices = [idx for idx, _, _ in current_prompts]  # 全局索引列表（按 batch 顺序）
        batch_prompts = [p + ''.join(r[0] for r in responses) for _, p, responses in current_prompts]
        full_responses = [''.join(r[0] for r in prev_resp) 
                          for (_, _, prev_resp) in current_prompts]
        processed_data = [
            prepare_input(p, full_resp, tokenizer=prm_tokenizer, step_token=args.step_word)
            for p, full_resp in zip(current_problems, full_responses)
        ]
        input_ids, steps, reward_flags = zip(*processed_data)
        max_embedding_tokens = 4096
        truncated_input_ids = [seq[:max_embedding_tokens] for seq in input_ids]
        rewards = prm_client.embeddings.create(
            input=truncated_input_ids,
            model=args.prm_name_or_path.split("/")[-1],
        )
        step_rewards_before = derive_step_rewards_vllm(rewards, reward_flags)
        # 防止空
        for r in step_rewards_before:
            if r == []:
                r.append(0.0)
        

        # 获取 state -> logits, values（对应 prev_indices 顺序）
        batch_states = get_state_from_batch(slm,slm_tokenizer,batch_prompts, slm_name=args.draft_model_name_or_path).to(device)  #[num_prompts, hidden_dim]
        logits = model_pai(batch_states)  # values: [len(prev_indices), 1] or [len(prev_indices),]
        values = model_v(batch_states)
        # 确保 values 为 [B,1]
        if values.dim() == 1:
            values = values.unsqueeze(1)
        values = values.to(device)

        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        # 这里按原代码使用 argmax 决策（actor 输出 deterministic action）
        # actions = torch.argmax(probs, dim=-1).to(device)  # shape: [B]
        actions = dist.sample()  #shape: [B]

        # 分组索引（基于 batch 内部位置）
        draft_pos = [i for i, a in enumerate(actions.tolist()) if a == 0]
        target_pos = [i for i, a in enumerate(actions.tolist()) if a == 1]

        # === 批量调用两个模型（按 batch 内位置），然后放回 new_responses（按 batch 内位置） ===
        new_responses_batch = [None] * len(batch_prompts)  # 临时按 batch 内位置存放模型返回
        # draft
        if draft_pos:
            draft_prompts = [batch_prompts[i] for i in draft_pos]
            draft_batch_responses = draft_client.completions.create(
                model=args.draft_model_name_or_path.split("/")[-1],
                prompt=draft_prompts,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens_per_call,
                stop=[args.step_word],
            ).choices
            draft_batch_responses = sorted(draft_batch_responses, key=lambda x: int(x.index))
            for p_idx, resp in zip(draft_pos, draft_batch_responses):
                new_responses_batch[p_idx] = resp
        # target
        if target_pos:
            target_prompts = [batch_prompts[i] for i in target_pos]
            target_batch_responses = target_client.completions.create(
                model=args.target_model_name_or_path.split("/")[-1],
                prompt=target_prompts,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens_per_call,
                stop=[args.step_word],
            ).choices
            target_batch_responses = sorted(target_batch_responses, key=lambda x: int(x.index))
            for p_idx, resp in zip(target_pos, target_batch_responses):
                new_responses_batch[p_idx] = resp

        # ===== 评估新响应（用 PRM）并记录 reward =====
        # full_responses 对应 prev_indices 顺序
        full_responses = [''.join(r[0] for r in prev_resp) + new_resp.text
                          for (_, _, prev_resp), new_resp in zip(current_prompts, new_responses_batch)]
        processed_data = [
            prepare_input(p, full_resp, tokenizer=prm_tokenizer, step_token=args.step_word)
            for p, full_resp in zip(current_problems, full_responses)
        ]
        input_ids, steps, reward_flags = zip(*processed_data)
        max_embedding_tokens = 4096
        truncated_input_ids = [seq[:max_embedding_tokens] for seq in input_ids]
        rewards = prm_client.embeddings.create(
            input=truncated_input_ids,
            model=args.prm_name_or_path.split("/")[-1],
        )
        step_rewards = derive_step_rewards_vllm(rewards, reward_flags)
        # 防止空
        for r in step_rewards:
            if r == []:
                r.append(0.0)
        # === 对 target 模型（action==1）的奖励减 delay_norm ===
        k = args.beta
        for i, r in enumerate(step_rewards):
            if actions[i].item() == 1:  # 1表示使用target model
                tokenizer = target_tokenizer
                num_tokens_star = len(tokenizer.encode(batch_prompts[i]))
                num_tokens_generate = len(tokenizer.encode(new_responses_batch[i].text))
                delayS = num_tokens_generate*dmodelS**2 + dmodelS*(2*num_tokens_star+num_tokens_generate-1)*num_tokens_generate/2
                delayL = num_tokens_generate*dmodelL**2 + dmodelL*(2*num_tokens_star+num_tokens_generate-1)*num_tokens_generate/2
                delay = k * delayL
                r[-1] = float(r[-1]) - k * delayL
            else:
                tokenizer = draft_tokenizer
                num_tokens_star = len(tokenizer.encode(batch_prompts[i]))
                num_tokens_generate = len(tokenizer.encode(new_responses_batch[i].text))
                delayS = num_tokens_generate*dmodelS**2 + dmodelS*(2*num_tokens_star+num_tokens_generate-1)*num_tokens_generate/2
                delayL = num_tokens_generate*dmodelL**2 + dmodelL*(2*num_tokens_star+num_tokens_generate-1)*num_tokens_generate/2
                delay = k * delayS
                r[-1] = float(r[-1]) - k * delayS

            global_idx = prev_indices[i]
            all_d1s[global_idx].append(delay)
            all_numtokens[global_idx].append(num_tokens_star)

        # 将 step reward 写回全局 all_rewards（按全局索引 prev_indices）
        now_prompts = []
        for (orig_idx, prompt, prev_responses), new_response, step_reward, step_reward_before in zip(current_prompts, new_responses_batch, step_rewards, step_rewards_before):
            all_rewards[orig_idx].append(step_reward[-1]-step_reward_before[-1])
            all_actions[orig_idx].append(actions[prev_indices.index(orig_idx)].item())
            all_states[orig_idx].append(batch_states[prev_indices.index(orig_idx)])
            all_values[orig_idx].append(values[prev_indices.index(orig_idx)])
            all_logits[orig_idx].append(logits[prev_indices.index(orig_idx)])
            now_prompts.append((orig_idx, prompt, prev_responses, new_response, actions[prev_indices.index(orig_idx)].item() == 0))

        # ===== 根据 terminate 条件，构建 next_prompts（全局索引） =====
        next_prompts = []
        next_problems = []
        v_prompts = []
        for orig_idx, prompt, prev_responses, response, used_draft in sorted(now_prompts, key=lambda x: x[0]):
            response_text = response.text + args.step_word
            client_id = 1 if used_draft else 2
            tokenizer = draft_tokenizer if client_id == 1 else target_tokenizer
            num_tokens = len(tokenizer.encode(response_text))

            # Update token counts
            if client_id == 1:
                token_counts[orig_idx] = (token_counts[orig_idx][0] + num_tokens,
                                          token_counts[orig_idx][1],
                                          token_counts[orig_idx][2])
            else:
                token_counts[orig_idx] = (token_counts[orig_idx][0],
                                          token_counts[orig_idx][1] + num_tokens,
                                          token_counts[orig_idx][2])

            step_info[orig_idx].append((num_step, client_id))

            full_responses = prev_responses + [(response_text, client_id)]
            full_responses_text = ''.join(r[0] for r in full_responses)
            v_prompts.append((orig_idx, prompt, full_responses))

            # terminate conditions (按照你原逻辑)
            if (response.stop_reason is None) \
                    or len(draft_tokenizer.encode(prompt + full_responses_text)) >= args.max_tokens_per_call \
                    or len(target_tokenizer.encode(prompt + full_responses_text)) >= args.max_tokens_per_call \
                    or num_step >= args.max_steps - 1 \
                    or num_unchanged >= args.patience - 1:
                outputs[orig_idx] = full_responses_text[:-len(args.step_word)]
            else:
                next_prompts.append((orig_idx, prompt, full_responses))
                next_problems.append(problems[orig_idx])

        # 更新 current_prompts 到 next_prompts（用于下一轮）
        current_prompts = next_prompts
        current_problems = next_problems

        batch_prompts_next = [p + ''.join(r[0] for r in responses) for _, p, responses in v_prompts]
        # 获取 state -> logits, values（对应 prev_indices 顺序）
        batch_states_next = get_state_from_batch(slm,slm_tokenizer,batch_prompts_next, slm_name=args.draft_model_name_or_path).to(device)  #[num_prompts, hidden_dim]
        values_next = model_v(batch_states_next)
         # 确保 values 为 [B,1]
        if values_next.dim() == 1:
            values_next = values_next.unsqueeze(1)
        values_next = values_next.to(device)
        for orig_idx, _, _ in v_prompts:
            all_values_next[orig_idx].append(values_next[prev_indices.index(orig_idx)])

        # update termination bookkeeping
        if len(outputs) - len(current_prompts) > pre_num_finished:
            num_unchanged = 0
            pre_num_finished = len(outputs) - len(current_prompts)
        else:
            num_unchanged += 1
        print(f"#### Step {num_step}: Completed {pre_num_finished} / {len(outputs)}, #unchanged {num_unchanged} / {args.patience}")
        num_step += 1

    return outputs, token_counts, step_info, all_rewards, all_states, all_actions, all_values, all_logits, all_d1s, all_numtokens, all_values_next

def find_simpletasks(args, draft_client, target_client, prm_client, draft_tokenizer, target_tokenizer, prm_tokenizer, prompts, problems):
    outputs = [None] * len(prompts)  # Initialize with None for tracking
    token_counts = [(0, 0, 0) for _ in prompts]  # (draft_tokens, target_tokens, discarded_draft_tokens) for each prompt
    step_info = [[] for _ in prompts]  # List to store (step_num, client_id) for each prompt
    current_prompts = [(i, p, []) for i, p in enumerate(prompts)] # (index, prompt, responses)
    all_rewards = [[] for _ in prompts]  # List to store (step_num, client_id) for each prompt
    current_problems = problems
    num_step = 0
    pre_num_finished = 0
    num_unchanged = 0
   
    while current_prompts:
        batch_prompts = [p + ''.join(r[0] for r in responses) for _, p, responses in current_prompts]

        # Firstly generate with the draft model
        draft_responses = draft_client.completions.create(
            model=args.draft_model_name_or_path.split("/")[-1],
            prompt=batch_prompts,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens_per_call,
            stop=[args.step_word],
        ).choices

        draft_responses = sorted(draft_responses, key=lambda x: int(x.index))

        # Evaluate draft responses with PRM
        full_responses = [''.join(r[0] for r in prev_resp) + new_resp.text
                    for (_, _, prev_resp), new_resp in zip(current_prompts, draft_responses)]
        processed_data = [
            prepare_input(p, full_resp, tokenizer=prm_tokenizer, step_token=args.step_word) 
            for p, full_resp in zip(current_problems, full_responses)
        ]
       
        input_ids, steps, reward_flags = zip(*processed_data)
        # 截断 input_ids，防止 embedding 超过最大 token 限制（4096）
        max_embedding_tokens = 4096
        truncated_input_ids = [
            seq[:max_embedding_tokens]
            for seq in input_ids
        ]
        rewards = prm_client.embeddings.create(
            input=truncated_input_ids,
            model=args.prm_name_or_path.split("/")[-1],
        )
        step_rewards = derive_step_rewards_vllm(rewards, reward_flags) # list[list]
        for r in step_rewards:
            if r == []:
                r.append(0.0)

        good_prompts = []
        for (orig_idx, prompt, prev_responses), draft_response, step_reward in zip(current_prompts, draft_responses, step_rewards):
            all_rewards[orig_idx].append(round(step_reward[-1], 6))
            good_prompts.append((orig_idx, prompt, prev_responses, draft_response, True))

        if num_step == 0:
            simple_idxs = []
            for idx, reward_list in enumerate(all_rewards):
                if len(reward_list) == 0:
                    continue
                final_reward = reward_list[-1]
                if final_reward <= 0.8:
                    simple_idxs.append(idx)
        break
    return simple_idxs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="math500", type=str)
    parser.add_argument("--data_dir", default="./external/qwen25_math_evaluation/data", type=str)
    parser.add_argument("--draft_model_name_or_path", default="Qwen/Qwen2.5-Math-1.5B-Instruct", type=str)
    parser.add_argument("--draft_model_ip_address", default="http://localhost:12340/v1", type=str)
    parser.add_argument("--target_model_name_or_path", default="Qwen/Qwen2.5-Math-7B-Instruct", type=str)
    parser.add_argument("--target_model_ip_address", default="http://localhost:12341/v1", type=str)
    parser.add_argument("--prm_name_or_path", default="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B", type=str)
    parser.add_argument("--prm_ip_address", default="http://localhost:12342/v1", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="qwen25-math-cot", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
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
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_delta{args.prm_threshold}_maxsteps{args.max_steps}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def setup(args):
    # load model
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

    prm_client = OpenAI(
        api_key=openai_api_key,
        base_url=args.prm_ip_address,
    )
    prm_tokenizer = AutoTokenizer.from_pretrained(args.prm_name_or_path, trust_remote_code=True)


    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(main(draft_client, target_client, prm_client, draft_tokenizer, target_tokenizer, prm_tokenizer, data_name, args))

    # # add "avg" result to data_list and results
    # data_list.append("avg")
    # results.append({"acc": sum([result["acc"] for result in results]) / len(results),})

    # # print all results
    # pad = max([len(data_name) for data_name in data_list])
    # print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    # print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True

def main(draft_client, target_client, prm_client, draft_tokenizer, target_tokenizer, prm_tokenizer, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    # build samples
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

    samples = sorted(samples, key=lambda x: x['idx'])
    # repeat n times if n_sampling>1
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

    # remain_prompts: list of tuples (global_input_index, prompt_text)
    remain_prompts = [(i, prompt) for i, prompt in enumerate(input_prompts)]

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")

    # --------------------
    # prepare models & optim
    # --------------------
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    slm_name="Qwen/Qwen2.5-Math-1.5B-Instruct"
    slm = AutoModelForCausalLM.from_pretrained(
        slm_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        low_cpu_mem_usage=True
    )
    slm_tokenizer = AutoTokenizer.from_pretrained(slm_name)

    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    model_pai = Actor(1536).to(device)
    model_v = Critic(1536).to(device)
    model_delay = Delay(1536).to(device)
    model_randa = RandA(1536).to(device)

    # target_model is used only to snapshot policy at sampling time if needed. target_model = old policy.
    target_model = copy.deepcopy(model_pai).to(device)
    target_model.eval()

    model_pai.train()
    model_v.train()
    model_delay.train()
    model_randa.train()

    optimizer_pai = optim.Adam(model_pai.parameters(), lr=5e-5)
    optimizer_v = optim.Adam(model_v.parameters(), lr=5e-4)
    optimizer_delay = optim.Adam(model_delay.parameters(), lr=5e-4)
    optimizer_randa = optim.Adam(model_randa.parameters(), lr=5e-4)

    num_ep = getattr(args, "num_ep", 1000)
    difficult_idxs_list = []

    # --------------------
    # main training loop
    # --------------------
    for ep in range(num_ep):
        print("ep ", ep)

        # split large remain_prompts & samples into batches (keep consistent partitioning)
        remain_prompts_groups = [remain_prompts[i:i+4500] for i in range(0, len(remain_prompts), 4500)]
        samples_groups = [samples[i:i+4500] for i in range(0, len(samples), 4500)]

        for batch_idx, (remain_prompts_batch, samples_batch) in enumerate(zip(remain_prompts_groups, samples_groups)):
            print(f"  batch: {batch_idx}")

            # map local index within this batch -> global input_prompts index
            local_to_global_map = {local_idx: global_idx for local_idx, (global_idx, _) in enumerate(remain_prompts_batch)}

            # snapshot policy into target_model BEFORE sampling (so old logits can be computed from saved logits)
            # (we will still rely on saved all_logits returned from train_gainSAR)
            for param, target_param in zip(model_pai.parameters(), target_model.parameters()):
                target_param.data.copy_(param.data)

            # build prompts & problems arrays for this batch (local)
            prompts = [item[1] for item in remain_prompts_batch]  # local batch prompts (strings)
            problems = [sample["question"] for sample in samples_batch]
            assert len(prompts) == len(problems)

            # -------------------------
            # find simple tasks in this batch (local indices)
            # -------------------------
            if ep == 0:
                simple_idxs = find_simpletasks(
                    args,
                    draft_client,
                    target_client,
                    prm_client,
                    draft_tokenizer,
                    target_tokenizer,
                    prm_tokenizer,
                    prompts,
                    problems,
                )
                # ensure it's a list of local indices (ints)
                if simple_idxs is None:
                    simple_idxs = []
                difficult_idxs_list.append(simple_idxs)

            else:
                # reuse previously found simple indices for this batch if available
                if batch_idx < len(difficult_idxs_list):
                    simple_idxs = difficult_idxs_list[batch_idx]
                else:
                    simple_idxs = []

            if len(simple_idxs) == 0:
                print(f"    Warning: No difficult_idxs found for batch {batch_idx}, skip")
                continue

            # Project simple_idxs (local indices) into the local arrays
            valid_simple_idxs = [i for i in simple_idxs if 0 <= i < len(prompts)]
            if len(valid_simple_idxs) == 0:
                print(f"    Warning: valid_simple_idxs empty after filtering for batch {batch_idx}")
                continue

            prompts1 = [prompts[i] for i in valid_simple_idxs]
            problems1 = [problems[i] for i in valid_simple_idxs]
            current_prompts1 = [remain_prompts_batch[i] for i in valid_simple_idxs]  # elements are (global_idx, prompt)
            samples1 = [samples_batch[i] for i in valid_simple_idxs]

            # prepare input_prompts1 (global input_prompts strings) using local_to_global_map
            input_prompts1 = []
            for local_idx in valid_simple_idxs:
                global_input_idx = local_to_global_map.get(local_idx, None)
                if global_input_idx is None:
                    # shouldn't happen
                    input_prompts1.append("")
                else:
                    input_prompts1.append(input_prompts[global_input_idx])

            # -------------------------
            # collect rollouts for prompts1 using train_gainSAR (this function must return all_logits)
            # -------------------------
            outputs, token_counts, turn_info, all_rewards, all_states, all_actions, all_values, all_logits, all_delays, all_numtokens, all_values_next = train_gainSAR(
                ep,
                slm,
                slm_tokenizer,
                optimizer_pai,
                optimizer_v,
                model_pai,
                model_v,
                target_model,
                args,
                draft_client,
                target_client,
                prm_client,
                draft_tokenizer,
                target_tokenizer,
                prm_tokenizer,
                prompts1,
                problems1,
            )

            # ---------- Basic sanity checks ----------
            # ensure lists exist
            if any(x is None for x in [all_states, all_actions, all_values, all_logits]):
                print("    Error: train_gainSAR must return all_states/all_actions/all_values/all_logits")
                continue

            # each of these should be list per sample; convert to list-of-lists if needed
            # ensure same outer length (#samples)
            if not (len(all_states) == len(all_actions) == len(all_values) == len(all_logits) == len(all_rewards)):
                print("    Warning: lengths mismatch from train_gainSAR:",
                      len(all_states), len(all_actions), len(all_values), len(all_logits), len(all_rewards))
                # try to proceed but be careful

            # ------------- process outputs -> assemble end_prompts -------------
            remain_prompts_batch_next = []
            remain_codes = []
            end_prompts = []

            for idx_local, ((global_idx, query), output) in enumerate(zip(current_prompts1, outputs)):
                # global idx already comes from current_prompts1 entries
                output = output.rstrip()
                query_concat = query + output
                if args.prompt_type == "pal":
                    remain_prompts_batch_next.append((global_idx, query_concat))
                    if "```python" in output:
                        out_code = extract_program(query_concat)
                        remain_codes.append(out_code)
                    else:
                        remain_codes.append(output)
                elif args.prompt_type == "cot":
                    end_prompts.append((global_idx, query_concat))
                elif "boxed" not in output and output.endswith("```"):
                    program = extract_program(query_concat)
                    remain_prompts_batch_next.append((global_idx, query_concat))
                    remain_codes.append(program)
                else:
                    end_prompts.append((global_idx, query_concat))

            if len(remain_prompts_batch_next) == 0:
                print("    No remain prompts (batch finished early)")
                # continue to next batch
            else:
                # execute remain_codes (if any)
                if len(remain_codes) > 0:
                    remain_results = executor.batch_apply(remain_codes)
                    for k in range(len(remain_prompts_batch_next)):
                        gidx, q = remain_prompts_batch_next[k]
                        res, report = remain_results[k]
                        exec_result = res if res else report
                        if "pal" in args.prompt_type:
                            exec_result = "\\boxed{" + exec_result + "}"
                        exec_result = f"\n```output\n{exec_result}\n```\n"
                        q += exec_result
                        if max_func_call <= 1:
                            q += "\nReach max function call limit."
                        remain_prompts_batch_next[k] = (gidx, q)

            # unsolved
            if len(remain_prompts_batch_next) == 0 and len(end_prompts) == 0:
                # nothing left for this batch
                continue

            # combine end_prompts and remain_prompts_batch_next to produce codes
            combined_end_prompts = end_prompts + remain_prompts_batch_next
            combined_end_prompts = sorted(combined_end_prompts, key=lambda x: x[0])

            # ensure equal length to input_prompts1 (robust handling)
            if len(input_prompts1) != len(combined_end_prompts):
                min_len = min(len(input_prompts1), len(combined_end_prompts))
                input_prompts1 = input_prompts1[:min_len]
                combined_end_prompts = combined_end_prompts[:min_len]

            codes = []
            for i in range(len(input_prompts1)):
                gidx, end_prompt = combined_end_prompts[i]
                code = end_prompt.split(input_prompts1[i])[-1].strip()
                for stop_word in stop_words:
                    if stop_word in code:
                        code = code.split(stop_word)[0].strip()
                codes.append(code)

            # run executor to get predictions for codes
            results = [ run_execute(executor, code, args.prompt_type, data_name) for code in codes ]

            # assemble all_samples and evaluate
            all_samples = []
            for i_local, sample in enumerate(samples1):
                code = codes[i_local * args.n_sampling : (i_local + 1) * args.n_sampling]
                result_slice = results[i_local * args.n_sampling : (i_local + 1) * args.n_sampling]
                preds = [item[0] for item in result_slice]
                reports = [item[1] for item in result_slice]
                for j in range(len(preds)):
                    if sample["gt"] in ["A","B","C","D","E"] and preds[j] not in ["A","B","C","D","E"]:
                        preds[j] = choice_answer_clean(code[j])
                    elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                        preds[j] = "".join([c for c in preds[j] if c in ["A","B","C","D","E"]])

                if "prompt" in sample:
                    sample.pop("prompt")
                sample.update({
                    "code": code, "pred": preds, "report": reports,
                    "token_counts": token_counts[i_local], "turn_info": turn_info[i_local], "reward": all_rewards[i_local]
                })
                all_samples.append(sample)

            all_samples.extend(processed_samples)
            # correct_indices is global indexs
            all_samples, result_json, _, correct_indices = evaluate(
                samples=all_samples,
                data_name=data_name,
                prompt_type=args.prompt_type,
                execute=True,
            )

            # -------------------------
            # Reward mapping (local -> global)
            # all_rewards corresponds to prompts1 local order
            # We map each local_i to its global_idx via local_to_global_map and simple_idxs
            # -------------------------
            for local_i, reward in enumerate(all_rewards):
                # batch_local_idx: index in the batch (local index into 'prompts')
                batch_local_idx = valid_simple_idxs[local_i]  # because prompts1 built with valid_simple_idxs order
                global_idx = local_to_global_map.get(batch_local_idx, None)
                if global_idx is None:
                    print(f"    Warning: global_idx not found for local_i {local_i} in batch {batch_idx}, skip reward shaping")
                    continue
                if global_idx in correct_indices:
                    reward[-1] += 7.5
                else:
                    reward[-1] -= 5.0

            # compute average reward for logging
            avg_reward = float(sum(sum(r_list) for r_list in all_rewards) / max(1, len(all_rewards)))

            # -------------------------
            # Build targets for PPO update
            # all_states/all_actions/all_values/all_logits are list per sample (time-steps)
            # flatten them in time order across samples
            # -------------------------
            flat_states = [s.detach() for sample_states in all_states for s in sample_states]
            flat_rewards = [r for sample_rewards in all_rewards for r in sample_rewards]
            flat_numtokens = [n for sample_numtokens in all_numtokens for n in sample_numtokens]
            flat_actions = [a for sample_actions in all_actions for a in sample_actions]
            flat_values = [v for sample_values in all_values for v in sample_values]
            flat_values_next = [v for sample_values_next in all_values_next for v in sample_values_next]
            flat_logits = [l.detach() for sample_logits in all_logits for l in sample_logits]
            flat_delays = [d for sample_delays in all_delays for d in sample_delays]  # used earlier too
            if len(flat_actions) > 0:
                action1_count = sum(1 for action in flat_actions if action == 1)
                trajectory_lengths = [len(actions) for actions in all_actions]
                total_actions = sum(trajectory_lengths)
                action1_prob = action1_count / total_actions

                
                print(f"Epoch {ep} Batch {batch_idx}: 平均选择LLM的概率 = {action1_prob:.4f} ({action1_count}/{total_actions})")
            else:
                print(f"Epoch {ep} Batch {batch_idx}: 无动作数据")

            if len(flat_states) == 0:
                print("    Nothing to update (no states collected), skipping update")
                continue

            # convert to tensors
            states_tensor = torch.stack(flat_states).to(device)  # shape [T, state_dim]
            actions_tensor = torch.tensor(flat_actions, dtype=torch.int64, device=device)  # [T]
            delays_tensor = torch.tensor(flat_delays, dtype=torch.int64, device=device)  # [T]
            values_tensor = torch.stack(flat_values).squeeze(1).to(device) if isinstance(flat_values[0], torch.Tensor) else torch.tensor(flat_values, dtype=torch.float32, device=device)
            ys_tensor = None  # we'll compute return targets below

            # compute discounted returns (n-step bootstrapping using value at t+step_size if available)
            step_size = 3
            gamma = getattr(args, "gamma", 1.0)
            all_ys = []
            for sample_idx, sample_rewards in enumerate(all_rewards):
                # for each sample's timeline compute n-step returns using sample's values
                sample_values = all_values[sample_idx]
                for t in range(len(sample_rewards)):
                    y = 0.0
                    for k in range(step_size):
                        if t + k < len(sample_rewards):
                            y += (gamma ** k) * sample_rewards[t + k]
                        else:
                            break
                    # bootstrap from value at t+step_size if exists
                    if t + step_size < len(sample_values):
                        y += (gamma ** step_size) * float(sample_values[t + step_size].item())
                    all_ys.append(y)

            ys_tensor = torch.tensor(all_ys, dtype=torch.float32, device=device).detach()

            # compute advantages: adv = y - V(s)
            flat_values_tensor = torch.stack([v for sample_values in all_values for v in sample_values]).squeeze(1).to(device)
            advantages = ys_tensor - flat_values_tensor
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages_tensor = advantages.detach()

            # old_log_probs: compute from saved old logits (flat_logits)
            # make sure each old logit vector is of size num_actions
            with torch.no_grad():
                try:
                    old_logits_tensor = torch.stack([l.view(-1) if l.dim()==1 else l.squeeze(0) for l in flat_logits]).to(device)  # shape [T, num_actions]
                except Exception as e:
                    # fallback: try to convert list->tensor
                    old_logits_tensor = torch.stack([torch.tensor(l, dtype=torch.float32) if not isinstance(l, torch.Tensor) else l for l in flat_logits]).to(device)
                old_probs = torch.softmax(old_logits_tensor, dim=-1)
                old_dist = torch.distributions.Categorical(old_probs)
                old_log_probs = old_dist.log_prob(actions_tensor)

            # -------------------------
            # PPO update (multiple epochs)
            # -------------------------
            PPO_num = getattr(args, "PPO_num", 5)
            clip_epsilon = getattr(args, "clip_epsilon", 0.2)
            entropy_coef = getattr(args, "entropy_coef", 0.05)

            for _ in range(PPO_num):
                new_logits = model_pai(states_tensor)  # if Actor.forward returns (feature, logits) change accordingly
                # handle Actor that returns (feature, logits)
                if isinstance(new_logits, tuple) or isinstance(new_logits, list):
                    _, new_logits = new_logits
                new_probs = torch.softmax(new_logits, dim=-1)
                new_dist = torch.distributions.Categorical(new_probs)
                new_log_probs = new_dist.log_prob(actions_tensor)

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages_tensor
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_tensor
                loss_pai = -torch.min(surr1, surr2).mean()

                # optional entropy bonus
                if entropy_coef > 0:
                    entropy = new_dist.entropy().mean()
                    loss_entropy = entropy_coef * entropy
                    loss_paiall = loss_pai - loss_entropy

                optimizer_pai.zero_grad()
                loss_paiall.backward()
                torch.nn.utils.clip_grad_norm_(model_pai.parameters(), 0.5)
                optimizer_pai.step()

                # value update
                new_values = model_v(states_tensor).squeeze(1)
                loss_val = nn.functional.mse_loss(new_values, ys_tensor)
                optimizer_v.zero_grad()
                loss_val.backward()
                # torch.nn.utils.clip_grad_norm_(model_v.parameters(), 0.5)
                optimizer_v.step()

            loss = loss_pai + loss_val

            print(f"epoch {ep} batch {batch_idx} PPO_num {PPO_num} : Loss={loss.item():.6f}, Policy Loss={loss_pai.item():.6f}, Value Loss={loss_val.item():.6f}, Avg_reward={avg_reward:.4f},entropy={entropy:.4f}")

            llm_states = []
            llm_delays = []
            llm_numtokens = []
            llm_values = []
            llm_values_next = []
            llm_rewards = []

            for s, a, d, n, r, v_next, v in zip(flat_states, flat_actions, flat_delays, flat_numtokens, flat_rewards, flat_values_next, flat_values):
                if a == 1:
                    llm_states.append(s)
                    llm_delays.append(d)
                    llm_numtokens.append(n)
                    llm_values.append(v)
                    llm_values_next.append(v_next)
                    llm_rewards.append(r)
            if len(llm_states) == 0:
                print("    No LLM actions taken in this batch, skipping update")
                continue

            llm_advantages = []
            for v, v_next, r in zip(llm_values, llm_values_next, llm_rewards):
                adv = r + v_next.item() - v.item()
                llm_advantages.append(adv)
            llm_advantages = torch.tensor(llm_advantages, dtype=torch.float32).to(device)
            llm_rewards = torch.tensor(llm_rewards, dtype=torch.float32).to(device)
            llm_randas = torch.stack([llm_advantages, llm_rewards], dim=1)  # [batch_size, 2]
            llm_delays = torch.tensor(llm_delays, dtype=torch.float32).to(device)
            llm_numtokens = torch.tensor(llm_numtokens, dtype=torch.float32).to(device)
            llm_states = torch.stack(llm_states).to(device)
            for _ in range(100):
                randa = model_randa(llm_states).squeeze()
                loss_randa = nn.MSELoss()(randa, llm_randas)
                optimizer_randa.zero_grad()
                loss_randa.backward()
                optimizer_randa.step()

                d = model_delay(llm_states, llm_numtokens).squeeze()
                loss_d = nn.MSELoss()(d, llm_delays)
                optimizer_delay.zero_grad()
                loss_d.backward()
                optimizer_delay.step()

            print(f"epoch {ep} batch {batch_idx} Loss_randa={loss_randa.item():.6f}, Loss_delay={loss_d.item():.6f}")

        if ((ep + 1) % 10 == 0):
            torch.save(model_pai.state_dict(), f'model_pai_ep{ep+1}_{args.beta}.pth')
            torch.save(model_v.state_dict(), f'model_v_ep{ep+1}_{args.beta}.pth')
            torch.save(model_delay.state_dict(), f'model_d1_ep{ep+1}_{args.beta}.pth')
            torch.save(model_randa.state_dict(), f'model_randa_ep{ep+1}_{args.beta}.pth')

            # end of batch loopss
    return 0


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)