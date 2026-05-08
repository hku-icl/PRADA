import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from .grader import *
from .parser import *
from .utils import load_jsonl
from .python_executor import PythonExecutor


def evaluate(data_name, prompt_type, samples: list=None, file_path: str=None, max_num_samples=None, execute=False):
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(load_jsonl(file_path))
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]
    
    # parse gt
    for sample in samples:
        sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)
    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]

    scores = []
    timeout_cnt = 0 
    num_cpus = os.cpu_count()
    safe_workers = 1 
    with ProcessPool(max_workers=safe_workers) as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1) 

    idx = 0
    score_mat = []
    for sample in samples:
        sample['score'] = scores[idx: idx+len(sample['pred'])]
        assert len(sample['score']) == len(sample['pred'])
        score_mat.append(sample['score'])
        idx += len(sample['pred'])

    max_len = max([len(s) for s in score_mat])
    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

    # output mean of each column of scores
    col_means= np.array(score_mat).mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))

    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "empty_samples": len([s for s in samples if not s['pred'][-1]]),
        "acc": mean_score[0]
    }

    # each type score
    if "type" in samples[0]:
        type_scores = {}
        for sample in samples:
            if sample['type'] not in type_scores:
                type_scores[sample['type']] = []
            type_scores[sample['type']].append(sample['score'][-1])
        type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
        type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
        result_json['type_acc'] = type_scores

    print(result_json)

    # ✅ 收集正确与错误样本
    correct_strategies = []
    incorrect_indices = []
    correct_indices = []
    for sample in samples:
        if any(sample['score']):  # 正确样本
            correct_indices.append(sample["idx"])
        else:  # 错误样本
            incorrect_indices.append(sample["idx"])


    # result_json["num_correct"] = len(correct_strategies)
    # result_json["correct_strategies"] = correct_strategies

    # print(f"✅ 正确样本数量: {len(correct_indices)}")

    # out_path = f"correct_strategies_{data_name}.json"
    # if os.path.exists(out_path):
    #     with open(out_path, "r", encoding="utf-8") as f:
    #         old_data = json.load(f)
    # else:
    #     old_data = []

    # existing_keys = {(d["idx"], tuple(d["strategy"])) for d in old_data}
    # for cs in correct_strategies:
    #     key = (cs["idx"], tuple(cs["strategy"]))
    #     if key not in existing_keys:
    #         old_data.append(cs)
    #         existing_keys.add(key)

    # with open(out_path, "w", encoding="utf-8") as f:
    #     json.dump(old_data, f, indent=4, ensure_ascii=False)

    # ✅ 直接返回，不写入文件
    return samples, result_json, incorrect_indices, correct_indices
    # return samples, result_json


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_name", type=str, default="math")
#     parser.add_argument("--prompt_type", type=str, default="tool-integrated")
#     parser.add_argument("--file_path", type=str, default=None, required=True)
#     parser.add_argument("--max_num_samples", type=int, default=None)
#     parser.add_argument("--execute", action="store_true")
#     args = parser.parse_args()
#     return args


# if __name__ == "__main__":
#     args = parse_args()
#     samples, result_json, incorrect_samples = evaluate(
#         data_name=args.data_name,
#         prompt_type=args.prompt_type,
#         file_path=args.file_path,
#         max_num_samples=args.max_num_samples,
#         execute=args.execute
#     )

#     print("\n❌ 错误样本索引:")
#     print([x["idx"] for x in incorrect_samples])
