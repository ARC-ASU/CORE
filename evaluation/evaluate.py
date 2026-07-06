import argparse
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import random
from collections import Counter

from grader import *

from parser import *
from utils import load_jsonl
from python_executor import PythonExecutor

def get_majority_vote_answer(predictions, scores, ground_truth):
    """
    Get the self-consistent answer using majority voting.
    If there's a tie, randomly select from the tied candidates.
    """
    if not predictions:
        return None, False

    # Filter valid predictions (non-empty)
    valid_preds = [pred for pred in predictions if pred and pred.strip()]

    if not valid_preds:
        return None, False

    # Count frequencies of predictions
    pred_counts = Counter(valid_preds)
    max_count = max(pred_counts.values())

    # Find all predictions with maximum count
    top_candidates = [pred for pred, count in pred_counts.items() if count == max_count]

    # If there's a clear winner, return it
    if len(top_candidates) == 1:
        final_pred = top_candidates[0]
    else:
        # Random tie-breaking
        final_pred = random.choice(top_candidates)

    # Check if the majority vote is correct with timeout protection
    try:
        is_correct = math_equal(final_pred, ground_truth, timeout=True)
    except Exception as e:
        # If any error or timeout occurs, mark as incorrect
        is_correct = False

    return final_pred, is_correct


def evaluate(data_name, prompt_type, samples: list=None, file_path: str=None, max_num_samples=None, execute=False, self_consistent=False):
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

    with ProcessPool(max_workers=1) as pool:
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

    # Self-consistent evaluation: use majority voting
    if self_consistent:
        print("🗳️  Using self-consistent evaluation with majority voting...")
        self_consistent_scores = []
        for sample in tqdm(samples, desc="Majority Voting", unit="sample"):
            majority_pred, is_correct = get_majority_vote_answer(
                sample['pred'], sample['score'], sample['gt']
            )
            sample['self_consistent_pred'] = majority_pred
            sample['self_consistent_score'] = is_correct
            self_consistent_scores.append(is_correct)
        
        # Calculate self-consistent accuracy
        sc_accuracy = np.mean(self_consistent_scores) * 100
        
        result_json = {
            "num_samples": len(samples),
            "num_scores": len(scores),
            "timeout_samples": timeout_cnt,
            "empty_samples": len([s for s in samples if not s['pred'][-1]]),
            "acc": round(sc_accuracy, 1),  # Use self-consistent accuracy
            "self_consistent_acc": round(sc_accuracy, 1),
            "individual_sample_acc": list(np.round(np.array(score_mat).mean(axis=0) * 100, decimals=1))
        }
        
        return samples, result_json

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
    return samples, result_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--prompt_type", type=str, default="tool-integrated")
    parser.add_argument("--file_path", type=str, default=None, required=True)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate(data_name=args.data_name, prompt_type=args.prompt_type, file_path=args.file_path,
             max_num_samples=args.max_num_samples, execute=args.execute)
