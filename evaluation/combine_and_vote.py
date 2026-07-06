import argparse
import os
import json
from utils import load_jsonl, save_jsonl
from evaluate import evaluate

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Combine two sets of generated samples and evaluate them.")
    parser.add_argument("--input_path1", type=str, required=True, help="Path to the first .jsonl file of samples.")
    parser.add_argument("--input_path2", type=str, required=True, help="Path to the second .jsonl file of samples.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the combined and evaluated .jsonl file.")
    parser.add_argument("--data_name", type=str, required=True, help="Name of the dataset being evaluated.")
    parser.add_argument("--prompt_type", type=str, required=True, help="The prompt type used for generation.")
    return parser.parse_args()

def main():
    """Main function to combine, evaluate, and save samples."""
    args = parse_args()

    try:
        samples1 = list(load_jsonl(args.input_path1))
        samples2 = list(load_jsonl(args.input_path2))
    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}")
        return

    samples1_dict = {s['idx']: s for s in samples1}
    samples2_dict = {s['idx']: s for s in samples2}

    combined_samples = []
    all_idxs = sorted(list(set(samples1_dict.keys()) & set(samples2_dict.keys())))

    # Warn about missing indices
    missing_in_1 = set(samples2_dict.keys()) - set(samples1_dict.keys())
    if missing_in_1:
        print(f"Warning: {len(missing_in_1)} indices exist in file 2 but not in file 1. They will be skipped.")
    missing_in_2 = set(samples1_dict.keys()) - set(samples2_dict.keys())
    if missing_in_2:
        print(f"Warning: {len(missing_in_2)} indices exist in file 1 but not in file 2. They will be skipped.")

    for idx in all_idxs:
        s1 = samples1_dict[idx]
        s2 = samples2_dict[idx]

        combined_sample = s1.copy()
        combined_sample['code'].extend(s2['code'])
        combined_sample['pred'].extend(s2['pred'])
        if 'report' in s1 and 'report' in s2:
            combined_sample['report'].extend(s2['report'])
        
        combined_samples.append(combined_sample)

    if not combined_samples:
        print("Error: No common samples found to combine. Exiting.")
        return

    # Perform evaluation using the existing evaluate function
    final_samples, result_json = evaluate(
        samples=combined_samples,
        data_name=args.data_name,
        prompt_type=args.prompt_type,
        execute=True,
        self_consistent=True
    )
    
    # Save the combined results and metrics
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_jsonl(final_samples, args.output_path)
    
    # Construct metrics file path based on convention from math_eval.py
    base_name = os.path.basename(args.output_path).replace(".jsonl", "")
    metrics_path = os.path.join(output_dir, f"{base_name}_{args.prompt_type}_metrics.json")
    
    with open(metrics_path, "w") as f:
        json.dump(result_json, f, indent=4)
    
    print(f"✅ Combined evaluation complete for {args.data_name}.")
    print(f"   Combined samples saved to: {args.output_path}")
    print(f"   Final metrics saved to:  {metrics_path}")

if __name__ == "__main__":
    main()
