import random
import os
import argparse
import time
import importlib
import sys
from pathlib import Path
from datetime import datetime

from importlib import metadata
from packaging import version
from tqdm import tqdm


def _ensure_tokenizers(min_version: str = "0.20.0") -> None:
    """Ensure we are running with a recent enough tokenizers build.

    Older 0.19.x releases cannot parse the tokenizer.json produced by
    the concept-enhanced checkpoints, which triggers vLLM to crash
    during initialization. We try to import a newer wheel if the active
    environment ships an older version.
    """

    def _current_version() -> str | None:
        try:
            return metadata.version("tokenizers")
        except metadata.PackageNotFoundError:
            return None
        except Exception:
            return None

    loaded_module = sys.modules.get("tokenizers")
    if loaded_module is not None:
        loaded_version = getattr(loaded_module, "__version__", None)
        if loaded_version and version.parse(loaded_version) >= version.parse(min_version):
            return
        raise RuntimeError(
            f"tokenizers {loaded_version or 'unknown'} is already imported in this process "
            f"but is older than {min_version}. Please restart the interpreter after upgrading "
            "tokenizers, or unset pre-imports that pull it in."
        )

    current = _current_version()
    if current and version.parse(current) >= version.parse(min_version):
        return

    if current:
        print(
            f"⚠️ Detected tokenizers {current}, but >= {min_version} is required; "
            "trying to load a newer build."
        )
    else:
        print("⚠️ tokenizers not found; trying to load a bundled build.")

    candidate_roots = []
    override_env = os.getenv("TOKENIZERS_OVERRIDE_PATH")
    if override_env:
        candidate_roots.append(Path(override_env))

    script_dir = Path(__file__).resolve().parent
    vendor_dir = script_dir / "vendor"
    candidate_roots.append(vendor_dir)

    for root in candidate_roots:
        if not root:
            continue
        site_dir = None
        package_dir = None
        if root.is_dir():
            if root.name == "tokenizers" and (root / "__init__.py").exists():
                package_dir = root
                site_dir = root.parent
            elif (root / "tokenizers").is_dir():
                site_dir = root
                package_dir = root / "tokenizers"
        if site_dir is None or package_dir is None or not package_dir.exists():
            continue
        sys.modules.pop("tokenizers", None)
        path_str = str(site_dir)
        sys.path.insert(0, path_str)
        try:
            tokenizers = importlib.import_module("tokenizers")  # type: ignore
            loaded_version = getattr(tokenizers, "__version__", "0")
            if version.parse(loaded_version) >= version.parse(min_version):
                print(
                    f"✅ Using tokenizers {loaded_version} from {package_dir}"
                )
                return
            else:
                print(
                    f"⚠️ tokenizers {loaded_version} from {package_dir} is still "
                    f"older than {min_version}"
                )
        except Exception as exc:
            print(f"⚠️ Failed to import tokenizers from {package_dir}: {exc}")
        finally:
            if sys.path and sys.path[0] == path_str:
                sys.path.pop(0)
            else:
                try:
                    sys.path.remove(path_str)
                except ValueError:
                    pass

    raise RuntimeError(
        f"tokenizers >= {min_version} is required to load this checkpoint. "
        "Please install a newer build or set TOKENIZERS_OVERRIDE_PATH to a "
        "directory containing one."
    )


_ensure_tokenizers()

from vllm import LLM, SamplingParams

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
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
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--self_consistent", action="store_true", help="Enable self-consistent evaluation with majority voting")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization for vLLM")
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=None, help="Maximum model sequence length")
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs for vLLM.",
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
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    # only default relative paths under outputs/; absolute paths (e.g. a model
    # checkpoint dir) are used as-is to avoid creating a nested outputs/<abs> tree
    if not os.path.isabs(output_dir) and not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
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
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if args.use_vllm:
        # vLLM engine arguments (throughput-only tuning; does not change sampling semantics)
        # Values can be overridden by env vars to avoid hard-coding.
        def _get_env_float(key: str, default: float) -> float:
            try:
                return float(os.getenv(key, str(default)))
            except Exception:
                return default

        def _get_env_int(key: str, default: int) -> int:
            try:
                return int(os.getenv(key, str(default)))
            except Exception:
                return default

        def _get_env_bool(key: str, default: bool) -> bool:
            val = os.getenv(key)
            if val is None:
                return default
            return val.strip().lower() in ["1", "true", "yes", "y", "on"]

        # Sensible, conservative defaults; only impact scheduling/throughput.
        max_num_batched_tokens = _get_env_int("VLLM_MAX_BATCHED_TOKENS", 16384)
        max_num_seqs = _get_env_int("VLLM_MAX_SEQS", 1024)
        swap_space_gb = _get_env_float("VLLM_SWAP_SPACE_GB", 4.0)
        enable_chunked_prefill = _get_env_bool("VLLM_ENABLE_CHUNKED_PREFILL", True)
        enable_prefix_caching = _get_env_bool("VLLM_ENABLE_PREFIX_CACHING", False)

        llm_kwargs = {
            "model": args.model_name_or_path,
            "tensor_parallel_size": len(available_gpus) // args.pipeline_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "trust_remote_code": True,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            # Throughput-oriented knobs (no effect on sampling semantics):
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_num_seqs": max_num_seqs,
            "swap_space": swap_space_gb,
            "enable_chunked_prefill": enable_chunked_prefill,
            "enable_prefix_caching": enable_prefix_caching,
        }
        if args.enforce_eager:
            llm_kwargs["enforce_eager"] = True

        if args.max_model_len is not None:
            llm_kwargs["max_model_len"] = args.max_model_len
            print(f"🔧 Setting max_model_len to {args.max_model_len}")

        print(
            f"🔧 vLLM engine: gpu_mem_util={args.gpu_memory_utilization}, "
            f"max_num_batched_tokens={max_num_batched_tokens}, max_num_seqs={max_num_seqs}, "
            f"swap_space_gb={swap_space_gb}, chunked_prefill={enable_chunked_prefill}, "
            f"prefix_caching={enable_prefix_caching}"
        )
        # Be defensive in case older vLLM versions don't accept some kwargs.
        try:
            llm = LLM(**llm_kwargs)
        except TypeError as e:
            print(f"⚠️ vLLM constructor rejected some perf kwargs ({e}); retrying with minimal args")
            minimal_kwargs = {
                "model": args.model_name_or_path,
                "tensor_parallel_size": len(available_gpus) // args.pipeline_parallel_size,
                "pipeline_parallel_size": args.pipeline_parallel_size,
                "trust_remote_code": True,
                "gpu_memory_utilization": args.gpu_memory_utilization,
            }
            if args.enforce_eager:
                minimal_kwargs["enforce_eager"] = True
            if args.max_model_len is not None:
                minimal_kwargs["max_model_len"] = args.max_model_len
            llm = LLM(**minimal_kwargs)
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    else:
        try:
            llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=None,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
            )
        except Exception as e:
            print("❌ Error in loading model:")
            import traceback
            traceback.print_exc()
            import sys
            sys.exit(1)


    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(main(llm, tokenizer, data_name, args))

    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )

    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def is_multi_choice(answer):
    if not answer:
        return False
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(llm, tokenizer, data_name, args):
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

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
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

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

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

    # start inference
    # measure time use
    start_time = time.time()
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
        if args.use_vllm:
            outputs = llm.generate(
                prompts,
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens_per_call,
                    n=1,
                    stop=stop_words,
                    stop_token_ids=(
                        [151645, 151643]
                        if "qwen2" in args.model_name_or_path.lower()
                        else None
                    ),
                ),
            )

            outputs = sorted(
                outputs, key=lambda x: int(x.request_id)
            )  # sort outputs by request_id
            outputs = [output.outputs[0].text for output in outputs]
        else:
            outputs = generate_completions(
                model=llm,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_tokens_per_call,
                batch_size=16,
                stop_id_sequences=stop_words,
            )

        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif "boxed" not in output and output.endswith("```"):
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute the remain prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports})
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
        self_consistent=args.self_consistent,
    )

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)

    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )

    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
