# SFT Baseline (LLaMA-Factory)

The SFT baseline in our paper uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for supervised fine-tuning with LoRA. LLaMA-Factory is included as part of this repository (in `LLaMA-Factory/`).

## Installation

```bash
conda create -n core_sft python=3.11 -y && conda activate core_sft

# From the repository root: install pinned dependencies, then the embedded LLaMA-Factory
pip install -r requirements/requirements_sft.txt
cd fine-tune/LLaMA-Factory
pip install -e . --no-deps
cd ../..
```

## Data Preparation

Convert concept quizzes to LLaMA-Factory format:

```bash
cd fine-tune

python convert_data.py \
  --quizzes ../data/quizzes/concept_quizzes.jsonl \
  --output LLaMA-Factory/data/conceptandquiz_sft.json
```

The dataset is automatically registered in `LLaMA-Factory/data/dataset_info.json`.

## Training

```bash
cd fine-tune

# Qwen2-Math-7B SFT (2x GPUs, LoRA rank=8)
bash run_sft.sh configs/qwen2_math_7b_sft.yaml

# DeepSeek-R1-Distill-Qwen-1.5B
bash run_sft.sh configs/deepseek_1_5b_sft.yaml
```

## Configuration

Training configs are in `configs/`. Key hyperparameters (from Appendix B.1):

| Parameter | Qwen2-Math-7B | DeepSeek-1.5B |
|:---|:---|:---|
| Method | LoRA (rank=8, target=all) | LoRA (rank=8, target=all) |
| Learning rate | 5e-5 | 5e-5 |
| Batch size (effective) | 120 | 120 |
| Epochs | 15 | 15 |
| Scheduler | Cosine | Cosine |
| Warmup ratio | 0.1 | 0.1 |
| Precision | bf16 | bf16 |
| Distributed | DeepSpeed ZeRO-2 | DeepSpeed ZeRO-2 |
