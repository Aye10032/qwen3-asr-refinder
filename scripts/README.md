# Development scripts

This directory contains the dataset preparation, training, and model publishing workflows for Qwen3 ASR Refiner.

## Prepare the dataset locally

Build a compact, text-only derivative of
[`TaurenMountain/WenetSpeech-Formal`](https://huggingface.co/datasets/TaurenMountain/WenetSpeech-Formal).
The extractor streams only `original_text` and `target_text` from remote Parquet column ranges, so the 87 GB audio
corpus is never downloaded.

```bash
uv sync --all-groups
uv run python scripts/prepare_dataset.py prepare
```

The local dataset is written to `artifacts/WenetSpeech-Formal-Text/`, with one Parquet file per split.

```python
from datasets import load_dataset

dataset = load_dataset('artifacts/WenetSpeech-Formal-Text')
```

## Upload the dataset

```bash
uv run python scripts/prepare_dataset.py upload
```

The default destination is
[`Aye10032/WenetSpeech-Formal-Text`](https://huggingface.co/datasets/Aye10032/WenetSpeech-Formal-Text).

To prepare and upload in one command:

```bash
uv run python scripts/prepare_dataset.py all
```

## Train

Put `WANDB_API_KEY` in `.env`, then launch BF16 LoRA training on two GPUs:

```bash
uv sync --all-groups
uv run accelerate launch \
  --multi_gpu \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  scripts/train.py --model-id Qwen/Qwen3-1.7B
```

Metrics are logged to `martians_beasts_dragons/qwen3-asr-refinder`. The final LoRA adapter and checkpoints are
written to `artifacts/qwen3-1.7b-asr-refinder-lora/`.

The base model is configurable. Output directories and W&B run names are derived from the model name, so the adapters
can coexist:

```bash
# Qwen3-0.6B
uv run accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 \
  scripts/train.py --model-id Qwen/Qwen3-0.6B

# Qwen3-4B (start with a smaller per-device batch if GPU memory is limited)
uv run accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 \
  scripts/train.py --model-id Qwen/Qwen3-4B --batch-size 1 --gradient-accumulation-steps 32
```

These commands write to `artifacts/qwen3-0.6b-asr-refinder-lora/` and
`artifacts/qwen3-4b-asr-refinder-lora/`, respectively. Use `--output-dir` and `--run-name` to override the generated
names.

Run a short pilot before the full training run:

```bash
uv run accelerate launch \
  --multi_gpu \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  scripts/train.py --model-id Qwen/Qwen3-1.7B --max-train-samples 100000 --max-steps 500
```

## Export and upload

Merge the trained LoRA adapter into the base model, save complete BF16 Transformers weights, and upload them to
[`Aye10032/Qwen3-1.7B-ASR-Refiner`](https://huggingface.co/Aye10032/Qwen3-1.7B-ASR-Refiner):

```bash
uv run python scripts/export_model.py --base-model Qwen/Qwen3-1.7B
```

The merged model is also written to `artifacts/Qwen3-1.7B-ASR-Refiner/`. Run this as a single process after training,
not through `accelerate launch`.

Export another family member by changing the required base-model argument:

```bash
uv run python scripts/export_model.py --base-model Qwen/Qwen3-0.6B
uv run python scripts/export_model.py --base-model Qwen/Qwen3-4B
```

By default these publish to `Aye10032/Qwen3-0.6B-ASR-Refiner` and `Aye10032/Qwen3-4B-ASR-Refiner`. Use `--repo-id`
to publish under a different account or name.
