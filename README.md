# qwen3-asr-refinder

Build and publish a compact, text-only derivative of
[`TaurenMountain/WenetSpeech-Formal`](https://huggingface.co/datasets/TaurenMountain/WenetSpeech-Formal).
The extractor streams only `original_text` and `target_text` from remote Parquet column ranges, so the 87 GB audio
corpus is never downloaded.

## Prepare locally

```bash
uv sync --all-groups
uv run python scripts/prepare_dataset.py prepare
```

The local dataset is written to `artifacts/WenetSpeech-Formal-Text/`, with one Parquet file per split.

```python
from datasets import load_dataset

dataset = load_dataset('artifacts/WenetSpeech-Formal-Text')
```

## Upload

```bash
uv run python scripts/prepare_dataset.py upload
```

The default destination is [`Aye10032/WenetSpeech-Formal-Text`](https://huggingface.co/datasets/Aye10032/WenetSpeech-Formal-Text).

To prepare and upload in one command:

```bash
uv run python scripts/prepare_dataset.py all
```
