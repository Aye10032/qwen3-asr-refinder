import json
from pathlib import Path

import click
import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_DATASET_ID = 'Aye10032/WenetSpeech-Formal-Text'
DEFAULT_REPO_OWNER = 'Aye10032'


def model_name(model_id: str) -> str:
    """Return the repository name portion of a Hugging Face model ID."""
    return model_id.rstrip('/').rsplit('/', 1)[-1]


def adapter_path(model_id: str) -> Path:
    return Path('artifacts') / f'{model_name(model_id).lower()}-asr-refinder-lora'


def merged_model_name(model_id: str) -> str:
    return f'{model_name(model_id)}-ASR-Refiner'


def read_training_metadata(adapter_dir: Path) -> dict[str, str]:
    metadata_path = adapter_dir / 'training_metadata.json'
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding='utf-8'))


def adapter_base_model(adapter_dir: Path) -> str | None:
    metadata = read_training_metadata(adapter_dir)
    if metadata.get('model_id'):
        return metadata['model_id']

    configured_model = PeftConfig.from_pretrained(adapter_dir).base_model_name_or_path
    if configured_model and not Path(configured_model).is_absolute():
        return configured_model
    return None


def build_model_card(base_model: str, dataset_id: str, repo_id: str) -> str:
    repo_owner = repo_id.split('/', 1)[0]
    return f'''---
base_model: {base_model}
library_name: transformers
pipeline_tag: text-generation
license: apache-2.0
language:
  - zh
datasets:
  - {dataset_id}
tags:
  - qwen3
  - asr-post-processing
  - spoken-to-written
  - text-normalization
---

# Qwen3 ASR Refiner

Qwen3 ASR Refiner is a family of models that converts Chinese ASR transcripts and other spoken-style text into
concise, natural written Chinese while preserving the original meaning. All variants are fine-tuned on
[`{dataset_id}`](https://huggingface.co/datasets/{dataset_id}) with the same task definition and training recipe.

## Model family

| Variant | Base model | Model repository |
| --- | --- | --- |
| 0.6B | `Qwen/Qwen3-0.6B` | [`{repo_owner}/Qwen3-0.6B-ASR-Refiner`](https://huggingface.co/{repo_owner}/Qwen3-0.6B-ASR-Refiner) |
| 1.7B | `Qwen/Qwen3-1.7B` | [`{repo_owner}/Qwen3-1.7B-ASR-Refiner`](https://huggingface.co/{repo_owner}/Qwen3-1.7B-ASR-Refiner) |
| 4B | `Qwen/Qwen3-4B` | [`{repo_owner}/Qwen3-4B-ASR-Refiner`](https://huggingface.co/{repo_owner}/Qwen3-4B-ASR-Refiner) |

The LoRA adapter has been merged into the base model. This repository contains complete BF16 Transformers weights and
can be loaded directly without PEFT.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = '{repo_id}'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype='auto', device_map='auto')

messages = [
    {{
        'role': 'system',
        'content': '将中文口语转写改写为正式、自然的书面语。保持原意，不添加原文没有的信息，只输出改写后的文本。',
    }},
    {{'role': 'user', 'content': '呃这个事情吧我们之后再讨论一下。'}},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(text, return_tensors='pt').to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
answer = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(answer)
```

The source dataset is licensed under CC BY 4.0. Refer to its dataset card for attribution and citation information.
'''


@click.command()
@click.option(
    '--base-model',
    required=True,
    help='Base model ID, for example Qwen/Qwen3-1.7B.',
)
@click.option(
    '--adapter-dir',
    type=click.Path(path_type=Path),
    help='LoRA adapter directory. Defaults to the model-specific artifacts directory.',
)
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    help='Merged model directory. Defaults to artifacts/<model>-ASR-Refiner.',
)
@click.option('--repo-id', help='Destination Hugging Face repository. Defaults to Aye10032/<model>-ASR-Refiner.')
def export(base_model: str, adapter_dir: Path | None, output_dir: Path | None, repo_id: str | None) -> None:
    """Merge the LoRA adapter into Qwen3 and publish the full BF16 model."""
    load_dotenv()

    if adapter_dir is None:
        adapter_dir = adapter_path(base_model)

    recorded_base_model = adapter_base_model(adapter_dir)
    if recorded_base_model is not None and recorded_base_model != base_model:
        raise click.UsageError(
            f'Adapter was trained from {recorded_base_model}, but --base-model is {base_model}. '
            'Use the matching base model.'
        )

    model_output_name = merged_model_name(base_model)
    output_dir = output_dir or Path('artifacts') / model_output_name
    repo_id = repo_id or f'{DEFAULT_REPO_OWNER}/{model_output_name}'
    metadata = read_training_metadata(adapter_dir)
    dataset_id = metadata.get('dataset_id', DEFAULT_DATASET_ID)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map='cpu',
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload(progressbar=True, safe_merge=True)
    model.config.use_cache = True

    model.save_pretrained(output_dir, max_shard_size='5GB')
    tokenizer.save_pretrained(output_dir)
    (output_dir / 'README.md').write_text(build_model_card(base_model, dataset_id, repo_id), encoding='utf-8')

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=output_dir,
        commit_message='Upload merged BF16 model',
    )
    click.echo(f'Published https://huggingface.co/{repo_id}')


if __name__ == '__main__':
    export()
