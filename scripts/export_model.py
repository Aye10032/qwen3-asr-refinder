from pathlib import Path

import click
import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = 'Qwen/Qwen3-1.7B'
ADAPTER_DIR = Path('artifacts/qwen3-1.7b-asr-refinder-lora')
OUTPUT_DIR = Path('artifacts/Qwen3-1.7B-ASR-Refiner')
REPO_ID = 'Aye10032/Qwen3-1.7B-ASR-Refiner'
MODEL_CARD = '''---
base_model: Qwen/Qwen3-1.7B
library_name: transformers
pipeline_tag: text-generation
license: apache-2.0
language:
  - zh
datasets:
  - Aye10032/WenetSpeech-Formal-Text
tags:
  - qwen3
  - asr-post-processing
  - spoken-to-written
  - text-normalization
---

# Qwen3-1.7B ASR Refiner

This model converts Chinese ASR transcripts and other spoken-style text into concise, natural written Chinese while
preserving the original meaning. It was fine-tuned from
[`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) on
[`Aye10032/WenetSpeech-Formal-Text`](https://huggingface.co/datasets/Aye10032/WenetSpeech-Formal-Text).

The LoRA adapter has been merged into the base model. This repository contains complete BF16 Transformers weights and
can be loaded directly without PEFT.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = 'Aye10032/Qwen3-1.7B-ASR-Refiner'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype='auto', device_map='auto')

messages = [
    {
        'role': 'system',
        'content': '将中文口语转写改写为正式、自然的书面语。保持原意，不添加原文没有的信息，只输出改写后的文本。',
    },
    {'role': 'user', 'content': '呃这个事情吧我们之后再讨论一下。'},
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
@click.option('--adapter-dir', type=click.Path(path_type=Path), default=ADAPTER_DIR, show_default=True)
@click.option('--output-dir', type=click.Path(path_type=Path), default=OUTPUT_DIR, show_default=True)
@click.option('--repo-id', default=REPO_ID, show_default=True)
def export(adapter_dir: Path, output_dir: Path, repo_id: str) -> None:
    """Merge the LoRA adapter into Qwen3 and publish the full BF16 model."""
    load_dotenv()

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map='cpu',
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload(progressbar=True, safe_merge=True)
    model.config.use_cache = True

    model.save_pretrained(output_dir, max_shard_size='5GB')
    tokenizer.save_pretrained(output_dir)
    (output_dir / 'README.md').write_text(MODEL_CARD, encoding='utf-8')

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
