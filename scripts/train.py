import json
from pathlib import Path

import click
import torch
from accelerate import PartialState
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

DATASET_ID = 'Aye10032/WenetSpeech-Formal-Text'
SYSTEM_PROMPT = '将中文口语转写改写为正式、自然的书面语。保持原意，不添加原文没有的信息，只输出改写后的文本。'


def model_name(model_id: str) -> str:
    """Return the repository name portion of a Hugging Face model ID."""
    return model_id.rstrip('/').rsplit('/', 1)[-1]


def default_output_dir(model_id: str) -> Path:
    return Path('artifacts') / f'{model_name(model_id).lower()}-asr-refinder-lora'


def format_dataset(dataset: Dataset, tokenizer: AutoTokenizer, split: str) -> Dataset:
    def format_example(example: dict[str, str]) -> dict[str, str]:
        prompt_messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': example['original_text']},
        ]
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        text = tokenizer.apply_chat_template(
            [*prompt_messages, {'role': 'assistant', 'content': example['target_text']}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return {'prompt': prompt, 'completion': text[len(prompt) :]}

    return dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        desc=f'Formatting {split}',
    )


@click.command()
@click.option('--model-id')
@click.option('--dataset-id', default=DATASET_ID, show_default=True)
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    help='Adapter output directory. Defaults to artifacts/<model>-asr-refinder-lora.',
)
@click.option('--run-name', help='W&B run name. Defaults to <model>-lora.')
@click.option('--batch-size', type=int, default=4, show_default=True)
@click.option('--gradient-accumulation-steps', type=int, default=8, show_default=True)
@click.option('--max-train-samples', type=int)
@click.option('--max-steps', type=int, default=-1, show_default=True)
@click.option('--resume-from-checkpoint', type=click.Path(path_type=Path))
def train(
    model_id: str,
    dataset_id: str,
    output_dir: Path | None,
    run_name: str | None,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_train_samples: int | None,
    max_steps: int,
    resume_from_checkpoint: Path | None,
) -> None:
    """Fine-tune Qwen3 for spoken-to-written Chinese conversion."""
    load_dotenv()
    output_dir = output_dir or default_output_dir(model_id)
    run_name = run_name or f'{model_name(model_id).lower()}-lora'

    with PartialState().main_process_first():
        model_path = snapshot_download(model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_dataset = load_dataset(dataset_id, split='train')
    eval_dataset = load_dataset(dataset_id, split='validation')

    if max_train_samples is not None:
        train_dataset = train_dataset.select(range(max_train_samples))

    train_dataset = format_dataset(train_dataset, tokenizer, 'train')
    eval_dataset = format_dataset(eval_dataset, tokenizer, 'validation')

    training_args = SFTConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        model_init_kwargs={'dtype': torch.bfloat16, 'attn_implementation': 'sdpa'},
        bf16=True,
        tf32=True,
        max_length=256,
        packing=True,
        packing_strategy='wrapped',
        completion_only_loss=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=False,
        learning_rate=1e-4,
        lr_scheduler_type='cosine',
        warmup_steps=0.03,
        num_train_epochs=1,
        max_steps=max_steps,
        optim='adamw_torch_fused',
        logging_steps=10,
        eval_strategy='steps',
        eval_steps=100,
        save_strategy='steps',
        save_steps=100,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        report_to='wandb',
        ddp_find_unused_parameters=False,
        eos_token='<|im_end|>',
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules='all-linear',
    )
    trainer = SFTTrainer(
        model=model_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    # snapshot_download gives the trainer a local path. Restore the public model ID in
    # the PEFT metadata so downstream export can select and validate the right base.
    for peft_config in trainer.model.peft_config.values():
        peft_config.base_model_name_or_path = model_id

    trainer.train(resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None)
    trainer.save_model()
    if trainer.is_world_process_zero():
        metadata = {'model_id': model_id, 'dataset_id': dataset_id}
        (output_dir / 'training_metadata.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )


if __name__ == '__main__':
    train()
