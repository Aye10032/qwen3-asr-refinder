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

MODEL_ID = 'Qwen/Qwen3-1.7B'
DATASET_ID = 'Aye10032/WenetSpeech-Formal-Text'
OUTPUT_DIR = Path('artifacts/qwen3-1.7b-asr-refinder-lora')
SYSTEM_PROMPT = '将中文口语转写改写为正式、自然的书面语。保持原意，不添加原文没有的信息，只输出改写后的文本。'


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
@click.option('--model-id', default=MODEL_ID, show_default=True)
@click.option('--dataset-id', default=DATASET_ID, show_default=True)
@click.option('--output-dir', type=click.Path(path_type=Path), default=OUTPUT_DIR, show_default=True)
@click.option('--batch-size', type=int, default=4, show_default=True)
@click.option('--max-train-samples', type=int)
@click.option('--max-steps', type=int, default=-1, show_default=True)
@click.option('--resume-from-checkpoint', type=click.Path(path_type=Path))
def train(
    model_id: str,
    dataset_id: str,
    output_dir: Path,
    batch_size: int,
    max_train_samples: int | None,
    max_steps: int,
    resume_from_checkpoint: Path | None,
) -> None:
    """Fine-tune Qwen3 for spoken-to-written Chinese conversion."""
    load_dotenv()

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
        run_name='qwen3-1.7b-lora',
        model_init_kwargs={'dtype': torch.bfloat16, 'attn_implementation': 'sdpa'},
        bf16=True,
        tf32=True,
        max_length=256,
        packing=True,
        packing_strategy='wrapped',
        completion_only_loss=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8,
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
    trainer.train(resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None)
    trainer.save_model()


if __name__ == '__main__':
    train()
