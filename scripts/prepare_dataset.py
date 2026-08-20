import json
from pathlib import Path

import click
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import get_session, hf_raise_for_status
from tqdm.auto import tqdm

SOURCE_DATASET = 'TaurenMountain/WenetSpeech-Formal'
TARGET_DATASET = 'Aye10032/WenetSpeech-Formal-Text'
SPLITS = ('train', 'validation', 'test')
DEFAULT_OUTPUT_DIR = Path('artifacts/WenetSpeech-Formal-Text')
DATASET_CARD = '''---
license: cc-by-4.0
language:
  - zh
pretty_name: WenetSpeech-Formal Text Only
tags:
  - chinese
  - asr-post-processing
  - spoken-to-written
  - text-normalization
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.parquet
      - split: validation
        path: data/validation.parquet
      - split: test
        path: data/test.parquet
---

# WenetSpeech-Formal Text Only

This is a lightweight, text-only derivative of
[`TaurenMountain/WenetSpeech-Formal`](https://huggingface.co/datasets/TaurenMountain/WenetSpeech-Formal).
It removes the audio and retains the existing `original_text` and `target_text` columns without modification.

Source revision: `{source_revision}`

## Statistics

| Split | Rows |
| --- | ---: |
| train | {train_rows} |
| validation | {validation_rows} |
| test | {test_rows} |
| **Total** | **{total_rows}** |

The complete text-only Parquet dataset is approximately {total_size_mib} MiB.

## Usage

```python
from datasets import load_dataset

dataset = load_dataset('Aye10032/WenetSpeech-Formal-Text')
print(dataset['train'][0])
```

## Source, license, and citation

This is a format-only derivative and does not introduce new annotations, methodology, or research claims.
There is no separate citation requested for this text-only mirror.

The source dataset is released under CC BY 4.0. For research use, follow the attribution and citation guidance on
the original [`TaurenMountain/WenetSpeech-Formal`](https://huggingface.co/datasets/TaurenMountain/WenetSpeech-Formal)
dataset page.
'''


def read_text_shard(revision: str, filename: str) -> pa.Table:
    url = hf_hub_url(SOURCE_DATASET, filename, repo_type='dataset', revision=revision)
    response = get_session().head(url, follow_redirects=True)
    hf_raise_for_status(response)

    connection = duckdb.connect()
    connection.execute('LOAD httpfs')
    connection.execute('SET allow_asterisks_in_http_paths = true')
    table = connection.execute(
        'SELECT original_text, target_text FROM read_parquet(?)',
        [str(response.url)],
    ).to_arrow_table()
    connection.close()
    return table


def write_metadata(output_dir: Path, revision: str) -> None:
    split_stats = {
        split: {
            'rows': pq.ParquetFile(output_dir / 'data' / f'{split}.parquet').metadata.num_rows,
            'size_bytes': (output_dir / 'data' / f'{split}.parquet').stat().st_size,
            'shards': 1,
        }
        for split in SPLITS
    }
    metadata = {
        'source_dataset': SOURCE_DATASET,
        'source_revision': revision,
        'columns': ['original_text', 'target_text'],
        'splits': split_stats,
        'total_rows': sum(stats['rows'] for stats in split_stats.values()),
        'total_size_bytes': sum(stats['size_bytes'] for stats in split_stats.values()),
    }
    (output_dir / 'provenance.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )

    card = DATASET_CARD.format(
        source_revision=revision,
        train_rows=f"{split_stats['train']['rows']:,}",
        validation_rows=f"{split_stats['validation']['rows']:,}",
        test_rows=f"{split_stats['test']['rows']:,}",
        total_rows=f"{metadata['total_rows']:,}",
        total_size_mib=f"{metadata['total_size_bytes'] / 1024**2:.1f}",
    )
    (output_dir / 'README.md').write_text(card, encoding='utf-8')


@click.group()
def cli() -> None:
    """Prepare and publish the text-only dataset."""


@cli.command()
@click.option('--output-dir', type=click.Path(path_type=Path), default=DEFAULT_OUTPUT_DIR, show_default=True)
def prepare(output_dir: Path) -> None:
    """Download the two text columns and write one Parquet file per split."""
    info = HfApi().dataset_info(SOURCE_DATASET)
    filenames = sorted(sibling.rfilename for sibling in info.siblings if sibling.rfilename.endswith('.parquet'))
    data_dir = output_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    installer = duckdb.connect()
    installer.execute('INSTALL httpfs')
    installer.close()

    for split in SPLITS:
        split_files = [filename for filename in filenames if filename.startswith(f'{split}-')]
        tables = [
            read_text_shard(info.sha, filename)
            for filename in tqdm(
                split_files,
                desc=f'Downloading {split}',
                unit='shard',
                dynamic_ncols=True,
            )
        ]
        pq.write_table(
            pa.concat_tables(tables),
            data_dir / f'{split}.parquet',
            compression='zstd',
            compression_level=6,
            row_group_size=100_000,
        )

    write_metadata(output_dir, info.sha)
    click.echo(f'Prepared dataset at {output_dir}')


@cli.command()
@click.option('--output-dir', type=click.Path(path_type=Path), default=DEFAULT_OUTPUT_DIR, show_default=True)
@click.option('--repo-id', default=TARGET_DATASET, show_default=True)
def upload(output_dir: Path, repo_id: str) -> None:
    """Upload the prepared dataset to Hugging Face."""
    api = HfApi()
    api.create_repo(repo_id, repo_type='dataset', exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type='dataset',
        folder_path=output_dir,
        allow_patterns=['README.md', 'provenance.json', 'data/*.parquet'],
        commit_message='Update text-only WenetSpeech-Formal dataset',
    )
    click.echo(f'Published https://huggingface.co/datasets/{repo_id}')


@cli.command(name='all')
@click.option('--output-dir', type=click.Path(path_type=Path), default=DEFAULT_OUTPUT_DIR, show_default=True)
@click.option('--repo-id', default=TARGET_DATASET, show_default=True)
def prepare_and_upload(output_dir: Path, repo_id: str) -> None:
    """Prepare the dataset and upload it."""
    ctx = click.get_current_context()
    ctx.invoke(prepare, output_dir=output_dir)
    ctx.invoke(upload, output_dir=output_dir, repo_id=repo_id)


if __name__ == '__main__':
    cli()
