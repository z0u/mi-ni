"""
Shared, importable data preparation.

The corpus prep used to be copy-pasted into every notebook and experiment. It
lives here as one importable symbol so experiments *share* it: because the memo
key is content-addressed (the function's source + its args), the same
``prepare_data`` called with the same config fingerprints identically wherever
it runs. Once storage is project-scoped (see ``notes/storage-design.md``), that
identical key becomes a memo *hit* across experiments — prep runs once, ever.

    from experiment.corpus import prepare_data
    meta = app.run(prepare_data)            # notebook
    meta = ctx.run(prepare_data, role='prep')  # experiment DAG
"""

from __future__ import annotations

from mini import get_data_dir

# A HuggingFace mirror of Project Gutenberg; one big block of text.
PRIDE_AND_PREJUDICE_URL = (
    'https://huggingface.co/api/datasets/larenwell/book-gutenberg-train/parquet/default/train/0.parquet'
)


def download_pride_and_prejudice():
    """Download Pride and Prejudice from the Gutenberg HuggingFace dataset."""
    import ftfy
    import pandas as pd

    from experiment.config import DatasetMetadata

    df = pd.read_parquet(PRIDE_AND_PREJUDICE_URL, columns=['text'])
    text = df.iloc[0]['text']
    text, explanation = ftfy.fix_and_explain(text)
    metadata = DatasetMetadata(
        title='Pride and Prejudice',
        author='Jane Austen',
        url=PRIDE_AND_PREJUDICE_URL,
        fixes=explanation or [],
        total_chars=len(text),
    )
    return text, metadata


def prepare_data():
    """Download, tokenize, and save training data to the volume; return the corpus metadata."""
    from experiment.compute.data_pipelines import save_data
    from experiment.data.preparation import tokenize_data

    data_dir = get_data_dir()
    data, metadata = tokenize_data([download_pride_and_prejudice()])
    save_data(data, metadata, data_dir)
    return metadata
