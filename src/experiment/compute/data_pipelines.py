from pathlib import Path

import torch

from experiment.config import CorpusMetadata
from utils.param_types import validate_call


@validate_call
def save_data(data: torch.Tensor, metadata: CorpusMetadata, data_dir: Path):
    """Save tokenized data and metadata to the given directory."""
    prepared = data_dir / 'processed'
    prepared.mkdir(parents=True, exist_ok=True)
    torch.save(data, prepared / 'tokenized.pt')
    (prepared / 'metadata.json').write_text(metadata.model_dump_json())


@validate_call
def load_data(data_dir: Path) -> tuple[torch.Tensor, CorpusMetadata]:
    """Load tokenized data and metadata from the given directory."""
    prepared = data_dir / 'processed'
    data = torch.load(prepared / 'tokenized.pt')
    metadata = CorpusMetadata.model_validate_json((prepared / 'metadata.json').read_text())
    return data, metadata
