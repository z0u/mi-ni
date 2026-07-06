from pathlib import Path

import numpy as np
from jaxtyping import Int

from experiment.config import CorpusMetadata
from utils.param_types import validate_call


@validate_call
def save_data(data: Int[np.ndarray, " T"], metadata: CorpusMetadata, data_dir: Path):
    """Save tokenized data and metadata to the given directory."""
    prepared = data_dir / "processed"
    prepared.mkdir(parents=True, exist_ok=True)
    np.save(prepared / "tokenized.npy", data)
    (prepared / "metadata.json").write_text(metadata.model_dump_json())


@validate_call
def load_data(data_dir: Path) -> tuple[Int[np.ndarray, " T"], CorpusMetadata]:
    """Load tokenized data and metadata from the given directory."""
    prepared = data_dir / "processed"
    data = np.load(prepared / "tokenized.npy")
    metadata = CorpusMetadata.model_validate_json((prepared / "metadata.json").read_text())
    return data, metadata
