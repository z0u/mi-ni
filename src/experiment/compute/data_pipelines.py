import torch

from experiment.compute.app import data_dir
from experiment.config import CorpusMetadata
from utils.param_types import validate_call

prepared_data_path = data_dir / 'processed'


@validate_call
def save_data(data: torch.Tensor, metadata: CorpusMetadata):
    prepared_data_path.mkdir(parents=True, exist_ok=True)
    torch.save(data, prepared_data_path / 'tokenized.pt')
    with open(prepared_data_path / 'metadata.json', 'w') as f:
        f.write(metadata.model_dump_json())


@validate_call
def load_data() -> tuple[torch.Tensor, CorpusMetadata]:
    data = torch.load(prepared_data_path / 'tokenized.pt')
    with open(prepared_data_path / 'metadata.json', 'r') as f:
        metadata = CorpusMetadata.model_validate_json(f.read())
    return data, metadata
