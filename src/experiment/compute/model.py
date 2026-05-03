from pathlib import Path

import torch

from experiment.config import TrainingConfig
from experiment.model.gpt import GPT
from experiment.training.metrics import TrainingMetrics
from utils.param_types import validate_call


@validate_call
def save_checkpoint(
    model: GPT,
    config: TrainingConfig,
    metrics: TrainingMetrics | None,
    data_dir: Path,
) -> None:
    """Save a model checkpoint to the given directory."""
    model_path = data_dir / 'model' / 'checkpoint.pt'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model': model.state_dict(),
        'config': config.model_dump(mode='json'),
        'metrics': metrics.model_dump(mode='json') if metrics else None,
    }
    torch.save(checkpoint, model_path)


@validate_call
def load_checkpoint(data_dir: Path) -> tuple[GPT, TrainingConfig, TrainingMetrics | None]:
    """Load a model checkpoint from the given directory."""
    model_path = data_dir / 'model' / 'checkpoint.pt'
    checkpoint = torch.load(model_path, map_location='cpu')
    config = TrainingConfig.model_validate(checkpoint['config'])
    model = GPT(config.model)
    model.load_state_dict(checkpoint['model'])

    metrics = checkpoint.get('metrics', None)
    metrics = TrainingMetrics.model_validate(metrics) if metrics else None

    return model, config, metrics
