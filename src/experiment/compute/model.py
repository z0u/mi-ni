import torch

from experiment.compute.app import data_dir
from experiment.config import TrainingConfig
from experiment.model.gpt import GPT
from experiment.training.metrics import TrainingMetrics
from utils.param_types import validate_call

model_path = data_dir / 'model' / 'checkpoint.pt'


@validate_call
def save_checkpoint(
    model: GPT,
    config: TrainingConfig,
    metrics: TrainingMetrics | None = None,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model': model.state_dict(),
        'config': config.model_dump(mode='json'),
        'metrics': metrics.model_dump(mode='json') if metrics else None,
    }
    torch.save(checkpoint, model_path)


@validate_call
def load_checkpoint() -> tuple[GPT, TrainingConfig, TrainingMetrics | None]:
    checkpoint = torch.load(model_path, map_location='cpu')
    config = TrainingConfig.model_validate(checkpoint['config'])
    model = GPT(config.model)
    model.load_state_dict(checkpoint['model'])

    metrics = checkpoint.get('metrics', None)
    metrics = TrainingMetrics.model_validate(metrics) if metrics else None

    return model, config, metrics
