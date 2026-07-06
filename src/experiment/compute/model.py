import json
from pathlib import Path

import equinox as eqx
import jax.random as jr

from experiment.config import TrainingConfig
from experiment.model import LanguageModel, build_model
from experiment.training.metrics import TrainingMetrics
from utils.param_types import validate_call


@validate_call
def save_checkpoint(
    model: LanguageModel,
    config: TrainingConfig,
    metrics: TrainingMetrics | None,
    data_dir: Path,
) -> None:
    """Save a model checkpoint to the given directory.

    The file is a JSON header line (config and metrics) followed by the
    serialized model arrays.
    """
    model_path = data_dir / "model" / "checkpoint.eqx"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "config": config.model_dump(mode="json"),
        "metrics": metrics.model_dump(mode="json") if metrics else None,
    }
    with open(model_path, "wb") as f:
        f.write((json.dumps(header) + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


@validate_call
def load_checkpoint(data_dir: Path) -> tuple[LanguageModel, TrainingConfig, TrainingMetrics | None]:
    """Load a model checkpoint from the given directory."""
    model_path = data_dir / "model" / "checkpoint.eqx"
    with open(model_path, "rb") as f:
        header = json.loads(f.readline().decode())
        config = TrainingConfig.model_validate(header["config"])
        # Build a skeleton with the right structure, then fill in the saved arrays.
        model = build_model(config.model, key=jr.key(0))
        model = eqx.tree_deserialise_leaves(f, model)

    metrics = header.get("metrics", None)
    metrics = TrainingMetrics.model_validate(metrics) if metrics else None

    return model, config, metrics
