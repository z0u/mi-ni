"""Model variants, selected by `ModelConfig.architecture`.

- `gpt`: baseline pre-norm transformer with additive residuals (the default).
- `ngpt`: normalized transformer on the unit hypersphere ('crude' / 'full').

Both share the primitives in `_shared` and subclass `LanguageModel`.
"""

from jaxtyping import PRNGKeyArray

from experiment.config import ModelConfig
from experiment.model._shared import Generation, LanguageModel, SingleGeneration
from experiment.model.gpt import GPT
from experiment.model.ngpt import NGPT

__all__ = ["GPT", "NGPT", "Generation", "LanguageModel", "SingleGeneration", "build_model"]


def build_model(config: ModelConfig, *, key: PRNGKeyArray) -> LanguageModel:
    """Construct the model variant named by `config.architecture`."""
    match config.architecture:
        case "gpt":
            return GPT(config, key=key)
        case "ngpt":
            return NGPT(config, key=key)
