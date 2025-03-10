from ftfy import ExplanationStep
from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt

from utils.param_types import IntX8, IntX32, IntX64, ZeroToOne


class ModelConfig(BaseModel, validate_assignment=True):
    vocab_size: IntX64
    """Vocabulary size"""

    block_size: IntX64
    """Maximum sequence length"""

    n_embd: IntX8
    """Embedding dimension"""

    n_head: IntX8
    """Number of attention heads per layer"""

    n_head_dim: IntX8
    """QKV dimension per-head, usually n_embd // n_head"""

    n_ff: IntX32
    """MLP dimensions, usually 4 * n_embd"""

    n_layer: PositiveInt
    """Number of transformer blocks"""

    dropout: ZeroToOne
    """Dropout rate"""


class DataConfig(BaseModel, validate_assignment=True):
    batch_size: PositiveInt
    """Batch size per iteration"""

    oversample: PositiveFloat
    """Increase the number of training samples per epoch by this factor"""

    train_split: ZeroToOne
    """Fraction of data to use for training"""


class TokenizerConfig(BaseModel, validate_assignment=True):
    vocabulary: list[str]
    """Unordered list of distinct tokens in the vocabulary"""

    @property
    def vocab_size(self) -> int:
        """Number of distinct tokens in the vocabulary"""
        return len(self.vocabulary)


class DatasetMetadata(BaseModel, validate_assignment=True):
    title: str

    author: str | None = None

    url: str | None = None
    """Where the dataset was downloaded from"""

    fixes: list[ExplanationStep]
    """List of fixes applied to the dataset"""

    total_chars: NonNegativeInt
    """Total number of characters in the dataset"""

    language: str | None = None
    """Language of the dataset"""


class CorpusMetadata(BaseModel, validate_assignment=True):
    tokenizer_config: TokenizerConfig
    """The tokenizer configuration used to encode the corpus"""

    total_tokens: NonNegativeInt
    """Total number of tokens in the corpus"""

    total_chars: NonNegativeInt
    """Total number of characters in the corpus"""

    sources: list[DatasetMetadata]
    """List of sources for the corpus"""


class OptimizerConfig(BaseModel, validate_assignment=True):
    weight_decay: ZeroToOne
    """Weight decay rate"""

    learning_rate: ZeroToOne
    """Learning rate"""

    betas: tuple[ZeroToOne, ZeroToOne]
    """Betas for the Adam optimizer"""


class SchedulerConfig(BaseModel, validate_assignment=True):
    epochs: PositiveInt
    """Number of epochs to train for"""

    warmup_epochs: NonNegativeFloat
    """Number of epochs to reach max learning rate"""

    min_lr_factor: ZeroToOne
    """Minimum learning rate as factor of the nominal learning rate"""


class TrainingConfig(BaseModel, validate_assignment=True):
    model: ModelConfig
    tokenizer: TokenizerConfig
    data: DataConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
