import logging

import torch
from jaxtyping import Int

from experiment.config import CorpusMetadata, DatasetMetadata, TokenizerConfig
from experiment.data.tokenizer import CharTokenizer
from utils.param_types import validate_call

log = logging.getLogger(__name__)


@validate_call
def tokenize_data(sources: list[tuple[str, DatasetMetadata]]) -> tuple[Int[torch.Tensor, ' T'], CorpusMetadata]:
    text = ''.join(source[0] for source in sources)
    # Create character-level encoder/decoder specific to this dataset.
    config = TokenizerConfig(vocabulary=sorted(set(text)))
    tokenizer = CharTokenizer(config)

    # Tokenizer expects a batch
    log.info(f'Tokenizing {len(sources)} sources with {len(text)} characters')
    tokens = tokenizer.encode([text])[0]
    data = torch.tensor(tokens, dtype=torch.long)
    log.info(f'Tokenized {len(data)} tokens')

    metadata = CorpusMetadata(
        tokenizer_config=config,
        total_tokens=len(data),
        total_chars=len(text),
        sources=[source[1] for source in sources],
    )
    return data, metadata
