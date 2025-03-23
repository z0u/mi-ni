from experiment.config import TokenizerConfig
from utils.param_types import validate_call


class CharTokenizer:
    """A simple character-level tokenizer."""

    @validate_call
    def __init__(self, config: TokenizerConfig):
        # Use the empty string as a padding token
        vocabulary = {''} | set(config.vocabulary)
        self.vocabulary = sorted(vocabulary)
        self.vocab_size = len(self.vocabulary)
        self.stoi = {ch: i for i, ch in enumerate(self.vocabulary)}
        self.itos = {i: ch for i, ch in enumerate(self.vocabulary)}

    @classmethod
    @validate_call
    def from_string(cls, string: str):
        """Create a tokenizer from a string."""
        vocabulary = set(string)
        return cls(TokenizerConfig(vocabulary=sorted(vocabulary)))

    @validate_call
    def encode(self, texts: list[str], block_size: int | None = None) -> list[list[int]]:
        """Encode a batch of texts into token sequences, padded to the same length."""
        tokens = []
        for t in texts:
            ts = [self.stoi[c] for c in t]
            tokens.append(ts[:block_size])

        # Pad each sequence with zeros to make uniform length
        if block_size is not None:
            max_len = block_size
        else:
            max_len = max((len(ts) for ts in tokens), default=0)
        return [[0] * (max_len - len(ts)) + ts for ts in tokens]

    @validate_call
    def decode_each(self, tokens: list[list[int]]) -> list[list[str]]:
        """Decode a batch of tokens, returning a batch of individual decoded tokens (string fragments)."""
        decoded = []
        for ts in tokens:
            decoded.append([self.itos.get(i, '') for i in ts])
        return decoded

    @validate_call
    def decode(self, tokens: list[list[int]]) -> list[str]:
        """Decode a batch of tokens, returning a batch of fully-decoded strings."""
        return [''.join(t) for t in self.decode_each(tokens)]
