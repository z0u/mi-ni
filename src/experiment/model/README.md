# GPT-2 model architecture

This is an implementation of the GPT-2 architecture. It's based on [nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/model.py) (Karpathy, 2022) with modifications and prose to make it easier for me to understand.

My key takeaways:

- The QKV values can be computed simultaneously, because V is the value that a token _would_ contribute if its key matched the query.
- The attention dimensions (QKV) are unrelated to the embedding dimension, but they are usually smaller than the embeddings.
- The sequence length is not an inhent part of the model. Originally GPT-2 used learned positional encodings, but this code uses RoPE — which makes it possible to extend the context length after training[^length].
- The output sequence $y$ is indeed the input sequence $x$ shifted by one. Therefore, the tokens are effectively shifted forward by the attention mechanism, and by the time they reach the final layer they have been shifted by one.


[^length]: Especially in conjunction normalization that forces model vectors only store direction (i.e. unit length) as in nGPT.


## [Attention](attention.py)

### Rotary positional encodings

Rotary positional encodings (RoPE) use quite a different scheme from earlier encodings schemes. GPT-2 (which nanoGPT replicates) uses learned embeddings, and the original Attention is All You Need paper used sinusoidal patterns — but in both cases, they were applied to the _input_ token embeddings. RoPE uses sinusoidal patterns, but they are used in attention space ($Q$, $K$) rather than input embedding space.

Crucially, RoPE applies sequence position information _as rotation_ of the queries and keys, rather than being simply added to the token embeddings. This means the attention mechanism can learn positional relationships in a way that is more compatible with the "direction-as-meaning" embeddings. It also generalises to longer sequences than were seen during training. However, that generalisation is still limited by the kind of normalisation that is applied in the `Block` (layer norm): layer norm is non-geometric, so it distorts the directional interpretation of the vectors.

The particular rotation used in RoPE is interesting: the vector components are rotated in pairs, where each pair is considered to be a 2D vector on a plane. The planes are all distinct, so while the embedding as a whole can be considered to be a single direction vector, it's not rotated as a whole (around another axis with as many dimensions). I wonder if doing so would improve things further?


### Causal self-attention

This module takes in a sequence of token-level embeddings (either from the previous layer or from the input), allows them to communicate with each other, and outputs new embeddings. The transformed embeddings are similar to the inputs, in that they exist in the same latent space, but each one now contains some context from tokens earlier in the sequence (Sanderson, 2024a). For example, if "the blue chair" was three tokens, then after passing through a self-attention layer the embeddings could have more nuanced meanings such as "the", "blue (as an adjective)", "chair (which is blue)".

#### Attention as information retrieval

The attention mechanism works like a look-up table (LUT), where each token embedding can be used to "look up" contextual information. But unlike a regular LUT, this one looks up information from the entire sequence at once, weighting each earlier contribution by how much it is relevant to the current token. To do this, we first convert incoming embeddings to queries, keys and values ($q$, $k$, and $v$, per token):

- Queries _ask_ "Which earlier[^causal] tokens are relevant to me?"
- Keys _match_ "My token is relevant to later[^causal] queries that are like me"
- Values _offer_ "_If_ a later[^causal] query matches my key, here's the context I can provide..." (Sanderson, 2024a).

It's surprising that the value can be computed up-front, even before the queries have been compared to other tokens' keys! It's possible because the context that a token would provide is always the same, no matter what the query is — but the extent to which that value will be influence the output will depend on how closely the key matches the query.

All of this happens in a different latent space than the token embeddings. In particular:
- $Q$ and $K$ must share the same latent space, so that queries can be compared to keys. It need not be the same length as the token embeddings, and is usually much smaller.
- $V$ has its own latent space. It need not be the same length as the token embeddings or $Q$ and $K$, although in practice it's usually the same length as $Q$ and $K$.

#### Attention heads

Reading the previous section, you might think that each token can only provide one piece of contextual information to later tokens. We get around that by doing the same thing multiple times, and then combining the results. We call the logical Q-K-V operation an "attention head", and package them up into an internal batch dimension called "head" so they can be computed in parallel.

[^causal]: This is for causal self-attention. For cross-attention, all other tokens are queried.


## [MLP](mlp.py)

The multilayer perceptron (MLP) (aka feedforward layer) is an OG[^og] deep learning pattern for nonlinear learned data transformations. That doesn't tell us much about what it does here but [3b1b has a great video on it](https://www.3blue1brown.com/lessons/mlp) (Sanderson, 2024). The structure of an MLP in a transformer does this:

1. First, it projects the input to a larger dimension (arbitrarily 4x the embedding size)
2. Apply GELU activation to introduce non-linearity, i.e. curves and bends that let the network learn more complex patterns, like a smooth switch.
3. Project back down to the original embedding size, to be compatible with later layers.

[^og]: From 1958!

### What it does

Exactly what is going on in the large matrices that project up and then back down is anyone's guess. But essentially, it takes each embedding from the attention block — which by now contains contextual information from earlier embeddings in the sequence — and it _does knowledge_ to it. That is, it looks at the embedding and says something about it that it thinks will be useful for later layers. Note the subtlety in that last sentence: "something" is another embedding (that has meaning) and "will be useful" are both things that the MLP learned during training.

### What it doesn't

The MLP operates on each token embedding individually: at this point, there is no communication between tokens; that happens in the `CausalSelfAttention` module. Also, it does not _add_ knowledge to the embedding; it outputs an entirely new embedding, which is later added to the residual stream in the `Block`.


## [Transformer block](block.py)

A transformer "block":
1. Uses multi-headed attention to pass context between tokens (see _Causal self-attention_, above)
2. Adds the attention output back to the original embeddings (residual connection)
3. Adds the knowledge it has learned to the contextualized tokens with a simple feed-forward network (see _Multilayer perceptron_, above)
4. Adds the MLP output back to its input (another residual connection)

Layer normalization is applied before each sub-module to keep activations well-behaved during training.

These blocks are then stacked as "layers"; see _GPT_ below.

### Residual connection

The residual stream has a couple of motivations. Numerically, it's needed so that the gradients can propagate all the way back through the network. Without it, earlier layers would struggle to learn anything. What's more interesting is that it allows some of the original information to flow past each layer — otherwise, the meaning of the token would be completely replaced by whatever the attention mechanism thought was useful, which may starve later layers of the information _they_ need.


## [Decoder-only transformer](gpt.py)

Here we tie together all the previous modules into "the transformer". Having discussed those other pieces already, this part is straightforward. The GPT module:

1. Prepares the input embeddings:
   - Converts token indices to the learned embeddings with a literal look-up table
   - Adds additional embeddings for the position of each token within the sequence (elementwise addition, i.e. without any extra embedding dimensions)
2. Pushes those embeddings through the transformer blocks (layers), one after the other
3. Normalizes and projects the output embeddings to logits.

### Interpretation of the logits

During training, the "labels" $y$ are the same as the input sequence $x$ but shifted by one (so that $y_t=x_{t+1}$). The output logits have a length equal to the vocabulary size[^vocab], so we can interpret them as meaning "the likelihood that token $k$ comes next". This isn't intrinsic to logits: we cause them to have this meaning when we define the loss function. However, the loss is not an intrinsic part of the model, and it's actually specified in the training code further down.

With the logits defined like this, generation is simple: we could just pick the most likely token (corresponding to the logit with the highest value) and output that. That would produce fairly formulaic text, though, so we convert the logits to a probability distribution (using softmax) and sample from it.

[^vocab]: I.e. the number of distinct token values.
