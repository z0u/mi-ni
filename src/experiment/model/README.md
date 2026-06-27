# nGPT model architecture

This started as an implementation of the GPT-2 architecture, based on [nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/model.py) (Karpathy, 2022), with prose to make it easier for me to understand. It has since been converted to **nGPT** (Loshchilov et al., 2024), the _normalized_ transformer, which keeps every vector on the unit hypersphere so that meaning is carried purely by direction.

The change is more than cosmetic. In a normal transformer, layer norm rescales activations in a way that distorts their direction (see _Rotary positional encodings_, below). nGPT instead L2-normalizes everything — token embeddings, the residual stream, the attention queries and keys, and even the weight matrices — so a vector's length is fixed at 1 and only its angle can vary. The benefits reported in the paper are faster convergence and better length generalisation; for us, the appeal is that the geometry now matches the "direction-as-meaning" intuition we keep appealing to.

My key takeaways:

- The QKV values can be computed simultaneously, because V is the value that a token _would_ contribute if its key matched the query.
- The attention dimensions (QKV) are unrelated to the embedding dimension, but they are usually smaller than the embeddings.
- The sequence length is not an inhent part of the model. Originally GPT-2 used learned positional encodings, but this code uses RoPE — which makes it possible to extend the context length after training[^length].
- The output sequence $y$ is indeed the input sequence $x$ shifted by one. Therefore, the tokens are effectively shifted forward by the attention mechanism, and by the time they reach the final layer they have been shifted by one.


[^length]: Especially in conjunction with the normalization that forces model vectors to store only direction (i.e. unit length), as in nGPT — which is exactly what this model now does.


## [Attention](attention.py)

### Rotary positional encodings

Rotary positional encodings (RoPE) use quite a different scheme from earlier encodings schemes. GPT-2 (which nanoGPT replicates) uses learned embeddings, and the original Attention is All You Need paper used sinusoidal patterns — but in both cases, they were applied to the _input_ token embeddings. RoPE uses sinusoidal patterns, but they are used in attention space ($Q$, $K$) rather than input embedding space.

RoPE applies sequence position information _as rotation_ of the queries and keys, rather than being simply added to the token embeddings. This means the attention mechanism can learn positional relationships in a way that is more compatible with the "direction-as-meaning" embeddings. It also generalises to longer sequences than were seen during training. The original GPT-2 limited that generalisation by applying layer norm in each `Block`: layer norm is non-geometric, so it distorts the directional interpretation of the vectors. Because we have moved to nGPT, there is no layer norm to fight against — rotation and normalisation are both geometric operations on the sphere, so they compose cleanly.

In nGPT, the queries and keys are L2-normalized (per head) immediately after RoPE. Normalisation and rotation commute — rotating a unit vector leaves it a unit vector — so the order doesn't matter mathematically. Once normalized, a query·key dot product is just a cosine in the range $[-1, 1]$. That is too flat for softmax to be selective, so we scale it back up by a single learnable scalar temperature `s_qk` (initialised to the $\sqrt{d_k}$ that standard attention would use), the mirror image of the usual $1/\sqrt{d_k}$ down-scaling. nGPT proper learns this as a per-channel vector; we found a lone scalar is enough.

The particular rotation used in RoPE is interesting: the vector components are rotated in pairs, where each pair is considered to be a 2D vector on a plane. The planes are all distinct, so while the embedding as a whole can be considered to be a single direction vector, it's not rotated as a whole (around another axis with as many dimensions). I wonder if doing so would improve things further?


### Causal self-attention

This module takes in a sequence of token-level embeddings (either from the previous layer or from the input), allows them to communicate with each other, and outputs new embeddings. The transformed embeddings are similar to the inputs, in that they exist in the same latent space, but each one now contains some context from tokens earlier in the sequence (Sanderson, 2024a). For example, if "the blue chair" was three tokens, then after passing through a self-attention layer the embeddings could have more nuanced meanings such as "the", "blue (as an adjective)", "chair (which is blue)".

#### Attention as information retrieval

The attention mechanism works like a look-up table (LUT), where each token embedding can be used to "look up" contextual information. But unlike a regular LUT, this one: 1. Is computed on the fly based on the incoming token embedding, and 2. Looks up information from the entire sequence at once, weighting each earlier contribution by how much it is relevant to the current token.

We first convert incoming embeddings to queries, keys and values ($q$, $k$, and $v$, per token):

- Queries _ask_ "Which earlier[^causal] tokens are relevant to me?"
- Keys _match_ "My token is relevant to later[^causal] queries that are like me"
- Values _offer_ "_If_ a later[^causal] query matches my key, here's the context I can provide..." (Sanderson, 2024a).

It's surprising that the value can be computed up-front, even before the queries have been compared to other tokens' keys! It's possible because the context that a token would provide does not depend what the query is — but the extent to which that value influences the output will depend on how closely the key matches the query.

All of this happens in a different latent space than the token embeddings. In particular:
- $Q$ and $K$ must share the same latent space, so that queries can be compared to keys. It need not be the same length as the token embeddings, and is usually much smaller.
- $V$ has its own latent space. It need not be the same length as the token embeddings or $Q$ and $K$, although in practice it's usually the same length as $Q$ and $K$.

#### Attention heads

Reading the previous section, you might think that each token can only provide one piece of contextual information to later tokens. We get around that by doing the same thing multiple times, and then combining the results. We call the logical Q-K-V operation an "attention head", and package them up into an internal batch dimension called "head" so they can be computed in parallel.

[^causal]: This is for causal self-attention. For cross-attention, all other tokens are queried.


## [MLP](mlp.py)

The multilayer perceptron (MLP) (aka feedforward layer) is an OG[^og] deep learning pattern for nonlinear learned data transformations. That doesn't tell us much about what it does here but [3b1b has a great video on it](https://www.3blue1brown.com/lessons/mlp) (Sanderson, 2024). The structure of an MLP in a transformer does this:

1. Projects the input to a larger vector (arbitrarily 4x the embedding size)
2. Applies GELU activation to introduce non-linearity, i.e. curves and bends that let the network learn more complex patterns, like a smooth switch.
3. Projects back down to the original embedding size, to be compatible with later layers.

There is one nGPT wrinkle worth calling out. The input to the MLP is a unit vector, and (after normalisation) so are the rows of the up-projection. Their dot products therefore land around $1/\sqrt{d}$ — close enough to zero that GELU would behave like a straight line, and the non-linearity we went to the trouble of adding would do nothing. To fix this we multiply the up-projection by a $\sqrt{n_\text{embd}}$ baseline (and a single learnable scalar `s_u`), pushing the pre-activations out to roughly unit scale where GELU's curve actually bends. nGPT proper uses a SwiGLU here with its own gate scaling; we keep the simpler GELU and just restore its operating range.

[^og]: From 1958!

### What it does

Exactly what is going on in the large matrices that project up and then back down is anyone's guess. But essentially, it takes each embedding from the attention block — which by now contains contextual information from earlier embeddings in the sequence — and it _does knowledge_ to it. That is, it looks at the embedding and says something about it that it thinks will be useful for later layers. Note the subtlety in that last sentence: "something" is another embedding (that has meaning) and "will be useful" are both things that the MLP learned during training.

### What it doesn't

The MLP operates on each token embedding individually: at this point, there is no communication between tokens; that happens in the `CausalSelfAttention` module. Also, it does not _add_ knowledge to the embedding; it outputs an entirely new embedding, which is then added to the residual stream in the `Block`.


## [Transformer block](block.py)

A transformer "block":
1. Uses multi-headed attention to pass context between tokens (see _Causal self-attention_, above)
2. Adds the attention output back into the residual stream and re-normalizes (a normalized residual update)
3. Adds the knowledge it has learned to the contextualized tokens with a simple feed-forward network (see _Multilayer perceptron_, above)
4. Adds the MLP output back in and re-normalizes (another normalized residual update)

Because the hidden state arrives already normalized, there is no layer norm before each sub-module — the unit-length constraint does the job that layer norm used to.

These blocks are then stacked as "layers"; see _GPT_ below.

### Residual connection on the sphere

In a standard transformer the residual update is a plain addition: `h ← h + attn(h)`. That has a couple of motivations. Numerically, it lets gradients propagate all the way back through the network; without it, earlier layers would struggle to learn anything. What's more interesting is that it allows some of the original information to flow past each layer — otherwise the meaning of the token would be completely replaced by whatever the attention mechanism thought was useful, which may starve later layers of the information _they_ need.

nGPT keeps that spirit but reframes it geometrically. Adding two unit vectors generally lands you off the sphere, so after a _gated_ step we re-normalize, projecting straight back onto it:

$$h \leftarrow \text{Norm}\big(h + \alpha\,\text{sub}(h)\big)$$

The gate $\alpha$ is a single learnable scalar per sub-module (a ReZero/LayerScale-style gate). It matters more than it looks. With unit-norm weights and a unit-norm input, $\text{sub}(h)$ already has norm $\approx 1$ — the same scale as the residual — so an _ungated_ sum would rotate the hidden state roughly 45° per layer and overwrite the token's identity within a few layers. The small gate (initialised at 0.05) holds each step to a few degrees, letting information flow past each layer the way the additive residual does in a standard transformer.

The full nGPT recipe goes one step further: it takes the step _towards_ the sub-module's normalized output, $h \leftarrow \text{Norm}\big(h + \alpha\,(h_\text{sub} - h)\big)$, with $\alpha$ a learned per-**channel** vector (the paper's _eigen learning rates_). We deliberately strip that back to a scalar. The [notebook sweep](../../../docs/gpt-sweep/report.py) shows the simplification is free at this scale: the scalar gate matches per-channel nGPT, while dropping the gate entirely (a raw additive retraction) breaks training outright. The learned gates also settle near $1/n_\text{layer}$ on average — the value that keeps the twelve-layer stack moving an $O(1)$ distance overall.


## [Decoder-only transformer](gpt.py)

Here we tie together all the previous modules into "the transformer". Having discussed those other pieces already, this part is straightforward. The GPT module:

1. Prepares the input embeddings:
   - Converts token indices to the learned embeddings with a literal look-up table, then normalizes each one onto the unit hypersphere so the residual stream starts on the sphere
   - Position is _not_ added here; it enters later as rotation inside attention (see _Rotary positional encodings_, above)
2. Pushes those embeddings through the transformer blocks (layers), one after the other
3. Projects the (already unit-length) output embeddings to logits, scaled by a single learnable temperature `s_z`

Because the hidden state is a unit vector by the time it reaches the head, a raw logit is again just a cosine in $[-1, 1]$ — `s_z` restores the dynamic range that the final softmax needs, exactly as `s_qk` did inside attention. (nGPT makes this a per-class vector; like the other scales we collapse it to one scalar.)

### Keeping the weights on the sphere

Normalizing the _activations_ is only half of nGPT. The weight matrices are constrained too: after every optimizer step we re-normalize each matrix that reads from or writes to the residual stream, so its rows (or columns) are unit vectors. This is enforced in the training loop rather than the forward pass — `model = model.normalize_weights()` after each optimizer step — and it means the matrices act as rotations on the sphere rather than arbitrary linear maps. A pleasant side effect is that weight decay becomes redundant: the norm is already pinned to 1, so there is nothing for decay to shrink.

### Interpretation of the logits

During training, the "labels" $y$ are the same as the input sequence $x$ but shifted by one (so that $y_t=x_{t+1}$). The output logits have a length equal to the vocabulary size[^vocab], so we can interpret them as meaning "the likelihood that token $k$ comes next". This isn't intrinsic to logits: we cause them to have this meaning when we define the loss function. However, the loss is not an intrinsic part of the model, and it's actually specified in the training code further down.

With the logits defined like this, generation is simple: we could just pick the most likely token (corresponding to the logit with the highest value) and output that. That would produce fairly formulaic text, though, so we convert the logits to a probability distribution (using softmax) and sample from it.

[^vocab]: I.e. the number of distinct token values.
