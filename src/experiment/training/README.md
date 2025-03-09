## [Optimizer](optimizer.py)

Optimizer: Use an Adam with weight decay (AdamW) for SGD.

We only want to apply weight decay to weight matrices — not biases or scalar parameters. It so happens that we can distinguish these with a simple test:
- Weight matrices (e.g. in Linear and Embedding layers) are at least rank 2
- Bias and scale vectors are rank 1, and scalars[^scalars] are rank 0.

This may look fragile, but it's actually quite robust: it captures the fundamental difference between weight matrices and bias vectors without relying on naming conventions.

[^scalars]: In practice, we don't have any scalars.


## [Learning rate schedule](scheduler.py)

Linear warmup and then cosine decay, so that:
- The parameters don't get jolted at the start by a high learning rate
- Learning steps become finer as we approach the bottom of the basin in the loss landscape (although for simplicity we don't measure that, and we just assume that we'll be somewhere near the bottom of a basin near the final epoch).
