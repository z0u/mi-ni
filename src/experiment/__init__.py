import os

# XLA:CPU's concurrency-optimized scheduler trades memory for parallelism,
# which lets the O(T²) attention buffers of every layer be live at once —
# easily exceeding the RAM of a modest dev machine. Prefer the memory-frugal
# scheduler. Takes effect when the XLA backend initializes (i.e. on first use
# of jax after this package is imported); no effect on GPU/TPU.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_enable_concurrency_optimized_scheduler=false")
