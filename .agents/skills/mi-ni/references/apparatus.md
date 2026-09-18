mi-ni provides experiment infrastructure via the `Apparatus` class. Its interface is similar to an Executor:

```py
class Apparatus:
    volume: Volume
    """Storage available to functions run by this Apparatus."""

    def run(self, fn, *args, **kwargs) -> R:
        """Run a single function and return its result."""

    async def arun(self, fn, *args, **kwargs) -> R:
        """Run a single function and return its result, asynchronously."""

    def amap(self, fn, *iterables, kwargs) -> AsyncGenerator:
        """Map *fn* over one or more iterables."""

    def map(self, fn, *iterables, kwargs) -> Iterable:
        """Map *fn* over one or more iterables."""

    def before_each(self, hook) -> Apparatus:
        """Return a new Apparatus that runs *hook* before each job."""
```

An apparatus instance is usually named `app`. Usage:

```py
app = LocalApparatus("demo", max_workers=3)

# Step 1: write shared config to the volume
await app.arun(prep)

# Step 2: train (reads config, writes job output to volume, and returns metrics)
metrics = [x async for x in app.amap(train, [1, 2, 3, 4, 5])]

# Step 3: pull outputs back from the volume
await app.volume.download("outputs", f"/data/outputs")
```

The apparatus takes care of setting up the environment with Python packages and a volume to write to. To change the compute provider, just swap in another `Apparatus`, e.g. `ModalApparatus`.

## Selecting the backend at run time (reports)

A source-only script (one run from source rather than published, like `docs/gpt.py`) keeps the backend as a constant near the top, so switching is a one-line edit:

```py
APP_TYPE = "local"  # "local" or "modal" (pick modal for the GPU)
app = ModalApparatus("demo").w(gpu="L4") if APP_TYPE == "modal" else LocalApparatus("demo")
```

Then run it headless with `./go lit render docs/gpt.py`. A published report never launches compute at all: it reads results the experiment already produced (see `reports.md`). Confirm which backend actually ran from the logs: a Modal run prints `Creating Modal image …` then `Running … on Modal`; a local one prints `Running … locally`.

Always use the async methods `arun` and `amap` in a report and wherever there is an asynchronous context: Modal will complain otherwise. In other contexts, you can use the synchronous variants `run` and `map`, which are just wrappers provided for convenience.

Functions run by an apparatus can accept and return Python objects, as long as they can be pickled by cloudpickle. The function itself must also be pickleable, which means e.g. it must not close over things like file pointers. See the `modal` skill for more details.

Most context is passed in to the function explicitly, but the apparatus sets some global context variables — e.g. for progress reporting and volume configuration. Search for `contextvars` if you need to know more.
