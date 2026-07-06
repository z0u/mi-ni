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

An appartus instance is usually named `app`. Usage:

```py
app = LocalApparatus("demo", max_workers=3)

# Step 1: write shared config to the volume
await app.arun(prep)

# Step 2: train (reads config, writes job output to volume, and returns metrics)
metrics = [x async for x in app.amap(train, [1, 2, 3, 4, 5])]

# Step 3: pull outputs back from the volume
await app.volume.download("outputs", f"/data/outputs")
```

The apparatus takes care of setting up the environment with Python packages and a volume to write to.
To change the compute provider, just swap in another `Apparatus`, e.g. `ModalApparatus`.

## Selecting the backend at run time (notebooks)

A notebook can make the backend a runtime choice rather than hard-coding it, so the same `.py` runs locally while you iterate and on Modal for the real run:

```py
app_type = mo.cli_args().get("app", "local")   # a marimo radio in edit mode; a CLI arg when headless
app = ModalApparatus("demo").w(gpu="L4") if app_type == "modal" else LocalApparatus("demo")
```

Export it headless with `./go export`, passing notebook options after a `--`:

```bash
./go export docs/gpt.py -- --app=modal --arch=ngpt
```

The `--` delimits notebook options from `./go`'s own args (and `./go` forwards them to `marimo export`). It's optional — `./go export docs/gpt.py --app=modal` also works — but explicit is clearer.

**Syntax gotcha:** the options are flags — marimo's `cli_args()` only parses `--key=value` or `--key value`. A bare `key=value` parses to *nothing*, so the notebook silently falls back to its default (here `local`) with no error. Confirm which backend actually ran from the logs — a Modal run prints `Creating Modal image …` then `Running … on Modal`; a local one prints `Running … locally`.

Always use the async methods `arun` and `amap` in Marimo notebooks and wherever there is an asynchronous context: Modal will complain otherwise. In other contexts, you can use the synchronous variants `run` and `map`, which are just wrappers provided for convenience.

Functions run by an apparatus can accept and return Python objects, as long as they can be pickled by cloudpickle. The function itself must also be pickleable, which means e.g. it must not close over things like file pointers. See the `modal` skill for more details.

Most context is passed in to the function explicitly, but the apparatus sets some global context variables — e.g. for progress reporting and volume configuration. Search for `contextvars` if you need to know more.
