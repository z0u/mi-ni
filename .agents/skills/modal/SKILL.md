---
name: modal
description: Guidelines for using Modal in this project. Patterns for distributed processing, remote execution, and handling large objects. How to write functions that can be run both locally and remotely.
---

### Modal

Modal is a serverless platform for running Python code on managed cloud infrastructure.

Use Modal-compatible patterns for distributed processing.

#### Key resources

- **[Modal Guide](https://modal.com/docs/guide)** – Core concepts, custom images, GPUs, scaling strategies, job queues, and batch processing
- **[API Reference](https://modal.com/docs/reference)** – `modal.App`, decorators, data primitives, volumes, networking
- **[Examples](https://modal.com/docs/examples)** – Practical applications (LLM inference, training, batch processing, job queues)

#### Design patterns

**Serialization & pickling**
Most objects including custom functions and classes can be pickled and executed remotely. Returning large models from remote training functions may be infeasible due to size — instead, write them to persistent volume. After a sweep, consider running a separate function to aggregate results from the volume and return only the final summary or evaluation metrics.

Special objects like database connections, GPU contexts, and file descriptors are often environment-specific and should be created within the remote function, not passed in from local scope.

**Closures & scope**
Closures work with remote functions, but don't assume that global scope will be available on the remote container. Usually module-level imports work fine, but occasionally you may need to import within the function.

**Image & environment setup (mostly automatic)**
You normally don't write any `modal.Image` code. `ModalApparatus` builds the image for you, lazily, on the first spawn/map (read-only commands like `status`/`results` never build it). What it does, so you can rely on it:

- **Pins your deps from `pyproject.toml`.** It freezes the resolved versions (`uv_freeze`, all dependency groups except `local`/`dev`) so the remote matches your lockfile. The `cuda` group is included remotely (e.g. `jax[cuda12]`) while local installs stay CPU-only — so the same code runs CPU locally and picks up the GPU when one is attached.
- **Ships your source automatically.** It adds the project's top-level `src/` packages via `add_local_python_source(*project_packages())` — so a remote worker can `import experiment...`/`import utils...` with no manual mounting or packaging. New top-level packages under `src/` are discovered automatically.
- **Cached by content hash**, so it only rebuilds when deps or source change. The first build can take a few minutes; later runs are near-instant cache hits.
- **Caches Hugging Face downloads across containers.** Every remote function mounts a shared `mini-hf-cache` Volume with `HF_HOME` pointing at it, so `from_pretrained`/`hf_hub_download` in a multi-stage pipeline pulls a model once, not once per container. It's a disposable cache (deleting the Volume only costs re-downloads); locally, `~/.cache/huggingface` already persists so nothing changes.

To override (extra apt/pip, a custom base), pass `.w(image=my_image)`. See `make_image`/`requirements.py` and `ModalApparatus._ensure_image`. The HF cache mount and `HF_HOME` still apply with a custom image (the env var rides in a Secret, not the image).

**Running in restricted environments / cloud sandboxes**
- **TLS-inspecting proxies:** `ModalApparatus.__init__` calls `ensure_grpc_trusts_system_ca()` (`mini/_tls.py`) so Modal's gRPC trusts a corporate/sandbox proxy CA. `bin/modal` applies the same fix (via `mini._modal_cli`) before handing off to modal's own CLI, so it also works behind these proxies — prefer it over a bare `modal` install, which does *not* apply this.
- **Auth:** `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` env vars (or `./go auth`, which writes `~/.modal.toml`).
- **Egress:** the worker pulls datasets/results over the network; ensure the Modal + storage domains listed in the README are allowed.

#### Common patterns in this project

- Use `ModalApparatus` to run functions remotely on Modal.
- In modern Modal, ~~"stubs"~~ are now called "apps".
- Do not use Modal's `@app.function()` directly in user code: it creates tight coupling with Modal. Use `ModalApparatus`' `.run(fn)` or `.map(fn)` (or the async variants `.arun(fn)` and `.amap(fm)`) instead so users can easily switch to other execution backends.
