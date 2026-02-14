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

**Image & environment setup**
Define container images with explicit dependencies using `modal.Image`. Use pinned versions to ensure pickling compatibility between local and remote environments. mi-ni provides utilities for building images with pinned dependencies from `pyproject.toml`; see `requirements.py` and the `ModalApparatus`.

#### Common patterns in this project

- Use `ModalApparatus` to run functions remotely on Modal.
- In modern Modal, ~~"stubs"~~ are now called "apps".
- Do not use Modal's `@app.function()` directly in user code: it creates tight coupling with Modal. Use `ModalApparatus.run(fn)` or `ModalApparatus.wrap(fn)` instead so users can easily switch to other execution backends.
