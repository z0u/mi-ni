# mi-ni — AI Research Template

> **<ruby>見<rt>み</rt>に</ruby> /mi·ni/** — _with intent to see_ [^etymology]

[^etymology]: From 見に行く (mi-ni iku), meaning "to go for the purpose of seeing something." This library is about small AI experiments—quick, lightweight explorations to try and see what happens.

&nbsp;

This is a template repository for doing AI research. Features:

- **Local Python notebooks**
- **Remote per-function GPU compute** [^modal]
- **Inline visualization** with remote-to-local callbacks
- **AI-assisted coding** with Copilot/VS Code

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/z0u/mi-ni)

&nbsp;

![Screen recording of a notebook cell in VS Code, with code to run a distributed training job and an inline loss chart that updates in real-time.](https://github.com/user-attachments/assets/6eb94b46-0b8f-4cd2-b200-abcec86c88cd)

Above: screen recording of a local notebook running a remote training job. `train()` is a function that runs in the cloud (with a GPU), and `track()` is a function that runs locally — even when called from `train`!

<details><summary>Code for the above demo</summary>

The code shown in the screen recording is [^recording-correction]:

```python
@run.hither
async def track(loss: float):
    history.append(loss)
    plot(history)

@run.thither(gpu='L4')
async def train(epochs: int, track):
    for _ in range(epochs):
        track(some_training_function())
    print('Training complete')

async with run(), track as callback:
    await train(25, callback)
```

Read about how it works in [doc/hither-thither.md](doc/hither-thither.md).

[^recording-correction]: The recording contained a mistake: it assigned the return value of `train()` to a variable called `model`, but the function doesn't actually return anything! It could, though; see [the Getting Started notebook][getting-started].
</details>

<details><summary>More cool features</summary>

- [Dev container][dc] for a consistent environment, both locally and in [Codespaces][codespaces]
- ML stack ([PyTorch, Polars, etc.](pyproject.toml))
- Modern package management with [uv]
- Pre-configured for good engineering practices: tests, linting, type-checking (optional!)
</details>

[^modal]: [Modal] is used for remote compute. They charge per-second, billed for the duration of your function.


## Getting started

First, [open in GitHub Codespaces](https://codespaces.new/z0u/mi-ni). Then:

```bash
./go install cpu  # CPU deps for local venv
./go auth         # Authenticate with Modal for remote compute
```

Open the [Getting Started notebook][getting-started] and try it out (choose `.venv/bin/python3` as the kernel). For a more complete example, have a look at the [nanoGPT notebook](nanogpt.ipynb).

[getting-started]: getting-started.ipynb
[codespaces]: https://github.com/features/codespaces


<details><summary>Virtual environment</summary>

The Python environment is configured when the dev container is created.

Use [uv] to add and remove packages, and to run scripts:

```bash
uv add plotly --group local
uv run python example.py
```
</details>

<details>
<summary>Restarting the language server (VS Code)</summary>

If you open a Python file before the setup is complete, you may need to restart the Python language server.

- Open a `.py` or `.ipynb` file
- Open the command pallette with <kbd>⇧</kbd><kbd>⌘</kbd><kbd>P</kbd> or <kbd>Ctrl</kbd><kbd>Shift</kbd><kbd>P</kbd>
- Run _Python: Restart Language Server_.
</details>

[dc]: https://containers.dev
[Modal]: https://modal.com
[uv]: https://astral.sh/uv


## Contributing & licence

This project is dedicated to the public domain [^unlicense][^attrib]. In your own experiments, there's no need to contribute back! The code is yours to modify as you please.

If you do want to contribute to _this template_, then fork it as usual. Before making a pull request, run:

```bash
./go check
```

[^not-fork]: Since your project isn't a fork, you don't need to worry about keeping the code in sync, and you can add and remove Python packages as you wish.

[^unlicense]: Technically, the licence is the [Unlicense](https://unlicense.org), which is about as close as you can get to "do whatever you want".

[^attrib]: Exception: Code in `src/experiment` is derived from [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy and is subject to MIT license terms. See the [LICENSE](LICENSE) file for details.
