import marimo

__generated_with = '0.20.1'
app = marimo.App(width='medium')

with app.setup(hide_code=True):
    import marimo as mo  # noqa: F401
    import json
    import tempfile
    import time
    from pathlib import Path
    from mini import LocalApparatus, ModalApparatus  # noqa: F401
    from mini import emit_progress, get_data_dir


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Getting started

    This is a notebook that demonstrates basic use of the Apparatus. An Apparatus is like a [thread pool](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor), but it abstracts away the distribution pattern.

    General workflow:

    1. Write general Python functions:
        ```py
        def f(x):
            ...
        ```

    2. Create an apparatus and map over data:

        ```py
        app = LocalApparatus('experiment-1', num_workers=2)
        results = app.map(f, data)
        ```

    3. Swap out the apparatus depending on your workload:

        ```py
        app = ModalApparatus('experiment-1').w(gpu='A100', max_containers=30)
        results = app.map(f, big_data)
        ```

    Each apparatus comes with a *volume* — a shared storage area that persists across function calls. Functions call `get_data_dir()` to read and write files in it, so you can chain steps together: one function prepares data, the next consumes it.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We'll define two functions that pass data through the volume. First, `prep` writes shared configuration; then `train` reads it, runs a mock workload, and saves per-item results.
    """)
    return


@app.function
def prep() -> str:
    """Write shared configuration to the volume."""
    data_dir = get_data_dir()
    config = {'learning_rate': 0.001, 'epochs': 10, 'batch_size': 32}
    (data_dir / 'config.json').write_text(json.dumps(config, indent=2))
    return f'Wrote config to {data_dir / "config.json"}'


@app.function
def train(x: int) -> int:
    """Read config, run a mock workload, and save results to the volume."""
    data_dir = get_data_dir()
    config = json.loads((data_dir / 'config.json').read_text())

    k = 10
    for i in range(k):
        time.sleep(1 / k)
        emit_progress(i + 1, 10, message=f'processing item {x}')

    result = x * config['epochs']

    output_dir = data_dir / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f'result_{x}.txt').write_text(f'input: {x}\nresult: {result}\nlr: {config["learning_rate"]}')
    return result


@app.cell(hide_code=True)
def _(app_type):
    mo.md(f"""
    {app_type}
    """)
    return


@app.cell
async def main(app_type):
    if app_type.value == 'local':
        executor = LocalApparatus('mi-ni-getting-started', max_workers=3)
    else:
        executor = ModalApparatus('mi-ni-getting-started').w(max_containers=3)

    print(f'Using {executor}')

    # Step 1: write shared config to the volume
    print(await executor.arun(prep))

    # Step 2: train (reads config, writes per-item results to volume)
    results = [x async for x in executor.amap(train, [1, 2, 3, 4, 5])]
    print('Results:', results)

    # Step 3: pull outputs back from the volume
    with tempfile.TemporaryDirectory() as tmp:
        await executor.volume.download('outputs', f'{tmp}/outputs')
        print('\nVolume outputs:')
        for p in sorted(Path(tmp, 'outputs').iterdir()):
            print(f'\n--- {p.name} ---')
            print(p.read_text())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Utilities
    """)
    return


@app.cell
def _():
    app_type = mo.ui.dropdown(
        label='App type',
        options=['local', 'modal'],
        value=mo.cli_args().get('app', 'local'),
    )
    return (app_type,)


if __name__ == '__main__':
    app.run()
