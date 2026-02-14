import marimo

__generated_with = '0.19.9'
app = marimo.App(width='medium')

with app.setup(hide_code=True):
    import marimo as mo  # noqa: F401
    import time
    from mini import LocalApparatus, ModalApparatus  # noqa: F401
    from mini import emit_progress


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
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's start by defining a mock training function.
    """)
    return


@app.function
def train(x: int) -> int:
    """A task that reports progress."""
    k = 10
    for i in range(k):
        time.sleep(1 / k)
        emit_progress(i + 1, 10, message=f'processing item {x}')
    return x * 2


@app.cell(hide_code=True)
def _(app_type):
    mo.md(f"""
    {app_type}
    """)
    return


@app.cell
async def main(app_type):
    if app_type.value == 'local':
        executor = LocalApparatus('experiment-1', max_workers=3)
    else:
        executor = ModalApparatus('experiment-1').w(max_containers=3)

    print(f'Using {executor}')
    # Use async map because Marimo uses an async context
    results = [result async for result in executor.amap(train, [1, 2, 3, 4, 5])]
    print('Results:', results)
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
