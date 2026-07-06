import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup(hide_code=True):
    import marimo as mo  # noqa: F401
    import numpy as np

    from subline.series import Series
    from subline.subline import Subline


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Subline

    Subline draws per-token sparklines directly beneath the text they describe.
    Each segment is aligned to its token, so you can see at a glance which tokens
    carry the most information — without losing the context of the surrounding text.

    The visualization adapts to your system light/dark preference automatically.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Single series

    The simplest case: one value per token, plotted as a sparkline beneath each character.
    Here we show a rough hand-crafted surprise signal for the sentence "The quick brown fox."
    Spaces and common letters score low; the rarer "q" and "x" score high.
    """)
    return


@app.cell
def _():
    text = "The quick brown fox"
    tokens = list(text)

    # Rough per-character surprisal (higher = more unexpected)
    surprise = np.array([
        0.40, 0.20, 0.15,             # The
        0.05,                         # (space)
        0.80, 0.20, 0.25, 0.20, 0.50, # quick
        0.05,                         # (space)
        0.30, 0.25, 0.20, 0.35, 0.20, # brown
        0.05,                         # (space)
        0.30, 0.20, 0.90,             # fox
    ])  # fmt: skip

    mo.Html(Subline().plot(tokens, [Series(surprise, label="Surprisal")]))
    return surprise, tokens


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Multiple series

    Multiple series overlay on the same sparkline, letting you compare two models
    side by side, or pair a metric with its complement. A dashed line keeps them
    visually distinct.
    """)
    return


@app.cell
def _(surprise, tokens):
    rng = np.random.default_rng(0)
    model_a = np.clip(surprise + rng.normal(0, 0.08, len(tokens)), 0, 1)
    model_b = np.clip(surprise * 0.55 + rng.normal(0, 0.05, len(tokens)), 0, 1)

    mo.Html(
        Subline().plot(
            tokens,
            [
                Series(model_a, label="Model A"),
                Series(model_b, label="Model B", dasharray="2"),
            ],
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Wrapping long text

    When the text exceeds `chars_per_line`, Subline wraps it automatically.
    Each line gets its own sparkline, and the token alignment is preserved
    across the break.
    """)
    return


@app.cell
def _():
    long_text = "To be or not to be, that is the question: whether tis nobler in the mind to suffer"
    long_tokens = long_text.split()
    long_tokens = [long_tokens[0]] + [f" {tok}" for tok in long_tokens[1:]]
    rng2 = np.random.default_rng(1)
    vals = np.abs(rng2.normal(0.35, 0.2, len(long_tokens))).clip(0, 1)

    mo.Html(Subline(chars_per_line=50).plot(long_tokens, [Series(vals, label="Surprisal")]))
    return


if __name__ == "__main__":
    app.run()
