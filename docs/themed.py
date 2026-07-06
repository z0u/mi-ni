import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup(hide_code=True):
    import marimo as mo  # noqa: F401
    import matplotlib.pyplot as plt
    import numpy as np

    from mini.reports import report_bundle, use_publisher
    from mini.vis import themed
    from mini.vis.theme import light_dark

    # Externalize every themed figure to a file beside the exported HTML, referenced
    # by a relative URL — keeps the report light, and `build_site` repoints those URLs
    # at the bucket (one <base> tag) when publishing. No publisher → figures inline.
    use_publisher(report_bundle(__file__))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Themed plots

    `themed` wraps a plot function to render in both light and dark modes,
    producing a single HTML element that switches on `prefers-color-scheme`.
    The same function runs twice — once per theme — so you can use
    `light_dark()` inside to pick theme-dependent values.

    It has three call patterns.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Plain decorator

    The simplest form: `@themed` with no arguments.
    """)
    return


@app.cell
def _():
    x = np.linspace(0, 2 * np.pi, 300)

    @themed
    def plot_plain() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(x, np.sin(x), color=light_dark("#1a5f8a", "#6ab0d4"), lw=2)
        ax.set_title("sin(x)")
        return fig

    mo.Html(plot_plain())
    return (x,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Decorator factory

    Pass keyword arguments to set `alt_text`, `max_width`, or custom styles.
    This is the form you want when defining a standalone plot function.
    """)
    return


@app.cell
def _(x):
    @themed(alt_text="sin and cos")
    def plot_factory() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 3))
        color_sin = light_dark("#1a5f8a", "#6ab0d4")
        color_cos = light_dark("#8a3a1a", "#d49a6a")
        ax.plot(x, np.sin(x), color=color_sin, lw=2, label="sin")
        ax.plot(x, np.cos(x), color=color_cos, lw=2, label="cos")
        ax.legend()
        return fig

    mo.Html(plot_factory())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Direct call

    Useful for one-off plots, or when wrapping a function defined elsewhere.
    """)
    return


@app.cell
def _(x):
    def _plot_raw() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(x, np.sin(x) * np.exp(-x / 6), color=light_dark("#2a6e3a", "#7ad49a"), lw=2)
        ax.set_title("Damped sine")
        return fig

    mo.Html(themed(_plot_raw, alt_text="Damped sine wave")())
    return


if __name__ == "__main__":
    app.run()
