`mini.vis` interface:

```py
def themed(plot: Callable[P, Figure]) -> Callable[P, str]:
    """Wrap a plot function to apply styles. Returns HTML."""

def light_dark[T](light: T, dark: T) -> T:
    """Pick a value based on the active theme (like CSS ``light-dark()``)."""
```

`themed` wraps a plot function to render in both light and dark modes, producing
a single HTML element that switches on `prefers-color-scheme`. The same function
runs twice — once per theme — so you can use `light_dark()` inside to pick
theme-dependent values. It can be used as a decorator with or without arguments:

```py
@themed(alt_text='Plot of a sine wave')
def plot_factory() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x, np.sin(x), color=light_dark('#1a5f8a', '#6ab0d4'), lw=2)
    ax.set_title('sin(x)')
    return fig

mo.Html(plot_factory())
```
