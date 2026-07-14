Visualize metrics as sparklines under text.

![Screenshot of text that reads, "A long time ago, in a galaxy somewhat far away..." A sparkline beneath the text shows that the word "somewhat" is clearly out of distribution (i.e. unexpected) in this context.](../../doc/subline.svg)

## Styling

The SVG carries its own theme (light/dark aware) via CSS custom properties. To
restyle without editing the library, pass `css` — appended after the built-in
styles, so a later rule overrides at equal specificity:

```python
Subline(css="svg { --bg-color: light-dark(#fff, #181c1a); }").plot(text, series)
```

Overridable properties include `--bg-color`, `--col-text`, `--col-baseline`,
`--col-series-1..5`, and `--blend-mode`.


## Citation

If you use this visualization in your research, please cite:

```bibtex
@software{text_metrics_viz,
  author = {Sandy Fraser},
  title = {Subline: A Text Metrics Visualizer},
  year = {2025},
  url = {https://github.com/z0u/subline}
}
```
