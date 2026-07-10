import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup(hide_code=True):
    import json
    import tempfile
    from pathlib import Path

    import marimo as mo  # noqa: F401
    import matplotlib.pyplot as plt
    import numpy as np

    from mini.reports import report_bundle, use_publisher
    from mini.store import project_store
    from mini.vis import light_dark, themed

    # Externalize every themed figure to a file beside the exported HTML, referenced
    # by a relative URL — keeps the report light, and `build_site` repoints those URLs
    # at the bucket (one <base> tag) when publishing. No publisher → figures inline.
    use_publisher(report_bundle(__file__))

    # Sweep axes (kept in sync with experiment.py), and per-arch plot colours.
    LRS = ["3e-3", "1e-2", "4e-2"]
    ARCHS = ["baseline", "nGPT", "nGPT (scalar)"]
    ARCH_COLORS = {"baseline": "tab:gray", "nGPT": "tab:red", "nGPT (scalar)": "tab:green"}

    # The residual-bug ablation: the scalar-gate nGPT across widths, in three
    # recipes. In reading order — the additive step with a fixed gate (exposes the
    # bug), the additive step with our learnable width-scaled gate (masks it), and
    # the normalized-LERP fix.
    WIDTHS = [32, 64, 128]
    RESID = [
        ("fixed", "additive, fixed α (bug)"),
        ("add", "additive, learnable α"),
        ("norm", "normalized LERP (fix)"),
    ]

    def resid_colors() -> dict[str, str]:
        return dict(
            fixed=light_dark("#d1495b", "#e06c7d"),  # red — the bug, exposed
            add=light_dark("#8d99ae", "#a7b1c2"),  # grey — additive, masked by the gate
            norm=light_dark("#1b998b", "#2ec4b6"),  # teal — the fix
        )

    # The experiment publishes its curves to the project-scoped store under this
    # name (see experiment.py); we resolve them by name at export time, so no data
    # file is committed to Git. The store is the HF bucket when configured, else local.
    CURVES_REF = "reports/gpt-sweep/curves"

    def load_curves() -> dict[str, list[float]]:
        """Resolve `{key: [val_loss per epoch]}` from the store, or `{}` if unpublished."""
        store = project_store()
        art = store.get_ref(CURVES_REF)
        if art is None:
            return {}
        with tempfile.TemporaryDirectory() as d:
            return json.loads(store.get(art, Path(d) / "curves.json").read_text())

    def plateau(curves: dict[str, list[float]], key: str) -> float:
        """Converged loss: mean of the last 10 epochs (per-epoch eval noise is ~±0.08)."""
        return float(np.mean(curves[key][-10:]))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Architecture sweep: GPT versus nGPT

    A controlled comparison of the baseline LayerNorm GPT against
    [nGPT](https://arxiv.org/abs/2410.01131), and a second sweep that pins down a
    geometry bug in our simplified nGPT residual — found, and fixed in one line.

    This is a **report**: it reads results the experiment already produced and
    renders them. The experiment itself is [`experiment.py`](./experiment.py), an
    importable `main(ctx)` DAG (one data-prep step, then a fifteen-cell GPU sweep)
    run from the CLI on Modal L4s (the `train` role binds the GPU + timeout):

    ```bash
    bin/mini run docs/gpt-sweep/experiment.py --app modal --max-containers 15
    ```

    On completion the experiment publishes its val-loss curves to the project
    store under a stable name, and this report resolves them by that name when it
    renders — so the data lives in the durable store (the HF bucket), not in Git.
    """)
    return


@app.cell(hide_code=True)
def _():
    curves = load_curves()
    return (curves,)


@app.cell(hide_code=True)
def _(curves):
    mo.stop(
        not curves,
        mo.md(
            "No results yet — run the experiment (it publishes its curves to the store on completion):\n\n"
            "```bash\nbin/mini run docs/gpt-sweep/experiment.py --app modal --max-containers 15\n```"
        ),
    )
    flat = {(a, lr): v for a in ARCHS for lr in LRS if (v := curves.get(f"{a}|{lr}"))}
    best_arch, best_lr = min(flat, key=lambda k: min(flat[k]))
    mo.md(
        f"**Best run:** {best_arch} at peak LR {best_lr} — val_loss "
        f"**{min(flat[(best_arch, best_lr)]):.2f}** over {len(flat)} architecture×LR cells."
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Does normalization help? nGPT versus the LayerNorm baseline

    nGPT added a fair amount of machinery, so it's worth checking how it compares
    against the baseline. The two architectures don't want the same learning rate:
    normalization rescales the effective gradients, which shifts nGPT's useful LR
    band upward. To isolate the architecture, we sweep both across the same three
    peak learning rates, with the same warmup-then-cosine schedule, on the same
    data. The only difference within each panel is LayerNorm versus the hypersphere.

    - **baseline** — standard pre-norm transformer (LayerNorm + additive residual).
    - **nGPT** — normalized transformer as published (per-channel eigen learning rates).
    - **nGPT (scalar)** — a single learnable scalar gate per sub-module instead of
      per-channel weights.

    Both nGPT arms use the corrected normalized-LERP residual (see the next
    section); at this width the correction is invisible, so this is a clean
    architecture comparison.
    """)
    return


@app.cell(hide_code=True)
def _(curves):
    mo.stop(not curves)

    @themed(
        alt_text="Three panels, one per peak learning rate. Each plots validation loss versus epoch "
        "for the baseline, nGPT, and nGPT (scalar) architectures, with the minimum of each curve marked. "
        "At the lowest learning rate the baseline edges ahead; at higher rates the two nGPT variants reach "
        "slightly lower loss and track each other almost exactly."
    )
    def plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
        for ax, lr in zip(axes, LRS, strict=True):
            for arch in ARCHS:
                ys = curves.get(f"{arch}|{lr}")
                if not ys:
                    continue
                ax.plot(range(1, len(ys) + 1), ys, color=ARCH_COLORS[arch], lw=1.5, label=arch)
                best = min(range(len(ys)), key=ys.__getitem__)
                ax.scatter([best + 1], [ys[best]], color=ARCH_COLORS[arch], s=18, zorder=5)
            ax.set_title(f"peak LR = {lr}")
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("val_loss")
        axes[0].legend()
        fig.tight_layout()
        return fig

    mo.Html(plot())
    return


@app.cell(hide_code=True)
def _(curves):
    mo.stop(not curves)

    def _cell(arch: str, lr: str) -> str:
        ys = curves.get(f"{arch}|{lr}")
        return f"{min(ys):.2f}" if ys else "—"

    # Bold the best (lowest) architecture in each LR column.
    def _row(lr: str) -> str:
        vals = {a: min(curves[f"{a}|{lr}"]) for a in ARCHS if curves.get(f"{a}|{lr}")}
        best = min(vals, key=lambda a: vals[a]) if vals else None
        cells = " | ".join(f"**{_cell(a, lr)}**" if a == best else _cell(a, lr) for a in ARCHS)
        return f"| {lr} | {cells} |"

    table = "\n".join(
        ["| peak LR | baseline | nGPT | nGPT (scalar) |", "| --- | --- | --- | --- |", *(_row(lr) for lr in LRS)]
    )
    mo.md(f"""
    Best (minimum) validation loss per cell:

    {table}

    Two things stand out. **nGPT needs a higher learning rate to pay off** — at the
    lowest rate it trails the baseline, but given enough LR it reaches a slightly
    lower loss. And the **scalar-gate simplification matches the full per-channel
    variant** almost exactly: the per-channel granularity buys nothing at this scale.

    The paper's headline "trains several times faster" and its stability advantages
    don't reproduce at this toy scale — the LayerNorm baseline never threatens to
    diverge — so at this scale nGPT is a small but real improvement in final loss.
    The load-bearing piece is the scalar gate on the *normalized* residual (see the
    [model notes](../../src/experiment/model/README.md)); the rest of nGPT's
    machinery is optional here.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The residual bug: a width-gated failure the gate was hiding

    The nGPT residual should step toward the sub-module's **normalized** output,
    `h ← Norm(h + α·(Norm(sub(h)) − h))`, which makes `α` a true interpolation
    fraction. We had simplified it to a raw additive step, `h ← Norm(h + α·sub(h))`,
    on the assumption that `sub(h)` has norm ≈ 1. It doesn't: the MLP scales its
    pre-activations by a √n_embd baseline (to keep GELU in range), so
    `‖MLP(h)‖ ∝ √n_embd`. The *effective* rotation is `α·‖sub(h)‖`, which **grows
    with width** — a fixed `α` can't control it.

    Whether that bites depends on the gate. Our `α` is *learnable* and
    reparametrized by √n_embd, so it can shrink as width grows and quietly absorb
    the error — which is why the sweep above (all width 32) saw nothing. To separate
    the geometry from the gate, we sweep width {32, 64, 128} (with `n_ff = 4·n_embd`,
    so the MLP grows too) at depth 12 and LR 1e-2, in three recipes: the additive
    step with a **fixed** `α = 1/n_layer` (exposes the bug), the additive step with
    our **learnable** `α` (masks it), and the **normalized LERP** (removes it).
    """)
    return


@app.cell(hide_code=True)
def _(curves):
    mo.stop(not curves or not any(f"width|fixed|{w}" in curves for w in WIDTHS))
    fixed_128 = plateau(curves, "width|fixed|128")
    norm_128 = plateau(curves, "width|norm|128")
    norm_spread = max(plateau(curves, f"width|norm|{w}") for w in WIDTHS) - min(
        plateau(curves, f"width|norm|{w}") for w in WIDTHS
    )
    mo.md(
        f"**A fixed gate exposes the bug; the fix removes it.** With a fixed step size the additive "
        f"residual falls apart as width grows — converged loss climbs to **{fixed_128:.2f}** nats/char "
        f"at width 128. The normalized-LERP fix reaches **{norm_128:.2f}** there, flat across all three "
        f"widths (spread just {norm_spread:.02f} nats/char). Same learning rate, same everything else; "
        f"one line of geometry."
    )
    return


@app.cell(hide_code=True)
def _(curves):
    mo.stop(not curves or not any(f"width|fixed|{w}" in curves for w in WIDTHS))

    @themed(
        name="width-gating",
        alt_text=(
            "Two panels sharing a y-axis. Left, converged validation loss against width (32, 64, 128 on a "
            "log axis) for three residual recipes. The additive step with a fixed gate (red) starts near "
            "1.5 at width 32 and climbs steeply to about 3.1 at width 128; the additive step with a "
            "learnable gate (grey) and the normalized-LERP fix (teal) both stay flat and low near 1.35 at "
            "every width. Right, validation loss against epoch at width 128: the fixed-gate recipe falls "
            "during warmup then rises and sticks near 3.1, while the learnable-gate and normalized recipes "
            "fall smoothly together to about 1.34."
        ),
    )
    def _plot() -> plt.Figure:
        colors = resid_colors()
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)

        # Left: converged loss vs width.
        for arm, label in RESID:
            ys = [plateau(curves, f"width|{arm}|{w}") for w in WIDTHS]
            lw = 2.6 if arm in ("fixed", "norm") else 1.8
            ax_l.plot(WIDTHS, ys, "o-", color=colors[arm], lw=lw, label=label)
        ax_l.set(xlabel="width (n_embd)", ylabel="converged val_loss (nats/char)", xscale="log")
        ax_l.set_xticks(WIDTHS, [str(w) for w in WIDTHS])
        ax_l.minorticks_off()
        ax_l.grid(alpha=0.3)
        ax_l.legend(fontsize=8)

        # Right: convergence at the failing width 128.
        for arm, label in RESID:
            ys = curves[f"width|{arm}|128"]
            lw = 2.0 if arm in ("fixed", "norm") else 1.4
            ax_r.plot(range(1, len(ys) + 1), ys, color=colors[arm], lw=lw, label=label)
        ax_r.axvline(10, color="#8888", lw=1, ls=":", label="end of LR warmup")
        ax_r.set(title="width 128", xlabel="epoch")
        ax_r.grid(alpha=0.3)
        ax_r.legend(fontsize=8)
        fig.tight_layout()
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(curves):
    mo.stop(not curves or not any(f"width|fixed|{w}" in curves for w in WIDTHS))
    _f = {w: plateau(curves, f"width|fixed|{w}") for w in WIDTHS}
    _a = {w: plateau(curves, f"width|add|{w}") for w in WIDTHS}
    _n = {w: plateau(curves, f"width|norm|{w}") for w in WIDTHS}
    mo.md(f"""
    With a **fixed** step the additive residual is fine narrow and breaks wide:
    {_f[32]:.2f} → {_f[64]:.2f} → {_f[128]:.2f} nats/char across widths 32 / 64 / 128.
    At width 128 it never really trains — the hidden state over-rotates through every
    layer until the token's identity is lost. That's the failure the normalized
    residual exists to prevent.

    Two things rescue it, for different reasons. Our **learnable** `α` masks the bug
    ({_a[32]:.2f} → {_a[64]:.2f} → {_a[128]:.2f}): reparametrized by √n_embd, it
    shrinks as width grows and happens to cancel the `‖sub(h)‖` growth — but that's
    the gate laundering a geometry error, not a correct residual. The **normalized
    LERP** removes the error at the source ({_n[32]:.2f} → {_n[64]:.2f} → {_n[128]:.2f}):
    normalizing `sub(h)` makes `α` a true interpolation fraction, so the per-layer
    rotation is ≈ `α` regardless of width or gate.

    This is why the correction matters even though the architecture sweep couldn't
    see it: the scalar-gate nGPT we ship is only safe because *something* controls the
    residual geometry, and the normalized LERP does it by construction rather than by
    luck. The additive forms survive as `config.normalize_sublayer=False` (with
    `learnable_alpha=False` for the fixed gate), kept solely to reproduce this failure;
    the normalized LERP is the default. See the
    [model notes](../../src/experiment/model/README.md) for the geometry.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## References

    Loshchilov, I., Hsieh, C.-P., Sun, S., & Ginsburg, B. (2024). nGPT: Normalized
    transformer with representation learning on the hypersphere. _arXiv_.
    https://arxiv.org/abs/2410.01131

    Karpathy, A. (2022). nanoGPT [Computer software]. GitHub.
    https://github.com/karpathy/nanoGPT
    """)
    return


if __name__ == "__main__":
    app.run()
