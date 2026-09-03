"""
Utilities for working with matplotlib stylesheets, and drawing primitives it lacks.
"""

import math
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Mapping

import matplotlib as mpl
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import ArrayLike


type Stylesheet = Literal["base", "light", "dark", "transparent"] | Mapping[str, str]

type AxesRow = Sequence[Axes]
"""The axes array from ``plt.subplots(1, n)``, or a flattened grid — ``cast`` to it."""

type AxesGrid = Sequence[Sequence[Axes]]
"""The axes array from ``plt.subplots(m, n)`` — ``cast`` to it to keep ``axes[i][j]`` typed."""


@contextmanager
def use_style(*styles: Stylesheet):
    """Apply matplotlib styles.

    When *theme* is given, :func:`light_dark` and :func:`current_theme` will resolve against it inside the block.
    """
    with mpl.rc_context():
        stylesheet_dir = Path(__file__).parent / "mplstyles"
        for style in styles:
            if isinstance(style, Mapping):
                plt.style.use(dict(style))
            else:
                plt.style.use(stylesheet_dir / f"{style}.mplstyle")
        yield


def _step_path(x: np.ndarray, y: np.ndarray, half_widths: np.ndarray, breaks: set[int]) -> MplPath:
    """Plateau-and-riser path through *(x, y)*, each riser a cubic of the given half-width."""
    dx = (x[-1] - x[0]) / (len(x) - 1)
    verts, codes = [(x[0] - dx / 2, y[0])], [MplPath.MOVETO]
    for i in range(len(x) - 1):
        m, h = (x[i] + x[i + 1]) / 2, half_widths[i]
        if i in breaks:  # close this plateau at its right shoulder, restart at the next's left — a gap, no riser
            verts += [(m - h, y[i]), (m + h, y[i + 1])]
            codes += [MplPath.LINETO, MplPath.MOVETO]
        else:
            verts += [(m - h, y[i]), (m, y[i]), (m, y[i + 1]), (m + h, y[i + 1])]
            codes += [MplPath.LINETO, *[MplPath.CURVE4] * 3]
    verts.append((x[-1] + dx / 2, y[-1]))
    codes.append(MplPath.LINETO)
    return MplPath(np.array(verts), codes)


def _fillet_run(d: float, r: float, psi: float, rooms: tuple[float, float], inset: float) -> tuple[float, float, float]:
    """The straight run's span and the two hand-over insets, for arcs of radius *r* turning through *psi*.

    Each arc leaves its plateau ``r·sin ψ`` before its hand-over, which sits ``e·r`` inside the shoulder line, so it reaches ``r·(sin ψ − e)`` into the plateau. The inset *e* is the design value where the plateau has room for that, and grows just enough to keep the arc within its half of the plateau where it hasn't: with no plateau at all the arc curls to horizontal exactly at the shoulder. The run covers what the insets leave of the span *d*.
    """
    e = tuple(max(inset, math.sin(psi) - room / r) for room in rooms)
    return d - (e[0] + e[1]) * r, e[0], e[1]


def _fillet_solve(
    d: float, dy: float, r: float, rooms: tuple[float, float], inset: float
) -> tuple[float, float, float] | None:
    """``(ψ, e_a, e_b)`` for a filleted riser of shoulder span *d* and rise *dy* with arcs of radius *r*, or None if they don't fit.

    Each arc turns through ψ from its plateau and hands over to the straight run ``r·(1 − cos ψ)`` above (or below) it; the run then covers the remaining rise over its span: ``run(ψ)·tan ψ + 2r·(1 − cos ψ) = dy``. The rise is increasing in ψ (its derivative is ``run·sec²ψ`` plus a non-negative term), so ψ is found by bisection over the angles whose run is non-negative. Where the run closes before the rise is met the arcs would have to cross, and the radius is too large for this riser; where it closes exactly at ψ = π/2 the run is vertical and covers any rise.
    """
    if _fillet_run(d, r, math.pi / 2, rooms, inset)[0] >= 0:
        psi_max = math.pi / 2
    else:  # the run closes before the arcs reach vertical: that angle is the steepest the riser can be
        lo, hi = 0.0, math.pi / 2
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if _fillet_run(d, r, mid, rooms, inset)[0] >= 0 else (lo, mid)
        psi_max = lo
        if 2 * r * (1 - math.cos(psi_max)) < dy:
            return None

    def rise(psi: float) -> float:
        return max(_fillet_run(d, r, psi, rooms, inset)[0], 0.0) * math.tan(psi) + 2 * r * (1 - math.cos(psi))

    lo, hi = 0.0, psi_max
    for _ in range(48):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if rise(mid) < dy else (lo, mid)
    psi = (lo + hi) / 2
    return psi, *_fillet_run(d, r, psi, rooms, inset)[1:]


_FILLET_INSET = 2 / 3
"""How far, as a fraction of the radius, each arc's hand-over to the straight run sits inside the shoulder line, given room.

At 1 a vertical riser's arc starts exactly at the shoulder, so the whole transition sits inside the ramp footprint. At 0 the hand-over is on the shoulder line and the arcs reach a full ``r·sin ψ`` into the plateaus. Values between trade a little plateau for a run that keeps more of the footprint; two thirds looked best across steepnesses. Where the plateau is too short for that reach, the inset grows so the arc stops at the plateau's midpoint (see :func:`_fillet_run`).
"""


def _fillet_path(
    a: np.ndarray, b: np.ndarray, ends: np.ndarray, breaks: set[int], radius: float, inset: float = _FILLET_INSET
) -> MplPath:
    """Step path whose risers are straight lines eased into the plateaus by circular arcs of *radius*.

    Everything here is in one uniform coordinate system (display space, in practice), which is what makes the arcs circles. *a[i]* and *b[i]* are riser *i*'s shoulders and *ends* the path's extremities. Each arc hands over to the straight run ``inset·r`` inside its shoulder line and leaves its plateau ``r·sin ψ`` before that, so a vertical riser gets a quarter-circle and a shallow one a sliver, with the run spanning the same horizontal distance whatever the steepness (see :func:`_fillet_run`). An arc never reaches past the midpoint of its plateau: where the plateau is too short, the hand-over moves inward instead, and with no plateau at all (``ramp=1``) the arc curls to horizontal at the shared sample point, so a peak is rounded by the two arcs meeting there. The radius is capped per riser where even that leaves the run no room; a riser with no span (``ramp=0``) stays sharp.
    """
    xa, xb, x0, x1 = a[:, 0], b[:, 0], ends[0, 0], ends[1, 0]
    left_room = np.abs(xa - np.concatenate([[x0], xb[:-1]])) / 2  # half the plateau before each riser
    right_room = np.abs(np.concatenate([xa[1:], [x1]]) - xb) / 2  # and after it
    verts, codes = [ends[0]], [MplPath.MOVETO]
    for i, ((ax_, ay), (bx, by)) in enumerate(zip(a, b, strict=True)):
        if i in breaks:
            verts += [(ax_, ay), (bx, by)]
            codes += [MplPath.LINETO, MplPath.MOVETO]
            continue
        sx, sy = np.sign(bx - ax_) or 1.0, np.sign(by - ay) or 1.0
        d, dy, rooms = abs(bx - ax_), abs(by - ay), (float(left_room[i]), float(right_room[i]))
        r, sol = radius, _fillet_solve(d, dy, radius, rooms, inset) if dy > 0 and radius > 0 else None
        if dy > 0 and radius > 0 and sol is None:  # too large for this riser: bisect the radius down to what fits
            lo, hi = 0.0, radius
            for _ in range(30):
                mid = (lo + hi) / 2
                fit = _fillet_solve(d, dy, mid, rooms, inset)
                lo, hi, sol = (mid, hi, fit) if fit is not None else (lo, mid, sol)
            r = lo
        if sol is None:
            verts += [(ax_, ay), (bx, by)]
            codes += [MplPath.LINETO, MplPath.LINETO]
            continue
        psi, ea, eb = sol
        c, sn = np.cos(psi), np.sin(psi)
        k = 4 / 3 * np.tan(psi / 4) * r  # cubic handle length that best approximates a circular arc of angle ψ
        px, qx = ax_ + sx * ea * r, bx - sx * eb * r  # where the arcs hand over to the straight run
        p = (px, ay + sy * r * (1 - c))
        q = (qx, by - sy * r * (1 - c))
        verts += [
            (px - sx * r * sn, ay),
            (px - sx * (r * sn - k), ay),
            (p[0] - sx * k * c, p[1] - sy * k * sn),
            p,
            q,
            (q[0] + sx * k * c, q[1] + sy * k * sn),
            (qx + sx * (r * sn - k), by),
            (qx + sx * r * sn, by),
        ]
        codes += [MplPath.LINETO, *[MplPath.CURVE4] * 3, MplPath.LINETO, *[MplPath.CURVE4] * 3]
    verts.append(ends[1])
    codes.append(MplPath.LINETO)
    return MplPath(np.array(verts), codes)


class _FilletStepPatch(PathPatch):
    """A filleted step line or band, laid out in display space each time it is drawn.

    A circle in data coordinates is an ellipse on the page unless the axes happen to be isotropic, so the fillets are built from the data transform's *current* output — after layout, resizing, and any change of limits — and the patch itself carries the identity transform. *ys* holds one series for a line, or (upper, lower) for a band.
    """

    def __init__(self, ax: "Axes", x, ys, half_widths, breaks, radius_pt, transform, **kwargs):
        super().__init__(MplPath([(0.0, 0.0)]), **kwargs)
        self._layout = (ax, np.asarray(x, float), [np.asarray(y, float) for y in ys], half_widths, breaks, radius_pt)
        self._data_transform = transform
        self.set_transform(mtransforms.IdentityTransform())

    def get_path(self) -> MplPath:
        ax, x, ys, hs, breaks, radius_pt = self._layout
        dx, m = (x[-1] - x[0]) / (len(x) - 1), (x[:-1] + x[1:]) / 2
        to_display = self._data_transform.transform
        paths = []
        for y in ys:
            a = to_display(np.column_stack([m - hs, y[:-1]]))
            b = to_display(np.column_stack([m + hs, y[1:]]))
            ends = to_display([(x[0] - dx / 2, y[0]), (x[-1] + dx / 2, y[-1])])
            paths.append(_fillet_path(a, b, ends, breaks, radius_pt * ax.figure.dpi / 72))
        return paths[0] if len(paths) == 1 else _band_path(*paths)


def _add_step_patch(ax: "Axes", x, ys, hs, breaks, fillet, style: dict) -> PathPatch:
    """Add a step line (one series) or band (upper, lower) to *ax*, filleted or S-curved."""
    if fillet is None:
        paths = [_step_path(x, y, hs, breaks) for y in ys]
        patch = PathPatch(paths[0] if len(paths) == 1 else _band_path(*paths), **style)
    else:
        transform = style.pop("transform", ax.transData)
        patch = _FilletStepPatch(ax, x, ys, hs, breaks, fillet, transform, **style)
        # The identity transform hides the patch from autoscaling; report the data extent ourselves.
        dx = (x[-1] - x[0]) / (len(x) - 1)
        ax.update_datalim([(x[0] - dx / 2, min(map(np.min, ys))), (x[-1] + dx / 2, max(map(np.max, ys)))])
    ax.add_patch(patch)
    return patch


def _strokes(path: MplPath) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a path into its connected strokes, one (vertices, codes) pair per MOVETO."""
    verts, codes = np.asarray(path.vertices, float), np.asarray(path.codes, int)
    starts = [*np.flatnonzero(codes == MplPath.MOVETO), len(codes)]
    return [(verts[a:b], codes[a:b]) for a, b in zip(starts[:-1], starts[1:], strict=True)]


def _band_path(upper: MplPath, lower: MplPath) -> MplPath:
    """Close each of *upper*'s strokes onto the matching one of *lower*, giving filled ribbons.

    The lower edge is walked backwards. Reversing a stroke's vertices reverses its segments, and each segment's codes are the same read either way (a lone LINETO, or a CURVE4 triple), so reversing the code list after the leading MOVETO re-describes the same curve.
    """
    verts, codes = [], []
    for (uv, uc), (lv, lc) in zip(_strokes(upper), _strokes(lower), strict=True):
        verts += [*uv, *lv[::-1], uv[0]]
        codes += [*uc, MplPath.LINETO, *lc[:0:-1], MplPath.CLOSEPOLY]
    return MplPath(np.array(verts), codes)


def smooth_step_band(
    ax: "Axes",
    x: "ArrayLike",
    lower: "ArrayLike",
    upper: "ArrayLike",
    *,
    ramp: "float | ArrayLike" = 1.0,
    breaks: "Iterable[int] | None" = None,
    fillet: float | None = None,
    **kwargs,
) -> PathPatch:
    """Fill the ribbon between two :func:`smooth_step` curves.

    Both edges are stepped the same way, so the band keeps the plateau-and-riser shape of the lines it belongs to and never disagrees with them about where a value starts and stops. Use it for a spread the reader should take in as a thickness rather than read off — a min–max over seeds, a quantile range — behind the series it belongs to. Pass a scalar *lower* for the common case of an area down to a baseline. Pass the same *ramp* and *fillet* (a corner radius in points; see :func:`smooth_step`) as the line it sits behind.
    """
    x = np.asarray(x, float)
    if len(x) < 2:
        raise ValueError("smooth_step_band needs at least two samples")
    lo, hi = (np.broadcast_to(np.asarray(v, float), x.shape) for v in (lower, upper))
    dx = (x[-1] - x[0]) / (len(x) - 1)
    hs = np.broadcast_to(ramp, len(x) - 1) * dx / 2
    brk = set(breaks or ())
    return _add_step_patch(ax, x, [hi, lo], hs, brk, fillet, dict(lw=0) | kwargs)


def smooth_step_area(
    ax: "Axes",
    x: "ArrayLike",
    y: "ArrayLike",
    *,
    baseline: float = 0.0,
    **kwargs,
) -> PathPatch:
    """Fill between a :func:`smooth_step` curve and *baseline* — :func:`smooth_step_band` with a flat floor.

    Use it for a summary the reader should be able to take in without tracing anything — the total, the mean of the series drawn over it — in a neutral fill, so it reads as ground rather than as one more series.
    """
    return smooth_step_band(ax, x, baseline, y, **kwargs)


def smooth_step(
    ax: "Axes",
    x: "ArrayLike",
    y: "ArrayLike",
    *,
    ramp: "float | ArrayLike" = 1.0,
    breaks: "Iterable[int] | bool | None" = None,
    elide: "Iterable[int] | None" = None,
    fade: "float | str" = 0.3,
    fillet: float | None = None,
    **kwargs,
) -> PathPatch:
    """Draw a step plot whose risers are S-curves rather than vertical jumps.

    Each riser is a cubic whose control points both sit at its midpoint, level with their respective endpoints. That gives horizontal tangents at every sample, long flat shoulders, and the steepest slope at the crossover — twice the straight-line slope — so the curve reads as flowing while each value still gets a moment of rest. Because the control points never leave the [y_i, y_{i+1}] band, the curve is monotone between samples: no spline overshoot. By default each riser spans the whole gap between samples; *ramp* < 1 narrows it to that fraction of the gap, inserting flat plateaus like a classic step plot. Compare ``ax.step``, whose vertical risers carry no rate information and pile up exactly on top of each other where two series agree.

    Use it for point measurements on a regular ordinal axis — character positions, epochs, layer depths — where a straight line between samples would wrongly suggest the value passes through the intermediate values. *x* must be evenly spaced.

    *ramp* may be a scalar or one value per riser (length ``len(x) - 1``), so individual transitions flatten or square off independently: 1 is a pure ramp with no rest, 0 is a vertical step. Use a per-riser ramp when neighbours differ in kind — draw the jump across a run of unsampled positions as a smooth slide (→1) while keeping genuine neighbours as discrete plateaus (< 1). Any two adjacent ramps must sum to ≤ 2, which holds for values in [0, 1].

    *fillet* swaps the S for a straight riser whose two corners are rounded by circular arcs of that radius, in points — the corner-radius idea from vector editors. The straight run makes the slope constant across most of the transition, so the rate a reader estimates from the line is the chord slope, where the S peaks at twice it; prefer it when the risers carry rate information the reader will compare, and the S when the transitions are just connective. The arcs are circles on the page, whatever the axes' aspect, because the path is laid out in display space at draw time (so the returned patch carries the identity transform, and a *transform* keyword is applied to the data instead). The straight run spans the footprint set by *ramp* less a fixed fraction of the radius at each end, and the arcs curl out from there, so risers of different steepness read as covering the same horizontal distance and a vertical one stays inside its footprint. Where a plateau is too short for the arcs to reach into it (at *ramp* = 1 there are none), each arc curls to horizontal at the plateau's midpoint instead, so neighbouring risers meet with level tangents and a peak is rounded; a radius too large for its riser is capped so the run never inverts.

    *breaks* holds indices *i* after which to leave a gap instead of a riser: sample *i*'s plateau ends at its right shoulder and sample *i+1*'s begins at its left one, with nothing between. Use it where consecutive samples are the same underlying point (e.g. two ordinal slots that alias onto one character) so the connecting riser would span zero real distance. The gap is the riser's footprint, so it's widest when the plateaus are narrow — pair breaks with *ramp* < 1 to keep a visible flat on each side.

    *elide* holds indices *i* whose riser spans ground the samples never covered — a run of unmeasured positions between two measured ones. Those risers are drawn at *fade* times the line's opacity, so interpolated stretches carry visibly less ink than the measurements they join, and the eye stops reading them as another step. The faded riser is a second, lighter artist beneath the returned one (which is broken at those gaps), so the effect is real transparency: it needs no opaque mask matched to the background, and survives a transparent figure over any page color. Where *breaks* and *elide* name the same riser, the break wins and nothing is drawn.

    Pass *fade* as a color instead to draw those risers opaque in that color — see :func:`~mini.vis.mix` for computing one. Worth it where elided risers from several lines land on top of each other: translucent copies accumulate into something darker than any of them, which reads as emphasis where the data only agreed.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        raise ValueError("smooth_step needs at least two samples")
    brk, eli = set(range(len(x))) if breaks is True else set(breaks or ()), set(elide or ())
    dx = (x[-1] - x[0]) / (len(x) - 1)
    hs = np.broadcast_to(ramp, len(x) - 1) * dx / 2  # riser half-widths (one per gap)
    style = dict(fill=False, capstyle="round", joinstyle="round") | kwargs

    if eli - brk:  # a fainter companion keeps the elided risers; the solid path drops them
        quiet = (
            {"color": fade, "alpha": None} if isinstance(fade, str) else {"alpha": fade * (kwargs.get("alpha") or 1.0)}
        )
        _add_step_patch(ax, x, [y], hs, brk, fillet, style | quiet)
    return _add_step_patch(ax, x, [y], hs, brk | eli, fillet, style)


def smooth_step_marks(
    ax: "Axes",
    x: "ArrayLike",
    y: "ArrayLike",
    *,
    breaks: "Iterable[int] | None" = None,
    riser_weight: float = 0.5,
    riser_alpha: float = 0.5,
    **kwargs,
) -> PathPatch:
    """A :func:`smooth_step` whose weight sits on the plateaus, with the risers drawn faint.

    Two strokes: the full path at *riser_weight* times the line width and *riser_alpha* times its opacity, and over it the same line at full weight broken at every riser. A row then reads as a run of level marks with a hint of the path between them, rather than as a curve that happens to be flat in places. Use it where the x axis is a handful of discrete sites and the measurements are the plateaus; where a riser can span unprobed ground the reader has to judge, keep :func:`smooth_step`'s solid risers (or *elide* those stretches).

    *breaks* and the other keywords pass through to both strokes. Returns the marks' patch.
    """
    x = np.asarray(x, float)
    lw = kwargs.pop("lw", kwargs.pop("linewidth", 1.0))
    alpha = kwargs.pop("alpha", None) or 1.0
    smooth_step(ax, x, y, breaks=breaks, lw=lw * riser_weight, alpha=alpha * riser_alpha, **kwargs)
    return smooth_step(ax, x, y, breaks=set(breaks or ()) | set(range(len(x) - 1)), lw=lw, alpha=alpha, **kwargs)
