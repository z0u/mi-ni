"""The step path is a drawing convention, so pin what each of its gap kinds does to the ink."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.path import Path as MplPath

from mini.vis.plt import _FILLET_INSET, smooth_step, smooth_step_area, smooth_step_band

Y = [0.0, 1.0, 0.5, 0.5]


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def verts(patch) -> np.ndarray:
    """The patch's (N, 2) vertex array. Matplotlib types it as the much looser `ArrayLike`."""
    return np.asarray(patch.get_path().vertices)


def code_count(patch, code) -> int:
    """How many of the patch's vertices carry a given path code."""
    return int(np.sum(patch.get_path().codes == code))


def subpaths(patch) -> int:
    """How many disconnected strokes the patch draws — one more than the gaps in it."""
    return code_count(patch, MplPath.MOVETO)


def test_plain_step_is_one_unbroken_stroke(ax):
    patch = smooth_step(ax, range(len(Y)), Y)
    assert subpaths(patch) == 1
    assert len(ax.patches) == 1


def test_breaks_split_the_stroke_and_add_no_ink(ax):
    patch = smooth_step(ax, range(len(Y)), Y, breaks=[1])
    assert subpaths(patch) == 2
    assert len(ax.patches) == 1


def test_elide_lifts_the_riser_onto_a_fainter_companion(ax):
    patch = smooth_step(ax, range(len(Y)), Y, elide=[1], fade=0.25, alpha=0.8)
    ghost, solid = ax.patches
    assert solid is patch
    assert subpaths(solid) == 2  # the elided riser is gone from the solid stroke...
    assert subpaths(ghost) == 1  # ...and only the ghost still carries it
    assert ghost.get_alpha() == pytest.approx(0.2)  # fade × the line's own alpha


def test_a_color_fade_draws_the_elided_riser_opaque(ax):
    """So that several lines' elided risers landing together don't build up into emphasis."""
    smooth_step(ax, range(len(Y)), Y, elide=[1], fade="#aabbcc", color="#123456")
    ghost, solid = ax.patches
    assert ghost.get_alpha() is None and to_hex(ghost.get_edgecolor()) == "#aabbcc"
    assert to_hex(solid.get_edgecolor()) == "#123456"


def test_a_break_beats_an_elision_on_the_same_riser(ax):
    """Both say "these two samples are not neighbours"; the break says draw nothing at all."""
    patch = smooth_step(ax, range(len(Y)), Y, breaks=[1], elide=[1])
    assert subpaths(patch) == 2
    assert len(ax.patches) == 1


def test_elided_ghost_spans_exactly_the_riser_the_solid_stroke_dropped(ax):
    ramp = 0.5
    solid = smooth_step(ax, range(len(Y)), Y, ramp=ramp, elide=[1])
    ghost = ax.patches[0]
    # Riser 1 runs from x=1.5-ramp/2 to x=1.5+ramp/2. Both strokes reach its shoulders; only
    # the ghost carries the control points at the crossover, which is the riser itself.
    for xs in (verts(solid)[:, 0], verts(ghost)[:, 0]):
        assert np.isclose(xs, 1.5 - ramp / 2).any() and np.isclose(xs, 1.5 + ramp / 2).any()
    assert np.isclose(verts(solid)[:, 0], 1.5).sum() == 0
    assert np.isclose(verts(ghost)[:, 0], 1.5).sum() == 2


def test_band_traces_both_edges_and_closes(ax):
    lo, hi = [-0.1, 0.8, 0.4, 0.35], Y  # strictly under Y, so the ribbon never folds over
    patch = smooth_step_band(ax, range(len(Y)), lo, hi)
    ys = verts(patch)[:, 1]
    assert subpaths(patch) == 1
    assert np.isclose(ys.min(), min(lo)) and np.isclose(ys.max(), max(hi))
    assert code_count(patch, MplPath.CLOSEPOLY) == 1


def test_a_break_splits_the_band_into_two_ribbons(ax):
    patch = smooth_step_band(ax, range(len(Y)), 0.0, Y, breaks=[1])
    assert subpaths(patch) == 2
    assert code_count(patch, MplPath.CLOSEPOLY) == 2


def test_area_is_a_band_with_a_flat_floor(ax):
    area = smooth_step_area(ax, range(len(Y)), Y, baseline=0.25, ramp=0.5)
    band = smooth_step_band(ax, range(len(Y)), [0.25] * len(Y), Y, ramp=0.5)
    assert np.allclose(verts(area), verts(band))


def test_band_edges_follow_the_same_plateaus_as_the_line(ax):
    """The ribbon and the series it sits behind must agree on where a value starts and stops."""
    line = smooth_step(ax, range(len(Y)), Y, ramp=0.5)
    band = smooth_step_band(ax, range(len(Y)), 0.0, Y, ramp=0.5)
    upper = verts(band)[: len(verts(line))]
    assert np.allclose(upper, verts(line))


def bezier(p0, p1, p2, p3, n=50) -> np.ndarray:
    """Points along one cubic segment."""
    u = np.linspace(0, 1, n)[:, None]
    return (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * p1 + 3 * (1 - u) * u**2 * p2 + u**3 * p3


def test_fillet_risers_are_a_straight_run_between_two_arcs(ax):
    patch = smooth_step(ax, range(len(Y)), Y, ramp=0.5, fillet=4)
    assert code_count(patch, MplPath.CURVE4) == 6 * 2  # two cubic arcs per riser; the flat riser needs none
    assert subpaths(patch) == 1


def test_fillet_arcs_are_circles_on_the_page_whatever_the_aspect(ax):
    """The arc is laid out in display space, so its radius in pixels matches the one asked for in points."""
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 10)  # a strongly anisotropic data space
    radius_pt = 6
    patch = smooth_step(ax, [0, 1], [1.0, 9.0], ramp=0.5, fillet=radius_pt)
    ax.figure.canvas.draw()
    v = verts(patch)  # display coords: end, shoulder, c1, c2, p, q, c1, c2, shoulder, end
    r_px = radius_pt * ax.figure.dpi / 72
    lower_centre = v[1] + [0, r_px]  # the arc leaves the lower plateau at the shoulder, curving upward
    arc = bezier(*v[1:5])
    assert np.allclose(np.hypot(*(arc - lower_centre).T), r_px, rtol=2e-3)
    upper_centre = v[8] - [0, r_px]
    assert np.allclose(np.hypot(*(bezier(*v[5:9]) - upper_centre).T), r_px, rtol=2e-3)


def test_fillet_run_is_tangent_to_both_arcs(ax):
    patch = smooth_step(ax, [0, 1], [0.0, 1.0], ramp=0.6, fillet=5)
    ax.figure.canvas.draw()
    v = verts(patch)
    run = v[5] - v[4]
    for handle in (v[4] - v[3], v[6] - v[5]):  # the handles either side of the run point along it
        cross = run[0] * handle[1] - run[1] * handle[0]
        assert np.isclose(cross, 0, atol=1e-6 * np.linalg.norm(run) * np.linalg.norm(handle))
        assert np.dot(run, handle) > 0


def test_fillet_run_sits_a_fixed_inset_inside_the_shoulders_whatever_the_slope(ax):
    """Both hand-overs sit the same distance inside the shoulder lines, so every riser's run covers the same width."""
    inset_px = _FILLET_INSET * 6 * ax.figure.dpi / 72
    for rise in (0.05, 0.4, 1.0):
        patch = smooth_step(ax, [0, 1], [0.0, rise], ramp=0.5, fillet=6)
        ax.figure.canvas.draw()
        v = verts(patch)
        shoulders = ax.transData.transform([(0.25, 0), (0.75, 0)])[:, 0]
        assert np.allclose(v[[4, 5], 0], shoulders + [inset_px, -inset_px])


def test_a_vertical_riser_keeps_its_arcs_inside_the_footprint(ax):
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 1)  # a tall, narrow step: near-vertical on the page
    patch = smooth_step(ax, [0, 1], [0.0, 1.0], ramp=0.1, fillet=2)
    ax.figure.canvas.draw()
    v = verts(patch)
    shoulders = ax.transData.transform([(0.45, 0), (0.55, 0)])[:, 0]
    spill = (1 - _FILLET_INSET) * 2 * ax.figure.dpi / 72  # a quarter-circle pokes out by the un-inset part of r
    assert v[1, 0] >= shoulders[0] - spill - 0.5 and v[8, 0] <= shoulders[1] + spill + 0.5  # within half a pixel


def test_an_oversize_fillet_is_capped_by_the_plateaus(ax):
    y = [0.0, 1.0, 0.5, 0.2]
    patch = smooth_step(ax, range(len(y)), y, ramp=0.5, fillet=1000)
    ax.figure.canvas.draw()
    v = verts(patch)
    for i in range(len(y) - 1):  # each riser's vertices climb (or fall) monotonically
        ys = v[1 + 8 * i : 9 + 8 * i, 1]
        assert np.all(np.diff(ys) * np.sign(y[i + 1] - y[i]) >= -1e-9)
    assert np.all(np.diff(v[:, 0]) >= -1e-9)  # and neighbouring arcs never cross on a shared plateau


def test_without_plateaus_every_riser_is_still_filleted(ax):
    """At ramp=1 the plateaus have no width, so each arc curls to horizontal at the shared sample point instead."""
    y = [0.0, 1.0, 0.5, 0.2]
    patch = smooth_step(ax, range(len(y)), y, ramp=1.0, fillet=6)
    ax.figure.canvas.draw()
    v = verts(patch)
    assert code_count(patch, MplPath.CURVE4) == 6 * 3  # two arcs on every riser, the steep one included
    assert np.all(np.diff(v[:, 0]) >= -1e-6)  # the path never doubles back on itself
    samples = ax.transData.transform([(i, 0) for i in range(len(y))])[:, 0]
    for i in range(len(y) - 1):  # each arc ends where its sample sits, so neighbours meet there with level tangents
        assert v[1 + 8 * i, 0] >= samples[i] - 1e-6 and v[8 + 8 * i, 0] <= samples[i + 1] + 1e-6


def test_fillet_radius_shrinks_smoothly_as_the_plateaus_close(ax):
    """The peak's arcs at ramp=0.95 and ramp=1 are near neighbours, rather than one round and one sharp."""
    y = [0.0, 1.0, 0.0]
    v = {}
    for ramp in (0.95, 1.0):
        patch = smooth_step(ax, range(len(y)), y, ramp=ramp, fillet=6)
        ax.figure.canvas.draw()
        v[ramp] = verts(patch)
    r_px = 6 * ax.figure.dpi / 72
    extents = {ramp: w[8] - w[5] for ramp, w in v.items()}  # the peak's arc on the way up, hand-over to crest
    for extent in extents.values():
        assert extent[0] > 0.3 * r_px  # it has a visible size
    assert np.allclose(extents[0.95], extents[1.0], atol=0.1 * r_px)


def test_fillet_band_edges_follow_the_same_path_as_the_line(ax):
    """The band walks its lower edge backwards, which must re-describe the arcs too."""
    line = smooth_step(ax, range(len(Y)), Y, ramp=0.5, fillet=3)
    band = smooth_step_band(ax, range(len(Y)), 0.0, Y, ramp=0.5, fillet=3)
    floor = smooth_step(ax, range(len(Y)), [0.0] * len(Y), ramp=0.5, fillet=3)
    ax.figure.canvas.draw()
    n = len(verts(line))
    assert np.allclose(verts(band)[:n], verts(line))
    assert code_count(band, MplPath.CLOSEPOLY) == 1
    assert np.allclose(verts(band)[n:-1][::-1], verts(floor))


def test_fillet_patch_still_feeds_autoscale(ax):
    smooth_step(ax, range(len(Y)), Y, fillet=3)
    ax.autoscale_view()
    assert ax.get_xlim()[0] <= -0.5 and ax.get_xlim()[1] >= len(Y) - 0.5
    assert ax.get_ylim()[0] <= min(Y) and ax.get_ylim()[1] >= max(Y)
