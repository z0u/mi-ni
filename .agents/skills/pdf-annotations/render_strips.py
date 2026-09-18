r"""Render an annotated PDF into strips a vision model can read, plus a manifest.

Usage (pymupdf is not a project dependency, so run it under uvx; the proxy
env is needed in the remote sandbox, see SKILL.md):

    env NO_PROXY=localhost no_proxy=localhost \\
        uvx --with pymupdf python .claude/skills/pdf-annotations/render_strips.py in.pdf outdir

Writes, per page: `pNN.txt` (printed text), `pNN_sK.png` (strips), and one
`manifest.md` listing every strip with its ink count and colours, so the
transcriber can skip blank strips and knows where the marks are.

Ink from a reMarkable export is vector geometry in the page content stream, not
PDF annotation objects. A pen stroke arrives as a filled path made of many
curve segments; a highlighter stroke as a wide stroked line. Printed page
elements are single rectangles (backgrounds, badges, bars) or thin black
strokes (rules), so those are counted separately as "shapes" rather than ink,
and a coloured glyph that repeats with identical geometry (list bullets, icons)
is printed, since a hand never draws the same path twice.
Each page is cropped to its content (text plus ink) before it is cut into
strips, since a tall page is mostly empty below the text.
"""

import sys
from collections import Counter
from pathlib import Path

import pymupdf  # ty: ignore[unresolved-import]  (installed by uvx, not a project dependency)

ZOOM = 2.0  # 2x is legible for handwriting at ~450pt page width
STRIP_PT = 900  # height of one strip in points, before zoom
OVERLAP_PT = 40
MARGIN_PT = 24

NAMED = {
    (0.76, 0.19, 0.2): "red",
    (0.75, 0.5, 0.82): "purple",
    (0.16, 0.49, 0.23): "green",
    (0.62, 0.42, 0.0): "gold",
    (0.39, 0.45, 0.55): "slate",
    (1.0, 0.33, 0.81): "magenta",
}


def colour_name(rgb):
    key = tuple(round(c, 2) for c in rgb)
    return NAMED.get(key, f"rgb{key}")


def colour_of(d):
    return d.get("fill") if d["type"] == "f" else d.get("color")


def is_plain(col):
    return col is None or all(c > 0.98 for c in col) or all(c < 0.02 for c in col)


def is_ink(d):
    col = colour_of(d)
    if is_plain(col):
        return False
    if d["type"] == "f":
        return len(d["items"]) > 1  # a pen stroke is many curve segments; a printed box is one rect
    return (d.get("width") or 0) > 3  # a highlighter stroke is wide; a printed rule is hairline


def signature(d):
    return (round(d["rect"].width, 1), round(d["rect"].height, 1), len(d["items"]))


def box(d):
    return d["rect"] + (-1, -1, 1, 1)  # a horizontal stroke has a zero-height rect, which never "intersects"


def is_shape(d):
    return not is_plain(colour_of(d)) and not is_ink(d)


def main(pdf, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf)
    seen = Counter(signature(d) for page in doc for d in page.get_drawings())
    n_annots = sum(len(list(page.annots())) for page in doc)
    rows = []
    n_strips = 0
    for i, page in enumerate(doc):
        n = i + 1
        (out / f"p{n:02d}.txt").write_text(page.get_text())
        drawings = [d for d in page.get_drawings() if seen[signature(d)] < 3]
        ink = [d for d in drawings if is_ink(d)]
        shapes = [d for d in drawings if is_shape(d)]
        boxes = [pymupdf.Rect(b[:4]) for b in page.get_text("blocks")] + [box(d) for d in ink]
        if not boxes:
            rows.append((f"p{n:02d}", "-", 0, "", "", "blank page"))
            continue
        content = pymupdf.Rect(
            page.rect.x0, min(b.y0 for b in boxes) - MARGIN_PT, page.rect.x1, max(b.y1 for b in boxes) + MARGIN_PT
        )
        content &= page.rect
        y, k = content.y0, 0
        while y < content.y1 - OVERLAP_PT:
            k += 1
            y0, y1 = max(content.y0, y - OVERLAP_PT), min(content.y1, y + STRIP_PT + OVERLAP_PT)
            clip = pymupdf.Rect(content.x0, y0, content.x1, y1)
            name = f"p{n:02d}_s{k}"
            page.get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM), clip=clip).save(out / f"{name}.png")
            here = [d for d in ink if box(d).intersects(clip)]
            colours = sorted({colour_name(colour_of(d)) for d in here})
            shape_cols = sorted({colour_name(colour_of(d)) for d in shapes if box(d).intersects(clip)})
            has_text = any(pymupdf.Rect(b[:4]).intersects(clip) for b in page.get_text("blocks"))
            note = "ink" if here else ("printed text only" if has_text else "blank")
            rows.append((name, f"{y0:.0f}-{y1:.0f}pt", len(here), ", ".join(colours), ", ".join(shape_cols), note))
            n_strips += 1
            y += STRIP_PT
    lines = [
        f"# Strip manifest for {Path(pdf).name}",
        "",
        f"{len(doc)} pages, {n_strips} strips at {ZOOM}x, {n_annots} PDF annotation objects (if not zero, read those with pymupdf instead of the images). Ink is pen and highlighter strokes touching the strip; shapes are other coloured rectangles (a margin bar drawn with the rectangle tool lands here, and so do printed badges and admonition backgrounds).",
        "",
        "| strip | span | ink strokes | ink colours | shape colours | note |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    lines += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |" for r in rows]
    (out / "manifest.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
