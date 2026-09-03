"""Color arithmetic for figures that can't rely on transparency."""

from matplotlib.colors import to_hex, to_rgb

from .theme import light_dark


def page_color() -> str:
    """The color the current theme's figures are meant to be read against.

    Exported figures are transparent, so this is an assumption rather than a fact — but it is the same assumption the stylesheets already make when they pick a text color. Use it to *pre-compute* a color that alpha would otherwise produce, never to paint over something: a mask matched to the page fails on a page of another color, while a pre-computed color that meets a different page is merely a shade off.
    """
    return light_dark("#ffffff", "#111111")


def mix(base: str, over: str, t: float) -> str:
    """*over* laid on *base* at opacity *t*, flattened to one opaque color.

    Matplotlib composites in sRGB, so the result is what the translucent version would have looked like over *base* — with the difference that opaque ink doesn't accumulate. Reach for this where strokes coincide: several translucent copies of one color build up into a darker line, and a reader takes that extra weight for a signal, when all it means is that the values agreed.
    """
    (br, bg, bb), (orr, og, ob) = to_rgb(base), to_rgb(over)
    return to_hex((br + t * (orr - br), bg + t * (og - bg), bb + t * (ob - bb)))
