import xml.etree.ElementTree as ET
from itertools import count
from random import randbytes


def Element(parent: ET.Element | None, tag: str, text: str | None = None, **attrs) -> ET.Element:
    """Create an XML element with case-preserved attributes and optional text content."""
    attrs = {k.replace("_", "-"): format_float(v) if isinstance(v, float) else str(v) for k, v in attrs.items()}
    elem = ET.SubElement(parent, tag, attrs) if parent is not None else ET.Element(tag, attrs)
    if text is not None:
        elem.text = text
    return elem


def format_float(v: float):
    return f"{v:.4g}"


def gen_ids():
    """For SVG defs. Unique within a single run, and _likely_ to be unique across runs (~1% chance after 10k runs)."""
    counter = count()
    seed = randbytes(4).hex()
    while True:
        yield f"{seed}-{next(counter):x}"


id_sequence = gen_ids()
