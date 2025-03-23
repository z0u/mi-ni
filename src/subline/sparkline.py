import xml.etree.ElementTree as ET
from math import isnan
from typing import Sequence

import numpy as np

from subline.series import Series
from subline.types import TokenBB
from utils.dom import Element, id_sequence


class Sparkline:
    """A multi-series sparkline plot that aligns with text characters."""

    def __init__(self):
        self.series = []
        self.char_width = 8.4
        self.stroke_width = 1.0
        self.baseline_width = 3.0

    def add_series(self, values: np.ndarray, color: str, dasharray=''):
        """Add a new data series to the plot."""
        self.series.append(Series(values, color, dasharray=dasharray))
        return self

    def _create_path_data(
        self,
        values: Sequence[float],
        spans: Sequence[TokenBB],
        window: slice,
        h: float,
    ) -> str:
        """Create SVG path data for values, breaking at NaN values."""
        # Expand window to include one token of context on either side
        peek = slice(max(0, window.start - 1), window.stop + 1, None)

        # Start x at -width of the peek token if we have one
        x = -(spans[peek.start].width) if peek.start < window.start else 0

        points = []
        is_drawing = False

        for span, v in zip(spans[peek], values[peek], strict=True):
            first, last = span.first_char, span.last_char
            cp_dx1 = first
            cp_dx2 = span.width - last
            knot1 = (x + first - cp_dx1, x + first)
            knot2 = (x + last - cp_dx2, x + last)

            if isnan(v):
                is_drawing = False
            else:
                y = h - h * v
                cp, vertex = knot1
                if is_drawing:
                    points.append(f'S {cp:.1f},{y:.1f} {vertex:.1f},{y:.1f}')
                else:
                    points.append(f'M {vertex:.1f},{y:.1f}')
                    is_drawing = True

                if span.is_wide:
                    cp, vertex = knot2
                    points.append(f'S {cp:.1f},{y:.1f} {vertex:.1f},{y:.1f}')

            x += span.width

        return ' '.join(points)

    def _render_series(
        self,
        parent: ET.Element,
        values: Sequence[float],
        spans: list[TokenBB],
        window: slice,
        h: float,
        color: str,
        dasharray: str,
    ):
        """Render a single sparkline series."""
        path_data = self._create_path_data(values, spans, window, h)
        return Element(
            parent,
            'path',
            d=path_data,
            fill='none',
            stroke=color,
            stroke_width=self.stroke_width,
            stroke_dasharray=dasharray,
            style='mix-blend-mode: var(--blend-mode);',
        )

    def render(self, parent: ET.Element, spans: list[TokenBB], window: slice, x=0.0, y=0.0, h=20.0):
        """Render all series into the plot at the specified position."""
        # Create transform group if needed
        if x != 0.0 or y != 0.0:
            parent = Element(parent, 'g', transform=f'translate({x}, {y})')

        w = sum(span.width for span in spans[window])

        clip = Element(parent, 'clipPath', id=f'clip-{next(id_sequence)}')
        Element(clip, 'rect', x=0, y=-h, width=w, height=h * 2)

        # Render each series
        for series in self.series:
            path = self._render_series(
                parent=parent,
                values=series.values,
                spans=spans,
                window=window,
                h=h,
                color=series.color,
                dasharray=series.dasharray,
            )
            path.set('clip-path', f'url(#{clip.get("id")})')

        # Render token baseline, so it's clear where each one starts and ends
        segments = []
        x = 0.0
        for span in spans[window]:
            segments.append((x + span.first_char, x + span.last_char))
            x += span.width

        dx = 0.2
        Element(
            parent,
            'path',
            d=' '.join(
                f'M{first_char - dx:.1f},{h:.1f} L{last_char + dx:.1f},{h:.1f}' for first_char, last_char in segments
            ),
            fill='none',
            stroke='var(--col-baseline)',
            stroke_width=self.baseline_width,
            stroke_linecap='round',
            style='mix-blend-mode: var(--blend-mode);',
        )

        return self
