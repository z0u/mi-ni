import xml.etree.ElementTree as ET
from textwrap import dedent
from typing import Sequence

from subline.series import Series
from subline.sparkline import Sparkline
from subline.types import TokenBB
from utils.dom import Element
from utils.theming import svg_theme_toggle


class Subline:
    def __init__(self, chars_per_line: int = 80):
        self.chars_per_line = chars_per_line
        self.font_size = 14
        self.line_height = self.font_size
        self.line_gap = self.line_height
        self.char_width = 8.4  # Width of each character in SVG units
        self.sparkline_height = 20
        self.margin = 10  # Margin around visualization
        self.legend_height = self.line_height  # Height for legend area
        self.toggle_width = 20  # Space for theme toggle

        # Register SVG namespace for proper rendering
        ET.register_namespace('', 'http://www.w3.org/2000/svg')

    def _wrap_tokens(self, spans: list[TokenBB]) -> list[slice]:
        """Split tokens into lines based on total width, returning (start,end) indices."""
        lines: list[slice] = []
        line_start = 0
        current_width = 0

        for i, span in enumerate(spans):
            if current_width + span.width > self.chars_per_line * self.char_width:
                if line_start < i:  # Don't create empty lines
                    lines.append(slice(line_start, i))
                line_start = i
                current_width = span.width
            else:
                current_width += span.width

        if line_start < len(spans):
            lines.append(slice(line_start, len(spans)))

        return lines

    def _add_legend(self, svg: ET.Element, x: float, y: float, series: list[Series]) -> float:
        """Add a horizontal legend at the bottom of the SVG."""
        legend = Element(svg, 'g', transform=f'translate({x}, {y})')

        # Layout items horizontally with spacing
        item_spacing = 40
        curr_x = 0

        for i, s in enumerate(series):
            # Add line sample
            Element(
                legend,
                'line',
                x1=curr_x,
                y1=0,
                x2=curr_x + 20,
                y2=0,
                stroke=f'var(--col-series-{i + 1})',
                stroke_width=1,
                stroke_dasharray=s.dasharray,
                shape_rendering='crispEdges',
            )
            # Add label
            Element(
                legend,
                'text',
                text=s.label,
                x=curr_x + 25,
                y=4,
                font_family='system-ui',
                font_size=10,
                fill='var(--col-text)',
            )

            # Move to next item position
            curr_x += item_spacing + len(s.label) * 5

        # Return total width used
        return curr_x

    def _get_token_spans(self, tokens: list[str]) -> list[TokenBB]:
        """Calculate token bounding boxes in relative coordinates."""
        spans = []
        for token in tokens:
            width = len(token) * self.char_width
            first_char = self.char_width / 2
            middle = width / 2
            last_char = width - self.char_width / 2
            spans.append(TokenBB(width, first_char, middle, last_char))
        return spans

    def _add_text_line(
        self,
        parent: ET.Element,
        tokens: Sequence[str],
        window: slice,
        x: float,
        y: float,
    ):
        """Add a line of text with centered tokens."""
        line_tokens = tokens[window]

        if x != 0.0 or y != 0.0:
            parent = Element(parent, 'g', transform=f'translate({x}, {y})')

        # Add main text element with centered alignment
        baseline = self.font_size * -0.2  # Still need this offset for text positioning
        text_elem = Element(
            parent,
            'text',
            font_size=self.font_size,
            y=baseline,
            text_anchor='middle',
            fill='var(--col-text)',
        )

        # Track cumulative x position as we place tokens
        pos = 0
        for token in line_tokens:
            width = len(token) * self.char_width
            mid = pos + width / 2
            Element(text_elem, 'tspan', x=mid, text=token)
            pos += width

    def plot(self, tokens: str | list[str], series: list[Series]):
        """Generate and display an SVG visualization of text metrics."""
        if isinstance(tokens, str):
            tokens = list(tokens)

        # Split tokens into lines and calculate dimensions
        spans = self._get_token_spans(tokens)
        lines = self._wrap_tokens(spans)

        # Calculate heights (these won't change)
        full_line_height = self.line_height + self.sparkline_height + self.line_gap
        content_height = len(lines) * full_line_height
        total_height = content_height + 2 * self.margin + self.line_gap + self.legend_height

        # Create SVG root without viewBox initially
        svg = Element(
            None,
            'svg',
            xmlns='http://www.w3.org/2000/svg',
            style='background-color: var(--bg-color); box-shadow: 0 0 0 10px var(--bg-color);',
        )
        Element(
            svg,
            'style',
            text=dedent("""
                @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Mono:wght@100..900&family=Source+Code+Pro&display=swap');
                text {
                    font-family: "Source Code Pro", "Noto Sans Mono", monospace !important;
                    font-optical-sizing: auto;
                    font-weight: 400;
                    white-space: pre;
                }
            """),
        )

        # Add text content and sparklines
        sparkline = Sparkline()
        for i, s in enumerate(series):
            sparkline.add_series(
                s.values,
                color=f'var(--col-series-{i + 1})',
                dasharray=s.dasharray,
            )

        text_width = self.chars_per_line * self.char_width
        for i, window in enumerate(lines):
            y_offset = i * full_line_height + self.margin
            baseline = y_offset + self.font_size + 1
            self._add_text_line(svg, tokens, window, self.margin, baseline)
            sparkline.render(
                parent=svg,
                spans=spans,
                window=window,
                x=self.margin,
                y=baseline + 1,
                h=self.sparkline_height,
            )

        # Add legend and get its width
        legend_y = content_height + self.line_gap + self.legend_height / 2
        legend_width = self._add_legend(svg, self.margin, legend_y, series)

        # Calculate final width needed
        min_width = text_width + 2 * self.margin
        total_width = max(min_width, legend_width + self.toggle_width + 2 * self.margin)

        # Add theme toggle using final width
        svg_theme_toggle(
            svg,
            toggle_pos=(
                total_width - self.toggle_width / 2,
                total_height - self.toggle_width / 2,
            ),
            theme_vars={
                'col-series-1': ('#ef4444', '#ff7878'),
                'col-series-2': '#3b82f6',
                'col-series-3': ('#22c55e', '#45e881'),
                'col-series-4': ('#f97316', '#ffa261'),
                'col-series-5': ('#a855f7', '#d9b1ff'),
                'col-text': ('#666666', '#dddddd'),
                'col-baseline': ('#cccccc', '#666666'),
                'blend-mode': ('multiply', 'screen'),
            },
        )

        # Set viewBox now that we know final dimensions
        svg.set('viewBox', f'0 0 {total_width} {total_height}')

        return ET.tostring(svg, encoding='unicode')
