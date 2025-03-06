"""
Dynamic theming support for notebooks.

This is a collection of utilities for making figures display nicely in notebooks. They adapt to the user's light/dark mode preference automatically, and have a button to override the current setting.


## SVG light/dark theming

Suppose we have an SVG [etree](https://docs.python.org/3/library/xml.etree.elementtree.html) with a structure like this:

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">
  <rect x="10" y="10" width="180" height="80" fill="var(--bg-color)" rx="5" />
  <text x="40" y="40" fill="var(--text-color)">Theme toggle!</text>
</svg>
```

Let's add dynamic theme support, based on those CSS variables!

```python
svg = et.Element("svg", ...)

svg_theme_toggle(
    svg,
    toggle_pos=(50, 60),  # Position of the toggle button in SVG units
    theme_vars={ "text-color": ("purple", "hotpink") }
)
display(HTML(et.tostring(svg, encoding='unicode')))
```


## HTML light/dark theming

Suppose we have some HTML — maybe it contains a figure as PNG, or an SVG with a structure that we don't control.

```python
img_io = BytesIO()
fig.savefig(img_io, format='png')
png_str = base64.b64encode(img_io.getvalue()).decode()
html_str = f'<img src="data:image/png;base64,{png_str}"/>'
```

Let's add dynamic theming support to it!

```python
themed_html = html_theme_toggle(
    html_str,
    toggle_pos=Anchor(top='10px', right='10px'))
display(HTML(themed_html))
```

For best results:
- The un-themed source should be in light mode
- It should have a transparent background: `html_theme_toggle` will add a background to match the notebook environment. E.g. in Matplot, use ` plt.style.use('default')` and `fig.patch.set_alpha(0)`.
"""

from xml.etree import ElementTree as et
from typing import Mapping, Optional, TypeAlias
from textwrap import dedent
from dataclasses import dataclass, asdict


def detect_notebook_env():
    """Detect what kind of notebook we're running in."""
    import os
    import sys

    if 'SPACE_ID' in os.environ:
        return 'huggingface'
    elif 'google.colab' in sys.modules:
        return 'colab'
    elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        return 'kaggle'
    elif 'PAPERSPACE_NOTEBOOK_REPO' in os.environ:
        return 'paperspace'
    else:
        return 'other'


css_snippets = {
    'sr-only': """.sr-only {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0,0,0,0);
        white-space: nowrap;
        border: 0;
    }"""
}

LightDark: TypeAlias = tuple[str, str]
Theme: TypeAlias = Mapping[str, LightDark]
PartialTheme: TypeAlias = Mapping[str, str | LightDark]


_default_theme: PartialTheme = {
    'bg-color': ('#fff', '#2a2a2a'),
    'sun-color': 'goldenrod',
    'moon-color': 'silver',
}
match detect_notebook_env():
    case 'huggingface':
        _default_theme |= {'bg-color': ('white', '#0f0f11')}
    case 'colab':
        _default_theme |= {'bg-color': ('#f0f0f0', '#111')}
    case 'kaggle':
        _default_theme |= {'bg-color': ('#fff', 'rgb(32, 33, 36)')}
    case 'paperspace':
        _default_theme |= {'bg-color': ('#f0f0f0', '#111')}

_default_font = 'Segoe UI Symbol, DejaVu Sans, Noto Sans, ui-sans-serif, system-ui'


def create_theme(**theme_vars: str | LightDark) -> Theme:
    theme_vars = {**_default_theme, **theme_vars}
    return {k: (v, v) if isinstance(v, str) else v for k, v in theme_vars.items()}


def svg_theme_toggle(
    parent: et.Element,
    toggle_pos: tuple[float, float],
    theme_vars: PartialTheme,
    *,  # Force keyword args for the rest
    icon_font: str = _default_font,
    sun_size: float = 24,
    moon_size: float = 16,
) -> et.Element:
    """
    Create a theme toggle switch with sun/moon icons in an SVG document.

    Args:
        parent: Parent SVG element to attach to
        toggle_pos: (x,y) position for the toggle group
        theme_vars: Dict mapping CSS var names to (light, dark) value tuples
        icon_font: Font stack for icons
        sun_size: Font size for sun icon in px
        moon_size: Font size for moon icon in px

    Returns:
        The created toggle group element
    """
    theme_vars = create_theme(**theme_vars)

    # Add style element with our CSS rules
    style = et.SubElement(parent, 'style')

    # Build CSS rules
    css_rules = []

    # Light theme default (when checkbox unchecked)
    light_rules = [f'--{var}: {light_val};' for var, (light_val, _) in theme_vars.items()]
    css_rules.append(f"""
        svg:not(:has(input[data-light-dark-toggle]:checked)) {{
            {' '.join(light_rules)}
            --sun-opacity: 1;
            --moon-opacity: 0;
        }}
    """)

    # Dark theme when toggled (checkbox checked)
    dark_rules = [f'--{var}: {dark_val};' for var, (_, dark_val) in theme_vars.items()]
    css_rules.append(f"""
        svg:has(input[data-light-dark-toggle]:checked) {{
            {' '.join(dark_rules)}
            --sun-opacity: 0;
            --moon-opacity: 1;
        }}
        /* For content that we don't control, like matplot charts */
        svg:has(input[data-light-dark-toggle]:checked) .theme-content {{
            filter: var(--content-filter);
        }}
    """)

    # System preference - dark
    css_rules.append(f"""
        @media (prefers-color-scheme: dark) {{
            svg:not(:has(input[data-light-dark-toggle]:checked)) {{
                {' '.join(dark_rules)}
                --sun-opacity: 0;
                --moon-opacity: 1;
            }}
            svg:has(input[data-light-dark-toggle]:checked) {{
                {' '.join(light_rules)}
                --sun-opacity: 1;
                --moon-opacity: 0;
            }}
        }}
    """)

    # Shared styles for icons and transitions
    css_rules.append(f"""
        .icon {{
            font-family: {icon_font};
            text-anchor: middle;
            dominant-baseline: central;
            transition: fill 0.3s, opacity 0.3s;
        }}

        rect, circle, line, path {{
            transition: fill 0.3s, stroke 0.3s;
        }}
        svg {{
            transition: all 0.3s
        }}
    """)

    style.text = '\n'.join(css_rules)

    # Create toggle group with proper ARIA attributes
    x, y = toggle_pos
    toggle_group = et.SubElement(
        parent,
        'g',
        {
            'transform': f'translate({x},{y})',
            'role': 'switch',
            'aria-checked': 'false',
            'tabindex': '0',  # Make it focusable
        },
    )

    # Add sun icon
    et.SubElement(
        toggle_group,
        'text',
        {
            'class': 'icon',
            'x': '0',
            'y': '0',
            'font-size': f'{sun_size}px',
            'fill': 'var(--sun-color)',
            'opacity': 'var(--sun-opacity)',
            'aria-hidden': 'true',
        },
    ).text = '☀'

    # Add moon icon
    et.SubElement(
        toggle_group,
        'text',
        {
            'class': 'icon',
            'x': '0',
            'y': '0',
            'font-size': f'{moon_size}px',
            'fill': 'var(--moon-color)',
            'opacity': 'var(--moon-opacity)',
            'role': 'presentation',
            'aria-hidden': 'true',
        },
    ).text = '◑'

    # Add invisible checkbox using foreignObject
    fo = et.SubElement(
        toggle_group,
        'foreignObject',
        {
            'x': '-10',  # Offset to center over icons
            'y': '-10',
            'width': '20',
            'height': '20',
        },
    )

    # Note: We have to set the XHTML namespace for the input element
    input_el = et.Element(
        'input',
        {
            'xmlns': 'http://www.w3.org/1999/xhtml',
            'type': 'checkbox',
            'data-light-dark-toggle': '',
            'title': 'Toggle light/dark',
            'aria-label': 'Toggle between light and dark theme',
            'style': 'width: 20px; height: 20px; opacity: 0.001; cursor: pointer;',
        },
    )
    fo.append(input_el)

    return toggle_group


@dataclass
class Anchor:
    top: Optional[str] = None
    left: Optional[str] = None
    right: Optional[str] = None
    bottom: Optional[str] = None

    def __str__(self):
        return ' '.join(f'{prop}: {value};' for prop, value in asdict(self).items() if value is not None)


def html_theme_toggle(
    html_content: str,
    toggle_pos: Anchor,
    theme_vars: PartialTheme,
    *,
    icon_font=_default_font,
) -> str:
    """Wrap any SVG content with a theme toggle using HTML."""
    theme_vars = create_theme(**theme_vars)
    light_rules = [f'--{var}: {light_val};' for var, (light_val, _) in theme_vars.items()]
    dark_rules = [f'--{var}: {dark_val};' for var, (_, dark_val) in theme_vars.items()]

    # Template our wrapper HTML
    return dedent(f"""
    <div class="theme-wrapper" style="position: relative;">
        <style>
            .theme-wrapper:not(:has(input[data-light-dark-toggle]:checked)) {{
                /* Light theme (user's default) */
                {' '.join(light_rules)}
                --sun-opacity: 1;
                --moon-opacity: 0;
                --content-filter: none;
            }}

            .theme-wrapper:has(input[data-light-dark-toggle]:checked) {{
                /* Dark theme (overridden) */
                {' '.join(dark_rules)}
                --sun-opacity: 0;
                --moon-opacity: 1;
                --content-filter: invert(1) hue-rotate(180deg);
            }}

            @media (prefers-color-scheme: dark) {{
                .theme-wrapper:not(:has(input[data-light-dark-toggle]:checked)) {{
                    /* Dark theme (user's default) */
                    {' '.join(dark_rules)}
                    --sun-opacity: 0;
                    --moon-opacity: 1;
                    --content-filter: invert(1) hue-rotate(180deg);
                }}
                .theme-wrapper:has(input[data-light-dark-toggle]:checked) {{
                    /* Light theme (overridden) */
                    {' '.join(light_rules)}
                    --sun-opacity: 1;
                    --moon-opacity: 0;
                    --content-filter: none;
                }}
            }}

            .theme-wrapper {{
                background-color: var(--bg-color);
                transition: 0.3s background-color;
            }}
            .toggle-icons {{
                font-family: {_default_font};
                font-size: 24px;
                line-height: 1;
                &.light-icon {{
                    color: var(--sun-color);
                    opacity: var(--sun-opacity);
                }}
                &.dark-icon {{
                    color: var(--moon-color);
                    opacity: var(--moon-opacity);
                    /* Overlay the light icon */
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%) scale(0.8);
                }}
                transition: 0.3s opacity;
            }}
            .theme-content {{
                transition: 0.3s filter;
            }}
            {css_snippets['sr-only']}
        </style>

        <!-- Theme toggle -->
        <label class="theme-toggle" style="
                    position: absolute;
                    {str(toggle_pos)}
                    z-index: 1;
                    cursor: pointer;
                "
                title="Toggle light/dark">
            <input
                type="checkbox"
                data-light-dark-toggle
                style="position: absolute; opacity: 0;"
            >
            <span class="sr-only">Toggle between light and dark theme</span>
            <span class="toggle-icons light-icon" aria-hidden="true">☀</span>
            <span class="toggle-icons dark-icon" aria-hidden="true">◑</span>
        </label>

        <!-- Content with filter -->
        <div class="theme-content" style="filter: var(--content-filter);">
            {html_content}
        </div>
    </div>
    """).strip()


def fig_theme_toggle(fig, anchor=None, theme_vars=None, already_dark=False):
    import base64
    from io import BytesIO

    img_io = BytesIO()
    fig.savefig(img_io, format='png')
    png_str = base64.b64encode(img_io.getvalue()).decode()

    # If image is already dark, pre-invert
    style = 'style="filter: invert(1) hue-rotate(180deg)"' if already_dark else ''

    themed_html = html_theme_toggle(
        f'<img {style} src="data:image/png;base64,{png_str}"/>',
        toggle_pos=anchor or Anchor(top='10px', right='10px'),
        theme_vars=theme_vars or {},
    )
    return themed_html
