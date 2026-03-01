from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # c is defined globally.
    # See https://nbconvert.readthedocs.io/en/latest/config_options.html
    c: Any = None

assert c is not None, "Expected 'c' to be defined globally by nbconvert"

c.TemplateExporter.exclude_input_prompt = True
c.TemplateExporter.exclude_output_prompt = True
c.HTMLExporter.anchor_link_text = ''
