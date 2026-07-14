# This file is used to allowlist unused code in the project.
# https://github.com/jendrikseipp/vulture?tab=readme-ov-file#handling-false-positives
# type: ignore

# Context-manager and API-shape false positives: the signatures are fixed by the
# protocol being implemented, not by what the body reads.
exc_val  # unused variable (src/mini/progress_display.py — __exit__/__aexit__)
exc_tb  # unused variable (src/mini/progress_display.py — __exit__/__aexit__)
create_if_missing  # unused variable (tests/mini/test_apparatus.py — mimics modal.Volume.from_name)
Styler  # unused import (src/mini/temporal/dopesheet.py — TYPE_CHECKING-only, used in overload return)

# Pydantic metadata fields: written at construction, read only via serialization.
author  # unused variable (src/experiment/config.py)
fixes  # unused variable (src/experiment/config.py)
total_chars  # unused variable (src/experiment/config.py)
language  # unused variable (src/experiment/config.py)
total_tokens  # unused variable (src/experiment/config.py)
training_tokens  # unused variable (src/experiment/training/metrics.py)
val_loss  # unused variable (src/experiment/training/metrics.py)

# Logging config knobs: part of SimpleLoggingConfig's public surface.
_.base_level  # unused method (src/mini/logging.py)
_.to_stream  # unused method (src/mini/logging.py)
_.critical  # unused method (src/mini/logging.py)
_.trace  # unused method (src/mini/logging.py)
SimpleLoggingConfig  # unused class (src/mini/logging.py)

# Named marimo cells: invoked by the marimo app, invisibly to vulture.
configuration  # unused function (docs/gpt.py)

# Dormant infra, kept deliberately: library surface the demo notebooks don't
# happen to exercise.
as_df  # unused method (src/mini/temporal/dopesheet.py — public API, exercised downstream)
scale_report  # unused method (src/experiment/model/ngpt.py — nGPT diagnostic, exercised downstream)
EntropySeries  # unused class (src/subline/series.py)
Subline  # unused class (src/subline/subline.py)
lr_finder_search  # unused function (src/utils/lr_finder/lr_finder.py)
plot_lr_finder  # unused function (src/utils/lr_finder/vis.py)
group_properties_by_scale  # unused function (src/mini/temporal/vis.py)
