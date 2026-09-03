# type: ignore
# Vulture allowlist: names that are used, but in ways vulture can't see.
#
# Each entry names the file so the reason is checkable. Regenerate candidates with
# `./go check --dead` (scripts/deadcode.sh) and keep only the genuine false positives.

# Context-manager and API-shape false positives: the signatures are fixed by the
# protocol being implemented, not by what the body reads.
exc_val  # unused variable (src/mini/progress_display.py:115, 122 — __exit__/__aexit__)
exc_tb  # unused variable (src/mini/progress_display.py:115, 122 — __exit__/__aexit__)
create_if_missing  # unused variable (tests/mini/test_apparatus.py:271 — mimics modal.Volume.from_name)
Styler  # unused import (src/mini/temporal/dopesheet.py:13 — TYPE_CHECKING-only, used in overload return)
ArrayLike  # unused import (src/mini/vis/plt.py:22 — TYPE_CHECKING-only, used in string annotations)
local_store  # unused variable (tests/mini/test_store_gc.py — fixture requested for its side effect)

# Pydantic metadata fields: written at construction, read only via serialization.
author  # unused variable (src/experiment/config.py)
fixes  # unused variable (src/experiment/config.py)
total_chars  # unused variable (src/experiment/config.py)
language  # unused variable (src/experiment/config.py)
total_tokens  # unused variable (src/experiment/config.py)
training_tokens  # unused variable (src/experiment/training/metrics.py)
val_loss  # unused variable (src/experiment/training/metrics.py)

# Logging config knobs: part of SimpleLoggingConfig's public surface.
_.base_level  # unused method (src/mini/logging.py:67)
_.to_stream  # unused method (src/mini/logging.py:72)
_.critical  # unused method (src/mini/logging.py:77)
_.trace  # unused method (src/mini/logging.py:102)
SimpleLoggingConfig  # unused class (src/mini/logging.py:44)

# Named marimo cells: invoked by the marimo app, invisibly to vulture.
configuration  # unused function (docs/gpt.py)

# Dormant infra, kept deliberately: library surface the demo notebooks don't
# happen to exercise.
as_df  # unused method (src/mini/temporal/dopesheet.py — public API, exercised downstream)
scale_report  # unused method (src/experiment/model/ngpt.py — nGPT diagnostic, exercised downstream)
EntropySeries  # unused class (src/subline/series.py:26)
Subline  # unused class (src/subline/subline.py:11)
lr_finder_search  # unused function (src/utils/lr_finder/lr_finder.py:18)
plot_lr_finder  # unused function (src/utils/lr_finder/vis.py:10)
group_properties_by_scale  # unused function (src/mini/temporal/vis.py:41)
Debouncer  # unused class (src/mini/_debounce.py:16 — BackgroundEmitter took over the hot path)
