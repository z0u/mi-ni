# This file is used to allowlist unused code in the project.
# https://github.com/jendrikseipp/vulture?tab=readme-ov-file#handling-false-positives
# type: ignore


exception  # unused variable (src/mini/_state.py:22)
_.exception  # unused attribute (src/mini/experiment.py:282)
_.before_each  # unused method (src/mini/experiment.py:170)
_.after_each  # unused method (src/mini/experiment.py:170)
_.hither  # unused attribute (src/mini/experiment.py:94)

_.base_level  # unused method (src/utils/logging.py:66)
_.to_stream  # unused method (src/utils/logging.py:71)
_.critical  # unused method (src/utils/logging.py:76)
_.trace  # unused method (src/utils/logging.py:101)

_.format  # unused method (src/utils/logging.py:26)
bottom  # unused variable (src/utils/theming.py:290)
concise_logging  # unused function (src/utils/logging.py:38)
id_sequence  # unused variable (src/utils/dom.py:27)
left  # unused variable (src/utils/theming.py:288)
lr_finder_plot  # unused function (src/utils/lr_finder/vis.py:13)
lr_finder_search  # unused function (src/utils/lr_finder/lr_finder.py:17)
right  # unused variable (src/utils/theming.py:289)
svg_theme_toggle  # unused function (src/utils/theming.py:122)
top  # unused variable (src/utils/theming.py:287)
total_steps  # unused variable (src/utils/lr_finder/types.py:42)
