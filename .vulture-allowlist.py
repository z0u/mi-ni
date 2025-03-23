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
left  # unused variable (src/utils/theming.py:288)
right  # unused variable (src/utils/theming.py:289)
top  # unused variable (src/utils/theming.py:287)

EntropySeries  # unused class (src/subline/series.py:26)

author  # unused variable (src/experiment/config.py:57)
fixes  # unused variable (src/experiment/config.py:62)
language  # unused variable (src/experiment/config.py:68)
total_chars  # unused variable (src/experiment/config.py:65)
total_chars  # unused variable (src/experiment/config.py:79)
total_tokens  # unused variable (src/experiment/config.py:76)
training_tokens  # unused variable (src/experiment/training/metrics.py:8)
val_loss  # unused variable (src/experiment/training/metrics.py:7)
