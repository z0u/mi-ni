#!/bin/bash

set -euo pipefail



# Convert notebooks one at a time with ipynb_to_py.py
mkdir -p vulture_temp
for notebook in $(find notebooks/ -name "*.ipynb"); do
    # Get the base name of the notebook without the extension
    base_name=$(basename "$notebook" .ipynb)

    # Convert the notebook to a Python script
    python scripts/ipynb_to_py.py "$notebook" "vulture_temp/$base_name.py"
done


# Run vulture on the project, including the converted notebooks
vulture src/ tests/ vulture_temp/ --make-whitelist

# Clean up
rm -rf vulture_temp/
