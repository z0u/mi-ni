#!/usr/bin/env bash

set -euo pipefail

# Make the volume mounts writable.
sudo chown -R "$USER:$USER" /home/vscode/.cache/uv
sudo chown -R "$USER:$USER" .venv

uv venv --allow-existing
uv sync
