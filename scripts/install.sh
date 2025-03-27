#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RC_FILE="${SCRIPT_DIR}/.install.sh.rc"


save_device_type() {
  echo "$1" > "$RC_FILE"
}


validate_device() {
  case "$1" in
    cpu|gpu|any) ;;
    *)
      echo "Error: Invalid device type '$1'. Valid options are 'cpu', 'gpu', or 'any' (default for current platform)." >&2
      return 1
      ;;
  esac
  return 0
}


# Process options

previous_device=$(cat "$RC_FILE" 2>/dev/null || echo "")
if ! validate_device "$previous_device" 2>/dev/null; then
  previous_device="any"
fi

device=""

show_usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --device=cpu|gpu|any  set device to install libraries for (default: $previous_device)"
  echo "  --help                show this help message"
}

# Handle arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device=*)
      device="${1#*=}"
      validate_device "$device" || exit 1
      ;;
    --help|-h)
      show_usage
      exit 0
      ;;
    *)
      echo "Error: Unknown option '$1'" >&2
      show_usage
      exit 1
      ;;
  esac
  shift
done


if [[ "$device" == "" ]]; then
  device="$previous_device"
fi


validate_device "$device"
save_device_type "$device"


if [[ "$device" == "gpu" ]]; then
  torch_opt=--extra=torch-gpu
elif [[ "$device" == "cpu" ]]; then
  torch_opt=--extra=torch-cpu
else
  torch_opt=--extra=torch
fi

( set -x; uv sync --all-groups "$torch_opt" < /dev/null )


echo "✅ Installation complete"
