#!/bin/sh
# Double-click to start ImageMixer on macOS. All the work happens in launch.py.
cd "$(dirname "$0")" || exit 1
for py in python3 python; do
  if command -v "$py" >/dev/null 2>&1; then exec "$py" launch.py "$@"; fi
done
echo "Python 3.10 or newer is required. Install it from https://www.python.org/downloads/"
read -r _
