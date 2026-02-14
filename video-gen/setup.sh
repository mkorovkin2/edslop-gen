#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

if ! command -v python3.13 &>/dev/null; then
    echo "Error: python3.13 not found. Please install Python 3.13 first."
    exit 1
fi

echo "Creating virtual environment with Python 3.13..."
python3.13 -m venv "$VENV_DIR"

echo "Installing dependencies..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r requirements.txt

echo "Done. Activate with: source $VENV_DIR/bin/activate"
