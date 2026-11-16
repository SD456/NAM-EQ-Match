#!/usr/bin/env bash
# ============================================================
#  NAM EQ Matcher - one-click launcher for macOS
#  Double-click this file (first time: right-click > Open).
#  If macOS says it can't be executed, run once in Terminal:
#      chmod +x start-nam-eq-matcher-macos.command
# ============================================================
set -e
cd "$(dirname "$0")"

echo
echo " === NAM EQ Matcher ==="
echo

if ! command -v python3 >/dev/null 2>&1; then
    echo " [ERROR] Python 3 was not found."
    echo " Install it from https://www.python.org/downloads/ (3.10 or newer),"
    echo " then double-click this file again."
    read -r -p " Press Enter to close..."
    exit 1
fi

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo " [ERROR] Your Python is older than 3.10. Please install a newer one"
    echo " from https://www.python.org/downloads/ and try again."
    read -r -p " Press Enter to close..."
    exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    echo " First run: creating a private Python environment..."
    python3 -m venv .venv
fi

if [ ! -f ".venv/.deps_installed" ]; then
    echo
    echo " Installing dependencies - this one-time step can take several minutes..."
    echo
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
    touch .venv/.deps_installed
fi

echo
echo " Starting NAM EQ Matcher - your browser will open at http://127.0.0.1:7860"
echo " Keep this window open while using the app. Press Ctrl+C to stop."
echo
exec .venv/bin/python app.py
