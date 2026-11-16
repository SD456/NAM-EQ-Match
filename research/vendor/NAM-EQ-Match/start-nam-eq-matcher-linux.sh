#!/usr/bin/env bash
# ============================================================
#  NAM EQ Matcher - one-click launcher for Linux
#  Run with:   bash start-nam-eq-matcher-linux.sh
#  (or make it double-clickable: chmod +x start-nam-eq-matcher-linux.sh)
# ============================================================
set -e
cd "$(dirname "$0")"

echo
echo " === NAM EQ Matcher ==="
echo

if ! command -v python3 >/dev/null 2>&1; then
    echo " [ERROR] Python 3 was not found."
    echo " Install it with your package manager, e.g.:"
    echo "   sudo apt install python3 python3-venv python3-pip     (Debian/Ubuntu)"
    echo "   sudo dnf install python3 python3-pip                  (Fedora)"
    exit 1
fi

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo " [ERROR] Your Python is older than 3.10. Please install a newer one."
    exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    echo " First run: creating a private Python environment..."
    python3 -m venv .venv || {
        echo " [ERROR] venv creation failed. On Debian/Ubuntu run:"
        echo "   sudo apt install python3-venv"
        exit 1
    }
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
echo " Keep this terminal open while using the app. Press Ctrl+C to stop."
echo
exec .venv/bin/python app.py
