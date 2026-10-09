#!/bin/bash
# Starts Genjutsu Studio and opens it in your browser. Keep this window open while you use it.
cd "$(dirname "$0")" || exit 1
[ -x .venv/bin/python ] || { echo "Run setup first (setup.command)."; read -n1; exit 1; }
PORT="${PORT:-8790}"
( sleep 2; if [[ "$OSTYPE" == darwin* ]]; then open "http://127.0.0.1:$PORT"; elif command -v wslview >/dev/null; then wslview "http://127.0.0.1:$PORT"; else xdg-open "http://127.0.0.1:$PORT" >/dev/null 2>&1; fi ) &
echo "Genjutsu Studio: http://127.0.0.1:$PORT  (close this window to stop it)"
PORT="$PORT" exec .venv/bin/python studio.py
