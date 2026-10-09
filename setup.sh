#!/bin/bash
# Genjutsu Studio setup (macOS). Double-click this once. Takes ~5-10 min (downloads ~2-3 GB of AI libraries).
cd "$(dirname "$0")" || exit 1
set -e
echo "== Genjutsu Studio setup =="
if [[ "$OSTYPE" == darwin* ]]; then
  if ! command -v brew >/dev/null; then echo "Homebrew is needed first. Install it from https://brew.sh then run this again."; [ -t 0 ] && read -n1 -p "Press any key"; exit 1; fi
  for pkg in ffmpeg rubberband python@3.12; do brew list "$pkg" >/dev/null 2>&1 || brew install "$pkg"; done
  PY="$(brew --prefix)/opt/python@3.12/bin/python3.12"
else
  for tool in ffmpeg rubberband python3.12; do command -v "$tool" >/dev/null || { echo "Missing $tool. On Ubuntu/WSL: sudo apt install ffmpeg rubberband-cli python3.12 python3.12-venv"; exit 1; }; done
  PY="$(command -v python3.12)"
fi
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
[ -f .env ] || cp .env.example .env
chmod 600 .env
echo
echo "Done. Now double-click 'start.command' (Mac) or run ./start.sh (Linux/WSL)."
if [[ "$OSTYPE" == darwin* && -t 0 ]]; then read -n1 -p "Press any key to close"; fi
