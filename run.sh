#!/usr/bin/env bash
# Start the Gift Design Agent demo.
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip
  ./.venv/bin/pip install -r backend/requirements.txt
fi

echo "Open http://127.0.0.1:8000 in your browser"
./.venv/bin/python -m uvicorn server:app --app-dir backend --host 127.0.0.1 --port 8000
