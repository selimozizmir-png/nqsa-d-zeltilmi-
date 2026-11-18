#!/usr/bin/env bash
# İstikbal Göklerdedir - macOS / Linux launcher
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  echo "[İstikbal Göklerdedir] First-time setup, installing packages..."
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
else
  source .venv/bin/activate
fi
echo "[İstikbal Göklerdedir] Starting... http://localhost:8501 will open in your browser."
streamlit run app.py
