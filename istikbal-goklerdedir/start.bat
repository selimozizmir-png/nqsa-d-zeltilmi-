@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv (
    echo [Istikbal Goklerdedir] First-time setup, installing packages...
    py -3 -m venv .venv || python -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)
echo [Istikbal Goklerdedir] Starting... http://localhost:8501 will open in your browser.
streamlit run app.py
pause
