@echo off
echo Starting WildfireWatch Backend & GIS Frontend...
cd /d "%~dp0backend"
if not exist ".venv" (
    echo Creating Python virtual environment...
    python -m venv .venv
)
echo Installing dependencies...
.\.venv\Scripts\pip install -r requirements.txt -q
echo.
echo ====================================================================
echo WildfireWatch is running! Open http://127.0.0.1:8000/ in your browser
echo ====================================================================
echo.
.\.venv\Scripts\uvicorn main:app --host 127.0.0.1 --port 8000
pause
