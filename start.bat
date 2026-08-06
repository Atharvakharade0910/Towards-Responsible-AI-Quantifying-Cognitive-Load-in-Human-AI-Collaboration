@echo off
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
  echo Creating Python environment...
  python -m venv venv
  call "venv\Scripts\python.exe" -m pip install -r requirements.txt
)
start "" "http://127.0.0.1:8002/"
echo CogniTrack is running at http://127.0.0.1:8002/
echo Press Ctrl+C to stop it.
"venv\Scripts\python.exe" -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8002
pause
