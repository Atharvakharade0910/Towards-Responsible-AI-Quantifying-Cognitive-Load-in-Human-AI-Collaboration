@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo Project-local Python environment was not found at "%PYTHON_EXE%".
  pause
  exit /b 1
)
set "PYTHONPATH="
"%PYTHON_EXE%" -c "import uvicorn"
if errorlevel 1 (
  echo The project Python environment could not load Uvicorn. Repair .venv and install requirements.txt.
  pause
  exit /b 1
)
start "" "http://127.0.0.1:8002/"
echo Starting CogniTrack at http://127.0.0.1:8002/ ...
echo Press Ctrl+C to stop it.
"%PYTHON_EXE%" -m uvicorn main:app --app-dir backend --host 0.0.0.0 --port 8002
set "COGNITRACK_EXIT_CODE=%ERRORLEVEL%"
pause
exit /b %COGNITRACK_EXIT_CODE%
