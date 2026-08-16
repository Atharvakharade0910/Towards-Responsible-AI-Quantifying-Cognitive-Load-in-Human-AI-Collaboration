@echo off
cd /d "%~dp0..\backend"
..\venv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8002
pause
