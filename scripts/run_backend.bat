@echo off
cd /d "%~dp0.."
if not exist venv (
  python -m venv venv
)
call venv\Scripts\activate
pip install -r requirements.txt
if not exist .env copy .env.example .env
cd /d "%~dp0..\backend"
uvicorn main:app --reload --host 127.0.0.1 --port 8002
pause
