@echo off
cd /d "%~dp0"
if not exist venv (
  echo Membuat lingkungan Python, tunggu...
  python -m venv venv
  call venv\Scripts\activate
  pip install -r requirements.txt
) else (
  call venv\Scripts\activate
)
start "" http://127.0.0.1:8000
uvicorn main:app --host 127.0.0.1 --port 8000
pause
