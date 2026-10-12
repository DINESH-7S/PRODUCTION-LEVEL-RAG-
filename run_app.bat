@echo off
setlocal

cd /d "%~dp0"

echo Starting FastAPI on http://127.0.0.1:8000 ...
start "RAG FastAPI" cmd /k ".\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000"

ping 127.0.0.1 -n 6 >nul

echo Starting Streamlit on http://127.0.0.1:8501 ...
start "RAG Streamlit" cmd /k ".\.venv\Scripts\python.exe -m streamlit run dashboard.py --server.address 127.0.0.1 --server.port 8501"

ping 127.0.0.1 -n 5 >nul
start http://127.0.0.1:8501

echo.
echo App started.
echo UI:      http://127.0.0.1:8501
echo API:     http://127.0.0.1:8000
echo.
echo Close the two opened command windows to stop the app.

endlocal
