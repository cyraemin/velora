@echo off
rem Launch Velora in the browser using the project virtual environment.
cd /d "%~dp0.."
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless false
