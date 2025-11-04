@echo off
"%~dp0\.venv\Scripts\python.exe" -m streamlit run "%~dp0\dashboard_streamlit.py" --server.port 8501 --logger.level info > "%~dp0\streamlit_bg.log" 2>&1
