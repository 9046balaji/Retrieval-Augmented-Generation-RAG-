@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONIOENCODING=utf-8
echo Starting Multi-Query RAG Streamlit Application in 'heart' environment...
C:\ProgramData\anaconda3\envs\heart\python.exe -m streamlit run app.py
pause
