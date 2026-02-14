@echo off
echo Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Installation failed. Please check errors above.
    pause
    exit /b %errorlevel%
)

echo Starting Translation Data Builder...
python -m streamlit run app.py
pause
