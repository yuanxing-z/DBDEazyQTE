@echo off
REM Navigate to script directory
cd /d "%~dp0"

REM Initialize Conda for the current shell session
call "%USERPROFILE%\anaconda3\Scripts\activate.bat"

REM Activate your Conda environment (replace name if needed)
call conda activate DBDEazyQTE-main

REM Run the script
python DBDEazyQTE.py
