@echo off
cd /d "%~dp0"

REM 使用 conda.bat 启动环境
call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate DBDEazyQTE-main

REM 执行 Python 脚本
python DBDEazyQTE.py

pause
