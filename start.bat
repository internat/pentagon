@echo off
title Intelligent Data Analysis System - AutoML Dashboard
cd /d "%~dp0"
echo ============================================================
echo   INTELLIGENT DATA ANALYSIS SYSTEM - AutoML Dashboard
echo ============================================================
echo.
echo   Starting local server and opening the browser...
echo   Dashboard: http://localhost:5000
echo.
echo   Training / Statistics / Methodology / Examples
echo   Extended session (40+ min): python extended_training.py
echo.
start "" python start.py
timeout /t 3 /nobreak >nul
start http://localhost:5000
echo   Browser opened!
echo   Press any key to stop the server...
pause >nul
taskkill /f /im python.exe >nul 2>&1
echo   Server stopped.
