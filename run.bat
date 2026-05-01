@echo off
title Nifty Quant Screener — Launcher
cd /d "%~dp0"

echo.
echo  ================================================
echo   Nifty Quant Screener — Starting Apps
echo  ================================================
echo.
echo  [1] dashboard.py   http://localhost:8501
echo  [2] app.py         http://localhost:8502
echo.

:: Start each app in its own console window (cmd /k keeps it open for logs)
start "Nifty Dashboard  —  port 8501" cmd /k streamlit run dashboard.py --server.port 8501
timeout /t 3 /nobreak >nul
start "Quant Screener App  —  port 8502" cmd /k streamlit run app.py --server.port 8502

echo  Both servers are starting in separate windows.
echo  Close those windows to stop each server individually.
echo.
echo  ================================================
echo.
pause
