@echo off
echo ============================================================
echo STARTING TRADING SYSTEM
echo ============================================================
echo.
echo This will start TWO windows:
echo   1. Trading Bot (runs your TCN model)
echo   2. Dashboard (monitors performance)
echo.
echo Press any key to continue...
pause > nul

start "Trading Bot" cmd /k python run_bot.py
timeout /t 3 /nobreak > nul
start "Dashboard" cmd /k python run_dashboard.py

echo.
echo ============================================================
echo SYSTEM STARTED
echo ============================================================
echo.
echo - Trading Bot: Running in separate window
echo - Dashboard: http://localhost:8501
echo.
echo Close the windows to stop trading.
echo ============================================================

