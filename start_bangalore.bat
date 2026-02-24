@echo off
title Video Analytics - Bangalore Office
echo ============================================
echo   Video Analytics - Bangalore Office
echo   Dahua NVR Channel 8 (Sub Stream)
echo   NVR: 172.16.11.89
echo ============================================
echo.

cd /d "%~dp0"

echo Starting detection module...
python detect.py --config config_bangalore.yaml

pause
