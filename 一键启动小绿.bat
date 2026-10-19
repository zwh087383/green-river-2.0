@echo off
setlocal
pushd "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\start_windows.ps1"
if errorlevel 1 goto failed
popd
exit /b 0

:failed
echo Startup failed. Press any key to close this window.
pause >nul
popd
exit /b 1
