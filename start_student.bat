@echo off
REM Baqylau Student — desktop app (dev launch via Python).
REM For exams use BaqylauStudent.exe (build with build_exe.bat).
cd /d "%~dp0"
if exist "%~dp0BaqylauStudent.exe" (
  echo [Baqylau] Starting BaqylauStudent.exe ...
  start "" "%~dp0BaqylauStudent.exe" %*
  exit /b 0
)
where python >nul 2>nul || (echo [Baqylau] Python not found in PATH & pause & exit /b 1)
python main_gui.py %*
if errorlevel 1 pause
