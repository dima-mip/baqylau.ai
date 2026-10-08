@echo off
REM Baqylau Web — teacher/admin panel (light). Run on the teacher PC.
cd /d "%~dp0"
where python >nul 2>nul || (echo [Baqylau] Python not found in PATH & pause & exit /b 1)
echo [Baqylau] Panel at http://127.0.0.1:5050  (LAN: check ipconfig IPv4)
echo [Baqylau] First login: admin / admin123
python -m server.app
pause
