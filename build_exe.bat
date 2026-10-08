@echo off
REM Build BaqylauStudent.exe (Windows onefile, no console).
REM Run from baqylau folder. Takes 10-20 min first time.
cd /d "%~dp0"
where python >nul 2>nul || (echo [Baqylau] Python not found & pause & exit /b 1)
python -m pip install --quiet pyinstaller
if errorlevel 1 (echo [Baqylau] pip failed & pause & exit /b 1)
python -m PyInstaller --noconfirm --clean --onefile --windowed ^
  --name BaqylauStudent ^
  --add-data "ui\brand.svg;ui" ^
  --add-data "ui\bento_style.qss;ui" ^
  --add-data "yolov8n.pt;." ^
  --add-data "data\models\face_landmarker.task;data\models" ^
  --collect-data mediapipe ^
  --collect-data ultralytics ^
  --collect-binaries mediapipe ^
  --exclude-module brainflow ^
  --exclude-module torch.testing ^
  main_gui.py
if errorlevel 1 (echo [Baqylau] BUILD FAILED & pause & exit /b 1)
echo.
echo [Baqylau] DONE: dist\BaqylauStudent.exe
dir dist\BaqylauStudent.exe
pause
