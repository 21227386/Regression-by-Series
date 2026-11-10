@echo off
setlocal
cd /d "%~dp0"
py -3.12 -m venv .venv-win
if errorlevel 1 goto fail
.venv-win\Scripts\python.exe -m pip install -r requirements.txt pyinstaller==6.16.0
if errorlevel 1 goto fail
.venv-win\Scripts\python.exe -m unittest discover -s tests -v
if errorlevel 1 goto fail
.venv-win\Scripts\python.exe -m PyInstaller --noconfirm --clean --onefile --windowed --name "without year" --add-data "assets;assets" app.py
if errorlevel 1 goto fail
echo Built: dist\without year.exe
pause
exit /b 0
:fail
echo Build failed. See the error above.
pause
exit /b 1
