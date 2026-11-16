@echo off
rem ============================================================
rem  NAM EQ Matcher - one-click launcher for Windows
rem  First run: creates a virtual environment and installs
rem  dependencies (can take several minutes). After that it
rem  starts instantly and opens the app in your browser.
rem ============================================================
setlocal
cd /d "%~dp0"
title NAM EQ Matcher

echo.
echo  === NAM EQ Matcher ===
echo.

rem --- find Python ---------------------------------------------------------
where py >nul 2>nul
if %errorlevel%==0 goto have_py
where python >nul 2>nul
if %errorlevel%==0 goto have_python
echo  [ERROR] Python was not found on this computer.
echo.
echo  Please install Python 3.10 or newer from https://www.python.org/downloads/
echo  IMPORTANT: tick "Add Python to PATH" in the installer, then run this file again.
echo.
pause
exit /b 1

:have_py
set PYCMD=py -3
goto check_version

:have_python
set PYCMD=python

:check_version
%PYCMD% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not %errorlevel%==0 (
    echo  [ERROR] Your Python is older than 3.10. Please install a newer Python
    echo  from https://www.python.org/downloads/ and run this file again.
    pause
    exit /b 1
)

rem --- create venv on first run --------------------------------------------
if exist ".venv\Scripts\python.exe" goto deps
echo  First run: creating a private Python environment...
%PYCMD% -m venv .venv
if not %errorlevel%==0 (
    echo  [ERROR] Could not create the virtual environment.
    pause
    exit /b 1
)

:deps
set VPY=.venv\Scripts\python.exe
if exist ".venv\.deps_installed" goto run
echo.
echo  Installing dependencies - this one-time step can take several minutes...
echo.
"%VPY%" -m pip install --upgrade pip
"%VPY%" -m pip install -r requirements.txt
if not %errorlevel%==0 (
    echo  [ERROR] Dependency installation failed. Check your internet connection
    echo  and run this file again.
    pause
    exit /b 1
)
type nul > ".venv\.deps_installed"

:run
echo.
echo  Starting NAM EQ Matcher - your browser will open at http://127.0.0.1:7860
echo  Keep this window open while using the app. Close it to stop.
echo.
"%VPY%" app.py
pause
