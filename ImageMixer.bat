@echo off
rem Double-click to start ImageMixer. All the work happens in launch.py.
cd /d "%~dp0"
where py >nul 2>nul && (py -3 launch.py %* & goto done)
where python >nul 2>nul && (python launch.py %* & goto done)
echo Python 3.10 or newer is required. Install it from https://www.python.org/downloads/ (tick "Add python.exe to PATH").
:done
if errorlevel 1 pause
