@echo off
setlocal
set "ROOT=%~dp0"
if not exist "%ROOT%.venv\Scripts\python.exe" (
  echo Run setup_windows.ps1 first.
  pause
  exit /b 1
)
"%ROOT%.venv\Scripts\python.exe" "%ROOT%benchmark.py" --mode ranging --parallel-stereo %*
if errorlevel 1 (
  echo Benchmark failed. See the error above.
  pause
  exit /b 1
)
pause
