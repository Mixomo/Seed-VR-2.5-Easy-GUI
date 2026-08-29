@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%CD%"
set "UV_CACHE_DIR=%ROOT%\temp\uv-cache"
set "HF_HOME=%CD%\temp\hf-home"
set "HF_HUB_CACHE=%CD%\temp\hf-hub"
set "HF_XET_CACHE=%CD%\temp\hf-xet"
set "TEMP=%CD%\temp"
set "TMP=%CD%\temp"
set "HF_HUB_ENABLE_HF_XET=1"
if not exist "%ROOT%\runtime\uv\uv.exe" echo ERROR: Run install.bat first.& pause& exit /b 1
if not exist "%ROOT%\.venv\Scripts\python.exe" echo ERROR: Run install.bat first.& pause& exit /b 1
"%ROOT%\runtime\uv\uv.exe" run --python "%ROOT%\.venv\Scripts\python.exe" --no-sync --no-cache python -m app.main
exit /b %ERRORLEVEL%
