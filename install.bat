@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%CD%"
set "UV_EXE=%ROOT%\runtime\uv\uv.exe"
set "VENV=%ROOT%\.venv"
set "UV_CACHE_DIR=%ROOT%\temp\uv-cache"
set "HF_HOME=%CD%\temp\hf-home"
set "HF_HUB_CACHE=%CD%\temp\hf-hub"
set "HF_XET_CACHE=%CD%\temp\hf-xet"
set "PIP_CACHE_DIR=%CD%\temp\pip-cache"
set "TEMP=%CD%\temp"
set "TMP=%CD%\temp"
set "HF_HUB_ENABLE_HF_XET=1"

if not exist "%ROOT%\runtime\uv" mkdir "%ROOT%\runtime\uv"
if not exist "%ROOT%\temp" mkdir "%ROOT%\temp"
if not exist "%UV_EXE%" (
  echo Downloading self-contained uv...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $u='https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip'; $z=Join-Path $env:TEMP 'uv.zip'; Invoke-WebRequest -UseBasicParsing -Uri $u -OutFile $z; Expand-Archive -LiteralPath $z -DestinationPath (Join-Path $env:TEMP 'uv-extract') -Force; Copy-Item -LiteralPath (Join-Path $env:TEMP 'uv-extract\uv.exe') -Destination $env:UV_EXE -Force; Remove-Item -LiteralPath $z -Force; Remove-Item -LiteralPath (Join-Path $env:TEMP 'uv-extract') -Recurse -Force"
  if errorlevel 1 echo ERROR: Could not download local uv.& pause& exit /b 1
)
"%UV_EXE%" venv "%VENV%" --python 3.12 --seed
if errorlevel 1 echo ERROR: Could not create .venv.& pause& exit /b 1

if not exist "%CD%\backend\inference_cli.py" (
  where git >nul 2>nul || (
    echo Git for Windows is required to fetch the SeedVR backend.
    pause
    exit /b 1
  )
  git clone --depth 1 https://github.com/ByteDance-Seed/SeedVR.git backend
  if errorlevel 1 echo ERROR: Could not download SeedVR backend.& pause& exit /b 1
)
if not exist "%ROOT%\backend\src\optimization\gguf_dequant.py" (
  echo ERROR: The SeedVR backend has no GGUF implementation.
  echo Delete the backend folder and run install.bat again.
  pause
  exit /b 1
)

"%UV_EXE%" sync --python "%VENV%\Scripts\python.exe" --no-cache
if errorlevel 1 echo ERROR: Dependency installation failed.& pause& exit /b 1
if exist "%ROOT%\backend\requirements.txt" "%UV_EXE%" pip install --python "%VENV%\Scripts\python.exe" --no-cache -r "%ROOT%\backend\requirements.txt"
if errorlevel 1 echo ERROR: Backend dependency installation failed.& pause& exit /b 1

echo Verifying PyTorch CUDA runtime...
"%VENV%\Scripts\python.exe" -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda, 'available=', torch.cuda.is_available())"
if errorlevel 1 echo ERROR: PyTorch is incomplete or not importable.& pause& exit /b 1
echo Verifying GGUF support...
"%VENV%\Scripts\python.exe" -c "import gguf; print('GGUF support OK:', getattr(gguf, '__file__', 'installed'))"
if errorlevel 1 echo ERROR: GGUF support is not installed.& pause& exit /b 1

echo.
echo Seed-VR-2.5-Easy-GUI installation complete.
pause
