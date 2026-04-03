@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "COMFY_ROOT=%%~fI"
set "PYTHON_EXE=%COMFY_ROOT%\python_embeded\python.exe"

if not exist "%PYTHON_EXE%" (
  echo [OmniVoice][ERROR] Could not find ComfyUI python at:
  echo %PYTHON_EXE%
  echo.
  echo Move this BAT into your ComfyUI custom node folder, then run again.
  pause
  exit /b 1
)

echo [OmniVoice][INFO] Using Python:
echo %PYTHON_EXE%
echo.

"%PYTHON_EXE%" -m pip install --no-deps "omnivoice>=0.1.0"
if errorlevel 1 (
  echo.
  echo [OmniVoice][ERROR] Failed to install omnivoice safely.
  pause
  exit /b 1
)

"%PYTHON_EXE%" -m pip install "huggingface_hub>=1.3.0,<2.0"
if errorlevel 1 (
  echo.
  echo [OmniVoice][ERROR] Failed to install huggingface_hub.
  pause
  exit /b 1
)

echo.
echo [OmniVoice][OK] Installed omnivoice and huggingface_hub.
pause
exit /b 0
