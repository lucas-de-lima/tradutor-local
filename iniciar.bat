@echo off
cd /d "%~dp0"
set "CUDA_DLLS=%~dp0.venv\Lib\site-packages\nvidia\cublas\bin;%~dp0.venv\Lib\site-packages\nvidia\cudnn\bin;%~dp0.venv\Lib\site-packages\nvidia\cuda_nvrtc\bin"
set "PATH=%CUDA_DLLS%;%PATH%"
.\.venv\Scripts\python.exe -u .\tradutor_local.py --language en --whisper-model large-v3 --output "FxSound Audio Enhancer"
if errorlevel 1 pause