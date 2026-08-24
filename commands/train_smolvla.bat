@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

for %%I in ("%~dp0..") do set "WORKSPACE=%%~fI"
cd /d "%WORKSPACE%"

if exist "%WORKSPACE%\venv\Scripts\python.exe" (
    set "PYTHON_EXE=%WORKSPACE%\venv\Scripts\python.exe"
) else if exist "%WORKSPACE%\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%WORKSPACE%\.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

rem Prefer the short-GOP copy when it has already been created.
if not defined DATASET_NAME (
    if exist "%WORKSPACE%\training\manual_data1_gop2\meta\info.json" (
        set "DATASET_NAME=manual_data1_gop2"
    ) else (
        set "DATASET_NAME=manual_data1"
    )
)

if not defined DATASET_ROOT set "DATASET_ROOT=%WORKSPACE%\training\%DATASET_NAME%"
if not defined OUTPUT_DIR set "OUTPUT_DIR=%WORKSPACE%\outputs\train\smolvla_%DATASET_NAME%"
if not defined STEPS set "STEPS=20000"
if not defined BATCH_SIZE set "BATCH_SIZE=16"
if not defined SAVE_FREQ set "SAVE_FREQ=5000"
if not defined LOG_FREQ set "LOG_FREQ=50"

rem Leave two logical CPUs for the trainer/OS and cap workers to avoid excessive
rem Windows process-spawn and video-decoder overhead.
if not defined NUM_WORKERS (
    set /a NUM_WORKERS=%NUMBER_OF_PROCESSORS%-2
    if !NUM_WORKERS! GTR 10 set "NUM_WORKERS=10"
    if !NUM_WORKERS! LSS 2 set "NUM_WORKERS=2"
)

rem TorchCodec is normally faster for random video access. Fall back cleanly
rem when its FFmpeg shared libraries or wheel are unavailable.
if not defined VIDEO_BACKEND (
    "%PYTHON_EXE%" -c "from torchcodec.decoders import VideoDecoder" >nul 2>&1
    if errorlevel 1 (
        set "VIDEO_BACKEND=pyav"
        echo TorchCodec is unavailable; using PyAV.
    ) else (
        set "VIDEO_BACKEND=torchcodec"
    )
)

if not exist "%DATASET_ROOT%\meta\info.json" (
    echo Dataset not found: "%DATASET_ROOT%\meta\info.json"
    exit /b 1
)

set "DRY_RUN_ARG="
if /I "%DRY_RUN%"=="1" set "DRY_RUN_ARG=--dry-run"

echo GPU:
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>nul
echo Dataset:      %DATASET_ROOT%
echo Output:       %OUTPUT_DIR%
echo Batch size:   %BATCH_SIZE%
echo Workers:      %NUM_WORKERS%
echo Video backend: %VIDEO_BACKEND%

"%PYTHON_EXE%" "%WORKSPACE%\scripts\train_smolvla.py" ^
    --dataset-name "%DATASET_NAME%" ^
    --dataset-root "%DATASET_ROOT%" ^
    --output-dir "%OUTPUT_DIR%" ^
    --steps "%STEPS%" ^
    --batch-size "%BATCH_SIZE%" ^
    --num-workers "%NUM_WORKERS%" ^
    --save-freq "%SAVE_FREQ%" ^
    --log-freq "%LOG_FREQ%" ^
    --video-backend "%VIDEO_BACKEND%" ^
    %DRY_RUN_ARG% %*

set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" echo Training exited with code %EXIT_CODE%.
exit /b %EXIT_CODE%
