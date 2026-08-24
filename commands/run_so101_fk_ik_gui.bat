@echo off
setlocal
cd /d "%~dp0.."
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" "scripts\so101_fk_ik_gui.py" %*
) else (
    python "scripts\so101_fk_ik_gui.py" %*
)
endlocal
