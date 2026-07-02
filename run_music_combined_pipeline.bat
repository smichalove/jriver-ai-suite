@echo off
:: Ensure working directory is the script's directory
cd /d "%~dp0"

:: Force Python to output stdout immediately without buffering
set PYTHONUNBUFFERED=1

echo =================================================================
echo Running Unified Music Ingest, VLM Scanning, and Curation Loop
echo =================================================================
echo(

..\ltx2_env\Scripts\python.exe run_music_combined_pipeline.py --dir "D:\Users\steven\Music" --limit-dirs 50 --batch-size 3 --max-workers 20 %*

if errorlevel 1 (
    echo [ERROR] Unified pipeline execution failed.
    pause
    exit /b 1
)

echo [SUCCESS] Unified pipeline completed successfully.
echo(
pause
