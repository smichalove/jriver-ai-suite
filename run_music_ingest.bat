@echo off
:: Ensure working directory is the script's directory
cd /d "%~dp0"

echo ===================================================
echo Running JRiver Music Library Ingestion (PostgreSQL)
echo ===================================================
echo Ingesting metadata from files and sidecars inside D:\Users\steven\Music...
echo(

..\ltx2_env\Scripts\python.exe ingest_music_library.py --root "D:\Users\steven\Music" --db-backend postgresql --max-workers 32

echo(
echo Process complete.
pause
