@echo off
:: Ensure working directory is the script's directory
cd /d "%~dp0"

echo ===================================================
echo Running Gemma 4 PostgreSQL Photo Database Chat REPL
echo ===================================================

:: 1. Check if the virtual environment activation script exists
if not exist "..\ltx2_env\Scripts\activate.bat" (
    echo [ERROR] Virtual environment activation script not found at:
    echo         ..\ltx2_env\Scripts\activate.bat
    echo Please verify that the environment directory is correctly set up.
    goto end
)

:: 2. Attempt to activate the virtual environment
echo Activating virtual environment...
call ..\ltx2_env\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate the virtual environment.
    goto end
)

:: 3. Verify python is available in path
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found in path after environment activation.
    goto end
)

:: 4. Check if the database chat script exists
if not exist "db_chat_repl.py" (
    echo [ERROR] db_chat_repl.py was not found in the current directory:
    echo         %CD%
    goto end
)

:: 5. Execute database chat REPL in PostgreSQL mode
echo Starting PostgreSQL database chat REPL...
set DB_BACKEND=postgresql
python db_chat_repl.py --remote %*

if errorlevel 1 (
    echo(
    echo [ERROR] The database chat script exited with errors - Code %ERRORLEVEL%.
) else (
    echo(
    echo [SUCCESS] Database chat session completed.
)

:end
:: Keep window open on completion/crash
echo(
pause
