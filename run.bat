@echo off
echo Starting CipherCore SDXL...

:: Define the virtual environment directory name (must match setup.bat)
set VENV_DIR=venv
set SCRIPT_NAME=main.py

:: 1. Check if the virtual environment exists
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo.
    echo [ERROR] Virtual environment not found in folder '%VENV_DIR%'.
    echo Please run setup.bat first to create the environment and install dependencies.
    goto end
)

:: 2. Activate the virtual environment
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate the virtual environment.
    goto end
)

:: 3. Run the main Python script
echo Launching the CipherCore SDXL application...
echo Please wait for the Gradio interface to load. This may take a moment.
python %SCRIPT_NAME%
if %errorlevel% neq 0 (
     echo [ERROR] The application exited with an error. See the console output above for details.
) else (
     echo Application finished successfully.
)

:: 4. Deactivate the virtual environment
echo Deactivating virtual environment...
call deactivate >nul 2>nul

:end
echo.
echo Press any key to close this window...
pause >nul
