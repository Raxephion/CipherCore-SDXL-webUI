@echo off
echo Setting up the environment for CipherCore SDXL (GPU)...

:: Define the virtual environment directory name
set VENV_DIR=venv

:: Change directory to the script's location
cd /d "%~dp0"

:: 1. Check if Python is available
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python not found.
    echo Please install Python 3.9+ and make sure it's added to your system's PATH.
    echo You can download it from: https://www.python.org/downloads/
    goto end
)
echo [OK] Found Python installation.

:: 2. Create virtual environment if it doesn't exist
if not exist %VENV_DIR% (
    echo Creating virtual environment in "%VENV_DIR%"...
    python -m venv %VENV_DIR%
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create the virtual environment.
        echo Please check your Python installation and script permissions.
        goto end
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment "%VENV_DIR%" already exists. Skipping creation.
)

:: 3. Activate the virtual environment
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate the virtual environment.
    goto end
)
echo Virtual environment activated.

:: 4. Install all dependencies, including PyTorch with CUDA support
echo.
echo Installing all required packages from requirements.txt...
echo This will download and install PyTorch with CUDA support, which can take some time.
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [!!! CRITICAL ERROR !!!] Failed to install dependencies.
    echo.
    echo This is often due to an NVIDIA driver or CUDA version mismatch.
    echo.
    echo --- TROUBLESHOOTING STEPS ---
    echo 1. CHECK DRIVERS: Open Command Prompt and run the command: nvidia-smi
    echo    Look at the "CUDA Version" displayed in the top right. This is the MAXIMUM version your driver supports.
    echo.
    echo 2. EDIT REQUIREMENTS: Open the 'requirements.txt' file in a text editor.
    echo    The first line is '--extra-index-url https://download.pytorch.org/whl/cu121'.
    echo    The 'cu121' part must match a version supported by your driver (e.g., 'cu118' for 11.8).
    echo    You can find other versions on the PyTorch website.
    echo.
    echo 3. VISIT PYTORCH: Go to https://pytorch.org/get-started/locally/ for the correct commands.
    echo.
    echo After making changes to requirements.txt, please run this setup file again.
    goto deactivate_and_end
)

echo.
echo --- SETUP COMPLETE ---
echo All packages, including PyTorch for GPU, have been installed successfully.
echo You can now run the application using 'run.bat' or by executing the python script directly.
echo.

goto end

:deactivate_and_end
echo Deactivating virtual environment due to an error.
call "%VENV_DIR%\Scripts\deactivate.bat" >nul 2>nul
echo.

:end
echo Press any key to exit...
pause >nul
