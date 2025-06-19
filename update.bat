@echo off
echo Updating CipherCore SDXL...

:: Define the virtual environment directory name (must match setup.bat)
set VENV_DIR=venv
:: !!! IMPORTANT: Replace this with your actual GitHub repository URL !!!
set REPO_URL=https://github.com/Raxephion/CipherCore-SDXL-WebUI
set TEMP_DIR=temp_update

:: Change directory to the script's location
cd /d "%~dp0"

:: Check if Git is available on the system PATH
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Git not found. Attempting to download the latest version directly...

    :: Create a temporary directory for the download
    mkdir %TEMP_DIR% >nul 2>nul

    :: Download the repository as a ZIP file
    echo Downloading latest code from %REPO_URL%...
    powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%REPO_URL%/archive/refs/heads/main.zip', 'repo.zip')"
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to download repository ZIP. Please check your internet connection.
        rmdir /s /q %TEMP_DIR% 2>nul
        goto end
    )

    :: Extract the ZIP file
    echo Extracting files...
    powershell -Command "Expand-Archive -Path 'repo.zip' -DestinationPath '%TEMP_DIR%'"
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to extract ZIP file. The download may be corrupt.
        del repo.zip >nul 2>nul
        rmdir /s /q %TEMP_DIR% 2>nul
        goto end
    )

    echo Overwriting old files with updated versions...
    robocopy "%TEMP_DIR%\%~n1-main" "." /e /move > nul
    
    :: Clean up temporary files and folders
    del repo.zip >nul 2>nul
    rmdir /s /q %TEMP_DIR% 2>nul
    
    echo Code updated successfully using direct download.
    echo For better update management, consider installing Git from https://git-scm.com/
    goto update_dependencies
)

echo [INFO] Git found. Pulling latest changes...

:: Check if the virtual environment exists before activating
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [ERROR] Virtual environment '%VENV_DIR%' not found. Please run setup.bat first.
    goto end
)

:: Activate the virtual environment to use its pip
call "%VENV_DIR%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate the virtual environment.
    goto end
)

:: Pull latest code from the GitHub repository
git pull
if %errorlevel% neq 0 (
    echo [ERROR] 'git pull' failed. This can happen if you have made local changes to the files.
    echo Please resolve the conflict manually or use 'git stash' to save your changes before pulling.
    goto deactivate_and_end
)
echo Code updated successfully from Git.

:update_dependencies
echo.
echo Checking for updated packages...
pip install -r requirements.txt --upgrade
if %errorlevel% neq 0 (
    echo [ERROR] Failed to update Python packages. See the error messages above.
    goto deactivate_and_end
)
echo All packages are up to date.

echo.
echo --- UPDATE COMPLETE ---
echo CipherCore SDXL and its dependencies have been successfully updated.
echo You can now run the application using 'run.bat'.
echo.

goto end

:deactivate_and_end
echo Deactivating virtual environment due to an error...
call deactivate >nul 2>nul
echo.

:end
echo Press any key to exit...
pause >nul
