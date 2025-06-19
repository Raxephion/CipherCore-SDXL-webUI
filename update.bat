@echo off
echo Updating the CipherCore SDXL Generator...

:: Define the virtual environment directory name (must match setup.bat and run.bat)
set VENV_DIR=venv
:: !!! IMPORTANT: Replace this with your actual GitHub repository URL !!!
set REPO_URL=https://github.com/Raxephion/CipherCore-SDXL-WebUI
set TEMP_DIR=temp_extraction

:: Change directory to the script's location
cd /d "%~dp0"

:: Check if Git is available
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Git not found. Attempting to download the latest code directly...

    :: Create a temporary directory for extraction
    mkdir %TEMP_DIR%
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create temporary directory.
        goto end
    )

    :: Download the repository as a ZIP file if Git is not installed
    powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%REPO_URL%/archive/refs/heads/main.zip', 'repo.zip')"
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to download repository as ZIP. Please check your internet connection.
        rmdir /s /q %TEMP_DIR% 2>nul
        goto end
    )

    echo Extracting ZIP file to temporary directory...
    powershell -Command "Expand-Archive -Path 'repo.zip' -DestinationPath '%TEMP_DIR%'"
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to extract ZIP file. The download may be corrupt.
        del repo.zip >nul 2>nul
        rmdir /s /q %TEMP_DIR% 2>nul
        goto end
    )

    echo Moving updated files into the current directory...
    :: This logic moves files from the extracted sub-folder to the project's root
    for /d %%d in (%TEMP_DIR%\*) do (
        for %%f in ("%%d\*") do (
            move /y "%%f" "." >nul
        )
        for /d %%s in ("%%d\*") do (
            move /y "%%s" "." >nul
        )
    )

    del repo.zip >nul 2>nul
    rmdir /s /q %TEMP_DIR% >nul 2>nul

    echo [OK] Successfully downloaded and extracted the latest code.
    echo For better version control, consider installing Git from https://git-scm.com/
    goto update_dependencies
)

echo [INFO] Git found.

:: Check if the virtual environment exists
if not exist %VENV_DIR%\Scripts\activate.bat (
    echo [ERROR] Virtual environment "%VENV_DIR%" not found.
    echo Please run setup.bat first to create the environment.
    goto end
)

:: Activate the virtual environment
call %VENV_DIR%\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    goto end
)

:: Pull latest code from the repository
echo Pulling latest code from GitHub...
git pull
if %errorlevel% neq 0 (
    echo [ERROR] Failed to pull latest code. This can happen if you have local changes.
    echo Please resolve any Git issues manually (e.g., using 'git stash').
    goto deactivate_and_end
)
echo Code updated successfully.

:update_dependencies
echo.
echo Installing/Upgrading dependencies from requirements.txt...
pip install -r requirements.txt --upgrade
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install/upgrade dependencies. See the output above for details.
    goto deactivate_and_end
)
echo Dependencies updated successfully.

echo.
echo --- UPDATE COMPLETE ---
echo The CipherCore SDXL application and its dependencies are now up-to-date.
echo.

goto end

:deactivate_and_end
:: Deactivate the virtual environment before exiting on error
echo Deactivating virtual environment...
call deactivate >nul 2>nul
echo.

:end
echo Press any key to exit...
pause >nul
