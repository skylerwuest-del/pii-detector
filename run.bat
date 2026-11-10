@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1
set "PYTHON=%CD%\.venv\Scripts\python.exe"

:menu
echo.
echo ========================================
echo  PII DETECTION AGENT
echo ========================================
echo [1] Mode 1 - Controlled Directory Scan
echo [2] Mode 2 - Full Computer Scan
echo [3] Generate data and train / retrain
echo [4] Run complete synthetic demo
echo [5] Open settings ^(config.py^)
echo [6] Install / repair dependencies
echo [0] Exit
echo.
choice /c 1234560 /n /m "Select an option: "
if errorlevel 7 goto done
if errorlevel 6 goto setup
if errorlevel 5 goto settings
if errorlevel 4 goto demo
if errorlevel 3 goto training
if errorlevel 2 goto mode2
goto mode1

:mode1
call :ready
if errorlevel 1 goto failed
"%PYTHON%" -c "from main import navigate; navigate()"
if errorlevel 1 goto failed
goto menu

:mode2
call :ready
if errorlevel 1 goto failed
"%PYTHON%" -c "from main import full_mode; full_mode()"
if errorlevel 1 goto failed
goto menu

:training
call :runtime
if errorlevel 1 goto failed
call :train
if errorlevel 1 goto failed
echo Training complete.
pause
goto menu

:demo
call :runtime
if errorlevel 1 goto failed
"%PYTHON%" demo.py
if errorlevel 1 goto failed
pause
goto menu

:settings
echo Edit config.py, save it, then return to this menu.
echo After changing model, encoder, or training settings, use option 3 to retrain.
start "" /wait notepad.exe "%CD%\config.py"
goto menu

:setup
call :environment
if errorlevel 1 goto failed
call :install
if errorlevel 1 goto failed
echo Dependencies are ready.
pause
goto menu

:ready
call :runtime
if errorlevel 1 exit /b 1
"%PYTHON%" -c "import config; import sys; sys.exit(0 if (config.OUTPUT_DIRECTORY / 'model.pt').is_file() else 1)"
if not errorlevel 1 exit /b 0
echo No trained checkpoint found. Generating synthetic data and training now...
call :train
exit /b %errorlevel%

:train
echo Generating synthetic data and replacing the trained checkpoint using config.py...
"%PYTHON%" generate_data.py
if errorlevel 1 exit /b 1
"%PYTHON%" train.py
exit /b %errorlevel%

:runtime
call :environment
if errorlevel 1 exit /b 1
"%PYTHON%" -c "import numpy; import torch; torch.from_numpy(numpy.zeros(1))" >nul 2>&1
if not errorlevel 1 exit /b 0
call :install
exit /b %errorlevel%

:environment
if exist "%PYTHON%" goto check_python
echo Creating local Python environment...
py -3 -c "import sys; sys.exit(not sys.version_info >= (3, 11))" >nul 2>&1
if errorlevel 1 goto fallback_python
py -3 -m venv .venv
if errorlevel 1 exit /b 1
goto check_python

:fallback_python
python -c "import sys; sys.exit(not sys.version_info >= (3, 11))" >nul 2>&1
if errorlevel 1 goto missing_python
python -m venv .venv
if errorlevel 1 exit /b 1

:check_python
"%PYTHON%" -c "import sys; sys.exit(not sys.version_info >= (3, 11))"
if errorlevel 1 goto missing_python
exit /b 0

:missing_python
echo Python 3.11 or newer is required. Install 64-bit Python with a compatible CPU PyTorch wheel.
echo Enable the Python launcher or add Python to PATH, then run this file again.
echo If .venv uses an older Python, rename it so a new environment can be created.
exit /b 1

:install
echo Installing CPU dependencies. Internet access is required for this step.
"%PYTHON%" -m pip install "torch>=2.9,<3" --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 exit /b 1
"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
"%PYTHON%" -c "import numpy; import torch; torch.from_numpy(numpy.zeros(1)); print('PyTorch', torch.__version__, 'is ready (CPU).')"
exit /b %errorlevel%

:failed
echo.
echo The operation failed. Review the error above.
pause
goto menu

:done
popd
endlocal
exit /b 0

