@echo off
REM setup.bat - Automated setup script for Windows
REM Run this after downloading all Python files

echo ========================================================================
echo STRIPE INVESTIGATION - AUTOMATED SETUP
echo ========================================================================
echo.

REM Check if Python is installed
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found!
    echo Please install Python 3.8+ from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)
python --version
echo.

REM Install required packages
echo [2/4] Installing required packages...
echo This may take a few minutes...
echo.

echo Installing NumPy, SciPy, Matplotlib, Pandas...
pip install numpy scipy matplotlib pandas --quiet
if %errorlevel% neq 0 (
    echo [WARNING] Some packages failed to install
    echo Try manually: pip install numpy scipy matplotlib pandas
) else (
    echo [OK] Standard packages installed
)
echo.

echo Installing CuPy for GPU acceleration...
echo Detecting CUDA version...
nvidia-smi | findstr "CUDA Version" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] nvidia-smi not found - cannot detect CUDA version
    echo Please manually install: pip install cupy-cuda11x
) else (
    nvidia-smi | findstr "CUDA Version"
    echo.
    echo Attempting to install CuPy for CUDA 11.x...
    pip install cupy-cuda11x --quiet
    if %errorlevel% neq 0 (
        echo [WARNING] CuPy installation failed
        echo Try manually: pip install cupy-cuda11x (or cupy-cuda12x)
        echo You can still run simulations on CPU (slower)
    ) else (
        echo [OK] CuPy installed successfully
    )
)
echo.

REM Test GPU setup
echo [3/4] Testing GPU setup...
if exist test_gpu_setup.py (
    python test_gpu_setup.py
) else (
    echo [WARNING] test_gpu_setup.py not found
    echo Make sure all Python files are in this directory
)
echo.

REM Create directories
echo [4/4] Creating output directories...
if not exist results mkdir results
if not exist results\yukawa_test mkdir results\yukawa_test
if not exist results\optimal_mu_search mkdir results\optimal_mu_search
if not exist plots mkdir plots
echo [OK] Directories created
echo.

REM Final message
echo ========================================================================
echo SETUP COMPLETE!
echo ========================================================================
echo.
echo Your system is ready to run simulations!
echo.
echo Quick test:
echo   python test_gpu_setup.py
echo.
echo Run scalar field test:
echo   python run_scalar_test.py
echo.
echo Read documentation:
echo   START_HERE.txt
echo   EXECUTIVE_SUMMARY.md
echo.
echo Check GPU status anytime:
echo   nvidia-smi
echo.
echo ========================================================================
echo.
pause