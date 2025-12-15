@echo off
REM ============================================================================
REM EEG Analysis - BCI System Prerequisites Checker
REM Purpose: Verify prerequisites for Python and C++ scripts before execution
REM ============================================================================

setlocal enabledelayedexpansion
set PASS=0
set FAIL=0

echo.
echo ============================================================================
echo                    PREREQUISITE CHECK REPORT
echo ============================================================================
echo.

REM Check Python Installation
echo [1/3] Checking Python Installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo     [FAIL] Python is not installed or not in PATH
    set /a FAIL+=1
) else (
    for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VER=%%i
    echo     [PASS] !PYTHON_VER!
    set /a PASS+=1
)
echo.

REM Check g++ Installation (for C++ compilation)
echo [2/3] Checking g++ Compiler...
g++ --version >nul 2>&1
if %errorlevel% neq 0 (
    echo     [FAIL] g++ is not installed or not in PATH
    set /a FAIL+=1
) else (
    for /f "tokens=1-3" %%i in ('g++ --version 2^>^&1 ^| findstr /R "g++"') do (
        echo     [PASS] g++ found
        set /a PASS+=1
        goto :cpp_check_done
    )
    :cpp_check_done
)
echo.

REM Check C++ eeg_core.cpp prerequisites
echo [3/3] Checking C++ Script: eeg_core.cpp
if not exist "eeg_core.cpp" (
    echo     [FAIL] eeg_core.cpp not found
    set /a FAIL+=1
) else (
    echo     [PASS] eeg_core.cpp found
    REM Check if DLL exists
    if not exist "eeg_processing.dll" (
        echo     [WARNING] eeg_processing.dll not compiled yet
        echo              >> Run: g++ -O3 -shared -static -o eeg_processing.dll eeg_core.cpp
    ) else (
        echo     [PASS] eeg_processing.dll exists
        set /a PASS+=1
    )
)
echo.

REM Check Python Package Dependencies
echo ============================================================================
echo                       PYTHON PACKAGE CHECK
echo ============================================================================
echo.

set PYTHON_PACKAGES=numpy scipy matplotlib scikit-learn mne PyQt6

for %%p in (%PYTHON_PACKAGES%) do (
    python -c "import %%p" >nul 2>&1
    if !errorlevel! neq 0 (
        echo [FAIL] Package not installed: %%p
        set /a FAIL+=1
    ) else (
        echo [PASS] Package installed: %%p
        set /a PASS+=1
    )
)
echo.

REM Check Python Scripts
echo ============================================================================
echo                       PYTHON SCRIPT CHECK
echo ============================================================================
echo.

set PYTHON_SCRIPTS=main.py GUI.py load_data_eeg_mne.py filtering_BPF_EEG.py CWT.py squaring_EEG.py average_all_EEG_trials.py moving_average_EEG.py percentage_ERD.py percentage_ERD_ERS.py csp_scratch.py ml_analysis.py logger_util.py

for %%s in (%PYTHON_SCRIPTS%) do (
    if not exist "%%s" (
        echo [FAIL] Script missing: %%s
        set /a FAIL+=1
    ) else (
        python -m py_compile "%%s" >nul 2>&1
        if !errorlevel! neq 0 (
            echo [FAIL] Script has syntax errors: %%s
            set /a FAIL+=1
        ) else (
            echo [PASS] Script OK: %%s
            set /a PASS+=1
        )
    )
)
echo.

REM Summary Report
echo ============================================================================
echo                          SUMMARY REPORT
echo ============================================================================
echo.
echo Total Checks: !PASS! Passed, !FAIL! Failed
echo.

if !FAIL! gtr 0 (
    echo [CRITICAL] !FAIL! prerequisite check(s) failed!
    echo.
    echo Recommended Actions:
    echo   - Install missing Python packages: pip install numpy scipy matplotlib scikit-learn mne PyQt6
    echo   - Compile C++ core: g++ -O3 -shared -static -o eeg_processing.dll eeg_core.cpp
    echo   - Ensure Python and g++ are in system PATH
    echo.
    pause
    exit /b 1
) else (
    echo [SUCCESS] All prerequisites met! System is ready.
    echo.
    pause
    exit /b 0
)
