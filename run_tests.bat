@echo off
setlocal EnableDelayedExpansion
title ASN - EMG Analysis System Validator
color 0B

:: ==============================================================================
::  ASN - MOVEMENT SIGNAL ANALYSIS PIPELINE VALIDATOR
::  Platform: Windows 11
::  Description: Automated health check, syntax verification, and unit testing.
:: ==============================================================================

:: --- 1. INITIALIZATION & LOGGING ---
set "LOGFILE=System_Check_Log.txt"
if exist "%LOGFILE%" del "%LOGFILE%"
echo ============================================================================== > "%LOGFILE%"
echo  ASN SYSTEM CHECK LOG - %DATE% %TIME% >> "%LOGFILE%"
echo ============================================================================== >> "%LOGFILE%"

echo.
echo  [SYSTEM] Initializing Validation Pipeline...
echo  [SYSTEM] Logs will be saved to: %LOGFILE%
echo.

:: --- 2. DEPENDENCY CHECK (LIBRARIES) ---
echo  ------------------------------------------------------------------------------
echo   PHASE 1: CHECKING DEPENDENCIES
echo  ------------------------------------------------------------------------------
echo.

set "MISSING_LIBS=0"
for %%L in (numpy matplotlib PyQt6 wfdb) do (
    <nul set /p="  > Checking %%L... "
    python -c "import %%L" 2>nul
    if !errorlevel! equ 0 (
        echo [OK]
        echo [PASS] Library '%%L' found. >> "%LOGFILE%"
    ) else (
        echo [MISSING]
        echo [FAIL] Library '%%L' is MISSING or corrupted. >> "%LOGFILE%"
        set /a MISSING_LIBS+=1
    )
)

if %MISSING_LIBS% gtr 0 (
    echo.
    echo  [CRITICAL] Some dependencies are missing. Please run: pip install [library_name]
    echo  [SYSTEM] Aborting process.
    pause
    exit /b 1
) else (
    echo.
    echo  [SUCCESS] All dependencies are installed.
)

:: --- 3. DATASET AVAILABILITY CHECK ---
echo.
echo  ------------------------------------------------------------------------------
echo   PHASE 2: CHECKING DATASET (S01)
echo  ------------------------------------------------------------------------------
echo.

if exist "S01.hea" (
    if exist "S01.dat" (
        echo   > Dataset S01 (Header/Dat) found. [OK]
        echo [PASS] Dataset S01 found. >> "%LOGFILE%"
    ) else (
        echo   > S01.dat is missing! [FAIL]
        echo [FAIL] S01.dat is missing. >> "%LOGFILE%"
    )
) else (
    echo   > S01.hea is missing! [FAIL]
    echo [FAIL] S01.hea is missing. >> "%LOGFILE%"
)

:: --- 4. SCRIPT ANALYSIS PIPELINE ---
echo.
echo  ------------------------------------------------------------------------------
echo   PHASE 3: SCRIPT SYNTAX AND RUNTIME SIMULATION
echo  ------------------------------------------------------------------------------
echo   NOTE: For GUI scripts, please CLOSE the window manually to proceed.
echo.

:: List of scripts in logical order
set "SCRIPTS=Load_and_Plot_Raw_Data.py Segmentation_Foot_Switch.py Filtering_BPF.py Denoising_DWT_EMG.py STFT_EMG.py CWT_EMG.py Threshold.py Result_Reporting.py GUI.py main.py"

set "ERRORS=0"

for %%S in (%SCRIPTS%) do (
    echo.
    echo   ----------------------------------------
    echo   TARGET: %%S
    echo   ----------------------------------------
    
    if exist "%%S" (
        :: A. Syntax Check
        <nul set /p="  > Syntax Check... "
        python -m py_compile "%%S" 2>> "%LOGFILE%"
        if !errorlevel! equ 0 (
            echo [OK]
            
            :: B. Runtime Simulation (Unit Test)
            echo     Executing Unit Test...
            echo     [OUTPUT START] >> "%LOGFILE%"
            echo     Target: %%S >> "%LOGFILE%"
            
            :: Capture output to log, but show error level status on screen
            python "%%S" >> "%LOGFILE%" 2>&1
            
            if !errorlevel! equ 0 (
                echo     > Runtime Status: [SUCCESS]
                echo     [OUTPUT END] Status: SUCCESS >> "%LOGFILE%"
            ) else (
                echo     > Runtime Status: [FAILURE] - Check Log!
                echo     [OUTPUT END] Status: FAILURE (Error Code: !errorlevel!) >> "%LOGFILE%"
                set /a ERRORS+=1
            )
        ) else (
            echo [SYNTAX ERROR]
            echo [FAIL] Syntax error in %%S >> "%LOGFILE%"
            set /a ERRORS+=1
        )
    ) else (
        echo   [FILE NOT FOUND]
        echo [FAIL] File %%S not found in directory. >> "%LOGFILE%"
        set /a ERRORS+=1
    )
)

:: --- 5. FINAL SUMMARY ---
echo.
echo ==============================================================================
echo  FINAL DIAGNOSTIC REPORT
echo ==============================================================================
if %ERRORS% equ 0 (
    color 0A
    echo  STATUS: SYSTEM HEALTHY
    echo  All scripts passed syntax and logic checks.
    echo  The pipeline is ready for deployment.
) else (
    color 0C
    echo  STATUS: ISSUES DETECTED
    echo  Total Errors Found: %ERRORS%
    echo  Please review "System_Check_Log.txt" for specific error tracebacks.
)

echo.
echo  Press any key to exit...
pause >nul