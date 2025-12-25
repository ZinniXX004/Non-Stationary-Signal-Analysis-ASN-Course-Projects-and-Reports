@echo off
setlocal EnableDelayedExpansion
title ASN - EMG Analysis System Validator V5 (Stable)
color 0B

:: ==============================================================================
::  MOVEMENT SIGNAL ANALYSIS PIPELINE VALIDATOR
::  Platform: Windows 11
::  Flattened logic using GOTO to eliminate False Positives.
:: ==============================================================================

:: --- 1. INITIALIZATION & LOGGING ---
set "LOGFILE=System_Check_Log.txt"
if exist "%LOGFILE%" del "%LOGFILE%"
echo ============================================================================== > "%LOGFILE%"
echo  ASN SYSTEM CHECK LOG - %DATE% %TIME% >> "%LOGFILE%"
echo ============================================================================== >> "%LOGFILE%"

echo.
echo  [SYSTEM] Initializing Validation Pipeline V5...
echo  [SYSTEM] Logs will be saved to: %LOGFILE%
echo.

:: --- 2. DEPENDENCY CHECK ---
echo  ------------------------------------------------------------------------------
echo   PHASE 1: CHECKING DEPENDENCIES
echo  ------------------------------------------------------------------------------
echo.

set "MISSING_LIBS=0"
call :CHECK_LIB numpy
call :CHECK_LIB matplotlib
call :CHECK_LIB PyQt6
call :CHECK_LIB wfdb

if !MISSING_LIBS! gtr 0 (
    echo.
    echo  [CRITICAL] Dependencies missing. Check log.
    echo  [SYSTEM] Aborting.
    pause
    exit /b 1
) else (
    echo.
    echo  [SUCCESS] All dependencies are installed.
)

:: --- 3. DATASET AVAILABILITY CHECK (S01 - S31) ---
echo.
echo  ------------------------------------------------------------------------------
echo   PHASE 2: CHECKING DATASETS (S01 to S31)
echo  ------------------------------------------------------------------------------
echo   Checking for .hea and .dat files...
echo.

set "MISSING_DATA=0"
set "FOUND_COUNT=0"

:: Loop from 1 to 31
for /L %%i in (1,1,31) do (
    call :CHECK_DATASET %%i
)

echo.
echo  [INFO] Total Valid Datasets Found: !FOUND_COUNT!
if !MISSING_DATA! gtr 0 (
    echo  [WARNING] !MISSING_DATA! datasets were incomplete or missing ^(See Log^).
) else (
    echo  [SUCCESS] All 31 datasets are present.
)

:: --- 4. SCRIPT ANALYSIS PIPELINE ---
echo.
echo  ------------------------------------------------------------------------------
echo   PHASE 3: SCRIPT SYNTAX AND RUNTIME SIMULATION
echo  ------------------------------------------------------------------------------
echo   NOTE: For GUI scripts, please CLOSE the window manually to proceed.
echo.

set "SCRIPT_ERRORS=0"

:: Execute tests sequentially
call :TEST_SCRIPT Load_and_Plot_Raw_Data.py
call :TEST_SCRIPT Segmentation_Foot_Switch.py
call :TEST_SCRIPT Filtering_BPF.py
call :TEST_SCRIPT Denoising_DWT_EMG.py
call :TEST_SCRIPT STFT_EMG.py
call :TEST_SCRIPT CWT_EMG.py
call :TEST_SCRIPT Threshold.py
call :TEST_SCRIPT Result_Reporting.py
call :TEST_SCRIPT GUI.py
call :TEST_SCRIPT main.py

:: --- 5. FINAL SUMMARY ---
echo.
echo ==============================================================================
echo  FINAL DIAGNOSTIC REPORT
echo ==============================================================================
if !SCRIPT_ERRORS! equ 0 (
    color 0A
    echo  STATUS: SYSTEM HEALTHY
    echo  All scripts passed syntax and logic checks.
    echo  The pipeline is ready for deployment.
) else (
    color 0C
    echo  STATUS: ISSUES DETECTED
    echo  Total Script Errors: !SCRIPT_ERRORS!
    echo  Please review "System_Check_Log.txt".
)

echo.
echo  Press any key to exit...
pause >nul
goto :EOF

:: ==============================================================================
::  SUBROUTINES (FUNCTIONS)
:: ==============================================================================

:CHECK_LIB
:: Usage: call :CHECK_LIB [LibraryName]
<nul set /p="  > Checking %1... "
python -c "import %1" 2>nul
if %errorlevel% equ 0 (
    echo [OK]
    echo [PASS] Library '%1' found. >> "%LOGFILE%"
) else (
    echo [MISSING]
    echo [FAIL] Library '%1' is MISSING. >> "%LOGFILE%"
    set /a MISSING_LIBS+=1
)
goto :EOF

:CHECK_DATASET
:: Usage: call :CHECK_DATASET [Number]
set "NUM=%1"
set "FILE_A=S%NUM%"
if %NUM% lss 10 (set "FILE_B=S0%NUM%") else (set "FILE_B=S%NUM%")

if exist "%FILE_A%.hea" (
    set "TARGET=%FILE_A%"
) else if exist "%FILE_B%.hea" (
    set "TARGET=%FILE_B%"
) else (
    echo [FAIL] Dataset S%NUM% missing. >> "%LOGFILE%"
    set /a MISSING_DATA+=1
    goto :EOF
)

if exist "%TARGET%.dat" (
    echo   > Found: %TARGET%
    echo [PASS] Dataset %TARGET% ^(Header/Dat^) found. >> "%LOGFILE%"
    set /a FOUND_COUNT+=1
) else (
    echo   > Found Header %TARGET% but MISSING .dat!
    echo [FAIL] Dataset %TARGET% missing .dat file. >> "%LOGFILE%"
    set /a MISSING_DATA+=1
)
goto :EOF

:TEST_SCRIPT
:: Usage: call :TEST_SCRIPT [ScriptName]
set "S_NAME=%1"
echo.
echo   ----------------------------------------
echo   TARGET: %S_NAME%
echo   ----------------------------------------

if not exist "%S_NAME%" (
    echo   [FILE NOT FOUND]
    echo [FAIL] File %S_NAME% not found. >> "%LOGFILE%"
    set /a SCRIPT_ERRORS+=1
    goto :EOF
)

:: --- A. SYNTAX CHECK ---
<nul set /p="  > Syntax Check... "
:: Reset error level
ver > nul
python -m py_compile "%S_NAME%" 2>> "%LOGFILE%"

:: Check Syntax Result using GOTO to avoid nested IF issues
if %errorlevel% neq 0 goto :SYNTAX_FAIL

echo [OK]

:: --- B. RUNTIME SIMULATION ---
echo     Executing Unit Test...
echo     [OUTPUT START: %S_NAME%] >> "%LOGFILE%"

:: Reset error level
ver > nul
python "%S_NAME%" >> "%LOGFILE%" 2>&1

:: Check Runtime Result
if %errorlevel% neq 0 goto :RUNTIME_FAIL

:: --- SUCCESS CASE ---
echo     > Runtime Status: [SUCCESS]
echo     [OUTPUT END: %S_NAME%] Status: SUCCESS >> "%LOGFILE%"
goto :EOF

:: --- FAILURE HANDLERS ---
:SYNTAX_FAIL
echo [SYNTAX ERROR]
echo [FAIL] Syntax error in %S_NAME%. >> "%LOGFILE%"
set /a SCRIPT_ERRORS+=1
goto :EOF

:RUNTIME_FAIL
echo     > Runtime Status: [FAILURE] - Check Log!
echo     [OUTPUT END: %S_NAME%] Status: FAILURE (Code: !errorlevel!) >> "%LOGFILE%"
set /a SCRIPT_ERRORS+=1
goto :EOF