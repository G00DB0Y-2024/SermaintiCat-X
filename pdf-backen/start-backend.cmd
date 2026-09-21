@echo off
REM ============================================================
REM  start-backend.cmd
REM  Purpose: Launch PDFAI FastAPI service.
REM  Usage  : Double-click (visible console) OR
REM           wrap with SilentLauncher-CMD.vbs for hidden auto-start.
REM ============================================================

setlocal

REM ---- Configuration ----------------------------------------
set "PROJECT_DIR=F:\MyProjects\PDFAI\pdf-backen"
set "PYTHON_EXE=python"
set "MAIN_SCRIPT=PdfBacken.py"
REM ------------------------------------------------------------

cd /d "%PROJECT_DIR%"

echo ============================================================
echo  PDFAI Backend Launcher
echo  Project : %PROJECT_DIR%
echo  Python  : %PYTHON_EXE%
echo  Port    : 8225
echo ============================================================
echo.

REM Run the server.
"%PYTHON_EXE%" "%MAIN_SCRIPT%"
set "EXITCODE=%ERRORLEVEL%"

echo.
echo ============================================================
echo  Server exited with code %EXITCODE%
echo  Press any key to close this window...
echo ============================================================
pause >nul
