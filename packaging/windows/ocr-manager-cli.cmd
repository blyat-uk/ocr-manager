@echo off
rem OCR Manager with a console: the same interpreter and flags as
rem "OCR Manager.exe", but output stays in this window (--version,
rem --self-test, --setup-engine, troubleshooting).
setlocal
set "OCR_MANAGER_BUNDLE=%~dp0"
set "OCR_MANAGER_BUNDLE=%OCR_MANAGER_BUNDLE:~0,-1%"
set "PATH=%OCR_MANAGER_BUNDLE%\bin;%PATH%"
"%OCR_MANAGER_BUNDLE%\python\python.exe" -s -E -B -X utf8 "%OCR_MANAGER_BUNDLE%\src\main.py" %*
exit /b %ERRORLEVEL%
