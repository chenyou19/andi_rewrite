@echo off
setlocal
set "FREESURFER_HOME=%~dp0"
set /p "SYNTHSTRIP_PYTHON="<"%~dp0python_path.txt"
if not exist "%SYNTHSTRIP_PYTHON%" (
  echo Configured Python executable does not exist: %SYNTHSTRIP_PYTHON% 1>&2
  exit /b 2
)
"%SYNTHSTRIP_PYTHON%" "%~dp0mri_synthstrip.py" %*
exit /b %errorlevel%
