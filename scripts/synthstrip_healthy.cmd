@echo off
setlocal
set "FREESURFER_HOME=C:\ML\tools\synthstrip"
"C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe" "C:\ML\tools\synthstrip\mri_synthstrip_windows_entry_v2.py" %*
exit /b %errorlevel%
