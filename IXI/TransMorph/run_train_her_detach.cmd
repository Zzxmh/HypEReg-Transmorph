@echo off
REM New CMD window; keeps running if you close IDE terminal (local). batch_size=2 in train_TransMorph_her.py
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_train_her_detach.ps1"
pause
