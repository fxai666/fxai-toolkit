@echo off
rem service control for the launcher page footer buttons
rem lives inside custom_nodes\fxai-toolkit so it syncs with the node pack
rem   ctl.cmd kill    stop the process listening on the port from fxai.json
rem   ctl.cmd start   start the service in the foreground (log stays here)
rem   ctl.cmd where   show detected ComfyUI root / config / script
chcp 65001 >nul
cd /d "%~dp0"
set "PY=K:\fxaitool\conda\env\comfyui13\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" "%~dp0ctl.py" %*
