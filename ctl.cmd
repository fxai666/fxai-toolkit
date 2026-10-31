@echo off
rem service control for the launcher page footer buttons
rem lives inside custom_nodes\fxai-toolkit so it syncs with the node pack
rem   ctl.cmd kill    stop the process listening on the port from fxai.json
rem   ctl.cmd start   start the service in the foreground (log stays here)
rem   ctl.cmd where   show detected ComfyUI root / config / script
rem
rem ctl.py launches the server with sys.executable, so PY must be the
rem interpreter that can actually run ComfyUI. Set FXAI_PYTHON to pin it;
rem otherwise fall back to whatever "python" resolves to on PATH.
chcp 65001 >nul
cd /d "%~dp0"
set "PY=%FXAI_PYTHON%"
if not defined PY set "PY=python"
"%PY%" "%~dp0ctl.py" %*
