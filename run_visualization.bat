@echo off
setlocal
cd /d "%~dp0\.."
echo Serving project at http://127.0.0.1:8000/
echo Open http://127.0.0.1:8000/visualization/
py -3 -m http.server 8000
