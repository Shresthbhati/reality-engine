@echo off
:: Reality Engine: double-click to start the full product (API + Home dashboard UI) and open it in the browser.
::   start.bat            start and open http://localhost:3100
::   start.bat stop       shut both servers down
::   start.bat --no-browser
call "%~dp0launch_ui.bat" %*
