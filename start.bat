@echo off
:: Double-click to start the whole app (API + UI) and open it in the browser.
:: Pass "stop" to shut it down, or "--no-browser" to skip opening the page.
call "%~dp0launch_ui.bat" %*
