@echo off
cd /d "%~dp0"
py -3.13 pinch_pop.py
if errorlevel 1 pause
