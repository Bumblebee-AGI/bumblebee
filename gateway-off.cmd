@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\gateway.ps1" off %*
