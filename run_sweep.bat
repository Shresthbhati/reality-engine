@echo off
cd /d C:\Users\shres\OneDrive\Desktop\codes\reality-engine\.claude\worktrees\reality-engine-docs-merge-da520d
python -m pytest -m "not slow" -q --tb=line -rf -o addopts= > C:\temp\sweep.txt 2> C:\temp\sweep_err.txt
echo EXITCODE=%ERRORLEVEL% >> C:\temp\sweep.txt