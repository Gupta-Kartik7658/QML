@echo off
REM Theory check, shot budget and hybrid test (reuse the caches of steps 1-2).
if "%WORKERS%"=="" set WORKERS=6
python run_theory_check.py --workers %WORKERS% || exit /b 1
python run_shot_budget.py --workers %WORKERS% || exit /b 1
python run_hybrid.py --workers %WORKERS%
