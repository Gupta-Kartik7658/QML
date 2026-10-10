@echo off
REM Main experiments, superconducting configuration (+ classical baselines, SFS-vs-exhaustive, noise sweep).
REM Run from the project folder:  scripts\1_main_superconducting.bat      (set WORKERS to change parallelism)
if "%WORKERS%"=="" set WORKERS=6
python check_env.py || exit /b 1
python run_xgb_superconducting.py --workers %WORKERS% || exit /b 1
python run_mlp_superconducting.py --workers %WORKERS% || exit /b 1
python run_svm_superconducting.py --workers %WORKERS% || exit /b 1
python run_sfs_vs_exhaustive.py --workers %WORKERS% --datasets creditcard banksim || exit /b 1
python run_noise_sweep.py --arch superconducting --workers %WORKERS% || exit /b 1
