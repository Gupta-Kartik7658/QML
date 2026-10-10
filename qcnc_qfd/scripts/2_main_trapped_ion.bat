@echo off
REM Trapped-ion configuration. --arch-only trains only the architecture-dependent (noisy) representations;
REM the ideal and classical rows are identical for both architectures and come from step 1.
if "%WORKERS%"=="" set WORKERS=6
python run_xgb_trapped_ion.py --workers %WORKERS% --arch-only || exit /b 1
python run_mlp_trapped_ion.py --workers %WORKERS% --arch-only || exit /b 1
python run_svm_trapped_ion.py --workers %WORKERS% --arch-only || exit /b 1
python run_noise_sweep.py --arch trapped_ion --workers %WORKERS% || exit /b 1
python run_qsvm_reference.py --workers %WORKERS% || exit /b 1
python run_noise_sweep.py --plot-only || exit /b 1
python analyze_results.py
