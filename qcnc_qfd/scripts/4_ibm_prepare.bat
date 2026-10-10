@echo off
REM Experiment venv: build the data bundles for the seven IBM hardware jobs of the paper.
python run_ibm_hardware.py prepare --dataset creditcard --fmaps Z ZZ || exit /b 1
python run_ibm_hardware.py prepare --dataset paysim --fmaps Z --subdir paysim_Z --val-pos 150 --test-pos 150 || exit /b 1
python run_ibm_hardware.py prepare --dataset banksim --fmaps Z --subdir banksim_Z --val-pos 150 --test-pos 150 || exit /b 1
python run_ibm_hardware.py prepare --dataset creditcard --fmaps Z --subdir creditcard_Z_fez || exit /b 1
python run_ibm_hardware.py prepare --dataset creditcard --fmaps ZZ --k 3 --subdir creditcard_ZZ_k3 --simulate-missing || exit /b 1
python run_ibm_hardware.py prepare --dataset creditcard --fmaps ZZ --k 5 --subdir creditcard_ZZ_k5 --simulate-missing || exit /b 1
