@echo off
REM Experiment venv, after all hardware jobs were fetched.
python run_ibm_hardware.py analyze
python run_ibm_hardware.py analyze --subdir paysim_Z
python run_ibm_hardware.py analyze --subdir banksim_Z
python run_ibm_hardware.py analyze --subdir creditcard_Z_fez
python run_ibm_hardware.py analyze --subdir creditcard_ZZ_k3
python run_ibm_hardware.py analyze --subdir creditcard_ZZ_k5
