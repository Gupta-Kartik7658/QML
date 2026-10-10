@echo off
REM IBM venv (Qiskit 2.x + qiskit-ibm-runtime, see ibm/README.md). Without arguments: dry run (cost estimate only).
REM   scripts\5_ibm_submit.bat --yes   submits the seven jobs (about 250 s of QPU time in total).
REM Afterwards:  python ibm\ibm_hardware.py fetch --all   (repeat until every job is DONE)
python ibm\ibm_hardware.py submit --backend ibm_marrakesh --fmaps Z --copies-z 14 %1
python ibm\ibm_hardware.py submit --backend ibm_marrakesh --fmaps ZZ --flatten --copies-zz 8 %1
python ibm\ibm_hardware.py submit --backend ibm_marrakesh --fmaps Z --copies-z 14 --subdir paysim_Z %1
python ibm\ibm_hardware.py submit --backend ibm_marrakesh --fmaps Z --copies-z 40 --subdir banksim_Z %1
python ibm\ibm_hardware.py submit --backend ibm_fez --fmaps Z --copies-z 14 --subdir creditcard_Z_fez %1
python ibm\ibm_hardware.py submit --backend ibm_marrakesh --fmaps ZZ --flatten --copies-zz 25 --subdir creditcard_ZZ_k3 %1
python ibm\ibm_hardware.py submit --backend ibm_marrakesh --fmaps ZZ --flatten --copies-zz 14 --subdir creditcard_ZZ_k5 %1
