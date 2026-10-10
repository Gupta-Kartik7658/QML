#!/usr/bin/env bash
# IBM venv. Without arguments: dry run. With --yes: submits. Then: python ibm/ibm_hardware.py fetch --all
python ibm/ibm_hardware.py submit --backend ibm_marrakesh --fmaps Z --copies-z 14 "$@"
python ibm/ibm_hardware.py submit --backend ibm_marrakesh --fmaps ZZ --flatten --copies-zz 8 "$@"
python ibm/ibm_hardware.py submit --backend ibm_marrakesh --fmaps Z --copies-z 14 --subdir paysim_Z "$@"
python ibm/ibm_hardware.py submit --backend ibm_marrakesh --fmaps Z --copies-z 40 --subdir banksim_Z "$@"
python ibm/ibm_hardware.py submit --backend ibm_fez --fmaps Z --copies-z 14 --subdir creditcard_Z_fez "$@"
python ibm/ibm_hardware.py submit --backend ibm_marrakesh --fmaps ZZ --flatten --copies-zz 25 --subdir creditcard_ZZ_k3 "$@"
python ibm/ibm_hardware.py submit --backend ibm_marrakesh --fmaps ZZ --flatten --copies-zz 14 --subdir creditcard_ZZ_k5 "$@"
