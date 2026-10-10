#!/usr/bin/env bash
set -e
python run_ibm_hardware.py prepare --dataset creditcard --fmaps Z ZZ
python run_ibm_hardware.py prepare --dataset paysim --fmaps Z --subdir paysim_Z --val-pos 150 --test-pos 150
python run_ibm_hardware.py prepare --dataset banksim --fmaps Z --subdir banksim_Z --val-pos 150 --test-pos 150
python run_ibm_hardware.py prepare --dataset creditcard --fmaps Z --subdir creditcard_Z_fez
python run_ibm_hardware.py prepare --dataset creditcard --fmaps ZZ --k 3 --subdir creditcard_ZZ_k3 --simulate-missing
python run_ibm_hardware.py prepare --dataset creditcard --fmaps ZZ --k 5 --subdir creditcard_ZZ_k5 --simulate-missing
