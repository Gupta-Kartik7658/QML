#!/usr/bin/env bash
# Main experiments, superconducting configuration. Run from the project folder: bash scripts/1_main_superconducting.sh
set -e; WORKERS=${WORKERS:-6}
python check_env.py
python run_xgb_superconducting.py --workers $WORKERS
python run_mlp_superconducting.py --workers $WORKERS
python run_svm_superconducting.py --workers $WORKERS
python run_sfs_vs_exhaustive.py --workers $WORKERS --datasets creditcard banksim
python run_noise_sweep.py --arch superconducting --workers $WORKERS
