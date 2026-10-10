#!/usr/bin/env bash
set -e; WORKERS=${WORKERS:-6}
python run_xgb_trapped_ion.py --workers $WORKERS --arch-only
python run_mlp_trapped_ion.py --workers $WORKERS --arch-only
python run_svm_trapped_ion.py --workers $WORKERS --arch-only
python run_noise_sweep.py --arch trapped_ion --workers $WORKERS
python run_qsvm_reference.py --workers $WORKERS
python run_noise_sweep.py --plot-only
python analyze_results.py
