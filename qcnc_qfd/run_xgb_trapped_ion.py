"""XGB on quantum-derived features — trapped_ion configuration.

Runs every dataset x feature map (Z, ZZ, Pauli) x seed, with Sequential Forward Selection,
ideal + trapped_ion noisy quantum feature extraction, classical baselines and full result logging.

    python run_xgb_trapped_ion.py --pilot          # quick check (minutes)
    python run_xgb_trapped_ion.py                  # full run (resumable: just run it again)
    python run_xgb_trapped_ion.py -h               # all options
"""
import qfd.threads  # noqa: F401  -- MUST be the first import (limits BLAS/OpenMP threads)

from qfd.pipeline import run_script

if __name__ == "__main__":
    run_script("xgb", "trapped_ion")
