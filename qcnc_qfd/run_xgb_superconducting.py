"""XGB on quantum-derived features — superconducting configuration.

Runs every dataset x feature map (Z, ZZ, Pauli) x seed, with Sequential Forward Selection,
ideal + superconducting noisy quantum feature extraction, classical baselines and full result logging.

    python run_xgb_superconducting.py --pilot          # quick check (minutes)
    python run_xgb_superconducting.py                  # full run (resumable: just run it again)
    python run_xgb_superconducting.py -h               # all options
"""
import qfd.threads  # noqa: F401  -- MUST be the first import (limits BLAS/OpenMP threads)

from qfd.pipeline import run_script

if __name__ == "__main__":
    run_script("xgb", "superconducting")
