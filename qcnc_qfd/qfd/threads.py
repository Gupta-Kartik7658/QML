"""Thread-pool limiting. MUST be imported before numpy/scipy/sklearn/xgboost.

Every runner script does `import qfd.threads` as its very first import. The value comes
from `--threads-per-worker N` on the command line (parsed directly from sys.argv here,
because argparse runs later) or the QFD_THREADS environment variable; default 1.
Child processes started with 'spawn' inherit these environment variables, which prevents
CPU oversubscription (6 workers x N BLAS threads).
"""
import os
import sys


def _requested_threads() -> int:
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--threads-per-worker" and i + 1 < len(argv):
            return max(1, int(argv[i + 1]))
        if a.startswith("--threads-per-worker="):
            return max(1, int(a.split("=", 1)[1]))
    return max(1, int(os.environ.get("QFD_THREADS", "1")))


N_THREADS = _requested_threads()
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = str(N_THREADS)
os.environ["QFD_THREADS"] = str(N_THREADS)
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("MPLBACKEND", "Agg")
