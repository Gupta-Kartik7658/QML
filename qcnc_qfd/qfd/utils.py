"""I/O, hashing, logging, timing and resource helpers (Windows + Linux safe)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


# ----------------------------------------------------------------------------- hashing
def _jsonable(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def stable_hash(obj: Any, n: int = 16) -> str:
    s = json.dumps(obj, sort_keys=True, default=_jsonable, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def array_hash(a: np.ndarray, n: int = 16) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()[:n]


def file_sha256(path: Path, cache_file: Path | None = None) -> str:
    """Full-file SHA-256, memoised by (path, size, mtime) because PaySim is ~470 MB."""
    path = Path(path)
    st = path.stat()
    key = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    memo = {}
    if cache_file is not None and cache_file.exists():
        try:
            memo = json.loads(cache_file.read_text())
        except Exception:
            memo = {}
    if key in memo:
        return memo[key]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if cache_file is not None:
        memo[key] = digest
        atomic_write_text(cache_file, json.dumps(memo, indent=1))
    return digest


# ----------------------------------------------------------------------------- atomic IO
def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(10):  # Windows: target may be briefly locked by a reader
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(0.2 * (attempt + 1))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=_jsonable))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_npz(path: Path, compress: bool = True, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".npz")
    os.close(fd)
    try:
        (np.savez_compressed if compress else np.savez)(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_csv_df(path: Path, df) -> None:
    atomic_write_text(path, df.to_csv(index=False))


# ----------------------------------------------------------------------------- logging
def get_logger(name: str, log_file: Path | None = None, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(processName)s | %(levelname)s | %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve()
                   for h in logger.handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)


# ----------------------------------------------------------------------------- timing/memory
class Timer:
    def __init__(self):
        self.t = {}

    @contextmanager
    def __call__(self, key: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.t[key] = self.t.get(key, 0.0) + time.perf_counter() - t0


def peak_memory_mb() -> float:
    """Peak resident memory of *this* process in MB (best effort, cross-platform)."""
    try:
        import psutil
        mi = psutil.Process().memory_info()
        if hasattr(mi, "peak_wset"):  # Windows
            return mi.peak_wset / 2**20
    except Exception:
        pass
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1024.0 if sys.platform != "darwin" else r / 2**20
    except Exception:
        return float("nan")


def current_memory_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except Exception:
        return float("nan")


def environment_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "hostname_hash": stable_hash(platform.node(), 8),  # anonymised (double-blind review)
        "threads_env": os.environ.get("OMP_NUM_THREADS"),
    }
    try:
        import psutil
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    for mod in ("numpy", "scipy", "sklearn", "xgboost", "pandas", "qiskit", "qiskit_aer",
                "qiskit_machine_learning", "matplotlib", "joblib"):
        try:
            m = __import__(mod)
            info[f"ver_{mod}"] = getattr(m, "__version__", "?")
        except Exception as e:  # pragma: no cover
            info[f"ver_{mod}"] = f"NOT AVAILABLE ({e.__class__.__name__})"
    return info


def quantum_versions() -> dict:
    out = {}
    for mod in ("qiskit", "qiskit_aer", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "na"
    return out


def limit_threadpools(n: int) -> None:
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=n)
    except Exception:
        pass
