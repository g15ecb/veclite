import ctypes
import os
import sys

import numpy as np

__all__ = ["open", "Database", "VecliteError", "ERRORS", "gpu_available"]

ERRORS = {
    -1: "null argument", -2: "bad row count", -3: "bad or mismatched dimension", -4: "bad k", -5: "bad metric",
    -6: "size overflow", -7: "query has NaN/inf", -8: "score is NaN/inf", -9: "zero query for cosine",
    -10: "bad chunk count", -11: "bad filter", -12: "vector has NaN/inf", -13: "zero vector", -14: "bad batch size",
    -15: "bad or mismatched storage type", -16: "value out of float16 range", -20: "I/O error", -21: "corrupt file",
    -22: "invalid state", -23: "unknown or deleted row id", -24: "reader error", -25: "bad argument",
    -30: "GPU error", -31: "no usable GPU",
}
METRICS = {"l2": 0, "dot": 1, "cosine": 2}
STORAGE = {"f32": 0, "f16": 1}
DEVICES = {"auto": 0, "cpu": 1, "gpu": 2}
_I, _I32, _P = ctypes.c_int64, ctypes.c_int32, ctypes.c_void_p


class VecliteError(RuntimeError):
    def __init__(self, code, what):
        super().__init__(f"{what}: {ERRORS.get(code, 'error')} ({code})")
        self.code = code


def _load():
    name = "libveclite.dylib" if sys.platform == "darwin" else "libveclite.so"
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.environ.get("VECLITE_LIB"), os.path.join(here, name),
                  os.path.join(here, "..", "..", "build", name)]
    for path in candidates:
        if path and os.path.exists(path):
            lib = ctypes.CDLL(path)
            break
    else:
        raise OSError(f"{name} not found; build it with scripts/build.sh or set VECLITE_LIB")
    sigs = {
        "veclite_open": [ctypes.c_char_p, _I, _I, _I32, _I, _P], "veclite_close": [_I],
        "veclite_append": [_I, _P, _I, _P], "veclite_delete": [_I, _I], "veclite_commit": [_I, _P],
        "veclite_checkpoint": [_I, _P], "veclite_reader_begin": [_I, _P], "veclite_reader_end": [_I, _I],
        "veclite_reader_search_dev": [_I, _I, _P, _I, _P, _P, _I, _I32, _I, _I32, _I, _P, _P, _P, _P],
        "veclite_info": [_I, _P], "veclite_gpu_available": [],
    }
    for fn, args in sigs.items():
        getattr(lib, fn).argtypes = args
        getattr(lib, fn).restype = _I32
    return lib


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = _load()
    return _LIB


def _check(rc, what):
    if rc != 0:
        raise VecliteError(rc, what)


def _ptr(a):
    return None if a is None else a.ctypes.data


def gpu_available():
    return _lib().veclite_gpu_available() == 1


def _device(device):
    if device not in DEVICES:
        raise ValueError(f"device must be one of {sorted(DEVICES)}")
    return device


class Database:
    def __init__(self, path, dim=0, storage=None, extent_rows=0, device="auto", gpu_min_batch=64):
        if storage is not None and storage not in STORAGE:
            raise ValueError(f"storage must be one of {sorted(STORAGE)}")
        if int(gpu_min_batch) < 1:
            raise ValueError("gpu_min_batch must be >= 1")
        self.device = _device(device)
        self.gpu_min_batch = int(gpu_min_batch)
        self.last_device = None
        if dim < 0:
            raise ValueError("dim must be >= 0 (0 = take it from an existing file)")
        out = np.zeros(1, dtype=np.int64)
        ext = extent_rows or (max(1024, (64 << 20) // (dim * 4)) if dim else 0)
        _check(_lib().veclite_open(os.fsencode(path), dim, ext, STORAGE.get(storage, -1), 0, _ptr(out)), "open")
        self._h = int(out[0])
        info = self.info()
        self.dim = info["dim"]
        self.storage = next(k for k, v in STORAGE.items() if v == info["storage"])

    def _handle(self):
        if self._h == 0:
            raise VecliteError(-22, "database is closed")
        return self._h

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._h:
            _check(_lib().veclite_close(self._h), "close")
            self._h = 0

    def add(self, vectors, tags=None, commit=True):
        v = np.ascontiguousarray(vectors, dtype=np.float32)
        if v.ndim == 1:
            v = v[None, :]
        if v.ndim != 2 or v.shape[1] != self.dim:
            raise ValueError(f"vectors must have shape (n, {self.dim})")
        t = np.zeros(len(v), dtype=np.int64) if tags is None else np.asarray(tags, dtype=np.int64).reshape(-1)
        if len(t) != len(v):
            raise ValueError("need one tag per vector")
        h, out, ids = self._handle(), np.zeros(1, dtype=np.int64), np.empty(len(v), dtype=np.int64)
        stride = self.dim * 4
        for i in range(len(v)):
            _check(_lib().veclite_append(h, v.ctypes.data + i * stride, int(t[i]), _ptr(out)), "add")
            ids[i] = out[0]
        if commit:
            self.commit()
        return ids

    def delete(self, row_id, commit=True):
        _check(_lib().veclite_delete(self._handle(), int(row_id)), "delete")
        if commit:
            self.commit()

    def commit(self):
        _check(_lib().veclite_commit(self._handle(), None), "commit")

    def checkpoint(self):
        _check(_lib().veclite_checkpoint(self._handle(), None), "checkpoint")

    def info(self):
        out = np.zeros(11, dtype=np.int64)
        _check(_lib().veclite_info(self._handle(), _ptr(out)), "info")
        keys = ["rows", "txn", "published", "checkpointed_txn", "wal_frames", "salt", "main_rows", "poisoned", "storage", "dim", "gpu_rows"]
        return dict(zip(keys, out.tolist()))

    def search(self, queries, k=10, tag=None, metric="cosine", device=None):
        dev = DEVICES[_device(device or self.device)]
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {sorted(METRICS)}")
        q = np.ascontiguousarray(queries, dtype=np.float32)
        single = q.ndim == 1
        q = q[None, :] if single else q
        if q.ndim != 2 or q.shape[1] != self.dim:
            raise ValueError(f"queries must have shape (nq, {self.dim})")
        nq = len(q)
        fo = ft = None
        if tag is not None:
            tags = np.broadcast_to(np.asarray(tag, dtype=object), (nq,))
            fo = np.array([t is not None for t in tags], dtype=np.uint8)
            ft = np.array([0 if t is None else int(t) for t in tags], dtype=np.int64)
        ids = np.empty((nq, k), dtype=np.int64)
        sc = np.empty((nq, k), dtype=np.float32)
        cnt = np.zeros(nq, dtype=np.int64)
        h, r, used = self._handle(), np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int32)
        _check(_lib().veclite_reader_begin(h, _ptr(r)), "search")
        try:
            _check(_lib().veclite_reader_search_dev(h, int(r[0]), _ptr(q), nq, _ptr(fo), _ptr(ft), k, METRICS[metric], 16,
                                                    dev, self.gpu_min_batch, _ptr(ids), _ptr(sc), _ptr(cnt), _ptr(used)), "search")
        finally:
            _lib().veclite_reader_end(h, int(r[0]))
        self.last_device = "gpu" if used[0] else "cpu"
        hits = [(ids[i, : cnt[i]].copy(), sc[i, : cnt[i]].copy()) for i in range(nq)]
        return hits[0] if single else hits


def open(path, dim=0, storage=None, device="auto", gpu_min_batch=64):
    return Database(path, dim, storage, device=device, gpu_min_batch=gpu_min_batch)
