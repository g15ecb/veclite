from __future__ import annotations

import ctypes
import os
import sys
import threading
import weakref
from typing import Optional, Sequence, Union

import numpy as np

__all__ = ["open", "Database", "VecliteError", "ERRORS", "gpu_available"]

ERRORS = {
    -1: "null argument", -2: "bad row count", -3: "bad or mismatched dimension", -4: "bad k", -5: "bad metric",
    -6: "size overflow", -7: "query has NaN/inf", -8: "score is NaN/inf", -9: "zero query for cosine",
    -10: "bad chunk count", -11: "bad filter", -12: "vector has NaN/inf", -13: "zero vector", -14: "bad batch size",
    -15: "bad or mismatched storage type", -16: "value out of float16 range", -20: "I/O error", -21: "corrupt file",
    -22: "invalid state", -23: "unknown or deleted key", -24: "reader error", -25: "bad argument", -26: "duplicate key",
    -30: "GPU error", -31: "no usable GPU",
}
METRICS = {"l2": 0, "dot": 1, "cosine": 2}
STORAGE = {"f32": 0, "f16": 1}
DEVICES = {"auto": 0, "cpu": 1, "gpu": 2}
MAX_K = 4096
MAX_BATCH = 4096
F16_MAX = 65504.0
_INFO = ["rows", "txn", "published", "checkpointed_txn", "wal_frames", "salt", "main_rows", "poisoned", "storage", "dim",
         "gpu_rows", "live_rows"]
_I, _I32, _P = ctypes.c_int64, ctypes.c_int32, ctypes.c_void_p

Tag = Optional[int]


class VecliteError(RuntimeError):
    def __init__(self, code: int, what: str, row: Optional[int] = None):
        where = "" if row is None else f" at row {row}"
        super().__init__(f"{what}: {ERRORS.get(code, 'error')}{where} ({code})")
        self.code = code
        self.row = row


def _load():
    name = "libveclite.dylib" if sys.platform == "darwin" else "libveclite.so"
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.environ.get("VECLITE_LIB"), os.path.join(here, name), os.path.join(here, "..", "..", "build", name)):
        if path and os.path.exists(path):
            lib = ctypes.CDLL(path)
            break
    else:
        raise OSError(f"{name} not found; build it with scripts/build.sh or set VECLITE_LIB")
    sigs = {
        "veclite_open": [ctypes.c_char_p, _I, _I, _I32, _I, _P], "veclite_close": [_I],
        "veclite_append": [_I, _P, _I, _P], "veclite_append_key": [_I, _P, _I, _I, _P], "veclite_delete": [_I, _I],
        "veclite_delete_key": [_I, _I], "veclite_key_lookup": [_I, _I, _P], "veclite_row_keys": [_I, _P, _I, _P],
        "veclite_commit": [_I, _P],
        "veclite_pending": [_I, _P], "veclite_rollback": [_I, _I],
        "veclite_checkpoint": [_I, _P], "veclite_reader_begin": [_I, _P], "veclite_reader_end": [_I, _I],
        "veclite_reader_search_dev": [_I, _I, _P, _I, _P, _P, _I, _I32, _I, _I32, _I, _P, _P, _P, _P],
        "veclite_info": [_I, _P], "veclite_gpu_available": [],
    }
    for fn, args in sigs.items():
        getattr(lib, fn).argtypes = args
        getattr(lib, fn).restype = _I32
    return lib


_LIB = None
_LIB_LOCK = threading.Lock()


def _lib():
    global _LIB
    with _LIB_LOCK:
        if _LIB is None:
            _LIB = _load()
    return _LIB


def _check(rc: int, what: str, row: Optional[int] = None) -> None:
    if rc != 0:
        raise VecliteError(rc, what, row)


def _ptr(a):
    return None if a is None else a.ctypes.data


def _choice(value: str, options: dict, name: str) -> str:
    if value not in options:
        raise ValueError(f"{name} must be one of {sorted(options)}")
    return value


def gpu_available() -> bool:
    return _lib().veclite_gpu_available() == 1


def _close_handle(h: int) -> None:
    _lib().veclite_close(h)


class Database:
    def __init__(self, path: Union[str, os.PathLike], dim: int = 0, storage: Optional[str] = None,
                 extent_rows: int = 0, device: str = "auto", gpu_min_batch: int = 64):
        if storage is not None:
            _choice(storage, STORAGE, "storage")
        if int(dim) < 0:
            raise ValueError("dim must be >= 0 (0 = take it from an existing file)")
        if int(gpu_min_batch) < 1:
            raise ValueError("gpu_min_batch must be >= 1")
        self.device = _choice(device, DEVICES, "device")
        self.gpu_min_batch = int(gpu_min_batch)
        self.last_device: Optional[str] = None
        self._lock = threading.RLock()
        self._h = 0
        out = np.zeros(1, dtype=np.int64)
        ext = extent_rows or (max(1024, (64 << 20) // (int(dim) * 4)) if dim else 0)
        _check(_lib().veclite_open(os.fsencode(path), int(dim), ext, STORAGE.get(storage, -1), 0, _ptr(out)), "open")
        self._h = int(out[0])
        self._finalizer = weakref.finalize(self, _close_handle, self._h)
        info = self.info()
        self.dim: int = info["dim"]
        self.storage: str = next(k for k, v in STORAGE.items() if v == info["storage"])

    def _handle(self) -> int:
        if self._h == 0:
            raise VecliteError(-22, "database is closed")
        return self._h

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __len__(self) -> int:
        return self.info()["live_rows"]

    def __contains__(self, key: int) -> bool:
        out = np.zeros(1, dtype=np.int64)
        with self._lock:
            rc = _lib().veclite_key_lookup(self._handle(), int(key), _ptr(out))
        if rc not in (0, -23):
            _check(rc, "contains")
        return rc == 0

    def close(self) -> None:
        with self._lock:
            if self._h:
                self._finalizer.detach()
                h, self._h = self._h, 0
                _check(_lib().veclite_close(h), "close")

    def _pending(self) -> int:
        out = np.zeros(1, dtype=np.int64)
        _check(_lib().veclite_pending(self._handle(), _ptr(out)), "pending")
        return int(out[0])

    def _rollback_to(self, mark: int) -> None:
        _check(_lib().veclite_rollback(self._handle(), mark), "rollback")

    @property
    def pending(self) -> int:
        with self._lock:
            return self._pending()

    def rollback(self) -> None:
        with self._lock:
            self._rollback_to(0)

    def add(self, vectors, tags: Optional[Sequence[int]] = None, keys: Optional[Sequence[int]] = None,
            commit: bool = True) -> np.ndarray:
        v = np.ascontiguousarray(vectors, dtype=np.float32)
        if v.ndim == 1:
            v = v[None, :]
        if v.ndim != 2 or v.shape[1] != self.dim:
            raise ValueError(f"vectors must have shape (n, {self.dim})")
        t = np.zeros(len(v), dtype=np.int64) if tags is None else np.asarray(tags, dtype=np.int64).reshape(-1)
        if len(t) != len(v):
            raise ValueError(f"need one tag per vector: {len(t)} tags for {len(v)} vectors")
        k = None if keys is None else np.asarray(keys, dtype=np.int64).reshape(-1)
        if k is not None:
            if len(k) != len(v):
                raise ValueError(f"need one key per vector: {len(k)} keys for {len(v)} vectors")
            uniq, first = np.unique(k, return_index=True)
            if len(uniq) != len(k):
                dup = int(np.setdiff1d(np.arange(len(k)), first)[0])
                raise VecliteError(-26, "add", dup)
        bad = np.flatnonzero(~np.isfinite(v).all(axis=1))
        if bad.size:
            raise VecliteError(-12, "add", int(bad[0]))
        if self.storage == "f16":
            big = np.flatnonzero((np.abs(v) > F16_MAX).any(axis=1))
            if big.size:
                raise VecliteError(-16, "add", int(big[0]))
        ids = np.empty(len(v), dtype=np.int64)
        out = np.zeros(1, dtype=np.int64)
        stride = self.dim * 4
        with self._lock:
            h = self._handle()
            mark = self._pending()
            try:
                for i in range(len(v)):
                    addr = v.ctypes.data + i * stride
                    rc = (_lib().veclite_append(h, addr, int(t[i]), _ptr(out)) if k is None
                          else _lib().veclite_append_key(h, addr, int(t[i]), int(k[i]), _ptr(out)))
                    _check(rc, "add", i)
                    ids[i] = out[0]
                if commit:
                    self.commit()
            except BaseException:
                self._rollback_to(mark)
                raise
        return ids if k is None else k.copy()

    def delete(self, keys: Union[int, Sequence[int]], commit: bool = True) -> None:
        ks = np.atleast_1d(np.asarray(keys, dtype=np.int64))
        with self._lock:
            h = self._handle()
            mark = self._pending()
            try:
                for i, key in enumerate(ks):
                    _check(_lib().veclite_delete_key(h, int(key)), "delete", i)
                if commit:
                    self.commit()
            except BaseException:
                self._rollback_to(mark)
                raise

    def commit(self) -> None:
        with self._lock:
            _check(_lib().veclite_commit(self._handle(), None), "commit")

    def checkpoint(self) -> None:
        with self._lock:
            _check(_lib().veclite_checkpoint(self._handle(), None), "checkpoint")

    def info(self) -> dict:
        out = np.zeros(len(_INFO), dtype=np.int64)
        with self._lock:
            _check(_lib().veclite_info(self._handle(), _ptr(out)), "info")
        return dict(zip(_INFO, out.tolist()))

    def search(self, queries, k: int = 10, tag: Union[Tag, Sequence[Tag]] = None, metric: str = "cosine",
               device: Optional[str] = None):
        dev = DEVICES[_choice(device or self.device, DEVICES, "device")]
        _choice(metric, METRICS, "metric")
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= MAX_K:
            raise ValueError(f"k must be an integer in [1, {MAX_K}]")
        q = np.ascontiguousarray(queries, dtype=np.float32)
        single = q.ndim == 1
        q = q[None, :] if single else q
        if q.ndim != 2 or q.shape[1] != self.dim:
            raise ValueError(f"queries must have shape (nq, {self.dim})")
        nq = len(q)
        if not 1 <= nq <= MAX_BATCH:
            raise ValueError(f"batch size must be in [1, {MAX_BATCH}]")
        fo = ft = None
        if tag is not None:
            tags = list(tag) if isinstance(tag, (list, tuple, np.ndarray)) else [tag] * nq
            if len(tags) != nq:
                raise ValueError(f"need one tag per query: {len(tags)} tags for {nq} queries")
            fo = np.array([t is not None for t in tags], dtype=np.uint8)
            ft = np.array([0 if t is None else int(t) for t in tags], dtype=np.int64)
        ids = np.empty((nq, int(k)), dtype=np.int64)
        sc = np.empty((nq, int(k)), dtype=np.float32)
        cnt = np.zeros(nq, dtype=np.int64)
        r, used = np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int32)
        with self._lock:
            h = self._handle()
            _check(_lib().veclite_reader_begin(h, _ptr(r)), "search")
            try:
                _check(_lib().veclite_reader_search_dev(h, int(r[0]), _ptr(q), nq, _ptr(fo), _ptr(ft), int(k), METRICS[metric], 16,
                                                        dev, self.gpu_min_batch, _ptr(ids), _ptr(sc), _ptr(cnt), _ptr(used)), "search")
            finally:
                _lib().veclite_reader_end(h, int(r[0]))
            self.last_device = "gpu" if used[0] else "cpu"
            flat = np.ascontiguousarray(np.concatenate([ids[i, : cnt[i]] for i in range(nq)]))
            keys = np.empty(len(flat), dtype=np.int64)
            _check(_lib().veclite_row_keys(h, _ptr(flat), len(flat), _ptr(keys)), "search")
        hits, at = [], 0
        for i in range(nq):
            hits.append((keys[at : at + cnt[i]].copy(), sc[i, : cnt[i]].copy()))
            at += int(cnt[i])
        return hits[0] if single else hits


def open(path: Union[str, os.PathLike], dim: int = 0, storage: Optional[str] = None, device: str = "auto",
         gpu_min_batch: int = 64) -> Database:
    return Database(path, dim, storage, device=device, gpu_min_batch=gpu_min_batch)
