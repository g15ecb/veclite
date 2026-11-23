import ctypes
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = ctypes.CDLL(os.environ.get("VDB_LIB") or os.path.join(ROOT, "build", "libvdb.dylib"))
I, I32, P = ctypes.c_int64, ctypes.c_int32, ctypes.c_void_p
SIGS = {
    "vdb_open": [ctypes.c_char_p, I, I, I32, I, P],
    "vdb_close": [I],
    "vdb_append": [I, P, I, P],
    "vdb_delete": [I, I],
    "vdb_commit": [I, P],
    "vdb_checkpoint": [I, P],
    "vdb_reader_begin": [I, P],
    "vdb_reader_end": [I, I],
    "vdb_reader_search": [I, I, P, I, P, P, I, I32, I, P, P, P],
    "vdb_info": [I, P],
}
for name, args in SIGS.items():
    getattr(LIB, name).argtypes = args
    getattr(LIB, name).restype = I32
INFO = ["rows", "ntx", "published", "hdr_txn", "wal_frames", "salt", "main_rows", "poisoned", "dtype"]
DTYPES = {"f32": 0, "f16": 1}


def addr(a):
    if a is None:
        return None
    if not a.flags["C_CONTIGUOUS"]:
        raise ValueError("array must be C-contiguous")
    return a.ctypes.data


def check(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} rc={rc}")


class Store:
    def __init__(self, path, dim, extent_rows=0, storage="f32"):
        out = np.zeros(1, dtype=np.int64)
        check(LIB.vdb_open(path.encode(), dim, extent_rows or max(1024, (64 << 20) // (dim * 4)), DTYPES[storage], 0, addr(out)), "open")
        self.h, self.dim = int(out[0]), dim
        self._id = np.zeros(1, dtype=np.int64)

    def close(self):
        check(LIB.vdb_close(self.h), "close")

    def append_rows(self, vecs, tags):
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self.dim or len(tags) != len(vecs):
            raise ValueError("vecs must be (n, dim) with one tag per row")
        base, stride = vecs.ctypes.data, self.dim * 4
        for i, t in enumerate(tags):
            check(LIB.vdb_append(self.h, base + i * stride, int(t), addr(self._id)), "append")

    def delete(self, row):
        check(LIB.vdb_delete(self.h, row), "delete")

    def commit(self):
        check(LIB.vdb_commit(self.h, None), "commit")

    def checkpoint(self):
        check(LIB.vdb_checkpoint(self.h, None), "checkpoint")

    def info(self):
        out = np.zeros(len(INFO), dtype=np.int64)
        check(LIB.vdb_info(self.h, addr(out)), "info")
        return dict(zip(INFO, out.tolist()))

    def search(self, qs, k, filter_tags=None, nchunks=16, metric=0):
        qs = np.ascontiguousarray(qs, dtype=np.float32)
        nq = len(qs)
        fo = ft = None
        if filter_tags is not None:
            fo = (np.asarray(filter_tags) >= 0).astype(np.uint8)
            ft = np.ascontiguousarray(filter_tags, dtype=np.int64)
        r = np.zeros(1, dtype=np.int64)
        check(LIB.vdb_reader_begin(self.h, addr(r)), "reader_begin")
        ids = np.empty((nq, k), dtype=np.int64)
        sc = np.empty((nq, k), dtype=np.float32)
        cnt = np.zeros(nq, dtype=np.int64)
        try:
            check(LIB.vdb_reader_search(self.h, int(r[0]), addr(qs), nq, addr(fo), addr(ft), k, metric, nchunks,
                                        addr(ids), addr(sc), addr(cnt)), "reader_search")
        finally:
            check(LIB.vdb_reader_end(self.h, int(r[0])), "reader_end")
        return [(ids[q, : cnt[q]], sc[q, : cnt[q]]) for q in range(nq)]


def remove_db(path):
    for p in (path, path + "-wal"):
        if os.path.exists(p):
            os.remove(p)
