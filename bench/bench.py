import ctypes
import json
import os
import statistics
import sys
import time

import faiss
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = ctypes.CDLL(os.path.join(ROOT, "build", "libvdb.dylib"))

F = ctypes.POINTER(ctypes.c_float)
I64 = ctypes.POINTER(ctypes.c_int64)
U8 = ctypes.POINTER(ctypes.c_uint8)

LIB.vdb_search.restype = ctypes.c_int32
LIB.vdb_search.argtypes = [F, ctypes.c_int64, ctypes.c_int64, I64, U8, F, ctypes.c_int64, ctypes.c_int32,
                           ctypes.c_int32, ctypes.c_int64, ctypes.c_int64, I64, F, I64]
LIB.vdb_scores.restype = ctypes.c_int32
LIB.vdb_scores.argtypes = [F, ctypes.c_int64, ctypes.c_int64, F, ctypes.c_int32, F]
LIB.vdb_validate_rows.restype = ctypes.c_int32
LIB.vdb_validate_rows.argtypes = [F, ctypes.c_int64, ctypes.c_int64, ctypes.c_int32, I64]


def ptr(a, t):
    if not a.flags["C_CONTIGUOUS"]:
        raise ValueError("array must be C-contiguous")
    return a.ctypes.data_as(t)


class Table:
    def __init__(self, data, tags, live):
        if data.dtype != np.float32 or data.ndim != 2:
            raise ValueError("data must be 2-D float32")
        if tags.dtype != np.int64 or tags.shape != (data.shape[0],):
            raise ValueError("tags must be int64 with one entry per row")
        if live.dtype != np.uint8 or live.shape != (data.shape[0],):
            raise ValueError("live must be uint8 with one entry per row")
        self.data = np.ascontiguousarray(data)
        self.tags = np.ascontiguousarray(tags)
        self.live = np.ascontiguousarray(live)
        bad = np.zeros(1, dtype=np.int64)
        rc = LIB.vdb_validate_rows(ptr(self.data, F), self.data.shape[0], self.data.shape[1], 0, ptr(bad, I64))
        if rc != 0:
            raise ValueError(f"vdb_validate_rows rc={rc} row={bad[0]}")

    def search(self, q, k, filter_tag=None, nchunks=16, metric=0):
        q = np.ascontiguousarray(q, dtype=np.float32)
        if q.shape != (self.data.shape[1],):
            raise ValueError("query dimension mismatch")
        ids = np.empty(k, dtype=np.int64)
        sc = np.empty(k, dtype=np.float32)
        cnt = np.zeros(1, dtype=np.int64)
        rc = LIB.vdb_search(ptr(self.data, F), self.data.shape[0], self.data.shape[1], ptr(self.tags, I64),
                            ptr(self.live, U8), ptr(q, F), k, metric, 0 if filter_tag is None else 1,
                            0 if filter_tag is None else filter_tag, nchunks, ptr(ids, I64), ptr(sc, F),
                            ptr(cnt, I64))
        if rc != 0:
            raise RuntimeError(f"vdb_search rc={rc}")
        return ids[: cnt[0]].copy(), sc[: cnt[0]].copy()

    def scores(self, q, metric=0):
        q = np.ascontiguousarray(q, dtype=np.float32)
        out = np.empty(self.data.shape[0], dtype=np.float32)
        rc = LIB.vdb_scores(ptr(self.data, F), self.data.shape[0], self.data.shape[1], ptr(q, F), metric,
                            ptr(out, F))
        if rc != 0:
            raise RuntimeError(f"vdb_scores rc={rc}")
        return out


def spec_topk(scores, mask, k):
    idx = np.nonzero(mask)[0]
    order = np.lexsort((idx, scores[idx]))
    return idx[order][:k], scores[idx][order][:k]


def check_conformance(t, q, k, filter_tag, nchunks):
    ids, sc = t.search(q, k, filter_tag, nchunks)
    s = t.scores(q)
    mask = t.live.astype(bool) if filter_tag is None else (t.live.astype(bool) & (t.tags == filter_tag))
    rid, rsc = spec_topk(s, mask, k)
    assert len(ids) == min(k, int(mask.sum())), "resultSizeIsMinKMatches"
    assert np.all(mask[ids]), "resultOnlyLiveAndMatching"
    for j in range(1, len(ids)):
        assert sc[j - 1] < sc[j] or (sc[j - 1] == sc[j] and ids[j - 1] < ids[j]), "resultSortedUnique"
    assert np.array_equal(ids, rid) and np.array_equal(sc, rsc), "resultIsExactTopK"
    d64 = ((t.data.astype(np.float64) - q.astype(np.float64)) ** 2).sum(axis=1)
    true_ids = np.nonzero(mask)[0][np.argsort(d64[mask], kind="stable")][:k]
    return len(set(ids.tolist()) & set(true_ids.tolist())) / max(1, len(true_ids))


def timeit(fn, reps):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def numpy_search(data, sqn, q, k, mask):
    d = sqn - 2.0 * (data @ q)
    if mask is not None:
        d = np.where(mask, d, np.inf)
    part = np.argpartition(d, k)[:k]
    return part[np.argsort(d[part])]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    dim = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    k = 10
    rng = np.random.default_rng(1)
    data = rng.standard_normal((n, dim), dtype=np.float32)
    tags = rng.integers(0, 100, n, dtype=np.int64)
    live = (rng.random(n) > 0.01).astype(np.uint8)
    t = Table(data, tags, live)
    qs = rng.standard_normal((8, dim), dtype=np.float32)

    recalls = []
    for nchunks in (1, 7, 16, 64):
        for ft in (None, 3):
            recalls.append(check_conformance(t, qs[0], k, ft, nchunks))
    sel_tags = np.where(tags < 10, 0, tags)
    t10 = Table(data, sel_tags, live)
    recalls.append(check_conformance(t10, qs[1], k, 0, 16))

    sqn = (data * data).sum(axis=1)
    index = faiss.IndexFlatL2(dim)
    index.add(data)
    live_b = live.astype(bool)
    res = {"n": n, "dim": dim, "k": k, "threads": os.cpu_count(), "min_recall_vs_float64": min(recalls), "ms": {}}
    reps = 20
    for label, table, ft, sel in (("all", t, None, None), ("10pct", t10, 0, 0), ("1pct", t, 3, 3)):
        mask = live_b if sel is None else (live_b & (table.tags == sel))
        q = qs[2]
        for nc in (8, 16, 64):
            res["ms"][f"mojo_{label}_chunks{nc}"] = timeit(lambda: table.search(q, k, ft, nc), reps)
        res["ms"][f"numpy_{label}"] = timeit(lambda: numpy_search(data, sqn, q, k, mask), reps)
        if sel is None:
            res["ms"][f"faiss_{label}"] = timeit(lambda: index.search(q[None, :], k), reps)
        else:
            sub = np.nonzero(mask)[0].astype(np.int64)
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(sub))
            res["ms"][f"faiss_{label}"] = timeit(lambda: index.search(q[None, :], k, params=params), reps)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
