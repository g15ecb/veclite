import ctypes
import json
import os
import statistics
import sys
import time

import faiss
import numpy as np

LIB = ctypes.CDLL(os.environ.get("VDB_LIB") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "libvdb.dylib"))
F, I64, U8 = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint8)
I, I32 = ctypes.c_int64, ctypes.c_int32
LIB.vdb_search.argtypes = [F, I, I, I64, U8, F, I, U8, I64, I, I32, I, I64, F, I64]
LIB.vdb_scores.argtypes = [F, I, I, F, I, I32, F]
LIB.vdb_validate_rows.argtypes = [F, I, I, I32, I64]
P = ctypes.c_void_p
LIB.vdb_encode.argtypes = [I32, F, I, I, P, I64]
LIB.vdb_search_ex.argtypes = [P, I32, I, I, I64, U8, F, I, U8, I64, I, I32, I, I64, F, I64]
for fn in (LIB.vdb_search, LIB.vdb_scores, LIB.vdb_validate_rows, LIB.vdb_encode, LIB.vdb_search_ex):
    fn.restype = I32
DTYPES = {"f32": 0, "f16": 1}


def ptr(a, t):
    if a is None:
        return None
    if not a.flags["C_CONTIGUOUS"]:
        raise ValueError("array must be C-contiguous")
    return a.ctypes.data_as(t)


def check(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} rc={rc}")


class Table:
    def __init__(self, data, tags, live, metric=0, storage="f32"):
        n = data.shape[0]
        if data.dtype != np.float32 or data.ndim != 2 or not data.flags["C_CONTIGUOUS"]:
            raise ValueError("data must be C-contiguous 2-D float32")
        if tags.dtype != np.int64 or tags.shape != (n,) or live.dtype != np.uint8 or live.shape != (n,):
            raise ValueError("tags must be int64[n] and live uint8[n]")
        self.data, self.tags, self.live, self.metric = data, tags, live, metric
        bad = np.zeros(1, dtype=np.int64)
        check(LIB.vdb_validate_rows(ptr(data, F), n, data.shape[1], metric, ptr(bad, I64)), f"validate row {bad[0]}")
        self.storage = storage
        self.dt = DTYPES[storage]
        self.enc = data if storage == "f32" else np.empty(data.shape, dtype=np.float16)
        if storage != "f32":
            check(LIB.vdb_encode(self.dt, ptr(data, F), n, data.shape[1], self.enc.ctypes.data, ptr(bad, I64)), f"encode row {bad[0]}")

    def queries(self, qs):
        qs = np.ascontiguousarray(qs, dtype=np.float32)
        if qs.ndim != 2 or qs.shape[1] != self.data.shape[1]:
            raise ValueError("queries must be (nq, dim)")
        return qs

    def search(self, qs, k, filter_tags=None, nchunks=16):
        qs = self.queries(qs)
        nq = qs.shape[0]
        fo = ft = None
        if filter_tags is not None:
            filter_tags = np.asarray(filter_tags)
            if filter_tags.shape != (nq,):
                raise ValueError("one filter tag per query")
            fo = (filter_tags >= 0).astype(np.uint8)
            ft = np.ascontiguousarray(filter_tags, dtype=np.int64)
        ids = np.empty((nq, k), dtype=np.int64)
        sc = np.empty((nq, k), dtype=np.float32)
        cnt = np.zeros(nq, dtype=np.int64)
        check(LIB.vdb_search_ex(self.enc.ctypes.data, self.dt, *self.data.shape, ptr(self.tags, I64), ptr(self.live, U8),
                                ptr(qs, F), nq, ptr(fo, U8), ptr(ft, I64), k, self.metric, nchunks,
                                ptr(ids, I64), ptr(sc, F), ptr(cnt, I64)), "vdb_search_ex")
        return [(ids[q, : cnt[q]], sc[q, : cnt[q]]) for q in range(nq)]

    def scores(self, qs):
        qs = self.queries(qs)
        out = np.empty((qs.shape[0], self.data.shape[0]), dtype=np.float32)
        check(LIB.vdb_scores(ptr(self.data, F), *self.data.shape, ptr(qs, F), qs.shape[0], self.metric, ptr(out, F)), "vdb_scores")
        return out


def conformance(t, qs, k, filter_tags):
    res = t.search(qs, k, filter_tags)
    spec = t.scores(qs)
    recall = []
    for q, (ids, sc) in enumerate(res):
        mask = t.live.astype(bool) & ((filter_tags[q] < 0) | (t.tags == filter_tags[q]))
        idx = np.nonzero(mask)[0]
        order = np.lexsort((idx, spec[q, idx]))
        assert len(ids) == min(k, len(idx)), "resultSizeIsMinKMatches"
        assert mask[ids].all(), "resultOnlyLiveAndMatching"
        assert all(sc[j - 1] < sc[j] or (sc[j - 1] == sc[j] and ids[j - 1] < ids[j]) for j in range(1, len(ids))), "resultSortedUnique"
        assert np.array_equal(ids, idx[order][:k]) and np.array_equal(sc, spec[q, idx][order][:k]), "resultIsExactTopK"
        single = t.search(qs[q : q + 1], k, filter_tags[q : q + 1])[0]
        assert np.array_equal(single[0], ids), "batch_correct"
        d64 = ((t.data[idx].astype(np.float64) - qs[q].astype(np.float64)) ** 2).sum(axis=1)
        truth = idx[np.argsort(d64, kind="stable")[:k]]
        recall.append(len(np.intersect1d(ids, truth)) / max(1, len(truth)))
    return min(recall)


def timeit(fn, reps=5):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    dim = int(sys.argv[2]) if len(sys.argv) > 2 else 1536
    batches = [int(b) for b in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 8, 32, 128, 512]
    k = 10
    rng = np.random.default_rng(1)
    data = rng.standard_normal((n, dim), dtype=np.float32)
    t = Table(data, rng.integers(0, 10, n, dtype=np.int64), (rng.random(n) > 0.01).astype(np.uint8))
    qs = rng.standard_normal((max(*batches, 16), dim), dtype=np.float32)
    mixed = np.where(np.arange(len(qs)) % 2 == 0, -1, np.arange(len(qs)) % 10)
    min_recall = min(conformance(t, qs[:8], k, mixed[:8]), conformance(t, qs[8:13], k, np.full(5, -1)))
    index = faiss.IndexFlatL2(dim)
    index.add(data)
    rows = []
    for b in batches:
        q = qs[:b]
        none = np.full(b, -1)
        tag3 = np.full(b, 3)
        sel = faiss.SearchParameters(sel=faiss.IDSelectorBatch(np.nonzero((t.tags == 3) & (t.live == 1))[0]))
        rows.append({
            "batch": b,
            "mojo_ms_per_q": timeit(lambda: t.search(q, k, none)) / b,
            "faiss_ms_per_q": timeit(lambda: index.search(q, k)) / b,
            "mojo_10pct_ms_per_q": timeit(lambda: t.search(q, k, tag3)) / b,
            "faiss_10pct_ms_per_q": timeit(lambda: index.search(q, k, params=sel)) / b,
            "mojo_mixed_filters_ms_per_q": timeit(lambda: t.search(q, k, mixed[:b])) / b,
        })
    print(json.dumps({"n": n, "dim": dim, "k": k, "gb": data.nbytes / 1e9, "min_recall_vs_float64": min_recall, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
