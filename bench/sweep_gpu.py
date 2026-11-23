import ctypes
import json
import os
import sys

import faiss
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import LIB, Table, check, timeit
from sweep_dtype import embeddings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
I, I32, P = ctypes.c_int64, ctypes.c_int32, ctypes.c_void_p
LIB.vdb_gpu_open.argtypes = [P, I32, I, I, P, P, P]
LIB.vdb_gpu_update.argtypes = [I, P, P]
LIB.vdb_gpu_close.argtypes = [I]
LIB.vdb_gpu_search.argtypes = [I, P, I, P, P, I, I32, P, P, P, P]
for fn in (LIB.vdb_gpu_open, LIB.vdb_gpu_update, LIB.vdb_gpu_close, LIB.vdb_gpu_search):
    fn.restype = I32


def addr(a):
    return None if a is None else a.ctypes.data


class GpuTable:
    def __init__(self, table):
        self.t = table
        out = np.zeros(1, dtype=np.int64)
        check(LIB.vdb_gpu_open(table.enc.ctypes.data, table.dt, *table.data.shape, addr(table.tags), addr(table.live), addr(out)), "gpu_open")
        self.h = int(out[0])

    def close(self):
        check(LIB.vdb_gpu_close(self.h), "gpu_close")

    def search(self, qs, k, filter_tags=None):
        qs = np.ascontiguousarray(qs, dtype=np.float32)
        nq = len(qs)
        fo = ft = None
        if filter_tags is not None:
            fo = (np.asarray(filter_tags) >= 0).astype(np.uint8)
            ft = np.ascontiguousarray(filter_tags, dtype=np.int64)
        ids = np.empty((nq, k), dtype=np.int64)
        sc = np.empty((nq, k), dtype=np.float32)
        cnt = np.zeros(nq, dtype=np.int64)
        check(LIB.vdb_gpu_search(self.h, addr(qs), nq, addr(fo), addr(ft), k, self.t.metric, addr(ids), addr(sc), addr(cnt), None), "gpu_search")
        return [(ids[q, : cnt[q]], sc[q, : cnt[q]]) for q in range(nq)]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    dims = [int(d) for d in (sys.argv[2] if len(sys.argv) > 2 else "768,1536,3072").split(",")]
    batches = (1, 8, 32, 128, 512)
    k = 10
    rows = []
    rng = np.random.default_rng(17)
    for dim in dims:
        data = embeddings(rng, n, dim)
        tags = rng.integers(0, 10, n, dtype=np.int64)
        live = np.ones(n, dtype=np.uint8)
        qs = embeddings(rng, max(batches), dim)
        index = faiss.IndexFlatIP(dim)
        index.add(data)
        for storage in ("f32", "f16"):
            cpu = Table(data, tags, live, metric=2, storage=storage)
            gpu = GpuTable(cpu)
            g = gpu.search(qs[:64], k)
            c = cpu.search(qs[:64], k)
            agree = float(np.mean([len(np.intersect1d(a[0], b[0])) / k for a, b in zip(g, c)]))
            for sel, ftag in ((1.0, None), (0.1, 3)):
                for b in batches:
                    q = qs[:b]
                    filt = None if ftag is None else np.full(b, ftag)
                    reps = 3 if b * dim * n > 3e10 else 5
                    row = {
                        "n": n, "dim": dim, "storage": storage, "selectivity": sel, "batch": b, "gpu_cpu_agreement": agree,
                        "gpu_ms_per_q": timeit(lambda: gpu.search(q, k, filt), reps) / b,
                        "cpu_ms_per_q": timeit(lambda: cpu.search(q, k, filt), reps) / b,
                    }
                    if storage == "f32" and ftag is None:
                        row["faiss_ms_per_q"] = timeit(lambda: index.search(q, k), reps) / b
                    rows.append(row)
                    print(row, flush=True)
            gpu.close()
        del index, data
    with open(os.path.join(ROOT, "results", "gpu_sweep.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
