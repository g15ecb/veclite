import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import Table, timeit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def embeddings(rng, n, dim):
    rank = max(16, dim // 16)
    basis = rng.standard_normal((rank, dim), dtype=np.float32)
    x = rng.standard_normal((n, rank), dtype=np.float32) @ basis + 0.3 * rng.standard_normal((n, dim), dtype=np.float32)
    return np.ascontiguousarray(x / np.linalg.norm(x, axis=1, keepdims=True), dtype=np.float32)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    dims = [int(d) for d in (sys.argv[2] if len(sys.argv) > 2 else "768,1536,3072").split(",")]
    batches = (1, 8, 32, 128)
    k = 10
    rows = []
    rng = np.random.default_rng(11)
    for dim in dims:
        data = embeddings(rng, n, dim)
        tags = rng.integers(0, 10, n, dtype=np.int64)
        live = np.ones(n, dtype=np.uint8)
        qs = embeddings(rng, max(batches), dim)
        tables = {s: Table(data, tags, live, metric=2, storage=s) for s in ("f32", "f16", "i8")}
        truth = tables["f32"].search(qs, k)
        for storage, t in tables.items():
            got = t.search(qs, k)
            recall = float(np.mean([len(np.intersect1d(a[0], b[0])) / k for a, b in zip(got, truth)]))
            identical = all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(got, truth))
            scan_bytes = t.enc.nbytes + (t.aux.nbytes if t.aux is not None else 0)
            for sel, ftag in ((1.0, None), (0.1, 3)):
                for b in batches:
                    q = qs[:b]
                    filt = None if ftag is None else np.full(b, ftag)
                    t.stats[:] = 0
                    t.search(q, k, filt)
                    cands = t.stats[0] / b
                    fallbacks = int(t.stats[1])
                    ms = timeit(lambda: t.search(q, k, filt), 3 if b * dim * n > 2e11 else 5) / b
                    rows.append({
                        "n": n, "dim": dim, "storage": storage, "selectivity": sel, "batch": b, "ms_per_q": ms,
                        "recall_vs_f32": recall, "identical_to_f32": identical, "scan_mb": scan_bytes / 1e6,
                        "candidates_per_q": cands, "fallbacks": fallbacks, "metric": "cosine",
                    })
                    print(rows[-1], flush=True)
        del tables, data
    with open(os.path.join(ROOT, "results", "dtype_sweep.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
