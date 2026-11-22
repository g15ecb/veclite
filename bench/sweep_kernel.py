import json
import os
import sys

import faiss
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import Table, timeit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    dims = [int(d) for d in (sys.argv[2] if len(sys.argv) > 2 else "768,1536,3072").split(",")]
    batches = (1, 8, 32, 128, 512)
    k = 10
    rows = []
    rng = np.random.default_rng(5)
    for dim in dims:
        data = rng.standard_normal((n, dim), dtype=np.float32)
        tags = rng.integers(0, 100, n, dtype=np.int64)
        t = Table(data, tags, np.ones(n, dtype=np.uint8))
        index = faiss.IndexFlatL2(dim)
        index.add(data)
        qs = rng.standard_normal((max(batches), dim), dtype=np.float32)
        for sel, match in ((1.0, None), (0.1, np.arange(10)), (0.01, np.array([3]))):
            params, tt = None, t
            if match is not None:
                sub = np.isin(tags, match)
                tt = Table(data, np.where(sub, 0, 1).astype(np.int64), np.ones(n, dtype=np.uint8))
                params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(np.nonzero(sub)[0].astype(np.int64)))
            for b in batches:
                q = qs[:b]
                ftag = None if match is None else np.zeros(b, dtype=np.int64)
                reps = 3 if b * dim * n > 2e11 else 5
                mojo = timeit(lambda: tt.search(q, k, ftag), reps) / b
                fa = timeit(lambda: index.search(q, k, params=params) if params else index.search(q, k), reps) / b
                rows.append({"n": n, "dim": dim, "selectivity": sel, "batch": b, "mojo_ms_per_q": mojo, "faiss_ms_per_q": fa})
                print(rows[-1], flush=True)
        del index, data, t
    with open(os.path.join(ROOT, "results", "kernel_sweep.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
