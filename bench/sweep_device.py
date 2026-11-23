import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import veclite
from bench import timeit
from sweep_dtype import embeddings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    dims = [int(d) for d in (sys.argv[2] if len(sys.argv) > 2 else "768,1536,3072").split(",")]
    batches = (1, 8, 32, 64, 128, 256, 512)
    rows = []
    rng = np.random.default_rng(23)
    for dim in dims:
        data = embeddings(rng, n, dim)
        qs = embeddings(rng, max(batches), dim)
        path = os.path.join(tempfile.mkdtemp(), "d.db")
        with veclite.open(path, dim=dim) as db:
            for lo in range(0, n, 8192):
                db.add(data[lo : lo + 8192], commit=False)
                db.commit()
            db.checkpoint()
            ref = db.search(qs[:64], k=10, device="cpu")
            got = db.search(qs[:64], k=10, device="gpu")
            agree = float(np.mean([len(np.intersect1d(a[0], b[0])) / 10 for a, b in zip(ref, got)]))
            for b in batches:
                q = qs[:b]
                row = {"n": n, "dim": dim, "batch": b, "gpu_cpu_agreement": agree}
                for dev in ("cpu", "gpu", "auto"):
                    row[f"{dev}_ms_per_q"] = timeit(lambda: db.search(q, k=10, device=dev), 3 if b * dim * n > 3e10 else 5) / b
                    row[f"{dev}_used"] = db.last_device
                rows.append(row)
                print(row, flush=True)
        os.remove(path)
        os.remove(path + "-wal")
    with open(os.path.join(ROOT, "results", "device_sweep.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
