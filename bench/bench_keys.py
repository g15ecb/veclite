import json
import os
import statistics
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import veclite

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fresh():
    return os.path.join(tempfile.mkdtemp(), "k.db")


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def median_time(fn, reps=5):
    fn()
    return statistics.median(timed(fn)[0] for _ in range(reps))


def main():
    dim = int(sys.argv[1]) if len(sys.argv) > 1 else 768
    sizes = [int(n) for n in (sys.argv[2] if len(sys.argv) > 2 else "10000,50000,200000").split(",")]
    rng = np.random.default_rng(9)
    rows = []
    for n in sizes:
        x = rng.standard_normal((n, dim), dtype=np.float32)
        keys = rng.permutation(np.arange(n, dtype=np.int64) * 7919 + 13)
        for mode in ("row_ids", "keys"):
            path = fresh()
            with veclite.open(path, dim=dim) as db:
                t_add, _ = timed(lambda: db.add(x, keys=None if mode == "row_ids" else keys))
                t_ckpt, _ = timed(db.checkpoint)
                probe = keys[: min(1000, n)] if mode == "keys" else np.arange(min(1000, n))
                assert sum(int(k) in db for k in probe) == len(probe)
                t_contains = median_time(lambda: sum(int(k) in db for k in probe))
                q = x[:32]
                assert len(db.search(q, k=10, device="cpu")) == 32
                t_search = median_time(lambda: db.search(q, k=10, device="cpu"))
            def reopen():
                with veclite.open(path) as d2:
                    assert len(d2) == n
            t_open = median_time(reopen, 3)
            wal_n = min(20000, n)
            path2 = fresh()
            with veclite.open(path2, dim=dim) as db3:
                db3.add(x[:wal_n], keys=None if mode == "row_ids" else keys[:wal_n])
            def recover():
                with veclite.open(path2) as d4:
                    assert len(d4) == wal_n
            t_recover = median_time(recover, 3)
            row = {"n": n, "dim": dim, "mode": mode, "add_rows_per_s": n / t_add, "open_checkpointed_s": t_open,
                   "recover_wal_rows": wal_n, "recover_wal_s": t_recover, "contains_us": t_contains / len(probe) * 1e6,
                   "search_32_ms_per_q": t_search / 32 * 1e3}
            rows.append(row)
            print(row, flush=True)
    with open(os.path.join(ROOT, "results", "keys_bench.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
