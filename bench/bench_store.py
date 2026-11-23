import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import Table
from store_api import Store, remove_db

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.environ.get("VECLITE_BENCH_DIR", os.path.join(ROOT, "scratch", "benchdb"))
RNG = np.random.default_rng(3)


def fresh(name, dim, storage="f32"):
    os.makedirs(DIR, exist_ok=True)
    path = os.path.join(DIR, name)
    remove_db(path)
    return path, Store(path, dim, storage=storage)


def med(fn, reps=5):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def load(st, n, batch, tags_mod=10):
    vecs = RNG.standard_normal((n, st.dim), dtype=np.float32)
    for lo in range(0, n, batch):
        st.append_rows(vecs[lo : lo + batch], np.arange(lo, min(n, lo + batch)) % tags_mod)
        st.commit()
    return vecs


def bench_commit():
    rows = []
    for dim in (768, 1536, 3072):
        for batch in (1, 4, 16, 64, 256, 1024):
            n = min(4096, max(256, batch * 16))
            path, st = fresh("commit", dim)
            vecs = RNG.standard_normal((n, dim), dtype=np.float32)
            t0 = time.perf_counter()
            for lo in range(0, n, batch):
                st.append_rows(vecs[lo : lo + batch], np.zeros(min(batch, n - lo), dtype=np.int64))
                st.commit()
            dt = time.perf_counter() - t0
            st.close()
            remove_db(path)
            rows.append({"dim": dim, "rows_per_commit": batch, "rows_per_s": n / dt, "ms_per_commit": dt * 1e3 / (n / batch), "mb_per_s": n * dim * 4 / dt / 1e6})
            print("commit", rows[-1], flush=True)
    return rows


def bench_checkpoint_recovery(dim=1536):
    rows = []
    for n in (1024, 4096, 16384):
        path, st = fresh("ckpt", dim)
        load(st, n, 1024)
        st.close()
        t0 = time.perf_counter()
        st = Store(path, dim)
        t_open = time.perf_counter() - t0
        t0 = time.perf_counter()
        st.checkpoint()
        t_ckpt = time.perf_counter() - t0
        info = st.info()
        assert info["hdr_txn"] == info["ntx"] and info["wal_frames"] == 0, info
        st.close()
        t0 = time.perf_counter()
        st = Store(path, dim)
        t_open_clean = time.perf_counter() - t0
        st.close()
        remove_db(path)
        mb = n * dim * 4 / 1e6
        rows.append({"dim": dim, "wal_rows": n, "wal_mb": mb, "recover_s": t_open, "checkpoint_s": t_ckpt, "open_after_ckpt_s": t_open_clean, "checkpoint_mb_per_s": mb / t_ckpt, "recover_mb_per_s": mb / t_open})
        print("ckpt", rows[-1], flush=True)
    return rows


def bench_search_tail(n=50000, dim=1536, k=10):
    rows = []
    qs = RNG.standard_normal((32, dim), dtype=np.float32)
    for frac in (0.0, 0.25, 0.5, 1.0):
        path, st = fresh("tail", dim)
        main_n = int(n * (1 - frac))
        vecs = load(st, main_n, 4096) if main_n else np.empty((0, dim), dtype=np.float32)
        if main_n:
            st.checkpoint()
        vecs = np.vstack([vecs, load(st, n - main_n, 4096)])
        info = st.info()
        assert info["main_rows"] == main_n and info["rows"] == n, info
        ref = Table(vecs, np.arange(n, dtype=np.int64) % 10, np.ones(n, dtype=np.uint8))
        for b in (1, 32):
            got = st.search(qs[:b], k)
            want = ref.search(qs[:b], k)
            assert all(np.array_equal(g[0], w[0]) for g, w in zip(got, want)), "store search != kernel on same rows"
            t_store = med(lambda: st.search(qs[:b], k)) / b
            t_kernel = med(lambda: ref.search(qs[:b], k)) / b
            rows.append({"n": n, "dim": dim, "wal_fraction": frac, "batch": b, "store_ms_per_q": t_store * 1e3, "kernel_ms_per_q": t_kernel * 1e3})
            print("tail", rows[-1], flush=True)
        st.close()
        remove_db(path)
    return rows


def bench_dtypes(n=50000, dim=1536, k=10):
    rows = []
    vecs = RNG.standard_normal((n, dim), dtype=np.float32)
    qs = RNG.standard_normal((32, dim), dtype=np.float32)
    ref = Table(vecs, np.arange(n, dtype=np.int64) % 10, np.ones(n, dtype=np.uint8))
    truth = ref.search(qs, k)
    for storage in ("f32", "f16"):
        path, st = fresh("dtype", dim, storage)
        t0 = time.perf_counter()
        for lo in range(0, n, 1024):
            st.append_rows(vecs[lo : lo + 1024], np.arange(lo, min(n, lo + 1024)) % 10)
            st.commit()
        t_ingest = time.perf_counter() - t0
        t0 = time.perf_counter()
        st.checkpoint()
        t_ckpt = time.perf_counter() - t0
        got = st.search(qs, k)
        recall = float(np.mean([len(np.intersect1d(a[0], b[0])) / k for a, b in zip(got, truth)]))
        row = {"storage": storage, "n": n, "dim": dim, "ingest_rows_per_s": n / t_ingest, "checkpoint_s": t_ckpt,
               "main_file_mb": os.path.getsize(path) / 1e6, "recall_vs_f32": recall}
        for b in (1, 32):
            row[f"search_ms_per_q_batch{b}"] = med(lambda: st.search(qs[:b], k)) / b * 1e3
        rows.append(row)
        print("dtype", row, flush=True)
        st.close()
        remove_db(path)
    return rows


def main():
    which = sys.argv[1:] or ["commit", "ckpt", "tail", "dtype"]
    out = {}
    if "commit" in which:
        out["commit"] = bench_commit()
    if "ckpt" in which:
        out["checkpoint_recovery"] = bench_checkpoint_recovery()
    if "tail" in which:
        out["search_tail"] = bench_search_tail()
    if "dtype" in which:
        out["store_dtype"] = bench_dtypes()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    path = os.path.join(ROOT, "results", "store_bench.json")
    merged = json.load(open(path)) if os.path.exists(path) else {}
    merged.update(out)
    with open(path, "w") as f:
        json.dump(merged, f, indent=1)


if __name__ == "__main__":
    main()
