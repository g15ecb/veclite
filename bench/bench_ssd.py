import ctypes
import fcntl
import json
import mmap
import statistics
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import F, I64, LIB, U8, check, ptr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.environ.get("VECLITE_SSD_DIR", os.path.join(ROOT, "scratch", "ssd"))
F_NOCACHE = 48
CHUNK_BYTES = 64 << 20
K = 10


def nocache_fd(path, flags):
    fd = os.open(path, flags, 0o644)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    return fd


LIBC = ctypes.CDLL("libc.dylib")
LIBC.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
LIBC.mincore.restype = ctypes.c_int


def resident_fraction(path):
    size = os.path.getsize(path)
    pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
    vec = ctypes.create_string_buffer(pages)
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), size, access=mmap.ACCESS_COPY)
        try:
            anchor = ctypes.c_char.from_buffer(mm)
            rc = LIBC.mincore(ctypes.addressof(anchor), size, vec)
            del anchor
            if rc != 0:
                raise OSError("mincore failed")
        finally:
            mm.close()
    return sum(b & 1 for b in vec.raw) / pages


def write_files(n, dim, seed):
    os.makedirs(DIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    rank = max(16, dim // 16)
    basis = rng.standard_normal((rank, dim), dtype=np.float32)
    paths = {s: os.path.join(DIR, f"d{dim}.{s}") for s in ("f32", "f16")}
    fds = {s: nocache_fd(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC) for s, p in paths.items()}
    sample = None
    try:
        for lo in range(0, n, 65536):
            m = min(65536, n - lo)
            x = rng.standard_normal((m, rank), dtype=np.float32) @ basis + 0.3 * rng.standard_normal((m, dim), dtype=np.float32)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            if sample is None:
                sample = x[:64].copy()
            os.write(fds["f32"], x.astype(np.float32).tobytes())
            os.write(fds["f16"], x.astype(np.float16).tobytes())
    finally:
        for fd in fds.values():
            os.fsync(fd)
            os.close(fd)
    return paths, sample


def search_chunk(buf, rows, dim, storage, qs, k, live):
    nq = len(qs)
    ids = np.empty((nq, k), dtype=np.int64)
    sc = np.empty((nq, k), dtype=np.float32)
    cnt = np.zeros(nq, dtype=np.int64)
    check(LIB.veclite_search_ex(buf.ctypes.data, 0 if storage == "f32" else 1, rows, dim, None, ptr(live, U8), ptr(qs, F), nq,
                            None, None, k, 2, 16, ptr(ids, I64), ptr(sc, F), ptr(cnt, I64)), "veclite_search_ex")
    return ids, sc, cnt


def merge(acc, ids, sc, cnt, base):
    for q in range(len(acc)):
        a_ids, a_sc = acc[q]
        n_ids = np.concatenate([a_ids, ids[q, : cnt[q]] + base])
        n_sc = np.concatenate([a_sc, sc[q, : cnt[q]]])
        order = np.lexsort((n_ids, n_sc))[:K]
        acc[q] = (n_ids[order], n_sc[order])


def median_run(fn, reps=3):
    runs = [fn() for _ in range(reps)]
    return runs[0][0], statistics.median(r[1] for r in runs)


def stream_search(path, n, dim, storage, qs, overlap=True):
    esz = 4 if storage == "f32" else 2
    rows_per = max(1024, CHUNK_BYTES // (dim * esz) // 1024 * 1024)
    if (rows_per * dim * esz) % mmap.PAGESIZE or dim % 8:
        raise ValueError("chunk reads must be page aligned for F_NOCACHE to bypass the cache")
    dt = np.float32 if storage == "f32" else np.float16
    bufs = [np.empty((rows_per, dim), dtype=dt) for _ in range(2)]
    live = np.ones(rows_per, dtype=np.uint8)
    acc = [(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)) for _ in qs]
    fd = nocache_fd(path, os.O_RDONLY)
    starts = list(range(0, n, rows_per))
    got = {}

    def read(i):
        lo = starts[i]
        rows = min(rows_per, n - lo)
        view = memoryview(bufs[i % 2]).cast("B")[: rows * dim * esz]
        done = 0
        while done < len(view):
            r = os.preadv(fd, [view[done:]], lo * dim * esz + done)
            if r <= 0:
                raise IOError(f"short read at row {lo}")
            done += r
        got[i] = rows

    t0 = time.perf_counter()
    t_read = 0.0
    try:
        read(0)
        for i in range(len(starts)):
            th = None
            if i + 1 < len(starts):
                if overlap:
                    th = threading.Thread(target=read, args=(i + 1,))
                    th.start()
            ids, sc, cnt = search_chunk(bufs[i % 2], got[i], dim, storage, qs, K, live)
            merge(acc, ids, sc, cnt, starts[i])
            if i + 1 < len(starts):
                if th is not None:
                    th.join()
                else:
                    r0 = time.perf_counter()
                    read(i + 1)
                    t_read += time.perf_counter() - r0
    finally:
        os.close(fd)
    return acc, time.perf_counter() - t0


def read_only(path):
    fd = nocache_fd(path, os.O_RDONLY)
    buf = bytearray(CHUNK_BYTES)
    t0 = time.perf_counter()
    total = 0
    try:
        while True:
            r = os.readv(fd, [buf])
            if r <= 0:
                break
            total += r
    finally:
        os.close(fd)
    dt = time.perf_counter() - t0
    return total / dt / 1e9


def load_nocache(path, n, dim, dt):
    data = np.empty((n, dim), dtype=dt)
    view = memoryview(data).cast("B")
    fd = nocache_fd(path, os.O_RDONLY)
    try:
        done = 0
        while done < len(view):
            r = os.preadv(fd, [view[done : done + CHUNK_BYTES]], done)
            if r <= 0:
                raise IOError("short read")
            done += r
    finally:
        os.close(fd)
    return data


def warm_search(path, n, dim, storage, qs):
    dt = np.float32 if storage == "f32" else np.float16
    data = load_nocache(path, n, dim, dt)
    live = np.ones(n, dtype=np.uint8)
    search_chunk(data, n, dim, storage, qs, K, live)
    t0 = time.perf_counter()
    ids, sc, cnt = search_chunk(data, n, dim, storage, qs, K, live)
    dtm = time.perf_counter() - t0
    return [(ids[q, : cnt[q]], sc[q, : cnt[q]]) for q in range(len(qs))], dtm


def main():
    gb = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    dims = [int(d) for d in (sys.argv[2] if len(sys.argv) > 2 else "1536,3072").split(",")]
    rows = []
    for dim in dims:
        n = int(gb * 1e9 / (dim * 4))
        paths, sample = write_files(n, dim, dim)
        qs = np.ascontiguousarray(sample[:32] + 0.05 * np.random.default_rng(1).standard_normal(sample[:32].shape, dtype=np.float32), dtype=np.float32)
        try:
            time.sleep(10)
            for storage in ("f32", "f16"):
                path = paths[storage]
                size_gb = os.path.getsize(path) / 1e9
                read_only(path)
                r1 = statistics.median(read_only(path) for _ in range(3))
                for b in (1, 8, 32):
                    res_before = resident_fraction(path)
                    cold, t_cold = median_run(lambda: stream_search(path, n, dim, storage, qs[:b]))
                    res_mid = resident_fraction(path)
                    cold_serial, t_serial = median_run(lambda: stream_search(path, n, dim, storage, qs[:b], overlap=False))
                    res_after = max(res_mid, resident_fraction(path))
                    r3 = read_only(path)
                    warm, t_warm = warm_search(path, n, dim, storage, qs[:b])
                    assert res_before < 0.02 and res_after < 0.02, f"file not cold: resident {res_before:.3f} -> {res_after:.3f}"
                    for a, w, s in zip(cold, warm, cold_serial):
                        assert np.array_equal(a[0], w[0]) and np.array_equal(a[1], w[1]), "streamed result != in-memory result"
                        assert np.array_equal(s[0], w[0]), "serial streamed result != in-memory result"
                    rows.append({
                        "dim": dim, "n": n, "storage": storage, "file_gb": size_gb, "batch": b,
                        "ssd_read_gb_s": r1, "ssd_read_gb_s_after_runs": r3, "resident_before": res_before, "resident_after": res_after,
                        "cold_ms_per_q": t_cold * 1e3 / b, "cold_serial_ms_per_q": t_serial * 1e3 / b,
                        "warm_ms_per_q": t_warm * 1e3 / b, "cold_effective_gb_s": size_gb / t_cold,
                    })
                    print(rows[-1], flush=True)
        finally:
            for p in paths.values():
                os.remove(p)
    with open(os.path.join(ROOT, "results", "ssd_bench.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
