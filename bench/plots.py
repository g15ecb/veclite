import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "graphs")


def load(name):
    path = os.path.join(RES, name)
    return json.load(open(path)) if os.path.exists(path) else None


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=130)
    plt.close(fig)
    print("wrote", name)


def kernel(rows):
    dims = sorted({r["dim"] for r in rows})
    sels = sorted({r["selectivity"] for r in rows}, reverse=True)
    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 4.2), sharey=True)
    for ax, dim in zip(axes, dims):
        for sel, color in zip(sels, ("C0", "C1", "C2")):
            sub = sorted((r for r in rows if r["dim"] == dim and r["selectivity"] == sel), key=lambda r: r["batch"])
            b = [r["batch"] for r in sub]
            ax.plot(b, [r["mojo_ms_per_q"] for r in sub], "-o", color=color, label=f"Mojo, {sel:.0%} match")
            ax.plot(b, [r["faiss_ms_per_q"] for r in sub], "--s", color=color, alpha=0.7, label=f"FAISS, {sel:.0%} match")
        ax.set(xscale="log", yscale="log", xlabel="queries per batch", title=f"n={sub[0]['n']:,}, dim={dim}")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("ms per query (lower is better)")
    axes[-1].legend(fontsize=8)
    save(fig, "kernel_batch_dim_selectivity.png")


def store(res):
    c = res.get("commit")
    if c:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for dim in sorted({r["dim"] for r in c}):
            sub = sorted((r for r in c if r["dim"] == dim), key=lambda r: r["rows_per_commit"])
            x = [r["rows_per_commit"] for r in sub]
            axes[0].plot(x, [r["rows_per_s"] for r in sub], "-o", label=f"dim {dim}")
            axes[1].plot(x, [r["ms_per_commit"] for r in sub], "-o", label=f"dim {dim}")
        axes[0].set(xscale="log", yscale="log", xlabel="rows per commit", ylabel="rows / s", title="Durable ingest throughput (F_FULLFSYNC)")
        axes[1].set(xscale="log", yscale="log", xlabel="rows per commit", ylabel="ms per commit", title="Commit latency")
        for ax in axes:
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
        save(fig, "store_commit.png")
    k = res.get("checkpoint_recovery")
    if k:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        x = [r["wal_mb"] for r in k]
        for key, label in (("recover_s", "open with WAL recovery"), ("checkpoint_s", "checkpoint"), ("open_after_ckpt_s", "open after checkpoint")):
            ax.plot(x, [r[key] for r in k], "-o", label=label)
        ax.set(xscale="log", yscale="log", xlabel="WAL size (MB, dim 1536)", ylabel="seconds", title="Recovery and checkpoint vs WAL size")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        save(fig, "store_checkpoint_recovery.png")
    t = res.get("search_tail")
    if t:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for b in sorted({r["batch"] for r in t}):
            sub = sorted((r for r in t if r["batch"] == b), key=lambda r: r["wal_fraction"])
            x = [r["wal_fraction"] for r in sub]
            ax.plot(x, [r["store_ms_per_q"] for r in sub], "-o", label=f"store, batch {b}")
            ax.plot(x, [r["kernel_ms_per_q"] for r in sub], "--", label=f"raw kernel, batch {b}")
        ax.set(yscale="log", xlabel="fraction of rows still in WAL (not checkpointed)", ylabel="ms per query", title=f"Search via snapshot reader (n={t[0]['n']:,}, dim={t[0]['dim']})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        save(fig, "store_search_wal_fraction.png")


def formal(res):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    s = res["wal_samples"]
    axes[0].plot([r["samples"] for r in s], [r["seconds"] for r in s], "-o")
    axes[0].set(xscale="log", yscale="log", xlabel="quint run samples (30 steps)", ylabel="seconds", title="WAL model: checking time vs samples")
    for nids in sorted({r["params"][1] for r in res["wal_size"]}):
        sub = sorted((r for r in res["wal_size"] if r["params"][1] == nids), key=lambda r: r["params"][0])
        axes[1].plot([r["params"][0] for r in sub], [r["traces_per_s"] for r in sub], "-o", label=f"{nids} row ids")
    axes[1].set(xlabel="WAL slots", ylabel="traces / s", title="WAL model: throughput vs instance size")
    axes[1].legend()
    tk = sorted(res["topk_size"], key=lambda r: r["params"][0])
    axes[2].plot([r["params"][0] for r in tk], [r["traces_per_s"] for r in tk], "-o", label="top-k model (2 queries, 3 chunks)")
    axes[2].set(xlabel="rows", ylabel="traces / s", title="Top-k model throughput; Lean: " + ", ".join(f"{r['file']} {r['seconds']:.1f}s" for r in res["lean"]))
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
    save(fig, "formal_checking.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    if (r := load("kernel_sweep.json")) is not None:
        kernel(r)
    if (r := load("store_bench.json")) is not None:
        store(r)
    if (r := load("formal_bench.json")) is not None:
        formal(r)


if __name__ == "__main__":
    main()
