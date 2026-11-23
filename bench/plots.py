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


def dtypes(rows):
    dims = sorted({r["dim"] for r in rows})
    style = {"f32": ("C0", "o"), "f16": ("C1", "s")}
    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 4.2), sharey=True)
    for ax, dim in zip(axes, dims):
        for storage, (color, marker) in style.items():
            for sel, ls in ((1.0, "-"), (0.1, "--")):
                sub = sorted((r for r in rows if r["dim"] == dim and r["storage"] == storage and r["selectivity"] == sel), key=lambda r: r["batch"])
                ax.plot([r["batch"] for r in sub], [r["ms_per_q"] for r in sub], ls, marker=marker, color=color,
                        label=f"{storage}, {sel:.0%} match")
        ax.set(xscale="log", yscale="log", xlabel="queries per batch", title=f"cosine, n={rows[0]['n']:,}, dim={dim}")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("ms per query (lower is better)")
    axes[-1].legend(fontsize=8)
    save(fig, "dtype_batch_dim.png")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for storage, (color, marker) in style.items():
        sub = sorted((r for r in rows if r["storage"] == storage and r["batch"] == 1 and r["selectivity"] == 1.0), key=lambda r: r["dim"])
        x = [r["dim"] for r in sub]
        base = {r["dim"]: r["ms_per_q"] for r in rows if r["storage"] == "f32" and r["batch"] == 1 and r["selectivity"] == 1.0}
        axes[0].plot(x, [base[r["dim"]] / r["ms_per_q"] for r in sub], "-" + marker, color=color, label=storage)
        axes[1].plot(x, [r["scan_mb"] for r in sub], "-" + marker, color=color, label=storage)
        axes[2].plot(x, [r["recall_vs_f32"] for r in sub], "-" + marker, color=color, label=f"{storage} recall@10")
    axes[0].set(xlabel="dim", ylabel="speed-up vs f32 (batch 1)", title="Single-query speed-up")
    axes[1].set(xlabel="dim", ylabel="MB scanned per query", yscale="log", title="Bytes scanned per query")
    axes[2].set(xlabel="dim", ylabel="recall@10 vs f32 exact", ylim=(0.99, 1.0005), title="f16 accuracy vs f32")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    save(fig, "dtype_tradeoffs.png")


def store_dtypes(rows):
    names = [r["storage"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].bar(names, [r["main_file_mb"] for r in rows], color=["C0", "C1"])
    axes[0].set(ylabel="MB", title=f"Main file size (n={rows[0]['n']:,}, dim={rows[0]['dim']})")
    axes[1].bar(names, [r["ingest_rows_per_s"] for r in rows], color=["C0", "C1"])
    axes[1].set(ylabel="rows / s", title="Durable ingest (1024 rows per commit)")
    w = 0.38
    xs = range(len(rows))
    axes[2].bar([x - w / 2 for x in xs], [r["search_ms_per_q_batch1"] for r in rows], w, label="batch 1")
    axes[2].bar([x + w / 2 for x in xs], [r["search_ms_per_q_batch32"] for r in rows], w, label="batch 32")
    axes[2].set_xticks(list(xs), names)
    axes[2].set(ylabel="ms per query", title="Search through snapshot reader")
    axes[2].legend()
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
    save(fig, "store_dtypes.png")


def ssd(rows):
    dims = sorted({r["dim"] for r in rows})
    fig, axes = plt.subplots(1, len(dims) + 1, figsize=(5 * (len(dims) + 1), 4.2))
    style = {"f32": "C0", "f16": "C1"}
    for ax, dim in zip(axes, dims):
        for storage, color in style.items():
            sub = sorted((r for r in rows if r["dim"] == dim and r["storage"] == storage), key=lambda r: r["batch"])
            b = [r["batch"] for r in sub]
            ax.plot(b, [r["cold_ms_per_q"] for r in sub], "-o", color=color, label=f"{storage}, from SSD (cache bypassed)")
            ax.plot(b, [r["warm_ms_per_q"] for r in sub], "--s", color=color, alpha=0.7, label=f"{storage}, in RAM")
        ax.set(xscale="log", yscale="log", xlabel="queries per batch", ylabel="ms per query",
               title=f"dim {dim}: {sub[0]['n']:,} rows, f32 file {sub[0]['file_gb'] * (1 if sub[0]['storage'] == 'f32' else 2):.1f} GB")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    ax = axes[-1]
    labels, ratios = [], []
    for dim in dims:
        for bsz in sorted({r["batch"] for r in rows}):
            f32 = next(r for r in rows if r["dim"] == dim and r["storage"] == "f32" and r["batch"] == bsz)
            f16 = next(r for r in rows if r["dim"] == dim and r["storage"] == "f16" and r["batch"] == bsz)
            labels.append(f"d{dim}\nb{bsz}")
            ratios.append(f32["cold_ms_per_q"] / f16["cold_ms_per_q"])
    ax.bar(labels, ratios, color="C1")
    ax.axhline(2, color="k", ls=":", label="2x = halved bytes")
    ax.set(ylabel="f16 speed-up over f32 from SSD", title="Out-of-RAM speed-up (higher is better)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, "ssd_f16_vs_f32.png")


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
        if r.get("store_dtype"):
            store_dtypes(r["store_dtype"])
    if (r := load("dtype_sweep.json")) is not None:
        dtypes(r)
    if (r := load("ssd_bench.json")) is not None:
        ssd(r)
    if (r := load("formal_bench.json")) is not None:
        formal(r)


if __name__ == "__main__":
    main()
