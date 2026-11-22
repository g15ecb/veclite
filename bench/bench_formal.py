import json
import os
import re
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORMAL = os.path.join(ROOT, "formal")
FLAGS = "ACK_BEFORE_SYNC = false, CKPT_IGNORES_READERS = false, NO_CHAIN = false, RESTART_EARLY = false, SKIP_DATA_SYNC = false, READER_NO_SLOT_BOUND = false, UNSYNCED_RESTART_HDR = false, TRUNCATE_ON_RECOVERY = true"


def run(cmd):
    t0 = time.perf_counter()
    out = subprocess.run(cmd, cwd=FORMAL, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    text = out.stdout + out.stderr
    if "No violation found" not in text and "depends on axioms" not in text:
        raise RuntimeError(f"{cmd} failed:\n{text[-2000:]}")
    m = re.search(r"at (\d+) traces/second", text)
    return dt, int(m.group(1)) if m else None


def quint(file, main, samples, steps=30):
    return run(["quint", "run", file, "--main", main, "--invariant", "inv", "--max-samples", str(samples), "--max-steps", str(steps), "--seed", "1"])


def main():
    res = {"wal_samples": [], "wal_size": [], "topk_size": [], "lean": []}
    for samples in (1000, 5000, 20000, 50000):
        dt, tps = quint("wal.qnt", "base", samples)
        res["wal_samples"].append({"samples": samples, "seconds": dt, "traces_per_s": tps})
        print(res["wal_samples"][-1], flush=True)
    gen = os.path.join(FORMAL, "bench_instances.qnt")
    mods = []
    for slots in (3, 4, 5, 6, 7):
        for nids in (2, 3, 4):
            mods.append((f"walS{slots}N{nids}", f"import wal(NIDS = {nids}, MAXTXN = {nids + 1}, SLOTS = {slots}, NREADERS = 2, {FLAGS}).* from \"./wal\""))
    for n in (4, 6, 8, 10):
        mods.append((f"topkN{n}", f"import topk(N = {n}, K = 3, NQ = 2, NCHUNKS = 3, MAXSCORE = 2, NTAGS = 2, TIE_BREAK = true, PRE_FILTER = true, LOCAL_K_DELTA = 0, QUERY_MIX = false).* from \"./topk\""))
    with open(gen, "w") as f:
        f.write("\n\n".join(f"module {name} {{\n  {imp}\n}}" for name, imp in mods) + "\n")
    try:
        for name, _ in mods:
            dt, tps = quint("bench_instances.qnt", name, 5000)
            key = "wal_size" if name.startswith("wal") else "topk_size"
            nums = [int(x) for x in re.findall(r"\d+", name)]
            res[key].append({"instance": name, "params": nums, "seconds": dt, "traces_per_s": tps})
            print(res[key][-1], flush=True)
    finally:
        os.remove(gen)
    for f in ("TopK", "Wal"):
        dt, _ = run(["lean", os.path.join("lean", f + ".lean")])
        res["lean"].append({"file": f, "seconds": dt})
        print(res["lean"][-1], flush=True)
    with open(os.path.join(ROOT, "results", "formal_bench.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
