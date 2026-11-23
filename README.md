# veclite

An embeddable, file-based vector database in the spirit of SQLite: exact nearest-neighbour search with metadata filters, written in Mojo and exposed as a C library (`libveclite.so`) with Python bindings.

- **Exact search.** Recall is always 1.0. There is no index to build or tune, and filters never cost recall.
- **Batched and filtered.** Each call takes many queries, each with its own tag filter. Rows that don't match are skipped, not scored.
- **Crash-safe single file + WAL.** SQLite-style write-ahead log with snapshot readers, checkpoints, and recovery from torn writes.
- **float32 or float16 storage**, chosen per collection.
- **Verified.** The algorithms are modelled in Quint and proved in Lean. The code is checked against those models with crash-simulation tests and deliberately broken builds (see [docs/DESIGN.md](docs/DESIGN.md)).

## Why

Most vector stores use approximate indexes (HNSW, IVF) that trade recall for speed, and lose recall badly under selective filters. For embeddings of 1536 dimensions and up, a well-tuned exact scan is competitive:

![Exact search vs FAISS](results/graphs/kernel_batch_dim_selectivity.png)

*100k rows, k = 10. veclite (solid) vs FAISS flat (dashed), by batch size and filter selectivity. veclite is faster for every filtered query and for unfiltered batches up to about 32. FAISS (BLAS) wins large unfiltered batches.*

The scan is bound by memory bandwidth, so float16 storage halves both the file and the search time once a collection no longer fits in RAM:

![float16 vs float32 from SSD](results/graphs/ssd_f16_vs_f32.png)

## Use it from Python (Linux)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install mojo==1.2.0 max==26.7.0 numpy
scripts/build.sh                 # -> build/libveclite.so
export PYTHONPATH=$PWD/python
```

```python
import numpy as np
import veclite

with veclite.open("docs.db", dim=1536, storage="f16") as db:
    ids = db.add(embeddings, tags=tenant_ids)        # durable on return
    db.delete(ids[0])

with veclite.open("docs.db") as db:
    hits, distances = db.search(query, k=10, tag=42)  # exact top-10 for tenant 42
    batch = db.search(queries, k=10)                  # one call, many queries
```

[`examples/quickstart.py`](examples/quickstart.py) is a runnable version, executed by the test suite. `libveclite.so` links to the Mojo runtime in the environment it was built in, so load it from that same environment.

## Platforms and tests

- **Linux** (arm64 and x86-64; glibc ≥ 2.34) and **macOS** (Apple silicon). The full suite passes on all three. x86-64 is tested under emulation in Docker. On macOS there are optional Apple GPU kernels; Linux builds are CPU-only.
- `scripts/check.sh` runs the C and Python test suites and checks that every deliberately broken build is caught.
- `scripts/check-linux.sh` runs the same suite in an Ubuntu 24.04 container.
- `scripts/check-formal.sh` runs the Quint models and Lean proofs.

## More

- [docs/DESIGN.md](docs/DESIGN.md): file format, WAL and recovery, formal property tables, GPU, float16/int8 findings, full benchmarks.
- `include/veclite.h`, `include/veclite_store.h`: the C API.
