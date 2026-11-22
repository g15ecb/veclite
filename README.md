# vecdb spike: SQLite-style file-based vector search in Mojo

Throwaway spike. It checks whether Mojo can be the engine for an embeddable, single-file vector database with exact search and metadata filtering.

## Decisions so far

1. **Storage:** our own single-file, page-based format with a write-ahead log. Not a SQLite extension.
2. **Index:** exact (brute-force) search only. See [Index decision](#index-decision).
3. **Filters:** equality on one metadata value per row (`tag == value`). Rows that are deleted are skipped like a filter miss.

## Index decision

- **Chosen: exact search.** Every query scans every eligible row. Results are always correct (recall 1.0), there's nothing to build or tune, and filters can't reduce recall.
- **Why not HNSW:** it's a graph walk, so it doesn't vectorise or run well on GPUs. Strict filters also break its recall, because the walk can't reach the matching rows.
- **Later, if needed: IVF (clustered scan).** It's still a scan, so the same kernel and filter path apply. Add it only when exact search misses a latency target (see results: about 4 ms per query at 1M x 128 on an M2 Pro).
- **Before IVF, try smaller vectors.** The scan is limited by memory bandwidth (about 120 GB/s measured), so storing float16 or int8 instead of float32 would cut time by up to 2x or 4x with no index at all.

## Layout

- `formal/topk.qnt`, `formal/check.sh`: Quint model of chunked, filtered top-k, with three mutant controls.
- `formal/lean/TopK.lean`: Lean 4 proof that the algorithm is correct for all sizes.
- `src/vdb.mojo`: the Mojo library (C ABI).
- `include/vdb.h`: C header.
- `tests/c_host.c`: C test program. It checks every error code, then runs 300 random trials against the spec.
- `bench/bench.py`: Python (ctypes) conformance checks and benchmarks against NumPy and FAISS.
- `check_all.sh`: runs everything above, in order.
- `results/`: benchmark JSON output.

## What the code must respect (model to code mapping)

| Model | Code |
|---|---|
| Lean `TotalOrd` order on (score, id) | `cand_le`: score, then row id |
| Lean `TotalOrd` needs a total order | Non-finite scores are rejected (`E_SCORE_NONFINITE`), so NaN can never reach the order. `vdb_validate_rows` rejects bad rows at ingest. Row ids are unique by construction (the row index). |
| Lean `ins` + `take k` (`bstep`) | `bstep`: insert before the first element it is `<=` to, cap at k |
| Lean `scanChunk` (filter, then bounded fold in row order) | `scan_chunk`: rows `lo..hi` in order, eligibility check before insert |
| Quint `chunkLo` / `chunkHi` | `c * n // nc`, `(c + 1) * n // nc` |
| Lean `mergeLocals`: any merge order is correct (`topk_correct` takes any permutation) | Merge runs in chunk order after `parallelize` |
| Quint invariants `resultIsExactTopK`, `resultSortedUnique`, `resultSizeIsMinKMatches`, `resultOnlyLiveAndMatching` | Asserted in `bench.py::check_conformance`. `c_host.c` checks exact equality with the spec. |
| Quint mutants (no tie-break, post-filter, local k-1) | Same mutants built with `mojo build -D VDB_MUTANT_*=true`. The tests must fail on each one. |

## Spikes

| # | Question | Verdict |
|---|---|---|
| S1 | Can Mojo build a C-ABI library that C and Python can call, with full input validation? | **VALIDATED** |
| S2a | Is a Mojo CPU exact-search kernel competitive with NumPy and FAISS? | **VALIDATED** |
| S2b | Can Mojo run the kernel on the Apple GPU? | **BLOCKED**: needs full Xcode |
| S3 | Filtered exact top-k is correct in every case (formal plus tests)? | **VALIDATED** |

### Results (M2 Pro, 12 cores, k = 10, one query, median ms)

| Dataset | Filter | Mojo | NumPy | FAISS flat |
|---|---|---|---|---|
| 1M x 128 | none | 4.2 | 15.6 | 8.9 |
| 1M x 128 | 10% of rows | 1.4 | 13.0 | 11.6 |
| 1M x 128 | 1% of rows | 0.20 | 11.5 | 1.8 |
| 200k x 768 | none | 5.1 | 12.1 | 11.1 |
| 200k x 768 | 10% of rows | 1.2 | 11.7 | 5.9 |
| 200k x 768 | 1% of rows | 0.16 | 11.4 | 0.51 |

- Recall against a float64 NumPy reference was 1.0 in every conformance case.
- Caveats: one query at a time. FAISS gets the filter through `IDSelectorBatch`, a hash lookup per row. The FAISS "none" row doesn't skip deleted rows. The number of chunks (8, 16 or 64) made little difference.

### What worked

- `mojo build --emit shared-lib` produces a 74 KB dylib with four exported C functions.
- The C test program passes 4,948 checks under ASan and UBSan, including every error code.
- All three code mutants are caught by the tests: no tie-break (1,962 failures), post-filter (47), local k-1 (35).
- An unfiltered scan reads about 120 GB/s, out of roughly 200 GB/s peak memory bandwidth.

### What didn't

- **Apple GPU:** Mojo detects the M2 Pro GPU, but compiling a kernel calls `xcrun metal`, which only ships with full Xcode. This Mac has the Command Line Tools only.
- **GPU payoff is limited here anyway.** A single unfiltered query is already limited by memory bandwidth, and the GPU shares the same unified memory. So one query can't get more than about 1.6x faster on the GPU. Batches of queries are where a GPU would help.

### Surprises

- Mojo 1.2 broke a lot from older docs:
  - The GPU and `parallelize` APIs moved to the `max` package; you need `pip install max` alongside `mojo`.
  - `def` no longer raises by default.
  - `Pointer` can't be null, so exported functions take `Int` addresses and check them for zero.
  - Closures need explicit capture lists (`{var}`).
- The dylib needs two Modular runtime dylibs, `libKGENCompilerRTShared` and `libAsyncRTMojoBindings`, through an absolute rpath into the venv. Shipping it needs bundled copies and an `@loader_path` rpath.

### Recommendation for the real build

- Keep exact search, with the per-chunk bounded top-k and the proven merge.
- Add float16 and int8 storage before considering GPU or IVF.
- Store metadata as columns (tag array plus deleted-row bitmap) beside the vector pages.
- At 1% selectivity, scanning rows to find matches costs more than the distance maths. A posting list per tag value, scanning only the matching rows, is the next filter step.
- Next formal work, before any code: a Quint model of the file format plus write-ahead log (one writer, many readers, crash at any write).

## Run

```
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install mojo==1.2.0 max==26.7.0 numpy faiss-cpu
./check_all.sh
python -I bench/bench.py 1000000 128
```
