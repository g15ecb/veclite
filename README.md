# vecdb spike: SQLite-style file-based vector search in Mojo

Throwaway spike. It checks whether Mojo can be the engine for an embeddable, single-file vector database. The database does exact, batched search with metadata filters, one filter per query.

## Decisions

1. **Storage:** our own single-file, page-based format with a write-ahead log. Not a SQLite extension.
2. **Index:** exact search only. See [Index decision](#index-decision).
3. **Filters:** equality on one tag per row, set per query (`tag == value`, or no filter). Deleted rows are skipped as a filter miss.
4. **Batching is built in.** The API takes `nq` queries. One query is just `nq = 1`.
5. **Workload:** embeddings with 1536 dimensions or more.

## Index decision

- **Chosen: exact search.** Recall is always 1.0, there's nothing to build or tune, and filters can't reduce recall.
- **Why not HNSW:** it's a graph walk, so it doesn't vectorise or batch well. Strict filters also break its recall.
- **Later, if needed: IVF (clustered scan).** It reuses the same tiled kernel and filter path.
- **Before IVF, try smaller vectors.** A single query is limited by memory bandwidth, so float16, int8 or binary storage cuts its time directly.

## Embedding dimensions (research)

| Provider / model | Default dimensions | Can be truncated to (Matryoshka) |
|---|---|---|
| OpenAI text-embedding-3-small / -large | 1536 / 3072 | yes, via the `dimensions` parameter |
| Google gemini-embedding-001 / -2 | 3072 | recommends 768, 1536 or 3072 |
| Cohere embed-v5.0 / v4.0 | 2048 / 1536 | 256 to 2048 |
| Voyage 4 series | 1024 | 256, 512, 2048 |
| Qwen3-Embedding 0.6B / 4B / 8B | 1024 / 2560 / 4096 | 32 to 4096 |

Sources: the OpenAI, Gemini, Cohere and Voyage docs, and the Qwen3-Embedding-8B model card.

- Gemini's published MTEB scores barely change between 1536 and 768 dimensions (68.17 vs 67.99).
- Hugging Face reports that binary quantization with rescoring keeps about 96% of retrieval quality at 32x less memory (mxbai-embed-large, 1024 dimensions).

### Where dimension, batching and the GPU meet (measured on an M2 Pro)

- **Time tracks bytes read, not dimension.** 500k x 1536 and 250k x 3072 are the same 3.07 GB, and both took 24.7 ms per single query.
- **One query is limited by memory bandwidth.** That holds at every dimension: each float read is used once. A GPU sharing the same memory can't do better than about 1.6x on one query.
- **A batch reuses every row it reads across all its queries,** so the work becomes limited by arithmetic instead. That's where accelerators pay off:

| Measured, 250k x 3072, k = 10 | ms per query |
|---|---|
| Mojo, batch 1 (memory-bound, about 117 GB/s) | 26.3 |
| Mojo, batch 32 or more (NEON, about 480 GFLOP/s) | 3.1 to 3.5 |
| FAISS flat, batch 512 (BLAS) | 1.22 |
| NumPy matmul through Accelerate (AMX matrix unit), batch 512, about 2.2 TFLOP/s | 0.68 |

- Mojo can't use the M2's AMX matrix unit: it isn't exposed to Mojo. Only Apple's Accelerate library reaches it.
- So for large batches on this Mac the ranking is: GPU (needs Xcode; about 6.8 TFLOPS on the 19-core M2 Pro, a figure from memory) > Accelerate/AMX > Mojo NEON.
- **Next accelerator spike:** batched GPU scoring, compared with the Accelerate path.

## Formal properties

### Quint model, `formal/topk.qnt`

The model has batched queries, each with its own filter. Chunks finish in any order.

| Property | Enforces | Why |
|---|---|---|
| `resultIsExactTopK` | For every query, the result set equals the k smallest eligible (score, id) pairs | This is what exact search means. It catches lost candidates from per-chunk truncation and filters applied in the wrong place. |
| `resultSortedUnique` | Results are strictly ascending by (score, id) | Output must be the same whatever order threads finish in. The id tie-break makes the order total. |
| `resultSizeIsMinKMatches` | Result length = min(k, number of eligible rows) | No short results when enough rows match (the filter-after-top-k bug), and no padding. |
| `resultOnlyLiveAndMatching` | Every returned row is live and passes that query's own filter | Deleted rows never come back, and one query's filter never leaks into another's. |
| `localsBounded` | Each per-chunk, per-query buffer holds at most k entries | The code allocates exactly chunks x queries x k entries. |
| Witnesses: `witnessDone`, `witnessTruncated`, `witnessTiesInResult`, `witnessFiltersDiffer` | Each must be reachable | Shows the checks aren't vacuous: truncation, ties and different per-query filters really happen in the runs. |
| Mutants: `NoTieBreak`, `PostFilter`, `LocalKMinusOne`, `QueryMix` | Each must violate `inv` | Each property above catches a known bug class. |

### Lean proof, `formal/lean/TopK.lean`

Valid for all sizes. No `sorry`; the only axioms are `propext` and `Quot.sound`.

| Theorem | States | Why |
|---|---|---|
| `TotalOrd` (assumption) | Scores are totally ordered by (score, id) | Everything below depends on it. This is why the code rejects NaN and infinite values, and why row ids are unique. |
| `topk_correct` | For any chunking and any merge order, merging per-chunk bounded top-k lists equals "sort the filtered rows, take k" | Justifies per-chunk buffers of size k, and shows `parallelize` completion order doesn't matter. |
| `fused_fold` | One loop over rows that updates every query's state equals separate per-query loops | The core reason batching is correct. |
| `batch_correct` | A fused batch scan with per-query filters gives each query exactly its single-query top-k | Justifies reading each row once for the whole batch. |
| `topk_unique` | Any sorted ordering of the filtered rows has the same top-k | The specification has exactly one answer. |
| `topk_length`, `topk_matches`, `topk_sorted` | The specification's size, filter and order | Same properties as Quint's, now proved for every size. |
| Mutant (k−1 local buffer) | Must fail to prove | The proof actually depends on the buffer size. |

### How the code conforms

| Formal definition | Code (`src/vdb.mojo`) |
|---|---|
| `TotalOrd` on (score, id) | `cand_le`. Non-finite queries, rows and scores are rejected, and `vdb_validate_rows` checks rows at ingest. |
| `ins` + `take k` (`bstep`) | `bstep` |
| `guardedStep` / `batchScanChunk` | `scan_qtile` / `do_tile`: rows ascending within a chunk, each query's own buffer, eligibility checked before insert |
| `mergeLocals` | Per-query merge after `parallelize` |
| Quint `chunkLo` / `chunkHi` | `c * n // nc`, `(c + 1) * n // nc` |
| Quint invariants | Asserted in `bench.py::conformance` and `tests/c_host.c` |
| Mutants | `mojo build -D VDB_MUTANT_*=true`. `check_all.sh` requires the tests to fail on each one. |

The tests also check two things the formal models assume:

- Batched scores are bit-identical to single-query scores.
- A batched search gives the same result as searching each query alone.

Both hold because every multiply-add is an explicit `fma`, so the result doesn't depend on tile shape.

## Status

| # | Question | Verdict |
|---|---|---|
| S1 | Mojo C-ABI library with full validation | **VALIDATED** |
| S2a | Batched CPU exact search, competitive | **VALIDATED** (Mojo wins up to batch 32; Accelerate/AMX wins large batches) |
| S2b | Apple GPU through Mojo | **BLOCKED**: needs full Xcode (`xcrun metal`) |
| S3 | Filtered batched top-k, formally verified | **VALIDATED** |
| S4 | File format and write-ahead log (Quint first) | Next |

### Results, 250k x 3072 (3.07 GB), k = 10, ms per query

| Batch | Mojo | FAISS | Mojo, 10% filter | FAISS, 10% filter | Mojo, mixed per-query filters |
|---|---|---|---|---|---|
| 1 | 26.3 | 65.4 | 4.24 | 19.5 | 26.5 |
| 8 | 4.59 | 16.3 | 0.80 | 2.36 | 4.67 |
| 32 | 3.13 | 22.3 | 0.68 | 2.99 | 3.17 |
| 128 | 3.36 | 2.06 | 0.64 | 3.32 | 3.35 |
| 512 | 3.45 | 1.22 | 0.65 | 3.44 | 3.49 |

- Recall against a float64 reference was 1.0.
- The C tests run 167,565 checks under ASan and UBSan. All four code mutants are caught.
- A register-tile sweep (queries x rows: 1x4, 4x2, 4x4, 8x2, 8x3, 6x4, 2x8) picked 4x4, tied with 8x2. Chunk count from 16 to 192 made no difference.
- The mixed-filter batch has half its queries unfiltered, so it costs the same as an unfiltered scan.

### Surprises

- **Clamping L2 hid NaN.** `max(x, 0)` returned 0 for NaN, so the NaN-row test failed. The code now clamps only when `x < 0`.
- **The first batched version was 2x slower at batch 1.** Row norms were computed in a separate pass, which read all the data twice. Norms are now computed inside the tile.
- **Mojo 1.2 API changes:** see the `mojo-development` skill.

## Run

```
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install mojo==1.2.0 max==26.7.0 numpy faiss-cpu
./check_all.sh
python -I bench/bench.py 250000 3072 1,8,32,128,512
```
