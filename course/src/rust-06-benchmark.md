# Benchmark Five Indexes on SIFT1M

> **Day 6**
>
> Complete [Compress IVFFlat with Product Quantization](./rust-07-ivfpq.md) first. Then compare Flat, IVFFlat, NSW,
> HNSW, and IVF-PQ in the same release-mode benchmark, using Euclidean search with `k = 100`.

Finish and test the benchmark before acquiring SIFT1M. Start with smoke mode to check the whole workflow on a smaller
base. Then use sampled mode to compare the five indexes at one million base vectors. The optional full run uses
every query in the corpus.

## Start from the Completed Indexes

Use the checkout where you completed Days 1–5. If you are starting from a fresh clone, finish those exercises first
and verify the earlier work from the repository root.

```sh
cargo xtask test-through day_05
```

Open `vector-db-starter/core/examples/recall.rs`. The benchmark support crate parses the command line, validates the
SIFT files, pairs selected queries with their truth rows, runs warm-up and timed searches, and supplies the quality
and percentile helpers. The example fixes the five configurations, output format, result validation, and IVF-PQ
byte accounting.

Start with the support crate's corpus-free checks.

```sh
cargo test -p vector-db-from-scratch-benchmark-support
```

Four `todo!()` calls remain in the benchmark example. The focused commands below select its two checkpoints
directly. The broader `cargo xtask` commands also build the DataFusion workspace for the course-wide checks.

## Keep the Five Configurations Fixed

| Index | Report configuration |
| --- | --- |
| Flat | `exact` |
| IVFFlat | `partitions=32,probes=6,iterations=12,seed=7` |
| NSW | `max_connections=12,ef_construction=64,ef_search_configured=40,ef_search_effective=100` |
| HNSW | `max_connections=12,ef_construction=64,ef_search_configured=40,ef_search_effective=100,max_level=12,seed=7` |
| IVF-PQ | `partitions=32,probes=6,iterations=12,subquantizers=4,codebook_size=16,rerank=100,seed=7` |

For NSW and HNSW, `search(query, 100)` uses `max(ef_search, k)`. The stored search width is 40 and the effective
width is 100.

## Checkpoint 1. Construct the Remaining Indexes

Run this command with your earlier work in place. Before editing Day 6, expect the test to stop at
`Day 6: build NSW with the benchmark configuration`.

```sh
cargo test -p vector-db-from-scratch-core-starter --example recall day_06::checkpoint_1
```

Replace the TODOs in `build_nsw`, `build_hnsw`, and `build_ivf_pq` with their matching constructors. Pass through
the supplied dataset, metric, and configuration, and return the constructor's result. The example prepares the
cloned dataset and configuration before timing each constructor. Keep that boundary so `build_s` measures the
constructor alone.

Retry the command above. It checks that each constructor preserves the given configuration and propagates errors.
A seed makes builds of your implementation repeatable. The tests allow valid implementations with different RNG
trajectories.

## Checkpoint 2. Select p50, p95, and p99

Run the second checkpoint. With the constructors filled in, expect the percentile tests to reach
`Day 6: select p50, p95, and p99 with the supplied nearest-rank helper`.

```sh
cargo test -p vector-db-from-scratch-core-starter --example recall day_06::checkpoint_2
```

Return the three percentiles in that order using `percentile` on the supplied sorted, nonempty duration slice.
For percentage `p` and `n` samples, nearest rank selects the zero-based position `ceil(p / 100 * n) - 1`, clamped
to the last sample.

Retry the second checkpoint. It checks percentile selection, truth selection, result validation, quality and
returned counts, report order, and IVF-PQ accounting.

## Finish the Corpus-Free Checks

The focused day and the cumulative gate compile the benchmark. The external SIFT1M tests are ignored by default.

```sh
cargo xtask test day_06
cargo xtask test-through day_06
```

## Acquire and Validate SIFT1M

Obtain SIFT1M and follow its published terms. Extract the corpus before running the benchmark.
[TexMex ANN corpus](http://corpus-texmex.irisa.fr/)

| File directly in the corpus directory | Records | Width | Exact bytes |
| --- | ---: | ---: | ---: |
| `sift_base.fvecs` | 1,000,000 | 128 `f32` values | 516,000,000 |
| `sift_query.fvecs` | 10,000 | 128 `f32` values | 5,160,000 |
| `sift_groundtruth.ivecs` | 10,000 | 100 `i32` row IDs | 4,040,000 |

The three extracted files take 525,200,000 bytes. Allow additional disk space for the archive, extraction, and
Cargo build products. For a full-base run, plan for at least 8 GiB of available memory and roughly 1 GiB of
working disk space beyond the corpus and build products. Runtime depends on the machine. All five indexes are
built sequentially and kept alive together.

Every mode scans all three files. The loader validates exact file sizes and record counts, little-endian dimension
headers, finite vectors, and in-range, unique ground-truth IDs. The executable expects one directory and an optional
mode flag. Status 2 indicates invalid arguments. Status 1 indicates an input or index error.

```text
usage: recall [--smoke|--sampled] <sift1m-dir>
```

If the loader reports a missing file, check which directory directly contains the three SIFT files. Archives can
extract into a nested directory. Retry the same command with the extracted directory as its final argument.

## Run Smoke, Then Sampled SIFT1M

Start with the smoke command once the tests pass.

```sh
cargo run --release -p vector-db-from-scratch-core-starter --example recall -- --smoke /absolute/path/to/sift1M
```

Then run the fixed sample for a full-base comparison on your machine.

```sh
cargo run --release -p vector-db-from-scratch-core-starter --example recall -- --sampled /absolute/path/to/sift1M
```

The optional full run uses the directory argument alone.

```sh
cargo run --release -p vector-db-from-scratch-core-starter --example recall -- /absolute/path/to/sift1M
```

| Mode | Base rows | Query rows | Exact top-100 truth |
| --- | ---: | ---: | --- |
| `sift1m-smoke` | First 10,000 | First 100 | Recomputed by Flat over the selected 10,000 rows |
| `sift1m-sampled` | All 1,000,000 | Fixed 1,000 | Supplied rows for the same query ordinals |
| `sift1m-full` | All 1,000,000 | All 10,000 | All supplied rows in file order |

Smoke checks integration at a smaller base size. Sampled mode runs a tenth as many timed queries per index as full
mode and still builds the same full-size indexes. The full run's `parity=bustub-sift1m` label identifies the
shared corpus, Euclidean ordering, `k`, and quality definitions. Implementation and machine details still matter
when comparing results. Smoke and sampled runs use `parity=non-parity`.

To identify the sample, the harness computes a SplitMix64 key from each zero-based query ordinal plus seed
`0x5eed`, takes the 1,000 smallest keys, and sorts the selected ordinals back into file order. It uses that same
list for the query vectors and their supplied truth rows. Every index sees the queries in that order. The workload
line prints `query_selection=splitmix64-key-rank-v1(seed=0x5eed)` and the sample fingerprint
`query_ordinals_fnv1a64=3daec467e7c9a97a`. The fingerprint hashes the ordered query ordinals as little-endian
64-bit integers with FNV-1a.

For a reference run, use the same command and arguments with package `vector-db-from-scratch-core`. Each invocation
performs its own loading and construction.

### Optional single-index smoke test

The external smoke tests check deterministic behavior and broad quality floors. Select one test if you need to
diagnose one index. Each selected test reads the corpus and may build its index several times.

```sh
SIFT1M_DIR=/absolute/path/to/sift1M \
  cargo test -p vector-db-from-scratch-core-starter --test sift_smoke \
  day_06::checkpoint_2::sift_ivf_pq_smoke -- --ignored --exact
```

The other test names end in `sift_flat_smoke`, `sift_ivf_flat_smoke`, `sift_nsw_smoke`, and `sift_hnsw_smoke`
under `day_06::checkpoint_2`. Set `SIFT1M_DIR` for the tests and pass the directory argument to the benchmark.

## Read the Report and Stage Timings

Progress goes to stderr. A noninteractive stream prints 0%, 25%, 50%, 75%, and 100% for counted stages. A terminal
redraws those milestones in place. File loading counts physical rows in all three modes. Smoke truth recomputation
counts the 100 selected queries. Each constructor prints start and completion messages.

Stdout is reserved for the complete validated report. An input, index, or result-validation error can leave progress
on stderr and still produce an empty stdout. Append `> report.txt` to save the report while watching progress.

The report starts with the workload and timing context. It gives the mode, row and query counts, dimension, metric,
`k`, truth source, query selection and fingerprint, warm-up count, timed passes, concurrency, and cache regime.
The `stages` line shows how the time was spent.

| Field | What it covers |
| --- | --- |
| `load_validate_s` | SIFT file loading and validation, query selection, and in-memory dataset preparation |
| `truth_s` | Selecting supplied truth or recomputing smoke truth with Flat |
| `build_s` | Each of the five constructors, shown in its index row |
| `warmup_s` | Warm-up phase wall time |
| `timed_search_wall_s` | Timed pass wall time across all five indexes, including loop, result collection, and progress |
| `validate_account_s` | Reading IVF-PQ byte counts, validating results, and preparing report rows |
| `total_s` | Elapsed time from starting the benchmark to preparing the validated report, before stdout |

Phase clocks can include progress and loop overhead. Each index's `search_s` is a sum of individual search calls within
the timed pass. Read Cargo build time and report writing separately from the benchmark total.

Warm-up runs the first `min(20, query_count)` queries through all five indexes. The timed pass uses every selected
query exactly once per index. Let `query_ordinal` be the zero-based position in the selected query order. The
starting index rotates at each position.

```text
(query_ordinal + offset) % 5
```

Only `search(query, 100)` is inside each latency sample. Result validation, quality calculation, latency sorting, and
formatting happen afterward. `serial_qps` divides the query count by the sum of that index's latency samples, with
one search call at a time. Use p50 and p95 to describe the distribution of those samples. Treat p99 as exploratory.
In sampled mode the slowest 1% is only about ten queries.

The index rows appear in `flat`, `ivf_flat`, `nsw`, `hnsw`, `ivf_pq` order.

```text
{name}: config={stable-config}, build_s={:.3}, search_s={:.3}, serial_qps={:.1}, exact_nn_hit@1={:.4}, exact_nn_hit@10={:.4}, exact_nn_hit@100={:.4}, recall@100={:.4}, returned_min={n}, returned_avg={:.1}, returned_max={n}, p50_ms={:.3}, p95_ms={:.3}, p99_ms={:.3}
```

## Interpret Quality and Under-fill

For each query, `recall@100` is the fraction of the exact top 100 IDs that the index returns. The denominator stays
100. If an index returns 50 valid rows and all are exact neighbors, recall is 0.5. Reported recall averages over all
selected queries. Use the returned-count fields to see when under-fill contributes to missed neighbors.

The `exact_nn_hit@1`, `exact_nn_hit@10`, and `exact_nn_hit@100` fields track whether the single first exact ID
appears within each returned prefix. Each hit is 0 or 1 for one query, and the fields average those hits across
queries. A wider prefix can only maintain or increase the hit rate.

The report checks row bounds, finite distances, distinct IDs, and public nearest-first `Neighbor` order. Shorter
results can pass validation. Flat must return 100 rows and reach 1.0 for every quality field. The ignored smoke
tests also require 100 distinct rows per configured index and enforce broad 0.05 `exact_nn_hit@100` and
`recall@100` floors for the approximate indexes as integration checks.

The final line compares the size of IVF-PQ codes and codebooks with the original vector components.

```text
ivf_pq search representation: codes_bytes={u64}, codebooks_bytes={u64}, search_bytes={u64}, full_vectors_bytes={u64}, compression={:.1}x
```

In sampled and full modes, 4,000,000 code bytes plus 8,192 codebook bytes give a 4,008,192-byte representation.
Compared with 512,000,000 full-vector component bytes, the ratio is `127.7x`. Its scope is limited to these
components. The original vectors for reranking, coarse centroids, IDs, list containers, allocator overhead, and
other live indexes also take space.

Compare recall and p50/p95 within the same workload. Save the report with the exact command, source revision,
CPU and memory, operating system, `rustc -Vv`, build profile, and any custom compiler flags. Each invocation uses
freshly built and warmed indexes, with one timed pass. Record the process count and starting cache conditions
when you repeat a comparison. The resulting measurements describe the machine, workload, and configuration you ran.

{{#include copyright.md}}
