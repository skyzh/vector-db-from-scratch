# Benchmark Five Indexes on SIFT1M

> **Day 6**
>
> Complete [Compress IVFFlat with Product Quantization](./rust-07-ivfpq.md) first. Then bring Flat, IVFFlat, NSW,
> HNSW, and IVF-PQ together in one release-mode benchmark over the external SIFT1M corpus.

This final day gives the five indexes the same work: Euclidean search with `k = 100`. With SIFT1M on disk, your first
useful run is smoke mode. It validates the complete corpus, keeps 10,000 base rows and 100 queries, recomputes exact
top-100 truth for that smaller base, builds every index, and reports latency and quality from the same searches.
Progress goes to standard error while the finished report goes to standard output, so you can redirect the report
without losing sight of a long-running phase.

The course does not contain a benchmark result to copy. You will finish the executable, check it without a corpus, and
then decide whether to run smoke mode or the much larger full experiment on your own machine.

## Start from the Completed Indexes

Your Day 5 starter already contains the five index implementations. Keep that boundary green from the repository root:

```sh
cargo xtask test-through day_05
```

Now open:

```text
vector-db-starter/core/examples/recall.rs
```

Most of the benchmark is supplied. The `vector-db-from-scratch-benchmark-support` crate parses the command line,
validates SIFT files, selects full or smoke sizes, balances warm-up and timed searches, computes quality, and selects
latency percentiles. The example fixes the five configurations, output format, result validation, and IVF-PQ byte
accounting.

Four `todo!()` calls are yours. Checkpoint 1 replaces the three constructor TODOs for NSW, HNSW, and IVF-PQ. Checkpoint
2 replaces `report_percentiles` with calls to the supplied nearest-rank helper. Nothing else in the example needs to be
designed for this day.

The support crate has a fast, corpus-free test suite. Run it before editing:

```sh
cargo test -p vector-db-from-scratch-benchmark-support
```

With your Days 1–5 work in place, the Day 6 selector should stop at the four new TODO boundaries:

```sh
cargo xtask test day_06
```

A fresh checkout still has earlier-course TODOs and will fail before it reaches this boundary. Either way, no SIFT1M
download is needed for the selector.

## Acquire and Validate SIFT1M

Obtain SIFT1M from the [TexMex ANN corpus](http://corpus-texmex.irisa.fr/) and follow the terms published there. The
course neither redistributes the corpus nor supplies an archive checksum or a separate dataset license claim.

Pass a directory that directly contains these three extracted files:

| File | Records | Width | Exact bytes |
| --- | ---: | ---: | ---: |
| `sift_base.fvecs` | 1,000,000 | 128 `f32` values | 516,000,000 |
| `sift_query.fvecs` | 10,000 | 128 `f32` values | 5,160,000 |
| `sift_groundtruth.ivecs` | 10,000 | 100 `i32` row IDs | 4,040,000 |

Both modes scan all three files before building an index. The loader checks every little-endian dimension header, the
exact record and byte counts, truncation and trailing bytes, finite vector components, and ground-truth IDs that are
nonnegative, in range, and unique within a row. A usage error exits with status 2; a data or index error exits with
status 1. The only public invocation shape is:

```text
usage: recall [--smoke] <sift1m-dir>
```

There is no synthetic fallback, arbitrary row limit, environment-variable run mode, or interactive prompt. Only the
ignored integration tests use `SIFT1M_DIR` to locate a developer's corpus.

## What Smoke Mode Changes

| Field | Full/default | Smoke |
| --- | --- | --- |
| Report labels | `mode=sift1m-full`, `parity=bustub-sift1m` | `mode=sift1m-smoke`, `parity=non-parity` |
| Base rows | 1,000,000 | first 10,000 |
| Queries | 10,000 | first 100 |
| Dimension, metric, `k` | 128, Euclidean, 100 | 128, Euclidean, 100 |
| Exact top-100 truth | supplied SIFT ground-truth row | Flat search over the selected 10,000 rows |

Smoke mode is not a miniature parity result. Its exact neighbors are recomputed after the base changes, so its quality
and timing describe the selected subset only. Full mode uses all one million base vectors, all 10,000 queries, and the
supplied exact top 100. Its parity label records the corpus, Euclidean ordering, `k = 100`, first-neighbor hit rates, and
top-100 overlap; it does not claim identical parameters, storage, floating-point paths, or timings across other
implementations.

## Keep the Five Configurations Fixed

| Index | Report configuration |
| --- | --- |
| Flat | `exact` |
| IVFFlat | `partitions=32,probes=6,iterations=12,seed=7` |
| NSW | `max_connections=12,ef_construction=64,ef_search_configured=40,ef_search_effective=100` |
| HNSW | `max_connections=12,ef_construction=64,ef_search=40,max_level=12,seed=7` |
| IVF-PQ | `partitions=32,probes=6,iterations=12,subquantizers=4,codebook_size=16,rerank=100,seed=7` |

These are the Rust course configurations, not universal tuning advice. One detail in the NSW row is easy to miss: the
stored search width is 40, but `search(query, 100)` uses `max(ef_search, k)`, so this benchmark actually explores with
an effective width of 100. The report records both numbers instead of presenting 40 as the work performed.

## Checkpoint 1: Construct the Remaining Indexes

Replace `build_nsw`, `build_hnsw`, and `build_ivf_pq` with their matching constructors. Pass through the supplied
dataset, metric, and configuration, and return constructor errors rather than switching to another index or setting.

The example prepares each immutable dataset clone, metric, and configuration before starting the clock. Preserve that
line: a `build_s` sample contains only the constructor. File loading, validation, query preparation, truth selection,
dataset cloning, and configuration construction remain outside it.

Run the constructor checkpoint:

```sh
cargo xtask test day_06::checkpoint_1
```

The tests accept more than one deterministic RNG trajectory. A seed must repeat within your implementation; it does not
make your IVFFlat centroids or HNSW levels match another implementation's internal samples.

## Checkpoint 2: Select p50 and p99

Replace `report_percentiles` with two calls to the supplied `percentile` helper. The input duration slice is already
sorted and nonempty. The helper uses nearest rank: for percentage `p` and `n` samples, it selects this zero-based
position, clamped to the last sample:

```text
ceil(p / 100 * n) - 1
```

Interpolation or a floor fraction of `n - 1` would describe a different statistic. When the p50 and p99 calls are in
place, run the complete example boundary:

```sh
cargo xtask test day_06::checkpoint_2
```

This checkpoint covers the constructors and percentile selection together with the fixed inventory, configurations,
mode-specific truth, result validation, quality and returned-count summaries, report order, and full-mode IVF-PQ
accounting.

## Follow Progress without Polluting the Report

The program writes best-effort progress to standard error and holds standard output until the entire report is valid.
Redirecting stdout therefore captures only the workload, five index rows, and IVF-PQ accounting. A bad input file,
constructor error, wrong result count, duplicate or out-of-range row, nonfinite distance, unordered result, or result
longer than `k` aborts before any stdout report line is printed. Progress already written to stderr may remain visible.

For a noninteractive stderr stream, each counted phase prints 0%, 25%, 50%, 75%, and 100% as separate lines. A terminal
redraws those milestones in place. Loading counts every physical row even in smoke mode because the complete files are
still validated; recomputing smoke truth counts the selected 100 queries. Builds can expose no useful fractional work,
so each prints only `start` and `complete`.

Warm-up completes 20 query rounds, or 100 searches across five indexes. The timed phase completes 100 rounds and 500
searches in smoke mode, or 10,000 rounds and 50,000 searches in full mode. Each milestone advances only after all five
indexes finish a round and their elapsed times have been captured. Progress writing is outside the samples, and a
closed or unwritable stderr stream does not abort the benchmark.

## Read the Measurement Loop

Warm-up uses the first `min(20, query_count)` queries, and the timed pass uses every selected query. Both rotate the
starting index so that no one implementation always runs first:

```text
(query_ordinal + offset) % 5
```

Only `search(query, 100)` is inside a latency sample. Result validation, quality calculation, latency sorting,
percentile selection, formatting, and printing happen afterward. Search errors are returned rather than skipped.
`search_s` is the sum of all per-query samples, and `qps` is `query_count / search_s`.

## Interpret Quality and Under-fill

For each query, `first_hit` follows one exact row: the first neighbor in the exact top 100. The three fields record how
far into the returned prefix the benchmark must look before finding it:

```text
first_hit@1      exact first neighbor appears at rank 1
first_hit@10     exact first neighbor appears somewhere in ranks 1..10
first_hit@100    exact first neighbor appears somewhere in ranks 1..100
```

Each field is binary for one query and averaged across all selected queries. Because each wider prefix contains the
narrower one, the final rates must satisfy:

```text
0 <= first_hit@1 <= first_hit@10 <= first_hit@100 <= 1
0 <= overlap@100 <= 1
```

`overlap@100` answers a different question: how many returned row IDs belong to the exact top 100? It always divides by
100. If an index returns 50 valid rows and all 50 are exact neighbors, its overlap is `0.5`, not `1.0`; the absent rows
count as misses. `returned_min`, `returned_avg`, and `returned_max` make that valid under-fill visible instead of hiding
it behind a quality average.

The report accepts between zero and `min(k, base_rows)` distinct, in-range rows in public nearest-first `Neighbor`
order, all with finite distances. Flat has the stronger contract: exactly 100 rows and `1.0` for every quality field.
The ignored external smoke tests are also deliberately stricter than the general report path: they require 100 distinct
rows from each configured index, plus exact Flat quality and broad `0.05` first-hit@100 and overlap@100 floors for the
approximate indexes. Those floors catch broken integrations; they are not production targets.

## Run Smoke, Then Full SIFT1M

After both checkpoints pass, supply the extracted corpus directory. Run the first command for the 10,000-row smoke
experiment. The second command is the optional full SIFT1M run:

```sh
cargo run --release -p vector-db-from-scratch-core-starter --example recall -- --smoke /absolute/path/to/sift1M
cargo run --release -p vector-db-from-scratch-core-starter --example recall -- /absolute/path/to/sift1M
```

You can compare behavior with the completed executable without reading its source:

```sh
cargo run --release -p vector-db-from-scratch-core --example recall -- --smoke /absolute/path/to/sift1M
cargo run --release -p vector-db-from-scratch-core --example recall -- /absolute/path/to/sift1M
```

The full run needs the extracted 525,200,000-byte corpus payload plus build products. Budget tens of minutes and several
GiB of working memory; at least 8 GiB of free memory and roughly 1 GiB of free disk beyond the corpus and build outputs
is a practical starting point, not a benchmark result or pass/fail threshold.

If you want to exercise one index with external data, the ignored tests expose it separately. For IVF-PQ:

```sh
SIFT1M_DIR=/absolute/path/to/sift1M \
  cargo test -p vector-db-from-scratch-core-starter --test sift_smoke \
  day_06::checkpoint_2::sift_ivf_pq_smoke -- --ignored --exact
```

The analogous names end in `sift_flat_smoke`, `sift_ivf_flat_smoke`, `sift_nsw_smoke`, and `sift_hnsw_smoke` under the
same `day_06::checkpoint_2` namespace.

## Read the Finished Report

Every successful run begins with one workload line:

```text
workload: mode={sift1m-full|sift1m-smoke}, parity={bustub-sift1m|non-parity}, rows={1000000|10000}, dimensions=128, queries={10000|100}, metric=euclidean, k=100, truth={supplied-sift1m-top-100|recomputed-flat-selected-base-top-100}
```

Five rows follow in `flat`, `ivf_flat`, `nsw`, `hnsw`, `ivf_pq` order:

```text
{name}: config={stable-config}, build_s={:.3}, search_s={:.3}, qps={:.1}, first_hit@1={:.4}, first_hit@10={:.4}, first_hit@100={:.4}, overlap@100={:.4}, returned_min={n}, returned_avg={:.1}, returned_max={n}, p50_ms={:.3}, p99_ms={:.3}
```

The final line is narrower than a memory measurement:

```text
ivf_pq search representation: codes_bytes={u64}, codebooks_bytes={u64}, search_bytes={u64}, full_vectors_bytes={u64}, compression={:.1}x
```

For the full data, four million code bytes plus 8,192 codebook bytes produce a 4,008,192-byte search representation.
Compared with 512,000,000 full-vector component bytes, the line prints `127.7x`. That ratio describes only codes plus
codebooks versus vector components. It excludes the original vectors retained for reranking, coarse centroids, row IDs,
list and graph containers, allocator overhead, and the other four live indexes. It is not resident memory or total-index
compression.

Record timings and quality only from a run you actually performed, together with its mode, machine, and fixed
configuration. One run cannot establish a universal fastest index, quality ranking, latency threshold, general
exactness, or graph connectivity.

## Finish the Course Boundary

The focused and cumulative gates compile the benchmark but skip the ignored SIFT1M tests:

```sh
cargo xtask test day_06
cargo xtask test-through day_06
```

Once they pass, you have a complete corpus-free implementation boundary. Smoke mode is the practical first external
check when you have acquired SIFT1M. The full-data command above remains optional: run it only with the corpus and
resources available, and treat any numbers it produces as observations from that machine and configuration—not as
measurements supplied or promised by the course.

{{#include copyright.md}}
