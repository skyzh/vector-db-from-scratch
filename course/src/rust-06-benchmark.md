# Benchmark Five Indexes on SIFT1M

> **Day 6**
>
> Complete [Compress IVFFlat with Product Quantization](./rust-07-ivfpq.md) first. Finish with one release-mode benchmark
> that compares Flat, IVFFlat, NSW, HNSW, and IVF-PQ on the external SIFT1M corpus under the same Euclidean queries and
> `k = 100` contract.

A search-time number is not useful by itself. An approximate index can look fast by returning the wrong neighbors, while
recall from another workload cannot explain the latency you measured. Day 6 therefore prints timing and result
quality from the same run.

The benchmark has two explicit modes. The default full run uses all one million SIFT base vectors, all 10,000 queries,
and the supplied first exact neighbor. The smaller `--smoke` run follows the same five-index code path but selects 10,000
base rows and 100 queries, then recomputes exact truth over that subset. It is quick external-data feedback, not a full
parity result.

## Start from the Completed Indexes

Your Day 5 starter already contains the five index implementations. Before opening the benchmark, keep their product
paths green from the repository root:

```sh
cargo xtask test-through day_05
```

Now open:

```text
vector-db-starter/core/examples/recall.rs
```

The supplied `vector-db-from-scratch-benchmark-support` crate owns command-line parsing, SIFT file validation, the full and smoke mode
sizes, cyclic warm-up and timing, rank-recall calculation, and nearest-rank percentile selection. The example already
owns the five configurations, report layout, result validation, and IVF-PQ accounting. You complete exactly four Day 6
ownership points:

1. construct NSW;
2. construct HNSW;
3. construct IVF-PQ; and
4. call the supplied percentile helper for p50 and p99.

Run the support tests before changing the example:

```sh
cargo test -p vector-db-from-scratch-benchmark-support
```

They use tiny little-endian fixtures and deliberately corrupted inputs, so they need no external download. The raw
starter's example test is expected to stop at a Day 6 `todo!()` until you finish both checkpoints below:

```sh
cargo xtask test day_06
```

## Acquire and Validate SIFT1M

Obtain SIFT1M from the [TexMex ANN corpus](http://corpus-texmex.irisa.fr/) and follow the terms published there. The
course does not redistribute or download the corpus, and it does not publish an archive checksum or a separate dataset
license claim.

Pass a directory that directly contains these three extracted files:

| File | Records | Width | Exact bytes |
| --- | ---: | ---: | ---: |
| `sift_base.fvecs` | 1,000,000 | 128 `f32` values | 516,000,000 |
| `sift_query.fvecs` | 10,000 | 128 `f32` values | 5,160,000 |
| `sift_groundtruth.ivecs` | 10,000 | 100 `i32` row IDs | 4,040,000 |

The loader scans and validates the complete files before the first index build in both modes. It checks each
little-endian dimension header, exact byte and record counts, truncation and trailing bytes, finite vector components,
and ground-truth IDs that are nonnegative, in range, and unique within a row. A usage error exits with status 2; a data
or index error exits with status 1. The public invocation is deliberately narrow:

```text
usage: recall [--smoke] <sift1m-dir>
```

There is no no-argument synthetic fallback, arbitrary row limit, environment-variable run mode, or interactive prompt.
Ignored integration tests alone use `SIFT1M_DIR` to find a developer's local corpus.

## Keep the Two Modes Distinct

| Field | Full/default | Smoke |
| --- | --- | --- |
| Report labels | `mode=sift1m-full`, `parity=bustub-sift1m` | `mode=sift1m-smoke`, `parity=non-parity` |
| Base rows | 1,000,000 | first 10,000 |
| Queries | 10,000 | first 100 |
| Dimension, metric, `k` | 128, Euclidean, 100 | 128, Euclidean, 100 |
| Exact first-neighbor truth | first supplied SIFT ground-truth ID | Flat search over the selected 10,000 rows |

The full label records parity with the BusTub course's corpus, Euclidean ordering, `k = 100`, and first-neighbor rank
recall. It does not claim identical index parameters, storage, floating-point paths, or timings across implementations.

**Prediction:** Smoke mode runs the same index implementations and report code. Why can its 10,000-row, 100-query result
still not stand in for the full SIFT1M parity run?

## Freeze the Five Configurations

Do not tune one index while leaving the others at the course defaults:

| Index | Report configuration |
| --- | --- |
| Flat | `exact` |
| IVFFlat | `partitions=32,probes=6,iterations=12,seed=7` |
| NSW | `max_connections=12,ef_construction=64,ef_search=40` |
| HNSW | `max_connections=12,ef_construction=64,ef_search=40,max_level=12,seed=7` |
| IVF-PQ | `partitions=32,probes=6,iterations=12,subquantizers=4,codebook_size=16,rerank=100,seed=7` |

These are fixed Rust course configurations, not a universal tuning recommendation or a promise of configuration parity
with the deprecated C++ implementation.

## Checkpoint 1: Construct the Remaining Indexes

Implement `build_nsw`, `build_hnsw`, and `build_ivf_pq` with the supplied dataset, Euclidean metric, and configuration.
Return constructor errors instead of substituting another configuration or index.

The surrounding code clones the immutable `Dataset` before each build and creates the metric and configuration before
starting the timer. Keep that boundary: each `build_s` measurement contains only the corresponding index constructor.
File I/O, validation, query preparation, truth selection, dataset cloning, and configuration construction are not build
time.

Before moving to reporting, rerun the invariant gate that permits more than one deterministic RNG trajectory:

```sh
cargo xtask test day_06::checkpoint_1
```

A seed promises repeatability within your implementation. It does not require your IVFFlat centroids or HNSW level
sequence to equal the reference implementation's internal samples.

## Checkpoint 2: Select p50 and p99

Implement `report_percentiles` by calling the supplied `percentile` helper on the sorted, nonempty duration slice. The
helper uses nearest rank. For percentage `p` and `n` samples, it selects this zero-based position, clamped to the final
sample:

```text
ceil(p / 100 * n) - 1
```

Do not replace it with interpolation or a floor fraction of `n - 1`; that would change the report contract. Once all
four ownership points are complete, run the five behavioral example tests:

```sh
cargo xtask test day_06::checkpoint_2
```

This gate pins the completed constructors and percentile selection, fixed inventory and configurations,
full-versus-smoke truth selection, result validation, rank-prefix averaging, report order, and full-mode IVF-PQ
accounting.

## Read the Supplied Measurement Loop

The support crate warms the first `min(20, query_count)` queries, then times every selected query. Warm-up and timed passes
both rotate the five indexes with:

```text
(query_ordinal + offset) % 5
```

Only `search(query, 100)` is inside each sample timer. Result validation, recall calculation, latency sorting,
percentile selection, formatting, and printing happen later. Search errors are returned rather than skipped.
`search_s` is the sum of all per-query search samples, and `qps` is `query_count / search_s`.

**Prediction:** Which of parsing, index construction, result validation, recall calculation, and printing belong outside
the search timer? Why would a faster row be uninterpretable if its recall fields were missing?

## Interpret First-neighbor Rank Recall

For each query, the benchmark chooses one exact nearest-neighbor row ID. It then asks whether that ID appears within the
first 1, 10, and 100 returned rows:

```text
R@1    exact first neighbor appears at rank 1
R@10   exact first neighbor appears somewhere in ranks 1..10
R@100  exact first neighbor appears somewhere in ranks 1..100
```

Each answer is binary for one query, and the report averages it across all selected queries. This is not set recall over
the exact top 100.

**Prediction:** How does “the exact first neighbor appears within the first 10 results” differ from “10 of the exact top
100 neighbors were recovered”?

Before recall is computed, every result must contain `min(k, base_rows)` distinct, in-range rows in public nearest-first
`Neighbor` order, with finite distances. The summary also requires:

```text
0 <= R@1 <= R@10 <= R@100 <= 1
```

Flat must report `1.0` at all three ranks.

**Prediction:** Why must widening the inspected prefix make rank recall monotonic? Name one result-order, duplicate-row,
or parser defect that could otherwise make the report untrustworthy.

## Run Smoke, Then Full SIFT1M

From the repository root, run the completed starter in release mode with an explicit corpus directory:

```sh
cargo run --release -p vector-db-from-scratch-core-starter --example recall -- --smoke /absolute/path/to/sift1M
cargo run --release -p vector-db-from-scratch-core-starter --example recall -- /absolute/path/to/sift1M
```

You can compare against the completed reference without reading its source:

```sh
cargo run --release -p vector-db-from-scratch-core --example recall -- --smoke /absolute/path/to/sift1M
cargo run --release -p vector-db-from-scratch-core --example recall -- /absolute/path/to/sift1M
```

The full run needs the extracted 525,200,000-byte corpus payload plus build products. Budget tens of minutes and several
GiB of working memory; a practical starting point is at least 8 GiB of free memory and roughly 1 GiB of free disk beyond
the extracted corpus and build outputs. These are planning guidelines, not benchmark results or pass/fail thresholds.

For one narrower external-data check, the supplied ignored tests expose each index separately. For example:

```sh
SIFT1M_DIR=/absolute/path/to/sift1M \
  cargo test -p vector-db-from-scratch-core-starter --test sift_smoke \
  day_06::checkpoint_2::sift_ivf_pq_smoke -- --ignored --exact
```

The analogous test names end in `sift_flat_smoke`, `sift_ivf_flat_smoke`, `sift_nsw_smoke`, and
`sift_hnsw_smoke` under the same `day_06::checkpoint_2` namespace. These
tests use the fixed smoke subset. Flat must match exact rank recall; approximate indexes must return ordered unique rows,
monotonic rank recall, same-implementation repeatability where seeded, and a broad `R@100 >= 0.05` floor. That floor is
a bug detector, not a production-quality target.

## Read the Report without Inventing Results

Every run begins with one workload line:

```text
workload: mode={sift1m-full|sift1m-smoke}, parity={bustub-sift1m|non-parity}, rows={1000000|10000}, dimensions=128, queries={10000|100}, metric=euclidean, k=100, truth={supplied-sift1m-first-neighbor|recomputed-flat-selected-base}
```

It then prints five rows in `flat`, `ivf_flat`, `nsw`, `hnsw`, `ivf_pq` order:

```text
{name}: config={stable-config}, build_s={:.3}, search_s={:.3}, qps={:.1}, r@1={:.4}, r@10={:.4}, r@100={:.4}, p50_ms={:.3}, p99_ms={:.3}
```

The final line isolates IVF-PQ search-representation accounting:

```text
ivf_pq search representation: codes_bytes={u64}, codebooks_bytes={u64}, search_bytes={u64}, full_vectors_bytes={u64}, compression={:.1}x
```

In full mode, 4,000,000 code bytes plus 8,192 codebook bytes make a 4,008,192-byte search representation. The comparison
against 512,000,000 full-vector component bytes prints `127.7x`. This is not resident memory or total-index compression:
it excludes retained vectors used for reranking, centroids, row IDs, list and graph containers, allocator overhead, and
the other four live indexes.

Record observed timings and rank recall only from a run you actually performed, together with its mode, machine, and
fixed configuration. Do not infer a universal fastest index, quality ranking, or latency threshold from this run.

## Day 6 Review

Run the Day 6 focused gate, then the complete cumulative course:

```sh
cargo xtask test day_06
cargo xtask test-through day_06
```

These commands compile but do not execute the ignored external-corpus SIFT1M
tests. Use the explicit `SIFT1M_DIR=... --ignored --exact` command above when
you have acquired the corpus.

After the release run you chose completes, explain:

- why all indexes must share data, queries, Euclidean metric, and `k = 100`;
- why full mode uses the supplied first neighbor while smoke mode recomputes truth over its selected base;
- why a seeded build must repeat within one implementation without copying reference centroids or levels;
- what belongs inside and outside constructor and search timers;
- how first-neighbor rank recall differs from top-100 set recall;
- why `R@1 <= R@10 <= R@100` must hold;
- why smoke output remains non-parity; and
- which bytes the IVF-PQ accounting includes and excludes.

Parameter sweeps, resident-memory measurement, multiple benchmark processes, and confidence intervals are useful next
steps. They are not evidence supplied by this single-process course benchmark.

{{#include copyright.md}}
