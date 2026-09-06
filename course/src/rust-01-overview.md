# Build Vector Search in Rust

<div class="warning">

**Course status:** All six required days are ready to implement. The repository includes starter code, focused tests,
and separate reference solutions. Day 6 uses a local copy of the external SIFT1M corpus; hosted tests do not download
or run it.

</div>

Start with a supplied product tour: launch an empty SQL session, create and populate an in-memory `points` table, attach an
IVFFlat index to its selected vector column, and watch `EXPLAIN` change without changing the nearest rows. Across the six
implementation days that follow, you will connect that table to DataFusion, implement the optimizer rule that selects
a safe vector-index scan, build
IVFFlat behind that rule, navigate a proximity graph with NSW, add HNSW hierarchy, compress residual candidate scoring
with IVF-PQ, and compare all five indexes on SIFT1M. The first five days return to runnable SQL so you can inspect how
the same product path changes as the index becomes more capable. The final day measures first-neighbor rank recall
and latency directly under one shared Euclidean, `k = 100` contract.

```sql
SELECT id, payload
FROM points
ORDER BY cosine_distance(embedding, [0.1, 0.2, 0.3])
LIMIT 10;
```

The [product tour](./rust-00-sql-shell.md) creates and fills the table, then runs a concrete query of this shape through the
supplied completed system before you edit anything. DataFusion's vector distance expression, bounded sort, and `LIMIT`
return an exact result; after you create a named index on the selected vector column, that unchanged SQL reaches the
course's vector-index scan. Day 1 then asks you to build the safe table, attachment, and planner path behind that
observation. Later, you will add IVFFlat as your own candidate selector behind the same interface.

## Where to Write Your Code

The repository-root Cargo workspace separates starter and reference trees:

```text
vector-db-starter/
  core/                      dataset, IVFFlat, NSW, HNSW, benchmark, and IVF-PQ TODOs
  datafusion/                Day 1 Arrow table and optimizer-rule TODOs
vector-db/
  core/                      completed core reference
  datafusion/                completed DataFusion reference
```

The product tour executes one supplied example from `vector-db/`; you do not need to inspect or modify that implementation.
After the tour, work in `vector-db-starter/` and implement its TODOs in day order. Keep the completed reference source
closed while you work through the exercises, as required by the starter's `AGENTS.md` files.

From the repository root, check that the untouched starter compiles:

```sh
cargo check -p vector-db-from-scratch-core-starter
cargo check -p vector-db-from-scratch-datafusion-starter
```

The focused tests initially stop at `todo!` calls. Each day names the exact tests that should pass before you move on,
then closes with `cargo x test-day N` for that day's work and `cargo x test-through N` for the cumulative course.

## One Query, Two Plans

Before index matching, the query is exact:

```text
SortExec: TopK(fetch=10), ...
  DataSourceExec: partitions=1, ...
```

An ordinary `MemTable` emits Arrow rows. DataFusion evaluates the distance function for every row and uses its own
bounded sort to produce the nearest ten.

On Day 1, you attach one index to an explicitly selected vector column, then implement a physical optimizer rule. It
accepts only one compatible distance ordering over that configured field with a literal query vector. The matched scan
asks the selected index for `LIMIT k` candidate row identities:

```text
SortExec: TopK(fetch=10), ...
  VectorIndexScanExec: index=flat, metric=Cosine, query_dim=3, fetch=Some(10), ordered=false
```

The starter's exact `FlatIndex` lets you exercise this rule on Day 1. Later days change only the selected index:

```text
SortExec: TopK(fetch=10), ...
  VectorIndexScanExec: index=ivf_flat, metric=Cosine, query_dim=3, fetch=Some(10), ordered=false
```

```text
SortExec: TopK(fetch=10), ...
  VectorIndexScanExec: index=nsw, metric=Cosine, query_dim=3, fetch=Some(10), ordered=false
```

```text
SortExec: TopK(fetch=10), ...
  VectorIndexScanExec: index=hnsw, metric=Cosine, query_dim=3, fetch=Some(10), ordered=false
```

The default plan retains DataFusion's bounded sort. The index selects candidates; `SortExec` owns SQL ordering. When the
selected index returns rows in the requested order, `SET vector_search.ordered = true` tells DataFusion it can skip this
final sort.

Filters, multiple sort keys, a non-literal query vector, another same-shaped vector column, the wrong distance function,
the wrong direction, or a dimension mismatch keep the exact plan. In particular, taking ANN top-k before applying a
filter can change the answer, so refusing that rewrite is a correctness requirement.

## Architecture

```text
ordinary MemTable --> selected-column attachment --> DataFusion optimizer --> VectorIndexScanExec
                                                                            |-- exact FlatIndex
                                                                            |-- your IvfFlatIndex
                                                                            |-- your IvfPqIndex
                                                                            |-- your NswIndex
                                                                            `-- your HnswIndex
```

The DataFusion crate owns Arrow conversion, SQL-pattern matching, plan properties, limits, and output batches. The core
crate owns dimensions, metrics, exact-search results, candidate selection, and deterministic result order. Later index
implementations will not import DataFusion.

This separation gives Days 1–5 two useful views of each checkpoint: small Rust tests isolate the algorithm, while
self-contained SQLLogicTests show that the Day 1 optimizer can reach it. Day 5 also keeps a focused
planner/EXPLAIN test for IVF-PQ; Day 6 brings every index into one fixed full-SIFT1M comparison and an explicitly
non-parity smoke mode.

## System Contract

1. **Dimension:** a dataset has one nonzero dimension; every stored vector and query matches it.
2. **Numeric domain:** stored values are finite `f32`, while metric accumulation uses `f64`. Cosine inputs have nonzero
   norm.
3. **Identity:** each core row offset maps through the attachment's checked snapshot location to the complete source row;
   no user field is row identity.
4. **Ordering:** lower internal distance is better. Ties use row offset. Dot product is negated at the metric boundary.
5. **Exact baseline:** exact search defines the expected result. When you report approximate latency, include recall from
   the same data, queries, metric, and `k`.
6. **SQL safety:** the optimizer selects an index only when expression, metric, direction, dimension, and limit match its
   contract. Unsupported shapes remain exact.

## Course Progression

| Day | Estimate | Before | After | Learner-owned files |
| --- | ---: | --- | --- | --- |
| [Product tour](./rust-00-sql-shell.md) | 10–15 minutes | The course has not yet shown a running database interface. | Starting from an empty session, you create and populate a table, run nearest-neighbor SQL, attach an IVFFlat index to a selected vector column, and make the plan change observable. | None; use the supplied `vector-db-from-scratch-datafusion` shell. |
| [1 — DataFusion table and optimizer](./rust-02-datafusion.md) | 3–4 hours | Vectors are Rust structs and DataFusion has no vector access path. | Rows become ordinary Arrow `MemTable` data; one attachment owns a selected vector field; a conservative physical rule selects its compatible index scan and preserves exact fallback. | `vector-db-starter/core/src/dataset.rs` and `vector-db-starter/datafusion/src/lib.rs` |
| [2 — IVFFlat](./rust-03-ivfflat.md) | 4–5 hours | A flat index handles matched SQL top-k queries exactly. | Seeded k-means, inverted lists, and `probes` create a measured recall/work tradeoff behind the same SQL query. | `vector-db-starter/core/src/{ivf,search}.rs` |
| [3 — NSW](./rust-04-nsw.md) | 4–5 hours | Candidate selection comes from centroid partitions. | Best-first traversal and bounded reciprocal graph insertion expose `ef_search` as a second recall/work tradeoff behind the same SQL query. | `vector-db-starter/core/src/{graph,nsw}.rs` |
| [4 — HNSW](./rust-05-hnsw.md) | 4–5 hours | Every graph query starts in one complete layer. | Seeded sparse layers route greedily into layer-zero beam search while preserving the same SQL and recall contracts. | `vector-db-starter/core/src/{graph,hnsw}.rs` |
| [5 — IVF-PQ](./rust-07-ivfpq.md) | 3–4 hours | HNSW completes the course's full-precision index set. | Residual PQ codes provide lookup-table candidate scoring, exact reranking, and explicit search-representation accounting. | `vector-db-starter/core/src/pq.rs` |
| [6 — Five-index SIFT1M benchmark](./rust-06-benchmark.md) | 1–2 hours plus the external run | Each index has been exercised separately. | Flat, IVFFlat, NSW, HNSW, and IVF-PQ share one full-SIFT1M Euclidean, `k = 100`, first-neighbor rank-recall, and latency contract. | `vector-db-starter/core/examples/recall.rs` |

Day 1 gives you an exact end-to-end query whose rows and physical plan you can inspect. Days 2–5 keep that SQL
interface and safety rule in place while changing how candidate rows are selected. Day 6 then compares all five
indexes without changing the SIFT1M data, queries, Euclidean metric, or `k = 100`; its smaller mode is labeled non-parity
because it recomputes truth over a 10,000-row subset.

After Day 6, you should be able to explain:

- how row identity survives conversion from Rust structs to core offsets and Arrow arrays;
- which physical expression shapes are safe to lower to a vector index;
- why DataFusion retains exact fallback for filtered or incompatible top-k queries;
- why the optimizer rule must exist before an approximate index can be exercised from SQL;
- why IVFFlat must rebuild list membership after its final centroid update; and
- how `probes` trades candidate work for recall without changing SQL;
- why NSW needs separate candidate and result frontiers;
- how reciprocal pruning preserves a bounded graph;
- why HNSW uses greedy upper layers and a layer-zero beam;
- how seeded promotion makes comparisons reproducible;
- why IVF-PQ separates coarse centroids, residual codebooks, approximate scoring, and exact reranking; and
- how supplied or recomputed exact first-neighbor truth, cyclic warm-up and timing order, and one shared workload make
  rank recall and latency interpretable together.

## Scope

These six days use an immutable in-memory collection and a readable Euclidean residual IVF-PQ implementation, but not
bit packing or optimized kernels. Online updates or deletes, index persistence, crash recovery, concurrent mutation,
filtered ANN, GPU kernels, distributed execution, general catalog semantics, and a network service remain outside this
implementation. The supplied shell's bounded `CREATE INDEX` bridge resolves eligible named or qualified in-memory tables,
supports multiple distinct attachments, and rejects writes that would stale an indexed snapshot; it is not a persistence,
online-maintenance, or general catalog subsystem. The final day also assumes a locally acquired SIFT1M directory;
the repository supplies parsers and tiny corruption fixtures, not the external corpus or benchmark results.

{{#include copyright.md}}
