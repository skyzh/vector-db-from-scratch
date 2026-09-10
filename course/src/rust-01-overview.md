# Build Vector Search in Rust

<div class="warning">

**Course status:** All six required days are ready to implement. The repository includes starter code, focused tests,
and separate reference solutions. Day 6 uses a local copy of the external SIFT1M corpus; hosted tests do not download
or run it.

</div>

Begin with the supplied product tour. You will open an empty SQL session, create an in-memory `points` table, run a
nearest-neighbor query, and attach an IVFFlat index to the table's vector column. `EXPLAIN` makes the change in scan
visible before you write any Rust.

The six implementation days then rebuild that path from the bottom up. Day 1 connects ordinary Arrow rows to
DataFusion and adds the optimizer rule that can select a vector index safely. Days 2–5 implement IVFFlat, NSW, HNSW,
and IVF-PQ behind the same query interface. Day 6 compares those four indexes with the exact flat baseline on SIFT1M.
The benchmark keeps Euclidean distance and `k = 100` fixed, and uses the same first-neighbor rank-recall definition and latency measurement procedure for all five indexes.

```sql
SELECT id, payload
FROM points
ORDER BY cosine_distance(embedding, [0.1, 0.2, 0.3])
LIMIT 10;
```

The [product tour](./rust-00-sql-shell.md) runs this shape of query through the supplied completed system. Before an index
is attached, DataFusion scans every row, so the fallback result is exact. The tour then creates an IVFFlat index with two
partitions and probes both of them. That particular indexed run still scores all five rows, although rows 3 and 5 tie for
the third slot and SQL has no secondary ordering key to break the tie. IVFFlat becomes approximate when it probes only a
subset of its partitions. In that case, DataFusion's final sort orders the candidates returned by the index; it cannot
recover rows that never entered the candidate set.

Day 1 asks you to build the table conversion, attachment, and planner path behind this observation. The later days
replace the selected index while keeping the SQL interface and safety rule intact.

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

The product tour executes a supplied example from `vector-db/`. Leave that completed implementation closed and
unchanged. Your work begins in `vector-db-starter/`, where the TODOs are arranged in day order. The `AGENTS.md` files in
the two starter crates state the same boundary.

From the repository root, check that the untouched starter compiles:

```sh
cargo check -p vector-db-from-scratch-core-starter
cargo check -p vector-db-from-scratch-datafusion-starter
```

The focused tests initially stop at `todo!` calls. Each day places its checkpoint command next to the code it exercises.
At the end of a day, run `cargo xtask test day_NN` for that day's work and `cargo xtask test-through day_NN` for the
cumulative course.

## One Query, Two Plans

Before index matching, the query is exact:

```text
SortExec: TopK(fetch=10), ...
  DataSourceExec: partitions=1, ...
```

An ordinary `MemTable` emits Arrow rows. DataFusion evaluates the distance function for every row and uses its own
bounded sort to produce the nearest ten. This is the exact fallback path.

On Day 1, you attach one index to an explicitly selected vector column, then implement a physical optimizer rule. It
accepts only one compatible distance ordering over that configured field with a literal query vector. The matched scan
asks the selected index for `LIMIT k` candidate row identities:

```text
SortExec: TopK(fetch=10), ...
  VectorIndexScanExec: index=flat, metric=Cosine, query_dim=3, fetch=Some(10), ordered=false
```

The starter's exact `FlatIndex` lets you exercise this rule on Day 1. Later days change the selected index:

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

The default plan retains DataFusion's bounded sort. The index chooses the candidate rows, and `SortExec` orders those
candidates for SQL. When an index guarantees that its output is already in the requested order,
`SET vector_search.ordered = true` lets DataFusion skip the final sort.

The optimizer keeps the exact plan for filters, multiple sort keys, a non-literal query vector, another same-shaped
vector column, the wrong distance function or direction, and dimension mismatches. This conservative behavior matters:
for example, taking ANN top-k before applying a filter can change the answer.

## Architecture

```text
ordinary MemTable --> selected-column attachment --> DataFusion optimizer --> VectorIndexScanExec
                                                                            |-- exact FlatIndex
                                                                            |-- your IvfFlatIndex
                                                                            |-- your IvfPqIndex
                                                                            |-- your NswIndex
                                                                            `-- your HnswIndex
```

The DataFusion crate owns Arrow conversion and the SQL-facing execution path: pattern matching, plan properties, limits,
and output batches. The core crate owns vector dimensions, metrics, search results, candidate selection, and
deterministic result order. The later index implementations do not need to import DataFusion.

This split gives you two ways to check Days 1–5. Small Rust tests isolate the algorithm, while self-contained
SQLLogicTests show that the Day 1 optimizer can reach it. Day 5 adds a focused planner/`EXPLAIN` test for IVF-PQ. Day 6
moves all five indexes into one full-SIFT1M comparison; its smaller smoke mode is explicitly not a parity run.

## Rules That Stay Fixed

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

Day 1 establishes the end-to-end path: a Rust row becomes a core offset and an Arrow row, the optimizer recognizes a
safe physical expression, and incompatible or filtered queries stay on the exact scan. This rule has to work before an
approximate index can be reached from SQL.

The next four days change how candidates are found. IVFFlat trains centroids, rebuilds list membership after the final
centroid update, and exposes `probes` as its recall/work control. NSW uses separate candidate and result frontiers while
reciprocal pruning keeps the graph bounded. HNSW adds seeded, reproducible promotion, greedy upper-layer routing, and a
layer-zero beam. IVF-PQ keeps coarse centroids, residual codebooks, approximate lookup-table scoring, and exact
reranking as distinct parts of the search.

The final benchmark holds the workload still while those choices change. Exact first-neighbor truth is supplied or
recomputed from the same data and queries, and every index uses the same Euclidean metric and `k = 100`. Cyclic warm-up
and timing order make the resulting rank-recall and latency numbers comparable.

## Scope

The course uses an immutable in-memory collection and a readable Euclidean residual IVF-PQ implementation. It does not
add bit packing or optimized kernels. Online updates and deletes, index persistence, crash recovery, concurrent
mutation, filtered ANN, GPU kernels, distributed execution, a general catalog, and a network service are outside the
implementation.

The supplied shell includes a narrow `CREATE INDEX` bridge for eligible named or qualified in-memory tables. It supports
multiple distinct attachments and rejects writes that would stale an indexed snapshot, but it is not a persistence or
online-maintenance subsystem. Day 6 assumes that you have acquired SIFT1M locally. The repository provides parsers and
small corruption fixtures, not the external corpus or benchmark results.

{{#include copyright.md}}
