# Make the SQL Path Reach Your Index Safely

> **Day 1**
>
> Start from the two `*-starter` crates. Finish with ordinary Arrow tables, one
> explicitly attached vector index, and a conservative DataFusion optimizer
> rule.

In the [product tour](./rust-00-sql-shell.md), you began with an empty session, created and populated `points`, then ran the
supplied shell before and after attaching an index to `embedding`. The SQL and nearest rows stayed fixed while the physical
leaf changed from `DataSourceExec` to `VectorIndexScanExec`. Day 1 opens that path: you will build the Arrow table,
bind one selected vector field to an index, and make the optimizer choose the new scan only when the query is safe.

Your first query uses the course's small three-column table:

```sql
SELECT id, payload
FROM points
ORDER BY cosine_distance(embedding, [1.0, 0.0, 0.0])
LIMIT 3;
```

Without an index match, DataFusion scans the `MemTable`, computes every
distance, and keeps the nearest three with a bounded sort:

```text
SortExec: TopK(fetch=3), ...
  DataSourceExec: partitions=1, ...
```

That plan is exact for every valid query. A vector index can select candidates
only when the SQL ordering refers to the same metric, literal, dimension,
direction, and configured vector column. A match changes the leaf while leaving
DataFusion's final sort in place by default:

```text
SortExec: TopK(fetch=3), ...
  VectorIndexScanExec: index=flat, metric=Cosine, query_dim=3, fetch=Some(3), ordered=false
```

Day 2 will put `index=ivf_flat` behind the same boundary.

## From the Product Tour to Your First Checkpoint

The product tour showed the complete path before asking you to build it. Keep
these boundaries separate as you work through the day:

| What you observed | What is supplied | What you implement |
| --- | --- | --- |
| Ordinary SQL creates and fills `points`, then scans it; the supplied `CREATE INDEX` bridge changes only the physical leaf. | The shell, bounded DDL bridge, metric math, exact `FlatIndex`, and shared attachment/lookup scaffolding. | Checkpoint 1 validates the core `Dataset`. |
| Both plans return the same rows and keep DataFusion's final sort. | Examples and tests that expose the plan and results. | Checkpoint 2 builds the introductory Arrow `MemTable`. |
| The indexed leaf is chosen only for the configured vector field and safe query shape. | The public attachment and optimizer interfaces. | Checkpoints 3–5 attach one field, match a safe top-k, then search and fetch source rows. |

You will modify:

```text
vector-db-starter/core/src/dataset.rs
vector-db-starter/datafusion/src/lib.rs
```

The starter exposes the same public API as the reference but leaves the Day 1
implementation points as TODOs. Metric math, the exact `FlatIndex`, shared
snapshot/lookup scaffolding, examples, and tests are ready. IVFFlat, NSW, HNSW,
and IVF-PQ remain later learner work. Do not modify public APIs or tests while
completing the exercises.

## Checkpoint 1: Validate the In-Memory Dataset

Implement the three TODOs in `vector-db-starter/core/src/dataset.rs`.

A dataset must be nonempty, have a fixed nonzero dimension, and contain only
finite `f32` values. `Dataset::try_new` reads the first row to establish the
dimension, rejects an empty dataset or zero-dimensional vector, then checks
every row for equal length and finite components. Store the vectors as
`Arc<[Vec<f32>]>`.

`validate_for_metric` rejects zero-norm stored rows for cosine distance.
`validate_query` checks dimension, finiteness, and the same cosine boundary
for a query. Use the existing `VectorError` variants.

```sh
cargo xtask test day_01::checkpoint_1
```

## Checkpoint 2: Build the Introductory MemTable

A vector index belongs to one field of an ordinary table. It does not own a
special `(id, payload, vector)` row format.

The small `VectorRow` and `vector_mem_table` helper remain the first example
because they make Arrow construction easy to inspect:

```text
id         UInt64
payload    Utf8
embedding  FixedSizeList<Float32, dimension>
```

Implement `vector_mem_table` in
`vector-db-starter/datafusion/src/lib.rs`.

Build a `Dataset` from the `VectorRow` embeddings to validate their shared
dimension. Create the three Arrow arrays in the same input order, assemble one
`RecordBatch`, then return an ordinary `MemTable`.

`FixedSizeListArray` stores vector components in one flat `Float32Array`.
For two three-dimensional rows, its child values are:

```text
[x0, y0, z0, x1, y1, z1]
 `---row 0--' `---row 1--'
```

Use `i32::try_from(dataset.dimension())` for Arrow's list width.

**Prediction:** What breaks if the payload array is reordered while the
embedding array keeps insertion order?

## Checkpoint 3: Attach One Selected Vector Column

The public indexing surface is more general. Register any `MemTable`, then
construct a `VectorIndexAttachment` with its table reference and selected
vector-column name:

```rust,ignore
let attachment = VectorIndexAttachment::try_new(
    &context,
    "documents",
    &table,
    "text_embedding",
    Metric::Euclidean,
    IndexConfig::Flat,
)
.await?;
let context = with_vector_indexes(&context, vec![attachment]);
```

The supplied SQL session resolves each accepted `CREATE INDEX` target into this same attachment constructor. The bridge is
already implemented; your Day 1 work is the attachment and execution path it calls.

The rich Day 1 test table deliberately puts ordinary scalar fields around
two vector fields:

```text
doc_key         Utf8
tenant_id       UInt32
price           Float64
inventory       Int32
text_embedding  FixedSizeList<Float32, 3>  <- selected
image_embedding FixedSizeList<Float32, 3>
active          Boolean
```

Both vector columns have the same type and width, but their nearest-neighbor
orders differ. A query ordered by `text_embedding` may use the attached index.
The same query shape over `image_embedding` must remain on DataFusion's exact
scan and return the image-vector ranking. No field name or ordinal is
inherently special; only the field selected by the attachment may use its
index.

The attachment snapshots the registered `MemTable` batches. It copies only the
selected vectors into the core `Dataset` and records a checked row location
for each dataset ordinal:

```text
index dataset ordinal -> snapshot RowId -> checked batch/row -> projected output
```

The source Arrow buffers remain shared with the ordinary `MemTable`. Scalar
columns and the unselected vector column stay normal table data. User columns
are never row identity.

DataFusion has no generic stable point-lookup API for arbitrary
`TableProvider` implementations. This adapter is therefore intentionally
limited to registered in-memory `MemTable` instances. A disk or distributed
provider would need its own stable row locator and lookup implementation.

An attachment must resolve the exact registered `MemTable` instance and the
configured field. The selected field must exist, be
`FixedSizeList<Float32>`, have a positive width, and contain no null list or
null element. Each source row must contribute exactly one dataset vector and
one checked snapshot row location.

A different positive list width is a valid schema choice; the core dataset takes
its dimension from the selected field. The SQL matcher later rejects a literal
whose width differs from that dataset. A zero-width selected field is invalid at
construction.

Implement `VectorIndexAttachment::try_new`.

1. Resolve the table reference and prove the supplied `Arc<MemTable>` is the
   registered provider.
2. Snapshot every partition and batch, requiring one shared schema.
3. Resolve only the configured vector-column name.
4. Validate its Arrow type, positive width, and non-null values.
5. Copy those selected vectors into `Dataset` in batch/row order.
6. Build the requested core index and record the corresponding checked row
   locations.

The rich-schema tests make the ownership rule observable: text and image vectors
have identical shapes but different rankings.

```sh
cargo xtask test day_01::checkpoint_3
```

## Checkpoint 4: Match and Rewrite One Safe Top-k

Implement `match_vector_order` and
`VectorIndexOptimizer::rewrite_sort`.

The optimizer may replace a scan only for one supported distance expression
over the configured vector field, a literal query vector, a compatible metric
and direction, a positive `LIMIT`, and a live source snapshot. Filters,
multiple sort keys, non-literal vectors, another vector field, wrong metrics or
directions, and invalid literals remain on DataFusion's exact scan and sort.

Unless ordered output is explicitly enabled for the session, DataFusion retains
the final bounded sort after the index selects candidates. Candidate order is
not automatically SQL order.

The matcher accepts only:

1. one physical sort expression;
2. Euclidean `array_distance`/`list_distance`, `cosine_distance`, or dot
   `inner_product`/`dot_product`;
3. ascending Euclidean/cosine or descending dot-product order;
4. one vector `Column` and one literal;
5. the exact configured vector-column name after projection;
6. a finite literal with the index dataset's dimension; and
7. a nonzero cosine literal.

DataFusion widens the fixed-size `Float32` list to `List<Float64>` for its
distance functions. `match_vector_column` accepts exactly that planner-added
cast, while `scalar_vector` admits only values that preserve their exact
`f32` representation.

The optimizer must also prove the physical `MemorySourceConfig` still matches
the attached table, snapshot, schema, projection, and unambiguous live provider.
On a match, construct `VectorIndexScanExec`; otherwise leave the plan
unchanged.

```sh
cargo xtask test day_01::checkpoint_4
```

## Checkpoint 5: Search, Fetch, and Preserve ORDER BY

Implement `VectorIndexScanExec::selected_rows` and
`ExecutionPlan::with_fetch`.

Search the selected index for at most `fetch` rows. Reject an index result that
does not resolve to the snapshot. The supplied lookup scaffolding reconstructs
the requested projection in index-result order.

For `ordered=true`, return the scan with its accepted ordering property. For
the default `ordered=false` path, clear that property and wrap the scan in
`SortExec::new(ordering, scan).with_fetch(Some(k))`. The index chooses
candidates; DataFusion still owns SQL's nearest-first result.

```sh
cargo xtask test day_01::checkpoint_5
```

The SQLLogicTest starts from an empty session: it creates and inserts the simple `points` table and a rich `documents`
table, then attaches indexes to the selected columns. `text_embedding` reaches `VectorIndexScanExec`, while
`image_embedding` stays on `DataSourceExec` and returns its different ranking.

## Day 1 Review

Run the Day 1 focused and cumulative gates:

```sh
cargo xtask test day_01
cargo xtask test-through day_01
```

After the core tests, `sql.rs`, and the Day 1 SQLLogicTest pass, explain:

- how the simple `VectorRow` helper becomes an ordinary `MemTable`;
- why an attachment owns exactly one configured vector field;
- how an index dataset ordinal resolves to a projected source row;
- why the same-shaped image-vector query cannot use the text-vector index;
- where DataFusion performs exact fallback and final ordering; and
- why the supplied session rejects changes to an indexed table instead of letting its attachment become stale;
- how later approximate indexes reuse this boundary without weakening it.

IVFFlat implementation, filtered pushdown, joins, general DDL/catalog semantics,
persistence, and disk row lookup remain outside Day 1. The product tour's
supplied bridge resolves an eligible in-memory table and selected vector column into the attachment path you implemented
here. It can hold multiple distinct attachments, but it rejects mutation of an indexed table and does not provide
persistence, automatic rebuilding, online maintenance, or a general catalog lifecycle.

{{#include copyright.md}}
