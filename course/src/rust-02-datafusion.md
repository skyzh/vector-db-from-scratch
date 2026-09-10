# Make the SQL Path Reach Your Index Safely

> **Day 1**
>
> Start from the two `*-starter` crates. Finish with ordinary Arrow tables, one
> explicitly attached vector index, and a conservative DataFusion optimizer
> rule.

The [product tour](./rust-00-sql-shell.md) began with an empty session and an ordinary in-memory `points` table. Its first
nearest-neighbor query used DataFusion's exact scan and returned rows 1, 2, and 3. After the tour attached an IVFFlat index
to `embedding`, the same SQL reached `VectorIndexScanExec` and returned 1, 2, and 3 again. Rows 3 and 5 tie for that final
slot, however, so SQL does not promise which one appears unless you add a secondary ordering key.

That small example probes both of its two partitions and therefore scores all five rows. IVFFlat is still an approximate
index when it probes only a subset of its partitions: it may omit a true neighbor before DataFusion sees the candidates.
The final sort orders the rows it receives; it does not make the candidate set exact.

Day 1 rebuilds the safe path beneath that tour. You will create an Arrow table, attach an exact `FlatIndex` to one selected
vector field, and teach DataFusion to use the new scan only when the physical query matches the attachment.

Your first query uses the course's small three-column table:

```sql
SELECT id, payload
FROM points
ORDER BY cosine_distance(embedding, [1.0, 0.0, 0.0])
LIMIT 3;
```

Before an index matches, DataFusion scans the `MemTable`, computes every distance, and keeps the nearest three with a
bounded sort:

```text
SortExec: TopK(fetch=3), ...
  DataSourceExec: partitions=1, ...
```

This fallback is exact for every valid query. An attachment may replace the leaf only when the SQL ordering uses its
configured vector column with the expected metric, literal, dimension, and direction. A safe match leaves DataFusion's
final sort in place by default:

```text
SortExec: TopK(fetch=3), ...
  VectorIndexScanExec: index=flat, metric=Cosine, query_dim=3, fetch=Some(3), ordered=false
```

Day 2 will put `index=ivf_flat` behind this same boundary.

## From the Product Tour to Your First Checkpoint

The shell and its narrow `CREATE INDEX` bridge are already complete. So are the metric implementations, the exact
`FlatIndex`, the public attachment and optimizer interfaces, and the snapshot lookup scaffolding. Examples and tests let
you inspect both the physical plan and the returned rows.

Your five checkpoints fill in the path between those supplied pieces. First validate the core `Dataset`, then turn the
small example into an Arrow `MemTable`. Next attach one selected field, recognize a safe top-k plan, and use index results
to fetch complete source rows in SQL order.

You will modify:

```text
vector-db-starter/core/src/dataset.rs
vector-db-starter/datafusion/src/lib.rs
```

The starter exposes the complete Day 1 API and marks your implementation points with TODOs. Work through those TODOs in
checkpoint order, leaving the public APIs and tests unchanged. IVFFlat, NSW, HNSW, and IVF-PQ belong to later days.

## Checkpoint 1: Validate the In-Memory Dataset

Implement the three TODOs in `vector-db-starter/core/src/dataset.rs`.

`Dataset::try_new` takes ownership of a nonempty set of finite `f32` vectors with one positive dimension. Use the first
row to establish that dimension, reject an empty dataset or zero-dimensional vector, and check every remaining row for
the same length and finite components. Store the validated vectors as `Arc<[Vec<f32>]>`.

`validate_for_metric` rejects zero-norm stored rows for cosine distance.
`validate_query` checks dimension, finiteness, and the same cosine boundary
for a query. Use the existing `VectorError` variants.

```sh
cargo xtask test day_01::checkpoint_1
```

## Checkpoint 2: Build the Introductory MemTable

A vector index belongs to one field of an ordinary table. The rest of the row keeps its normal Arrow types and layout.

The small `VectorRow` and `vector_mem_table` helper make the first conversion concrete:

```text
id         UInt64
payload    Utf8
embedding  FixedSizeList<Float32, dimension>
```

Implement `vector_mem_table` in `vector-db-starter/datafusion/src/lib.rs`.

Build a `Dataset` from the `VectorRow` embeddings so the core validation establishes their shared dimension. Create the
three Arrow arrays in input order, assemble one `RecordBatch`, and return it through an ordinary `MemTable`.

`FixedSizeListArray` stores vector components in one flat `Float32Array`.
For two three-dimensional rows, its child values are:

```text
[x0, y0, z0, x1, y1, z1]
 `---row 0--' `---row 1--'
```

Use `i32::try_from(dataset.dimension())` for Arrow's list width. Keep every array in the same row order: if the payload
array is reordered while the embeddings stay in insertion order, a query will return payloads that belong to different
vectors.

```sh
cargo xtask test day_01::checkpoint_2
```

## Checkpoint 3: Attach One Selected Vector Column

The small helper is only an introduction. The indexing surface accepts any registered `MemTable` and binds an index to
one named vector field. Construct that binding with `VectorIndexAttachment`:

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

The supplied SQL session uses this same constructor for every accepted `CREATE INDEX`. Its DDL bridge is already
implemented; your Day 1 work begins where that bridge hands off the table and selected field.

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

Both vector columns have the same type and width, but their nearest-neighbor orders differ. Attach the index to
`text_embedding` and that field's query may use it. The same query over `image_embedding` must keep DataFusion's exact
scan and return the image-vector ranking. Its shape alone is not enough: the attachment's selected field owns the index.

The attachment snapshots every batch in the registered `MemTable`. It copies the selected vectors into the core
`Dataset` and records where each dataset ordinal came from:

```text
index dataset ordinal -> snapshot RowId -> checked batch/row -> projected output
```

The source Arrow buffers remain shared with the `MemTable`; scalar columns and the unselected vector column stay ordinary
table data. A user column cannot stand in for row identity, so lookup follows the recorded snapshot location instead.

DataFusion has no generic stable point-lookup API for arbitrary `TableProvider` implementations. Day 1 therefore works
only with registered in-memory `MemTable` instances. A disk or distributed provider would need its own stable row locator
and lookup implementation.

Implement `VectorIndexAttachment::try_new`.

First resolve the table reference and confirm that the supplied `Arc<MemTable>` is the registered provider. Snapshot all
of its partitions and batches under one shared schema, then resolve the configured field by name. That field must be
`FixedSizeList<Float32>` with a positive width, no null lists, and no null elements.

Copy its vectors into `Dataset` in batch and row order. For every dataset ordinal, record the matching checked snapshot
location, then build the requested core index. The selected field determines the dataset dimension, so any positive list
width is valid here; the SQL matcher will reject a query literal with a different width. The rich-schema tests make the
ownership rule visible because the same-shaped text and image fields produce different rankings.

```sh
cargo xtask test day_01::checkpoint_3
```

## Checkpoint 4: Match and Rewrite One Safe Top-k

Implement `match_vector_order` and
`VectorIndexOptimizer::rewrite_sort`.

The optimizer may replace a scan only when it recognizes one supported distance expression over the configured vector
field, a literal query vector, a compatible metric and direction, a positive `LIMIT`, and the live source snapshot.

Match exactly one physical sort expression: ascending Euclidean `array_distance`/`list_distance` or `cosine_distance`, or
descending dot `inner_product`/`dot_product`. The expression must pair one vector `Column` with one literal. After any
projection, the column must still be the field selected by the attachment. The literal must be finite, match the index
dataset's dimension, and be nonzero for cosine distance.

DataFusion widens the fixed-size `Float32` list to `List<Float64>` for its
distance functions. `match_vector_column` accepts exactly that planner-added
cast, while `scalar_vector` admits only values that preserve their exact
`f32` representation.

The optimizer must also prove that the physical `MemorySourceConfig` still matches the attached table, snapshot, schema,
projection, and unambiguous live provider. Only then may it construct `VectorIndexScanExec`. Filters, multiple sort keys,
non-literal vectors, another vector field, the wrong metric or direction, and invalid literals all keep DataFusion's exact
scan and sort.

Unless ordered output is explicitly enabled for the session, retain the final bounded sort after the index selects its
candidates. The order returned by an index is not automatically SQL order.

```sh
cargo xtask test day_01::checkpoint_4
```

## Checkpoint 5: Search, Fetch, and Preserve ORDER BY

Implement `VectorIndexScanExec::selected_rows` and
`ExecutionPlan::with_fetch`.

Search the selected index for at most `fetch` rows. Every result must resolve through its checked location into the
snapshot; reject one that does not. The supplied lookup scaffolding then reconstructs the requested projection in
index-result order.

When `ordered=true`, the scan may expose its accepted ordering property. The default `ordered=false` path must clear that
property and wrap the scan in `SortExec::new(ordering, scan).with_fetch(Some(k))`. The index chooses candidates;
DataFusion still owns SQL's nearest-first result.

```sh
cargo xtask test day_01::checkpoint_5
```

The SQLLogicTest assembles the same path from an empty session. It creates and fills the simple `points` table and the rich
`documents` table, then attaches indexes to their selected fields. `text_embedding` reaches `VectorIndexScanExec`;
`image_embedding` stays on `DataSourceExec` and returns its own ranking.

## Day 1 Review

Run the Day 1 focused and cumulative gates:

```sh
cargo xtask test day_01
cargo xtask test-through day_01
```

At this point, trace one row through the whole system. `vector_mem_table` places the simple helper data in ordinary Arrow
arrays. An attachment snapshots one selected vector field, maps each index ordinal to a checked batch and row, and uses
that location to project the complete source row. A same-shaped vector field cannot borrow the attachment because its name
and ranking belong to a different field.

Then trace the plan boundary. Unsupported query shapes stay on DataFusion's exact scan and sort. A safe match lets the
index choose candidates, while the default path keeps DataFusion's final ordering. Later approximate indexes reuse this
boundary, but their candidate set can be incomplete before the final sort. The supplied session protects the snapshot by
rejecting changes to an indexed table instead of allowing its attachment to become stale.

IVFFlat implementation, filtered pushdown, joins, general DDL/catalog semantics, persistence, and disk row lookup remain
outside Day 1. The product tour's bridge can hold multiple attachments for eligible in-memory tables and selected fields;
it does not provide persistence, automatic rebuilding, online maintenance, or a general catalog lifecycle.

{{#include copyright.md}}
