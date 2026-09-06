# Try the Vector Database from SQL

Before you implement the table adapter or optimizer, use the supplied system once. You will create and populate an
ordinary in-memory table, run one nearest-neighbor query, attach an index to its vector column, and see the physical plan
change while the SQL result stays the same.

This tour uses the completed `vector-db-from-scratch-datafusion` example. You do not need to read or modify its source. Your own work begins
on Day 1.

## Launch the Supplied Shell

For an interactive run, start from the repository root:

```sh
cargo run -p vector-db-from-scratch-datafusion --example sql
```

The supplied DataFusion CLI starts with an empty course session and accepts semicolon-terminated SQL, including statements
that span multiple lines. For a repeatable first run from the repository root, paste the whole transcript below into your
terminal instead of entering the statements interactively:

```sh
cargo run -p vector-db-from-scratch-datafusion --example sql <<'SQL'
CREATE TABLE points (id BIGINT NOT NULL, payload VARCHAR NOT NULL, embedding REAL[3] NOT NULL);
INSERT INTO points VALUES (1, 'one', [1.0, 0.0, 0.0]), (2, 'two', [0.9, 0.1, 0.0]), (3, 'three', [0.0, 1.0, 0.0]), (4, 'four', [-1.0, 0.0, 0.0]), (5, 'five', [0.0, 0.0, 1.0]);
EXPLAIN SELECT id, payload FROM points ORDER BY cosine_distance(embedding, [1.0, 0.0, 0.0]) LIMIT 3;
SELECT id, payload FROM points ORDER BY cosine_distance(embedding, [1.0, 0.0, 0.0]) LIMIT 3;
CREATE INDEX points_embedding_idx ON points USING ivfflat (embedding);
EXPLAIN SELECT id, payload FROM points ORDER BY cosine_distance(embedding, [1.0, 0.0, 0.0]) LIMIT 3;
SELECT id, payload FROM points ORDER BY cosine_distance(embedding, [1.0, 0.0, 0.0]) LIMIT 3;
SQL
```

Before you run it, predict which rows should be nearest and why creating an index must not change them.

## Observe the Stable Query and Changing Plan

Before the index exists, DataFusion reads the ordinary in-memory table:

```text
SortExec: TopK(fetch=3), ...
  DataSourceExec: partitions=1, ...
```

The first query returns these rows:

```text
1  one
2  two
3  three
```

**Prediction:** The next command attaches an index, but the following `SELECT` is byte-for-byte identical. Which physical
plan leaf should change, and which three rows must not?

The `CREATE INDEX` statement builds the session's cosine IVFFlat index and attaches it to the vector column you selected.
The second `EXPLAIN` reaches the course-owned scan:

```text
SortExec: TopK(fetch=3), ...
  VectorIndexScanExec: index=ivf_flat, metric=Cosine, query_dim=3, fetch=Some(3), ordered=false
```

The second query is byte-for-byte the same SQL and returns the same three rows. The index changes how candidates reach
DataFusion's final sort; it does not change the query contract.

## Know What This Command Means

DataFusion parses and logically plans `CREATE INDEX`, but the pinned version does not provide a physical executor that can
build this course's index. The supplied shell therefore owns a bounded bridge from that statement to the course's existing
attachment path. The session is configured for cosine IVFFlat, while the statement supplies the index name, resolved table,
and selected column:

```sql
CREATE INDEX points_embedding_idx ON points USING ivfflat (embedding)
```

The name may be any unused index name, and the table may be bare or schema/catalog qualified. The bridge can attach indexes
to multiple distinct table/column pairs in one session. Each target must be a registered in-memory `MemTable`, and its
selected column must be a non-null `REAL[N]` vector with positive width. Duplicate names or attachments, missing tables or
columns, providers other than `MemTable`, nullable fields, vector fields with the wrong physical type or zero width, and an
index kind different from the session configuration are rejected before an attachment is installed.

**Prediction:** Suppose the session also contains another eligible table with a different vector field. Which table,
column, and index names must the bridge resolve from the SQL statement rather than hard-code from this `points` example?

An attachment is an immutable snapshot. After a table is indexed, `INSERT`, `ALTER TABLE`, and `DROP TABLE` against that
table are rejected instead of making the index stale. Writes to unrelated tables remain legal, as does `INSERT ... SELECT`
that reads indexed data into another table. The bridge does not add index persistence, `DROP INDEX`, automatic rebuilding,
or a general catalog lifecycle.

**Prediction:** Why must a later `INSERT` into the indexed table be rejected unless the table update and a rebuilt index
can become visible atomically?

That narrow boundary keeps the first experience concrete without turning the course into a parser or catalog project.
Next, [Day 1](./rust-02-datafusion.md) opens the path you just used: you will build the Arrow table, attach one selected
vector field, and make the optimizer choose `VectorIndexScanExec` only when doing so is safe.

{{#include copyright.md}}
