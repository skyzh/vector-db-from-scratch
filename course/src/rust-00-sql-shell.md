# Try the Vector Database from SQL

Start by running the supplied system. This gives you the complete SQL path before Day 1 asks you to build it: an ordinary
in-memory table answers an exact nearest-neighbor query, then an IVFFlat index changes how the database finds candidates.
You will compare the plans and results from both runs.

The tour uses the completed `vector-db-from-scratch-datafusion` example. Leave its source as it is for now; your own work
begins on Day 1.

## Launch the Supplied Shell

From the repository root, launch an interactive session with:

```sh
cargo run -p vector-db-from-scratch-datafusion --example sql
```

Each session starts empty. The supplied DataFusion CLI accepts semicolon-terminated SQL, including statements that span
multiple lines. For a repeatable first run, paste this entire transcript into your terminal:

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

Before you run it, predict which rows the exact scan should rank nearest. Afterward, compare that answer with the indexed
result and keep the approximate-search boundary in mind: the neighbors or their order may differ.

## Watch the Scan Change

The first `EXPLAIN` shows DataFusion reading the ordinary in-memory table:

```text
SortExec: TopK(fetch=3), ...
  DataSourceExec: partitions=1, ...
```

The exact query returns:

```text
1  one
2  two
3  three
```

The next command attaches an index. Although the following `SELECT` is byte-for-byte identical, its physical plan now
reaches the course-owned scan:

```text
SortExec: TopK(fetch=3), ...
  VectorIndexScanExec: index=ivf_flat, metric=Cosine, query_dim=3, fetch=Some(3), ordered=false
```

When the second `SELECT` executes this plan, the shell confirms the choice on standard error:

```text
Vector index selected: index=ivf_flat, metric=Cosine, query_dim=3, fetch=3, ordered=false
```

In this run, the indexed query returns rows 1, 2, and 3 in the same order as the exact scan. Treat that as one observation,
not an IVFFlat guarantee. The index retrieves an approximate candidate set, so membership and ordering may change.
DataFusion then applies the final sort to the candidates it received, using the same cosine distance and `LIMIT 3` from
the SQL.

## What `CREATE INDEX` Does Here

DataFusion parses and logically plans `CREATE INDEX`, but the pinned version does not provide a physical executor that can
build this course's index. The supplied shell handles that statement through a small bridge to the course's existing
attachment path. The session is configured for cosine IVFFlat; the statement supplies the index name, table, and vector
column:

```sql
CREATE INDEX points_embedding_idx ON points USING ivfflat (embedding)
```

The name may be any unused index name, and the table may be bare or schema/catalog qualified. A single session can attach
indexes to several table and column pairs because the bridge resolves the target from each SQL statement; it does not
hard-code the `points` example. Each table must be a registered in-memory `MemTable`, and the selected column must be a
non-null `REAL[N]` vector with positive width. The shell rejects duplicate names or attachments, missing tables or columns,
other provider types, nullable fields, incompatible vector fields, and an index kind that differs from the session
configuration.

An attachment is an immutable snapshot. After a table is indexed, `INSERT`, `ALTER TABLE`, and `DROP TABLE` against that
table are rejected instead of making the index stale. Writes to unrelated tables remain legal, as does `INSERT ... SELECT`
that reads indexed data into another table. A later insert would leave the snapshot behind, so the shell rejects it until
the table update and a rebuilt index could become visible together. Index persistence, `DROP INDEX`, automatic rebuilding,
and a general catalog lifecycle are outside this bridge.

Next, [Day 1](./rust-02-datafusion.md) opens the path you just ran. You will build the Arrow table, attach one vector field,
and make the optimizer choose `VectorIndexScanExec` only for a safe match.

{{#include copyright.md}}
