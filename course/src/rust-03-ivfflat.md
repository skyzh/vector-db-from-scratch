# Narrow the Search with IVFFlat

> **Day 2**
>
> Complete [Make the SQL Path Reach Your Index Safely](./rust-02-datafusion.md) first. Finish with a seeded IVFFlat
> index behind the same SQL top-k path, recall defined against exact search, and an explicit `probes` tradeoff.

Day 1 left you with an exact `FlatIndex` and a conservative DataFusion path that can use it. The SQL matcher, selected
vector column, checked row lookup, and final `SortExec` are already working. Day 2 changes only how the index
chooses candidates: IVFFlat groups dataset rows into inverted lists, then searches the lists nearest to the query.
It is a coarse quantization index: comparing against a small set of centroids chooses which full-precision vectors to
score.

## Start from the SQL Path You Already Own

The Day 2 SQL case keeps Day 1's table, matcher, lookup, and Euclidean query:

```sql
SELECT id, payload
FROM points
ORDER BY array_distance(embedding, [1.0, 1.0, 1.0])
LIMIT 5;
```

From the repository root, confirm the completed Day 1 case first:

```sh
cargo xtask test day_01
```

Now run the Day 2 case before implementing IVFFlat:

```sh
cargo xtask test day_02::checkpoint_5
```

This second command is your product-level expected failure. It uses the same DataFusion integration with
`IndexConfig::IvfFlat`, then reaches the unfinished IVFFlat constructor. At the end of the day, the same command must
reach this plan and return the five expected rows:

```text
SortExec: TopK(fetch=5), ...
  VectorIndexScanExec: index=ivf_flat, metric=Euclidean, query_dim=3, fetch=Some(5), ordered=false
```

Your work is limited to three functions:

```text
vector-db-starter/core/src/search.rs    recall_at_k
vector-db-starter/core/src/ivf.rs       IvfFlatIndex::try_new
vector-db-starter/core/src/ivf.rs       IvfFlatIndex::search_with_probes
```

The starter already supplies `Dataset`, `Metric`, `TopK`, `DeterministicRng`, the IVFFlat configuration and public index
shell, and the complete Day 1 DataFusion path. Keep those public APIs and the Day 1 tests unchanged.

## Checkpoint 1: Define Recall against Flat Search

Implement `recall_at_k` in `search.rs`. Recall is the fraction of expected top-k row offsets present in the approximate
top-k:

```text
expected = [0, 1, 2]
actual   = [0, 2, 9]
recall@3 = 2 / 3
```

Use row membership, not distance equality or result position. The denominator is the number of exact results available
within `k`, not always `k` itself. If exact search returns two rows for `k = 10`, an approximate index can recover at most
those two rows. Define recall as `1.0` when that denominator is zero; an empty request has missed nothing.

```sh
cargo xtask test day_02::checkpoint_1
```

This function gives the approximate result a correctness meaning. Timing and cross-index comparison remain separate;
the final benchmark will measure all five indexes under one shared workload.

## Checkpoint 2: Validate and Seed the Build

Implement the validation boundary at the start of `IvfFlatIndex::try_new`. The configuration must satisfy
`1 <= probes <= partitions <= rows` with `iterations > 0`. Call `dataset.validate_for_metric(metric)` before training so
cosine builds reject zero-norm rows just as exact search does.

```sh
cargo xtask test day_02::checkpoint_2
```

Once invalid configurations fail before any training work, initialize the centroids. The starter supplies
`DeterministicRng`; use it to shuffle row offsets, then copy the first `partitions` dataset rows. Sampling distinct offsets
avoids beginning with the same row twice.

For a tiny build with six rows and two partitions, the initial state is:

```text
dataset rows:     0 1 2 3 4 5
seeded centroids: two distinct shuffled row offsets
assignments:      unknown until the first assignment pass
```

The selected rows depend on both the seed and how your implementation consumes deterministic randomness. A second build
with the same implementation, data, metric, configuration, and seed must reproduce its centroids, lists, and results;
it does not need to copy the reference implementation's centroid identities.

**Prediction:** If another correct implementation consumes the seeded generator in a different deterministic order,
which properties must still hold even though its centroid row offsets can differ?

Before the index exists, all points belong to one unpartitioned dataset, so an exact query compares its target with every
point.

![Vectors before IVFFlat clustering](./vector-db/04-ivfflat-step1.svg)

K-means begins from the seeded centroids and alternates between assigning vectors to their nearest centroid and moving
each centroid to the mean of its assigned vectors. Each colored region will become one inverted list.

![K-means chooses centroids and their Voronoi regions](./vector-db/04-ivfflat-step2.svg)

## Checkpoint 3: Alternate Assignment and Update

For up to `iterations` rounds:

1. assign every vector to its nearest centroid;
2. stop early if the complete assignment vector did not change;
3. accumulate component-wise sums and counts for each partition; and
4. replace each non-empty centroid with its component-wise mean.

Keep sums in `f64`, as the starter's metric code does for distances. Every nearest-centroid decision uses
`Metric::distance`, including dot and cosine configurations.

```text
repeat up to iterations:
    next_assignments = nearest_centroid(row) for every row
    if next_assignments == assignments:
        stop
    assignments = next_assignments
    recompute each centroid from its assigned rows

rebuild lists once using the final centroids
```

After the last centroid update, assign every vector once more using the final centroids. The result must contain every
dataset row exactly once. An orphaned row is invisible to every query; a duplicate can occupy the result heap twice and
crowd out a distinct row. Both copies use the same immutable vector and metric, so they do not acquire different exact
distances.

![Every vector is assigned to its nearest centroid](./vector-db/04-ivfflat-step3.svg)

The final rebuild matters because the preceding assignment can describe centroid positions from the previous round.
Placing a row in the wrong list does not change exhaustive results when that row still appears exactly once, but it can
reduce recall when a query probes only some lists.

### Recover Empty and Zero-Mean Clusters

An empty cluster has no mean. Re-seed it from the row farthest from its nearest current centroid; do not divide by zero or
silently remove a partition.

Cosine adds a different boundary: nonzero assigned vectors can average to the zero vector. Normalize every nonzero cosine
centroid after the mean. If its norm is zero, replace it with an assigned dataset row, which has already passed Day 1's
nonzero-norm validation.

**Prediction:** The mean of `[1, 0]` and `[-1, 0]` is `[0, 0]`. What would cosine distance do with that centroid if you
kept it? Which already validated row can safely replace it?

Run the deterministic-build and zero-mean cases:

```sh
cargo xtask test day_02::checkpoint_3
```

## Checkpoint 4: Probe Lists at Query Time

Implement `search_with_probes`:

1. validate the query and `1 <= probes <= partitions`;
2. compute one `Neighbor` per centroid and sort centroids nearest-first;
3. visit row offsets from the first `probes` lists;
4. score those dataset rows with the original metric; and
5. feed every candidate into the existing `TopK` and return nearest-first.

Reject an invalid probe count before scanning any list. A request above the partition count is an error, not permission to
visit a uniquely ranked partition more than once.

Centroid assignment, centroid ranking, and candidate scoring must all use the same metric. Mixing metrics produces a
result set ordered by a criterion the probe loop never optimized. Do not return a separate top-k from each list: the SQL
query asks for the best `k` across the union of candidates.

The red vector below probes its nearest centroid's list. It can miss a closer point just across the partition boundary:

![Probing one centroid can miss a nearby point in another list](./vector-db/04-ivfflat-lookup.svg)

Probing the next-nearest list exposes those candidates. Increasing `probes` does more candidate work, but it is less
likely to miss a true neighbor:

![Probing two centroids expands the candidate set](./vector-db/04-ivfflat-lookup-2.svg)

Suppose ranked list IDs are `[2, 0, 1]` and their sizes are `[10, 40, 5]`. With `probes = 1`, search reads the five rows
in list 2. With `probes = 2`, it reads those five plus the ten rows in list 0. `k` controls retained output; `probes`
controls which candidates can enter it.

Approximate and exact recall runs must use identical data, queries, metric, and `k`. Changing any of these between runs
makes the recall number meaningless.

The decisive boundary is to probe every partition. IVFFlat then visits every dataset row and must produce the same complete
ordered result as `FlatIndex`, including tie order:

```sh
cargo xtask test day_02::checkpoint_4
```

If this fails, inspect list completeness, metric choice, heap retention, and final sorting. With every list open,
approximation is no longer an explanation.

## Checkpoint 5: Put IVFFlat behind the Same SQL

Return to the product-level case you ran at the start:

```sh
cargo xtask test day_02::checkpoint_5
```

The SQL text and the matcher you implemented on Day 1 are unchanged. DataFusion passes `LIMIT 5` through Day 1's
`with_fetch`. `VectorIndexScanExec` calls `IvfFlatIndex::search`, which uses the configured `probes`; the generic bounded
sort remains responsible for final SQL ordering. Unsupported query shapes still use the exact `DataSourceExec` path.

The test uses all three partitions, so its five returned rows must match exact search. This is an integration check, not
a claim that a smaller probe count always returns the same rows.

## Checkpoint 6: Run the Day 2 Product Loop

Run the supplied example after the focused core and SQL tests pass:

```sh
cargo run -p vector-db-from-scratch-datafusion-starter --example ivfflat_sql
```

The example issues one cosine top-k over the same five-row table through a Flat attachment and then a seeded IVFFlat
attachment with all partitions probed. Compare the two `EXPLAIN` leaves:

```text
VectorIndexScanExec: index=flat, metric=Cosine, query_dim=3, fetch=Some(3), ordered=false
VectorIndexScanExec: index=ivf_flat, metric=Cosine, query_dim=3, fetch=Some(3), ordered=false
```

Both runs keep DataFusion's final `SortExec` and return the same three rows. The product contract did not change; the
candidate-selection implementation did. Smaller probe counts expose the recall/work tradeoff you reasoned about above,
while Day 6 owns the release-mode latency comparison across all five indexes.

## Day 2 Review

Run the Day 2 focused gate, then the cumulative course through Day 2:

```sh
cargo xtask test day_02
cargo xtask test-through day_02
```

After the five Day 2 core tests, Day 2 SQLLogicTest, and product example pass, choose one concrete build and query
and explain:

- why the configuration is rejected before training;
- why list membership must be rebuilt after the final centroid update;
- how a dataset row flows from assignment to a probed list to `TopK`;
- why empty and zero-mean cosine clusters need different recovery logic;
- why probing every list is an exactness test; and
- how Day 1's optimizer rule reaches a new index without changing its SQL safety contract;
- which same-implementation properties a seed fixes without fixing the reference implementation's centroid identities.

Keep this checkpoint focused on in-memory IVFFlat. Persistent postings, online centroid retraining, product quantization,
cross-index timing, and reproducible latency targets remain outside Day 2.

{{#include copyright.md}}
