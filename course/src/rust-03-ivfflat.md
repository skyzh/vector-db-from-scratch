# Narrow the Search with IVFFlat

> **Day 2**
>
> Complete [Make the SQL Path Reach Your Index Safely](./rust-02-datafusion.md) first. Finish with a seeded IVFFlat
> index behind the same SQL top-k path, recall defined against exact search, and an explicit `probes` tradeoff.

Day 1 left you with an exact `FlatIndex` behind a conservative DataFusion path. The SQL matcher, selected vector column,
checked row lookup, and final `SortExec` are already working. Day 2 leaves that path intact and changes candidate
selection. IVFFlat groups rows into inverted lists around centroids, ranks those centroids for each query, and scores the
full-precision vectors in the selected lists.

## Start from the SQL Path You Already Own

The Day 2 SQL case keeps Day 1's table, matcher, lookup, and Euclidean query:

```sql
SELECT id, payload
FROM points
ORDER BY array_distance(embedding, [1.0, 1.0, 1.0])
LIMIT 5;
```

From the repository root, confirm that your completed Day 1 path is still green:

```sh
cargo xtask test day_01
```

Then run the Day 2 SQL case:

```sh
cargo xtask test day_02::checkpoint_5
```

It uses the existing DataFusion integration with `IndexConfig::IvfFlat` and currently stops at the unfinished IVFFlat
constructor. After the three learner-owned functions are complete, the same command reaches this plan and returns the
five expected rows:

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

The starter supplies `Dataset`, `Metric`, `TopK`, `DeterministicRng`, the IVFFlat configuration and public index shell,
and the complete Day 1 DataFusion path. Keep those public APIs and the Day 1 tests unchanged.

## Checkpoint 1: Define Recall against Flat Search

Implement `recall_at_k` in `search.rs`. Recall measures how many row offsets from the exact top-k also appear in the
approximate top-k:

```text
expected = [0, 1, 2]
actual   = [0, 2, 9]
recall@3 = 2 / 3
```

Compare row membership rather than distance equality or result position, and count each row at most once. The denominator
is the number of exact results available within `k`, which can be smaller than `k`. If exact search returns two rows for
`k = 10`, those two rows form the whole expected set. Define recall as `1.0` when that set is empty.

```sh
cargo xtask test day_02::checkpoint_1
```

This overlap gives the approximate result a correctness measure. Day 6 will handle timing and compare all five indexes
under one shared workload.

## Checkpoint 2: Validate and Seed the Build

Implement the validation boundary at the start of `IvfFlatIndex::try_new`. The configuration must satisfy
`1 <= probes <= partitions <= rows` with `iterations > 0`. Call `dataset.validate_for_metric(metric)` before training so
cosine builds reject zero-norm rows just as exact search does.

```sh
cargo xtask test day_02::checkpoint_2
```

After validation, initialize the centroids. Use the supplied `DeterministicRng` to shuffle the row offsets, then copy the
first `partitions` dataset rows. Each selected offset is distinct, so two centroids never begin from the same row.

For a tiny build with six rows and two partitions, the initial state is:

```text
dataset rows:     0 1 2 3 4 5
seeded centroids: two distinct shuffled row offsets
assignments:      unknown until the first assignment pass
```

The selected rows depend on both the seed and the order in which your implementation consumes the generator. Build the
same index twice and its centroids, lists, and results must match. Another correct implementation can consume the same
seed differently and choose different initial rows, so matching a reference centroid identity is not part of the
contract.

Before the index exists, all points belong to one unpartitioned dataset, so an exact query compares its target with every
point.

![Vectors before IVFFlat clustering](./vector-db/04-ivfflat-step1.svg)

K-means alternates between assigning every vector to its nearest centroid and moving each centroid to the mean of its
assigned vectors. Each colored region will become one inverted list.

![K-means chooses centroids and their Voronoi regions](./vector-db/04-ivfflat-step2.svg)

## Checkpoint 3: Alternate Assignment and Update

Run at most `iterations` rounds. In each round, assign every vector to its nearest centroid using `Metric::distance`. If
the complete assignment vector has not changed, training can stop. Otherwise, accumulate a component-wise sum and row
count for each partition, then replace each non-empty centroid with its mean. Keep the sums in `f64`, as the supplied
metric code does for distances; Euclidean, dot, and cosine builds must use their configured metric throughout.

```text
repeat up to iterations:
    next_assignments = nearest_centroid(row) for every row
    if next_assignments == assignments:
        stop
    assignments = next_assignments
    recompute each centroid from its assigned rows

rebuild lists once using the final centroids
```

After the last centroid update, assign every vector once more using the final centroids. The rebuilt lists must contain
every dataset row exactly once. An omitted row becomes invisible to every query. A duplicate can occupy the result heap
twice and crowd out a distinct row even though both copies have the same exact distance.

![Every vector is assigned to its nearest centroid](./vector-db/04-ivfflat-step3.svg)

The extra assignment matters because the preceding one described the centroid positions before their final update. A row
left in an old list is still found when every partition is probed, but it can be missed by a subset-probe query.

### Recover Empty and Zero-Mean Clusters

An empty cluster has no mean. Re-seed it from the dataset row farthest from its nearest current centroid so the configured
partition count stays intact.

Cosine needs a separate recovery. Nonzero assigned vectors can still average to the zero vector: `[1, 0]` and `[-1, 0]`
are the smallest example. Normalize a nonzero cosine centroid after computing its mean. When the mean has zero norm,
replace it with one of that cluster's assigned rows, whose nonzero norm was already checked by Day 1 validation. Keeping
the zero mean would make the next cosine-distance calculation invalid.

Run the deterministic-build and zero-mean cases:

```sh
cargo xtask test day_02::checkpoint_3
```

## Checkpoint 4: Probe Lists at Query Time

Implement `search_with_probes`. Validate the query and require `1 <= probes <= partitions` before scanning a list. Score
each centroid against the query and sort the resulting `Neighbor` values nearest-first. The first `probes` entries select
the lists to visit; score every row in their union with the original metric, feed it into the supplied `TopK`, and return
the retained neighbors nearest-first. A probe count above the partition count is an error, not permission to revisit a
partition.

Centroid assignment, centroid ranking, and candidate scoring all use the index metric. Mixing them would select lists for
one notion of distance and order their rows by another. Keep one `TopK` across the union of candidates because SQL asks
for the best `k` overall, not a separate result from each list.

The red vector below probes its nearest centroid's list. It can miss a closer point just across the partition boundary:

![Probing one centroid can miss a nearby point in another list](./vector-db/04-ivfflat-lookup.svg)

Probing the next-nearest list exposes more candidates. Increasing `probes` does more candidate work and makes a true
neighbor less likely to be missed:

![Probing two centroids expands the candidate set](./vector-db/04-ivfflat-lookup-2.svg)

Suppose the list sizes by ID are `[10, 40, 5]`, while the query ranks the IDs as `[2, 0, 1]`. With `probes = 1`, search
reads the five rows in list 2. With `probes = 2`, it also reads the ten rows in list 0. `probes` controls which candidates
can enter the heap; `k` controls how many of them remain in the result.

Measure recall against exact search with the same data, query, metric, and `k`, so candidate selection is the only changing
variable.

Probe every partition for the exactness boundary. IVFFlat then visits every dataset row and must produce the same complete
ordered result as `FlatIndex`, including tie order:

```sh
cargo xtask test day_02::checkpoint_4
```

If this case fails, inspect list completeness, metric choice, heap retention, and final sorting. Every row was available,
so subset probing cannot explain a difference.

## Checkpoint 5: Put IVFFlat behind the Same SQL

Return to the product-level case you ran at the start:

```sh
cargo xtask test day_02::checkpoint_5
```

The SQL text and Day 1 matcher are unchanged. DataFusion passes `LIMIT 5` through `with_fetch`, and
`VectorIndexScanExec` calls `IvfFlatIndex::search` with the configured `probes`. The generic bounded sort still produces
the final SQL order. Unsupported query shapes remain on the exact `DataSourceExec` path.

The test probes all three partitions, so its five returned rows match exact search. A smaller probe count may select a
different candidate set.

## Checkpoint 6: Run the Day 2 Product Loop

Run the supplied example after the focused core and SQL tests pass:

```sh
cargo run -p vector-db-from-scratch-datafusion-starter --example ivfflat_sql
```

The example runs one cosine top-k over the same five-row table, first through a Flat attachment and then through a seeded
IVFFlat attachment with all partitions probed. Compare the two `EXPLAIN` leaves:

```text
VectorIndexScanExec: index=flat, metric=Cosine, query_dim=3, fetch=Some(3), ordered=false
VectorIndexScanExec: index=ivf_flat, metric=Cosine, query_dim=3, fetch=Some(3), ordered=false
```

Both runs keep DataFusion's final `SortExec` and return the same three rows. Only candidate selection changed. Smaller
probe counts expose the recall/work tradeoff; Day 6 owns the release-mode latency comparison across all five indexes.

## Day 2 Review

Run the Day 2 focused gate, then the cumulative course through Day 2:

```sh
cargo xtask test day_02
cargo xtask test-through day_02
```

Choose one concrete build and query, then trace it from validation through the SQL result. Explain how the seed initializes
that build, why list membership is rebuilt after the final centroid update, and how one dataset row moves from assignment
to a probed list and into `TopK`. Account separately for an empty cluster and a zero-mean cosine cluster.

Finally, connect the core index to Day 1: the optimizer's safety rule is unchanged, the index supplies candidates, and the
final sort still belongs to DataFusion. Probing every list should recover Flat search, while subset probing may trade
recall for less candidate work. Persistent postings, online centroid retraining, product quantization, cross-index timing,
and reproducible latency targets remain for later work.

{{#include copyright.md}}
