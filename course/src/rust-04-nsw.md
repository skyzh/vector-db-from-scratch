# Navigate a Proximity Graph with NSW

> **Day 3**
>
> Complete [Narrow the Search with IVFFlat](./rust-03-ivfflat.md) first. You will replace centroid/list selection with graph
> reachability while keeping the SQL matcher, row lookup, and final top-k sort supplied.

## Move from Lists to a Graph

Day 2 ended with a five-row cosine query running through IVFFlat. From the repository root, run that product path once
more:

```sh
cargo run -p vector-db-from-scratch-datafusion-starter --example ivfflat_sql
```

The seeded IVFFlat plan contains `index=ivf_flat`, and its `LIMIT 3` result is:

```text
(1, one)
(2, two)
(3, three)
```

Today only the source of candidate row offsets changes. IVFFlat opens selected centroid lists; navigable small world
(NSW) search follows edges between nearby vectors. The table, query, matcher, row lookup, and final `SortExec` stay put.

The cumulative starter already contains the two files you will change:

```text
vector-db-starter/core/src/graph.rs
vector-db-starter/core/src/nsw.rs
```

Four TODOs form one path through the index: `search_layer` explores a graph, `prune_neighbors` bounds its degree,
`NswIndex::try_new` inserts the stored rows, and `NswIndex::search_with_ef` queries the result. Leave the starter's
`greedy_search`, HNSW, and IVF-PQ TODOs for later days. The crate-internal tests can exercise graph helpers without
making them public.

## Checkpoint 1: Search One Supplied Layer

An NSW graph has no centroid that points directly at the query. Search starts at one or more supplied entry points and
discovers only vertices connected to them.

![One entry point begins the NSW walk](./vector-db/05-nsw-explore-1.svg)

![A greedy step moves to a closer neighbor](./vector-db/05-nsw-explore-2.svg)

The walk needs three pieces of state. `C` is nearest-first, so its next item is the vertex to expand. `W` is bounded and
worst-first, so its top item is the first result to evict when a closer row arrives. `visited` ensures that each row is
measured and expanded at most once. Seed all three from the valid, unique entry points.

![Seed the candidate and result frontiers](./vector-db/05-nsw-explore-3.svg)

Here is a concrete trace. Suppose rows 0, 1, and 2 store the one-dimensional values 0, 1, and 2, with edges `0—1—2`.
Rows 3 and 4 form a separate component. For query 0, entry row 2, and width 3, the search first retains row 2 at
distance 2. Expanding row 2 discovers row 1 at distance 1; expanding row 1 then discovers row 0 at distance 0. `W`
finally returns rows 0, 1, and 2 in that order.

![Expand the nearest candidate and update both frontiers](./vector-db/05-nsw-explore-4.svg)

Revisiting row 2 through row 1 does nothing because it is already in `visited`. An expansion that adds nothing does not
end the whole search; another pending candidate may still open a useful path.

![Visited neighbors are not expanded twice](./vector-db/05-nsw-explore-5.svg)

No choice of width can make that entry at row 2 reach rows 3 and 4. A second entry point or an edge into their component
is required.

![A second entry point opens another region](./vector-db/05-nsw-explore-6.svg)

Whenever a closer row arrives, keep only the nearest `ef` rows in `W`.

![The result frontier keeps the nearest visited vertices](./vector-db/05-nsw-explore-7.svg)

Once `W` is full, stop only when the nearest pending candidate is **strictly worse** than `W.worst`.

![The nearest pending candidate is worse than the full result frontier](./vector-db/05-nsw-explore-8.svg)

A candidate equal to `W.worst` in public `(distance, row)` order must still be expanded because it may lead somewhere
better. Even this strict rule is approximate: a worse intermediate vertex can hide a path to a closer one.

```text
C = valid unique entry points as a min-heap by distance
W = the same points as a bounded max-heap by distance
visited = the same row offsets

while C is not empty:
    candidate = C.pop_nearest()
    if W is full and candidate is strictly worse than W.worst:
        break

    for neighbor in candidate.neighbors:
        if neighbor is outside allowed_rows or already visited:
            continue
        mark neighbor visited
        measure its distance once
        if W is not full or neighbor is better than W.worst:
            C.push(neighbor)
            W.push(neighbor)
            trim W to the search width

return W from nearest to farthest
```

Implement `search_layer` in `graph.rs`. Clamp `ef` to at least one and at most `allowed_rows`, and return no rows when
none are allowed. Ignore duplicate or out-of-range entry points. During insertion, `allowed_rows = r` means that only
the earlier rows `0..r` exist.

```sh
cargo xtask test day_03::checkpoint_1
```

Before the implementation, this command reaches the traversal TODO. Afterward it covers the trace above, row bounds,
disconnected components, duplicate and invalid entries, nearest-first uniqueness, and strict stopping.

## Checkpoint 2: Keep a Bounded Neighbor List

Rows enter the graph one at a time. Before row `r` can connect, search the graph of earlier rows with width
`ef_construction`, then select at most `max_connections` of the nearest candidates.

![Choose the new vector's nearest graph neighbors](./vector-db/05-nsw-insert-1.svg)

Each connection is reciprocal, so adding a new row can push an older endpoint past the degree cap.

![New reciprocal edges can exceed the degree cap](./vector-db/05-nsw-insert-2.svg)

Implement `prune_neighbors` in `graph.rs`. Deduplicate the supplied row offsets, order them by distance from the owner,
break distance ties by row offset, and truncate to `max_connections`. In the focused fixture, owner row 0 sees candidate
rows `[2, 1, 1, 3]`; rows 1 and 2 are equally distant, so a cap of two keeps `[1, 2]`.

![Choose the connections that survive pruning](./vector-db/05-nsw-insert-3.svg)

```sh
cargo xtask test day_03::checkpoint_2
```

The fixture is already self-free and isolates deduplication, ordering, tie-breaking, and the cap. The graph builder owns
the separate rule that a row never appears in its own adjacency list.

## Checkpoint 3: Build a Reciprocal Graph

Implement `NswIndex::try_new` in `nsw.rs`. Validate the stored vectors for the selected metric, then reject a graph
budget unless `max_connections > 0`, `ef_construction >= max_connections`, and `ef_search > 0`.

The first row becomes the initial entry point without a search. For every later row `r`, call `search_layer` with
`allowed_rows = r`, connect `r` to the nearest selected candidates in both directions, prune `r` and the older endpoints,
then make `r` the entry point for the next insertion. For example, if row 4 connects to rows 1 and 3, first add `4—1`
and `4—3`. If pruning row 1 then rejects row 4, remove the reverse `4 -> 1` edge as well.

![Pruning leaves a bounded reciprocal graph](./vector-db/05-nsw-insert-4.svg)

The finished graph must be deterministic, duplicate-free, self-free, reciprocal, and within the degree cap:

```sh
cargo xtask test day_03::checkpoint_3
```

This checkpoint covers stored-vector and configuration validation without calling `search_with_ef`, so construction
failures remain local.

## Checkpoint 4: Query with a Width Budget

Implement `NswIndex::search_with_ef`. Validate the query dimension, finite values, and selected metric, and reject a zero
search width. Start from the graph entry point and call `search_layer` with width `ef_search.max(k)`. Return at most `k`
neighbors, nearest-first.

The `.max(k)` floor keeps the result request separate from the exploration hint. A request for five rows with
`ef_search = 1` still needs room for five retained results. More width can expose more of the connected graph, but it
cannot cross a missing edge.

```sh
cargo xtask test day_03::checkpoint_4
```

The connected high-width fixture matches `FlatIndex`. Treat that as one observed result: NSW is not generally exact for
arbitrary data, widths, or disconnected graphs.

## Return to the Same SQL Product

With all four TODOs complete, run the supplied comparison:

```sh
cargo run -p vector-db-from-scratch-datafusion-starter --example nsw_sql
```

It executes the same five-vector cosine query twice. The first plan contains:

```text
VectorIndexScanExec: index=ivf_flat, metric=Cosine, query_dim=3, fetch=Some(3), ordered=false
```

The second contains:

```text
VectorIndexScanExec: index=nsw, metric=Cosine, query_dim=3, fetch=Some(3), ordered=false
```

Both retain the supplied `SortExec` and show the same three rows:

```text
(1, one)
(2, two)
(3, three)
```

Only the candidate route changed. The attachment chooses the index, the matcher recognizes the supported top-k shape,
row lookup resolves the returned offsets, and DataFusion performs the final sort. Equal rows in this example do not
establish general recall, work, or performance.

The Checkpoint 4 command also runs a separate SQLLogicTest with eight rows and `LIMIT 5`. It verifies the `index=nsw`
plan leaf, the supplied final sort, and its own five expected rows. Unsupported SQL shapes continue to use the supplied
exact `DataSourceExec` path.

## Finish Day 3

Run the focused Day 3 gate and then the cumulative course through Day 3:

```sh
cargo xtask test day_03
cargo xtask test-through day_03
```

The insertion and query traces now meet at the same graph boundary: opposite heap orderings choose what expands and what
survives; strict stopping bounds exploration; reciprocal pruning keeps both endpoints consistent; `ef_search.max(k)`
leaves room for the requested result; and connectivity decides which rows can be reached at all.

Day 3 deliberately builds one immutable graph layer. Hierarchy arrives with HNSW on Day 4. Deletion, concurrent
mutation, persistence, filtering pushdown, general DDL/catalog behavior, benchmarking, and neighbor diversification are
separate problems and do not change this checkpoint.

{{#include copyright.md}}
