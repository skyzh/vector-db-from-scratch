# Add Hierarchy with HNSW

> **Day 4**
>
> Complete [Navigate a Proximity Graph with NSW](./rust-04-nsw.md) first. You will turn that one-layer graph into a
> seeded hierarchy, route through its sparse upper layers, and run the same SQL top-k through `index=hnsw`.

## Start from the NSW Product

Day 3 ended with IVFFlat and NSW proposing candidates for the same five-row cosine query. Run that comparison again
from the repository root:

```sh
cargo run -p vector-db-from-scratch-datafusion-starter --example nsw_sql
```

The second plan contains `index=nsw`, and both indexes return:

```text
(1, one)
(2, two)
(3, three)
```

Keep that result in view while you build HNSW. The SQL matcher, source-row lookup, and final `SortExec` will not change.
Only the route that proposes candidate row offsets changes: NSW starts in one graph containing every row, while HNSW
makes coarse moves through sparse upper layers before searching the all-row graph at layer zero. The finished SQL plan
will say `index=hnsw`; DataFusion will still own the final ordering of the returned rows.

The cumulative starter leaves three Day 4 units unfinished:

```text
vector-db-starter/core/src/graph.rs        greedy_search
vector-db-starter/core/src/hnsw.rs         HnswIndex::try_new
vector-db-starter/core/src/hnsw.rs         HnswIndex::search_with_ef
```

Day 3 already supplied `search_layer`, `prune_neighbors`, deterministic `(distance, row)` ordering, and the DataFusion
boundary. `try_new` is one build operation: assign each row a level and connect that row before moving to the next one.

## Checkpoint 1: Route Through One Upper Layer

Layer zero contains every vector. A row promoted to level `L` also belongs to every layer below `L`, so each higher
layer is a smaller set of possible waypoints.

![Sparse HNSW layers route into the complete layer-zero graph](./vector-db/06-hnsw-architecture.svg)

A query enters at the global entry point in the highest layer. Within an upper layer, `greedy_search` looks at the
allowed neighbors and moves only when the best one strictly improves the public `(distance, row)` order. Distance ties
therefore prefer the lower row offset. Because every accepted move improves the total order, the walk must stop.

![HNSW descends through progressively denser layers](./vector-db/06-hnsw-explore.svg)

```text
current = distance(query, entry)
loop:
    next = minimum allowed neighbor by (distance, row)
    if next is strictly better than current:
        current = next
    else:
        return current.row
```

Implement `greedy_search` in `graph.rs`. During construction of row `r`, `allowed_rows = r` keeps the walk inside rows
that already exist. During a query, all stored rows are allowed. This helper returns one handoff row; it is not the
bounded multi-candidate search that produces the final top-k.

Run the focused helper test:

```sh
cargo xtask test day_04::checkpoint_1
```

The fixture starts at row 2. Row 1 wins an equal-distance tie by row offset. A closer row 3 is first outside
`allowed_rows`, then becomes reachable when the bound grows. Returning the starting row unconditionally, ignoring the
bound, or comparing distance without the row tie-break all fail here.

## Checkpoint 2: Build the Seeded Nested Graph

Implement `HnswIndex::try_new` in `hnsw.rs`. Begin by validating the stored vectors for the selected metric. The graph
budget is invalid when `max_connections` is zero, `ef_construction` is smaller than `max_connections`, `ef_search` is
zero, or `max_level` is zero.

For each dataset row, the supplied deterministic generator flips a seeded coin until the first failure or
`max_level`. A sampled level of one places the row in layers one and zero, not layer two. Rebuilding with the same
implementation and seed must reproduce the same levels and graph. A different valid implementation may consume random
values in another order, so the tests check repeatability and invariants rather than a reference level prefix.

![A new vector is promoted to level one and every lower layer](./vector-db/06-hnsw-insert-1.svg)

Each stored layer has one adjacency slot per dataset row. Extend the existing layers when a row arrives and create any
missing layers through its sampled level. A row outside a layer keeps an empty slot there. This makes membership visible
from both `levels[r]` and the layer storage, and it keeps upper-layer membership nested.

The first row needs no search. Put it in all of its included layers and make it the global entry point. Every later row
starts from that entry. Greedily cross layers above the new row's own level; then, in each layer the new row shares with
the existing graph, use Day 3's `search_layer` to choose nearby earlier rows. Add reciprocal edges, prune both endpoints
to `max_connections`, and remove the reverse edge whenever pruning rejects one direction.

![Search each included layer before connecting the new vector](./vector-db/06-hnsw-insert-2.svg)

```text
target_level = seeded_geometric_level()
entry = top entry point

for level above target_level, from highest down:
    entry = greedy_search(layer[level], new_vector, entry)

for shared level from min(highest, target_level) down to 0:
    candidates = search_layer(layer[level], new_vector, [entry], ef_construction)
    connect the nearest max_connections candidates in both directions
    prune every affected endpoint and remove rejected reciprocal edges
    entry = nearest candidate, when one exists

if target_level is above the previous highest level:
    make the new row the global entry point
```

Here is one concrete route. Suppose row 0 stores `[0]` at level two, row 1 stores `[4]` at level zero, and row 2 stores
`[8]` at level one, with the eligible rows connected in their shared layers. Now row 3, storing `[7]`, is promoted to
level one. It begins at row 0. Layer two has no better waypoint, so row 0 descends into layer one; there, the bounded
search reaches row 2 and connects the new row to that nearer candidate before construction continues at layer zero.
Because level one is not above the old top level, row 0 remains the global entry point. Later, a query for `[7.2]`
starts at row 0 in layer two, moves from row 0 toward row 2 in layer one, and hands row 2 to the wider layer-zero search.
The hierarchy shortened the route to a useful region; layer zero still decides the returned candidate set.

The finished graph stores dataset ordinals, not source row IDs. The supplied DataFusion adapter performs that mapping
after search. Within every layer, adjacency must stay degree-bounded, duplicate-free, self-free, and reciprocal. Update
the global entry point only when the new row creates a new top layer.

Run the construction test:

```sh
cargo xtask test day_04::checkpoint_2
```

It exercises invalid budgets and metric data, repeated seeded builds, nested membership, the degree cap, and reciprocal
edge cleanup. It also rejects injected self-edges; Checkpoint 3 is the first supplied gate that requires a positive
promoted level. The test deliberately permits any level sequence produced repeatably by a valid implementation.

## Checkpoint 3: Search from the Top Layer

Implement `HnswIndex::search_with_ef`. Validate the query dimension, finite values, and selected metric before routing,
and reject an explicit search width of zero.

Begin at the global entry point. Run `greedy_search` once in each upper layer, carrying its single returned row down to
the next layer. At layer zero, switch back to Day 3's `search_layer` with width `ef_search.max(k)`, then keep at most the
nearest `k` results.

```text
entry = top entry point
for level from highest down to 1:
    entry = greedy_search(layer[level], query, entry)

candidates = search_layer(
    layer[0],
    query,
    entry_points=[entry],
    width=max(k, ef_search),
)
return nearest k candidates
```

The `.max(k)` floor separates result count from exploration width. Asking for five rows with `ef_search = 1` still
requires room to retain five candidates.

Run the Checkpoint 3 gate:

```sh
cargo xtask test day_04::checkpoint_3
```

The gate covers query validation, zero width, nearest-first ordering, the width floor, upper-layer descent, and the
supplied DataFusion paths. On one connected fixture, a high-width HNSW search matches `FlatIndex`. That is a bounded
observation. This nearest-neighbor pruning rule can leave layer zero disconnected, and increasing the width cannot cross
an absent edge, so the course makes no general exactness, connectivity, recall, or performance claim.

## Return to the SQL Product

The Checkpoint 3 gate also runs a self-contained SQLLogicTest. It creates and populates its own table, records the exact
scan plan, attaches an HNSW index, and checks the new plan and rows. The indexed plan contains:

```text
VectorIndexScanExec: index=hnsw, metric=Euclidean, query_dim=3, fetch=Some(5), ordered=false
```

Its five-row Euclidean query returns:

```text
1 point-1
0 point-0
2 point-2
3 point-3
4 point-4
```

HNSW returns core dataset ordinals. The supplied adapter resolves them to snapshot source rows, and the supplied
`SortExec` performs the final SQL ordering. Unsupported query shapes continue through the exact scan path. These five
rows demonstrate the product handoff only; they do not turn the small fixture into a general recall or speed result.

## Finish Day 4

Run the focused Day 4 gate and then the cumulative course through Day 4:

```sh
cargo xtask test day_04
cargo xtask test-through day_04
```

The runner selects tests only through HNSW, so unfinished Day 5 IVF-PQ work stays outside this gate. At this point the
three implementations form one path: a strict greedy walk chooses each upper-layer handoff, seeded insertion builds
nested reciprocal layers from prior rows, and bounded layer-zero search returns the candidates that DataFusion maps and
sorts.

This course index remains immutable and in memory. Deletion, concurrent mutation, persistence, production neighbor
diversification, and adaptive search budgets would require a different learner contract; they are not hidden parts of
Day 4.

{{#include copyright.md}}
