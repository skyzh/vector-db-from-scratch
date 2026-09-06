# Add Hierarchy with HNSW

> **Day 4**
>
> Complete [Navigate a Proximity Graph with NSW](./rust-04-nsw.md) first. You will turn that one-layer graph into a
> seeded hierarchy, route through its sparse upper layers, and run the same SQL top-k through `index=hnsw`.

## Start from the NSW Product

Day 3 ended with one five-row table and one cosine-distance query running through IVFFlat and NSW. From the
repository root, run that supplied comparison again:

```sh
cargo run -p vector-db-from-scratch-datafusion-starter --example nsw_sql
```

The second plan contains `index=nsw`, and both indexes return:

```text
(1, one)
(2, two)
(3, three)
```

Day 4 keeps the SQL matcher, source-row lookup, and final `SortExec` fixed. You will change the core candidate
path: instead of starting every query in one graph that contains every row, HNSW first makes coarse moves through sparse
upper layers and then reuses Day 3's bounded search in the all-row layer-zero graph.

**Prediction:** In a before-and-after SQL comparison whose table and query are unchanged, which plan field should change?
Why must the returned rows still satisfy the same SQL ordering contract even though the route proposing them changes?

The cumulative starter leaves exactly three Day 4 units unfinished:

```text
vector-db-starter/core/src/graph.rs        greedy_search
vector-db-starter/core/src/hnsw.rs         HnswIndex::try_new
vector-db-starter/core/src/hnsw.rs         HnswIndex::search_with_ef
```

Day 3 already supplied `search_layer`, `prune_neighbors`, deterministic metric ordering, and the DataFusion boundary.
`try_new` is one complete build operation: level assignment and graph construction share the same insertion loop, so you
will implement and check them together.

## Checkpoint 1: Route Through One Upper Layer

Layer zero contains every vector. Each higher layer contains a progressively smaller subset, and a row promoted to level
`L` belongs to every layer from zero through `L`.

![Sparse HNSW layers route into the complete layer-zero graph](./vector-db/06-hnsw-architecture.svg)

A query begins at the global entry point in the highest layer. Within one upper layer, `greedy_search` repeatedly moves
to the best allowed neighbor only when that neighbor strictly improves the public `(distance, row)` order. Equal
geometric distance can therefore move to a lower row offset, but every move still decreases the total order and the walk
terminates.

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

Implement `greedy_search` in `graph.rs`. Respect `allowed_rows`: during construction, row `r` may route only through rows
`0..r`; during a query, every stored row is allowed. Do not turn this into a beam search. Upper layers choose one coarse
handoff, while layer zero will retain multiple candidates for top-k output.

Run the focused helper test:

```sh
cargo test -p vector-db-from-scratch-core-starter --lib \
  graph_tests::day_04_greedy_search_moves_on_public_tie_order_and_respects_bounds -- --exact
```

Its fixture begins at row 2. An equal-distance row 1 wins by row offset, while a closer row 3 is first excluded and then
admitted by changing `allowed_rows`. A no-op walk or a distance-only tie comparison fails at this checkpoint.

## Checkpoint 2: Build the Seeded Nested Graph

Implement the complete `HnswIndex::try_new` unit in `hnsw.rs`. Validate stored vectors for the selected metric and reject
an invalid graph budget:

- `max_connections` must be greater than zero;
- `ef_construction` must be at least `max_connections`;
- `ef_search` must be greater than zero; and
- `max_level` must be greater than zero.

For each dataset row, use the supplied deterministic generator to flip a seeded coin until the first failure or
`max_level`. A sampled level of one places the row in layers one and zero, but not layer two.

![A new vector is promoted to level one and every lower layer](./vector-db/06-hnsw-insert-1.svg)

As each row arrives, extend the adjacency storage of every existing layer and create missing layers through the sampled
level. Rows that do not belong to a layer keep an empty adjacency list there. That shape makes these two facts directly
inspectable:

- `levels[r]` is the highest layer containing row `r`; and
- membership is nested: appearing in an upper layer requires appearing in every lower layer.

The first row needs no search. Store it in every included layer and make it the entry point. For each later row, start
from the current global entry point. Greedily descend through layers above the new row's sampled level. At every layer the
new row joins, reuse Day 3's `search_layer` with `ef_construction`, connect the nearest
`max_connections` candidates, and prune reciprocal edges back to the cap.

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

Every layer must remain deterministic, degree-bounded, duplicate-free, self-free, and reciprocal. Keep core row values as
dataset ordinals; the supplied DataFusion adapter maps those ordinals through its snapshot row-ID boundary later.

Run the construction test:

```sh
cargo test -p vector-db-from-scratch-core-starter --test indexes \
  day_04_hnsw_rejects_invalid_configuration_and_builds_seeded_nested_layers -- --exact
```

It checks invalid budgets, same-implementation repeatability, nested membership, degree caps, and the absence of duplicate
or self-edges. It also checks every retained edge at both endpoints. A correct deterministic implementation may consume
randomness differently from the reference and therefore produce another valid level sequence and top layer; the tests do
not require the reference implementation's prefix. Forcing every sampled level to zero or injecting a self-edge still
fails here rather than being hidden by a later recall result.

**Prediction:** If a deterministic level sampler changes its RNG consumption order, which graph invariants and repeated
build observations must remain true even though the sampled level prefix may change?

## Checkpoint 3: Search from the Top Layer

Implement `HnswIndex::search_with_ef`. Validate the query for dimension, finite values, and the selected metric, and
reject a zero explicit search width.

Begin at the stored global entry point. Call `greedy_search` once per upper layer, from the top layer down through layer
one, carrying the returned row into the next layer. At layer zero, call Day 3's `search_layer` with width
`ef_search.max(k)`, then truncate the nearest-first result to `k`.

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

The `.max(k)` floor separates the requested result count from the exploration hint. A caller asking for five rows with
`ef_search = 1` still needs a result frontier capable of holding five rows.

Run the query test:

```sh
cargo test -p vector-db-from-scratch-core-starter --test indexes \
  day_04_hnsw_search_validates_widens_and_recovers_neighbors -- --exact
```

It checks query validation, zero width, result ordering, the `ef_search.max(k)` floor, and one connected high-width
fixture. Matching `FlatIndex` on that fixture is a bounded observation, not a promise that HNSW is exact for arbitrary
datasets or search budgets.

## Return to the SQL Product

First confirm that the unchanged Day 1 adapter can select the completed HNSW index:

```sh
cargo test -p vector-db-from-scratch-datafusion-starter --test sql day_04_hnsw_is_visible_in_explain
```

Then run the self-contained Day 4 SQLLogicTest:

```sh
cargo test -p vector-db-from-scratch-datafusion-starter --test sqllogictest day_04_hnsw_sql -- --exact
```

The fixture creates and populates its own table, checks the exact plan, attaches an HNSW index, and checks the changed
plan. Its indexed plan contains:

```text
VectorIndexScanExec: index=hnsw, metric=Euclidean, query_dim=3, fetch=Some(5), ordered=false
```

and its five-row Euclidean query returns:

```text
1 point-1
0 point-0
2 point-2
3 point-3
4 point-4
```

This small comparison shows the product handoff, not a performance or general-recall result. HNSW proposes core dataset
ordinals; the supplied adapter resolves them to source rows, and the supplied `SortExec` still owns final SQL ordering.
Unsupported SQL shapes continue to use the exact scan.

The fixture uses Euclidean distance and `LIMIT 5`; it verifies `index=hnsw`, the supplied final sort, and the expected
rows without depending on mutable interactive-shell state.

## Check the Course Through Day 4

Run the Day 4 focused gate, then the cumulative course through Day 4:

```sh
cargo x test-day 4
cargo x test-through 4
```

The runner selects only the tests assigned through HNSW, so unfinished Day 5
IVF-PQ work cannot turn this Day 4 gate red.

## Day 4 Review

Choose one insertion and one query and explain:

- why a promoted row must also belong to every lower layer;
- why the seed changes graph structure but must reproduce the same structure when repeated;
- why construction carries one entry point downward before connecting the new row;
- why query-time upper layers use greedy routing while layer zero keeps a bounded frontier;
- when the global entry point changes; and
- why equal rows in the supplied SQL comparison say nothing about general recall or speed.

The Day 4 index is immutable and in memory. Deletion, concurrent mutation, persistence, production
neighbor-diversification heuristics, and adaptive search budgets would change the learner contract rather than complete
this checkpoint.

{{#include copyright.md}}
