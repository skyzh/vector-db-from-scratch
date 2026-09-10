# Compress IVFFlat with Product Quantization

> **Day 5**
>
> Complete [Add Hierarchy with HNSW](./rust-05-hnsw.md) first. Build a residual IVF-PQ index that scores compact codes,
> reranks a shortlist with full vectors, and exposes the representation accounting used by the final benchmark.

IVFFlat narrows a query to a few lists, but scoring those lists still reads every component of every candidate vector.
On a large dataset, that inner loop can dominate the search. IVF-PQ keeps the coarse lists and replaces each candidate's
scoring representation with a short sequence of learned codeword IDs. It uses those IDs to choose a shortlist, then
returns to the original vectors for the final distances.

Follow one row through that path. Its IVF centroid `c` chooses the list. Subtracting `c` from the row vector `x` gives an
eight-dimensional residual, which we split into two four-dimensional pieces:

```text
residual = [ 0.7, -0.1, 0.3, 0.2 | -0.4, 0.8, 0.1, -0.2 ]
             subvector 0              subvector 1
```

Each slice position has its own codebook. With four codewords in each codebook, this residual is represented by two
choices:

```text
subvector 0 -> codeword 2
subvector 1 -> codeword 0
PQ code     -> [2, 0]
```

The course stores each ID as a `u8`, so this row contributes two code bytes. The codebooks are shared by every row. What
they approximate is the residual, not the original vector:

```text
r = x - c
```

That distinction matters during search. The IVF centroid decides which lists are probed; each PQ codeword approximates
one slice of a residual within those lists. Rebuilding with equal data and configuration must reproduce the same coarse
centroids, PQ codebooks, list sizes, codes, and search results.

For a query `q`, each probed list has its own centroid `c`. Subtract that same `c` from the query and compare each query
slice with the codewords for that position:

- an **IVF centroid** chooses the inverted list;
- a **PQ codeword** approximates one slice of the residual inside that list.

This is the residual IVF-PQ design introduced by
[Jégou, Douze, and Schmid](https://doi.org/10.1109/TPAMI.2010.57), often called IVFADC. The
[Faiss index guide](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes#summary-of-methods) describes the same
coarse-quantizer-plus-residual-PQ decomposition.

## Score the Codes, then Rerank

Build one squared-Euclidean lookup table for every slice position in the probed list:

```text
table[m][j] = squared_l2((query - coarse_centroid)[m], codebook[m][j])
```

The encoded row `[2, 0]` now needs two table reads instead of reading its eight stored components:

```text
score(code) = table[0][code[0]] + ... + table[M - 1][code[M - 1]]
```

The query remains full precision while the stored residual is quantized, which makes this asymmetric distance
computation. Apply the same lookup process to every encoded row in the probed lists and retain the best
`min(max(rerank, k), rows)` row offsets. Those offsets stay attached to their source rows when the original vectors are
read for exact Euclidean reranking:

```text
probed lists -> PQ score -> rerank shortlist -> exact distance -> top-k
```

With four subquantizers and sixteen codewords per codebook, one probed list builds 64 lookup entries. Each encoded row
then reads four entries and combines them into one approximate score. The base `Dataset` remains available because the
shortlist still needs exact reranking.

This also fixes the meaning of the byte counters. `encoded_bytes()` plus `codebook_bytes()` measures only the PQ search
representation. It does not include the retained full vectors, coarse centroids, row IDs, list allocations, or other
index and process overhead.

## Build IVF-PQ in Rust

You will modify:

```text
vector-db-starter/core/src/pq.rs
```

The starter already exposes `IvfPqConfig`, `IvfPqIndex`, its `VectorIndex` implementation, byte-accounting methods, and
the DataFusion `IndexConfig::IvfPq` path. Your two unfinished units are `IvfPqIndex::try_new` and
`IvfPqIndex::search_with_probes`; the first two checkpoints develop different parts of the same `try_new` implementation.

All commands on this page run against the cumulative starter workspace. Complete Days 1–4 first, because an untouched
starter reaches an earlier `todo!()` before it can exercise Day 5.

`IvfPqConfig` separates the main budgets:

| Field | Meaning |
| --- | --- |
| `partitions` | Coarse IVF lists |
| `probes` | Lists visited per query |
| `iterations` | Seeded k-means rounds |
| `subquantizers` | Equal residual slices |
| `codebook_size` | Codewords per slice |
| `rerank` | Full-precision shortlist budget |
| `seed` | Reproducible training seed |

This implementation accepts only `Metric::Euclidean`. Supporting cosine or inner product would change how vectors,
residuals, and codeword scores relate, so those metrics return a configuration error here.

## Checkpoint 1: Validate the Layout

Implement `IvfPqIndex::try_new`. Validate before training:

- `1 <= probes <= partitions <= rows` and `iterations > 0`;
- `subquantizers > 0` and the dimension divides evenly into that many slices;
- `2 <= codebook_size <= min(256, rows)`;
- `rerank > 0`; and
- the metric is Euclidean.

Run the focused validation boundary. Its supplied test checks the Euclidean-only metric and requires the subquantizer
count to divide the vector dimension:

```sh
cargo xtask test day_05::checkpoint_1
```

## Checkpoint 2: Train and Encode Residual Codebooks

Continue `try_new` by building the coarse partition with the configured partitions, probes, iterations, and seed. Once
its final centroids are known, assign every row again and compute `row - centroid`. That final reassignment gives every
row exactly one list and ensures its residual uses the centroid for that list. Checkpoint 2 is the first supplied test
that constructs a valid index and observes this path.

Split every residual into equal contiguous slices. For each subquantizer:

1. choose `codebook_size` distinct seeded residual rows;
2. copy that slice from each chosen row as an initial codeword;
3. assign every residual slice to its nearest codeword under squared Euclidean distance;
4. replace each non-empty codeword with the component-wise mean of its assignments; and
5. stop after convergence or `iterations` rounds.

When a cluster receives no residual slices, leave that codeword unchanged. Reuse the deterministic RNG from
`src/search.rs`, but derive a different deterministic seed for each subquantizer so their initial row choices are
independent. After training, encode every row with exactly one valid `u8` code for each slice position.

```sh
cargo xtask test day_05::checkpoint_2
```

This gate checks deterministic training, complete list membership, code layout, and byte accounting. At this point
`try_new` is complete; the index is built, but its search function remains the second starter `todo!()`.

## Checkpoint 3: Scan Codes and Rerank

Implement `search_with_probes` by following the query path from the opening trace:

1. validate the query, probe count, and nonzero rerank budget;
2. rank coarse centroids and visit the nearest lists;
3. build residual lookup tables for each visited list;
4. sum one table entry per code with a shortlist budget of at least `k`, even when `rerank < k`;
5. compute exact Euclidean distances for the shortlist row offsets; and
6. return exact top-k results in the public `(distance, row)` order.

Use `f64` for coarse selection, lookup sums, and exact rerank distances. A value crosses the public `Neighbor` boundary
only when it is finite and representable as `f32`; convert there and apply the public `(distance, row)` order. Keep the
row offset beside every code and every shortlisted distance. If unrepresentable exact distances leave fewer than
`min(k, rows)` valid results, return an error instead of a shortened answer.

Run the complete Checkpoint 3 gate:

```sh
cargo xtask test day_05::checkpoint_3
```

This checkpoint probes every list and reranks every row, so its result must match exact search for that bounded case.
The other cases cover large finite values, representation failures, public ordering, and the unchanged Day 1 adapter.
The physical plan names `index=ivf_pq`; the matcher remains conservative and DataFusion still applies the final bounded
sort.

## Checkpoint 4: Inspect the Search Representation

For any built index:

- `encoded_bytes()` counts stored PQ codes;
- `codebook_bytes()` counts shared PQ codeword components;
- `full_precision_bytes()` counts the retained dataset's vector components.

These counters let the final day print the retained vector components beside the codes and shared codebooks. Their ratio
is useful only for that representation accounting: it is not total-memory compression and it is not a measured speed or
quality result.

## Return to the SQL Product

Run the self-contained Day 5 SQLLogicTest:

```sh
cargo xtask test day_05::checkpoint_4
```

The fixture creates and fills its own eight-row table. Before it attaches the index, the plan contains:

```text
DataSourceExec: partitions=1, partition_sizes=[1]
```

After `CREATE INDEX ... USING ivfpq`, the same bounded top-k query contains:

```text
VectorIndexScanExec: index=ivf_pq, metric=Euclidean, query_dim=3, fetch=Some(5), ordered=false
```

and returns:

```text
1 point-1
0 point-0
2 point-2
3 point-3
4 point-4
```

The fixture gives you one deterministic handoff from an exact scan to the supplied IVF-PQ SQL adapter. The matcher still
falls back for unsupported query shapes, and the bounded final sort stays in the plan. Broader recall, latency, and
memory comparisons belong to the final benchmark workload.

## Check the Completed Day

Run the Day 5 focused gate, then the cumulative course through Day 5:

```sh
cargo xtask test day_05
cargo xtask test-through day_05
```

When both commands pass, trace one result all the way back: its coarse centroid chose a list, its residual slices chose
PQ codewords, lookup sums placed its row offset in the shortlist, and its retained original vector supplied the exact
distance used by the final ordering. The three byte counters describe the representations used along that path; they do
not measure the whole index.

Day 5 leaves bit-packed codes, cosine and inner-product support, optimized product quantization, SIMD table scans,
persistent layouts, separate training samples, and removing full vectors from memory for later work. The next chapter
uses a fixed external workload to make measured comparisons across all five indexes.

{{#include copyright.md}}
