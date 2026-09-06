# Compress IVFFlat with Product Quantization

> **Day 5**
>
> Complete [Add Hierarchy with HNSW](./rust-05-hnsw.md) first. Build a residual IVF-PQ index that scores compact codes,
> reranks a shortlist with full vectors, and exposes the representation accounting used by the final benchmark.

IVFFlat avoids comparing a query with every row, but it still reads every component of every vector in the probed lists.
Product quantization replaces that candidate-scoring representation with a short sequence of learned codeword IDs.

Day 5 follows the product-quantization design introduced by
[Jégou, Douze, and Schmid](https://doi.org/10.1109/TPAMI.2010.57). It combines a coarse IVF partition with product
quantization of residuals, the structure commonly called IVFADC or IVF-PQ. The
[Faiss index guide](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes#summary-of-methods) uses the same
coarse-quantizer-plus-residual-PQ decomposition.

## Split Residuals into Subvectors

Suppose a residual has eight dimensions. Split it into two four-dimensional subvectors:

```text
residual = [ 0.7, -0.1, 0.3, 0.2 | -0.4, 0.8, 0.1, -0.2 ]
             subvector 0              subvector 1
```

Train a separate codebook for each subvector position. If each codebook has four codewords, encoding chooses one ID from
each side:

```text
subvector 0 -> codeword 2
subvector 1 -> codeword 0
PQ code     -> [2, 0]
```

The course stores each ID as a `u8`, so an encoded row uses one byte per subquantizer. Shared codebooks add a fixed cost
rather than a full vector for every row.

IVF already assigns every vector `x` to a coarse centroid `c`. Encode the residual instead of the original vector:

```text
r = x - c
```

Keep the two learned structures distinct:

- an **IVF centroid** chooses the inverted list;
- a **PQ codeword** approximates one slice of the residual inside that list.

Equal data and configuration must produce equal coarse centroids, PQ codebooks, list sizes, codes, and search results.

## Score Codes, then Rerank

For each probed list, subtract its coarse centroid from the query and build one squared-Euclidean lookup table per
subquantizer:

```text
table[m][j] = squared_l2((query - coarse_centroid)[m], codebook[m][j])
```

A row's approximate score is a sum of table lookups:

```text
score(code) = table[0][code[0]] + ... + table[M - 1][code[M - 1]]
```

The query stays full precision while stored residuals are quantized, so this is asymmetric distance computation. Keep
the best `min(max(rerank, k), rows)` row offsets under the approximate score, then compute exact Euclidean distance from
the original dataset and select the final `k`:

```text
probed lists -> PQ score -> rerank shortlist -> exact distance -> top-k
```

The base `Dataset` remains available for exact reranking. Therefore `encoded_bytes()` plus `codebook_bytes()` describes
the PQ **search representation**, not total index or process memory. It excludes retained full vectors, coarse centroids,
row IDs, list allocations, and other overhead.

**Prediction:** With four subquantizers and sixteen codewords per codebook, how many table entries does one probed list
need? How many additions score one encoded row after the tables exist?

## Build IVF-PQ in Rust

You will modify:

```text
vector-db-starter/core/src/pq.rs
```

The starter already exposes `IvfPqConfig`, `IvfPqIndex`, its `VectorIndex` implementation, byte-accounting methods, and
the DataFusion `IndexConfig::IvfPq` path. Keep those public APIs, existing indexes, and tests unchanged.

All commands on this page exercise the cumulative starter workspace. Complete Days 1–4 first; an untouched
starter stops at an earlier `todo!()` before it reaches Day 5.

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

The index accepts only `Metric::Euclidean`. Cosine and inner-product quantization need additional representation and
scoring choices; returning plausible numbers would not establish a consistent metric contract.

## Checkpoint 1: Validate the Layout and Build Coarse Lists

Implement `IvfPqIndex::try_new`. Validate before training:

- `1 <= probes <= partitions <= rows` and `iterations > 0`;
- `subquantizers > 0` and the dimension divides evenly into that many slices;
- `2 <= codebook_size <= min(256, rows)`;
- `rerank > 0`; and
- the metric is Euclidean.

Build the coarse partition with the same partitions, probes, iterations, and seed. Assign every row against the final
centroids, then compute `row - centroid`. Rebuilding membership after the final centroid update preserves the complete
one-list-per-row invariant from Day 2.

Run the focused layout boundary:

```sh
cargo test -p vector-db-from-scratch-core-starter --test indexes day_05_ivf_pq_validates_its_euclidean_code_layout
```

## Checkpoint 2: Train and Encode Residual Codebooks

Split every residual into equal contiguous slices. For each subquantizer:

1. choose `codebook_size` distinct seeded residual rows;
2. copy that slice from each chosen row as an initial codeword;
3. assign every residual slice to its nearest codeword under squared Euclidean distance;
4. replace each non-empty codeword with the component-wise mean of its assignments; and
5. stop after convergence or `iterations` rounds.

Keep a codeword unchanged when its cluster is empty. Reuse the deterministic RNG from `src/search.rs`, deriving a
different deterministic seed for each subquantizer. Then encode every row with exactly one valid `u8` code per
subquantizer.

```sh
cargo test -p vector-db-from-scratch-core-starter --test indexes day_05_ivf_pq_build_is_seeded_and_codes_each_row
```

The test checks deterministic training, complete list membership, code layout, and byte accounting.

## Checkpoint 3: Scan Codes and Rerank

Implement `search_with_probes`:

1. validate the query, probe count, and nonzero rerank budget;
2. rank coarse centroids and visit the nearest lists;
3. build residual lookup tables for each visited list;
4. sum one table entry per code with a shortlist budget of at least `k`, even when `rerank < k`;
5. compute exact Euclidean distances for the shortlist row offsets; and
6. return exact top-k results in the public `(distance, row)` order.

Keep coarse selection, lookup scores, and exact rerank distances in `f64`. At the public `Neighbor` boundary, retain only
finite distances representable as `f32`, convert them, and apply the public tie-break. Row identity must remain attached
to each code through both candidate stages. If discarding an unrepresentable exact distance would leave fewer than
`min(k, rows)` results, return an error instead of silently returning an incomplete result.

Probe every list and rerank every row as an exactness boundary:

```sh
cargo test -p vector-db-from-scratch-core-starter --test indexes day_05_ivf_pq_full_scan_and_rerank_matches_exact_search
```

Then run the complete IVF-PQ core group:

```sh
cargo test -p vector-db-from-scratch-core-starter --test indexes day_05_ivf_pq_
```

These cases also cover large finite values, representability, and public ordering. Finally, confirm the unchanged
Day 1 adapter can select the completed Euclidean index:

```sh
cargo test -p vector-db-from-scratch-datafusion-starter --test sql day_05_ivf_pq_is_visible_in_explain
```

The physical plan names `index=ivf_pq` while retaining the same conservative matcher and final bounded sort.

## Checkpoint 4: Inspect the Search Representation

For any built index:

- `encoded_bytes()` counts stored PQ codes;
- `codebook_bytes()` counts shared PQ codeword components;
- `full_precision_bytes()` counts the retained dataset's vector components.

Do not describe the ratio between full-precision vector bytes and code-plus-codebook bytes as total-memory compression.
The final day prints the exact accounting beside the shared five-index benchmark, where its scope can be read with
the workload and search configuration.

## Return to the SQL Product

Run the self-contained Day 5 SQLLogicTest:

```sh
cargo test -p vector-db-from-scratch-datafusion-starter --test sqllogictest day_05_ivf_pq_sql -- --exact
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

This fixture verifies one deterministic handoff from an exact scan to the supplied IVF-PQ SQL adapter. It does not
establish external-corpus recall, latency, memory use, or general exactness.

## Day 5 Review

Run the Day 5 focused gate, then the cumulative course through Day 5:

```sh
cargo x test-day 5
cargo x test-through 5
```

After the IVF-PQ core tests and DataFusion plan check pass, explain:

- why IVF centroids and PQ codewords solve different parts of search;
- why stored and query residuals use the selected list's same coarse centroid;
- how asymmetric lookup tables avoid reconstructing every candidate;
- why reranking still needs the original vectors;
- which bytes the search-representation accounting includes and excludes; and
- why a compact representation alone does not establish a latency or recall ranking.

Keep Day 5 focused on an executable IVF-PQ mental model. Bit-packed codes, cosine or inner-product support,
optimized product quantization, SIMD table scans, persistent layouts, training samples separate from indexed rows, and
removing full vectors from memory remain outside its scope.

{{#include copyright.md}}
