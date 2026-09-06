# Vector Core Starter

Across the six Rust days, you will use this crate to:

1. validate the in-memory dataset used by the DataFusion table;
2. implement recall measurement and IVFFlat;
3. build and search a bounded-degree NSW graph;
4. add seeded hierarchy with HNSW;
5. compress IVFFlat residuals with product quantization and rerank a candidate
   shortlist with exact distances; and
6. compare Flat, IVFFlat, NSW, HNSW, and IVF-PQ on one Euclidean workload.

Start with the existing metric math, deterministic `FlatIndex`, and top-k
helpers. Run commands from the repository-root Cargo workspace.

For checkpoint `M` on Day `NN`, run `cargo xtask test day_NN::checkpoint_M`. At the
end of the day, run `cargo xtask test day_NN`, then `cargo xtask test-through day_NN`.
