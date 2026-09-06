use std::collections::HashSet;
use std::path::PathBuf;

use vector_benchmark_support::{Cli, Mode, load_sift1m, rank_recall};
use vector_core_starter::{
    Dataset, FlatIndex, HnswConfig, HnswIndex, IvfFlatConfig, IvfFlatIndex, IvfPqConfig,
    IvfPqIndex, Metric, Neighbor, NswConfig, NswIndex, VectorIndex,
};

const K: usize = 100;
const SEEDS: [u64; 3] = [7, 0x5eed, 0x9e37_79b9_7f4a_7c15];

struct Smoke {
    dataset: Dataset,
    queries: Vec<Vec<f32>>,
    exact_first: Vec<usize>,
}

fn smoke() -> Smoke {
    let directory = std::env::var_os("SIFT1M_DIR")
        .map(PathBuf::from)
        .expect("set SIFT1M_DIR to the directory containing the external SIFT1M files");
    let sift = load_sift1m(&Cli {
        mode: Mode::Smoke,
        data_dir: directory,
    })
    .unwrap();
    let dataset = Dataset::try_new(sift.base).unwrap();
    let exact = FlatIndex::try_new(dataset.clone(), Metric::Euclidean).unwrap();
    let exact_first = sift
        .queries
        .iter()
        .map(|query| exact.search(query, K).unwrap()[0].row)
        .collect();
    Smoke {
        dataset,
        queries: sift.queries,
        exact_first,
    }
}

fn search_all(index: &dyn VectorIndex, workload: &Smoke) -> Vec<Vec<Neighbor>> {
    workload
        .queries
        .iter()
        .map(|query| index.search(query, K).unwrap())
        .collect()
}

fn assert_quality(results: &[Vec<Neighbor>], workload: &Smoke, minimum_r100: f64) {
    assert_eq!(results.len(), workload.queries.len());
    let mut total_r100 = 0.0;
    for (neighbors, exact_first) in results.iter().zip(&workload.exact_first) {
        assert_eq!(neighbors.len(), K);
        assert!(neighbors.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(
            neighbors
                .iter()
                .all(|neighbor| neighbor.row < workload.dataset.len()
                    && neighbor.distance.is_finite())
        );
        assert_eq!(
            neighbors
                .iter()
                .map(|neighbor| neighbor.row)
                .collect::<HashSet<_>>()
                .len(),
            K
        );
        let rows = neighbors
            .iter()
            .map(|neighbor| neighbor.row)
            .collect::<Vec<_>>();
        let recall = rank_recall(&rows, *exact_first);
        assert!(recall.r1 <= recall.r10 && recall.r10 <= recall.r100);
        total_r100 += recall.r100;
    }
    assert!(total_r100 / results.len() as f64 >= minimum_r100);
}

mod day_06 {
    mod checkpoint_2 {
        use super::super::*;

        #[test]
        #[ignore = "requires external SIFT1M data through SIFT1M_DIR"]
        fn sift_flat_smoke() {
            let workload = smoke();
            let index = FlatIndex::try_new(workload.dataset.clone(), Metric::Euclidean).unwrap();
            assert_quality(&search_all(&index, &workload), &workload, 1.0);
        }

        #[test]
        #[ignore = "requires external SIFT1M data through SIFT1M_DIR"]
        fn sift_ivf_flat_smoke() {
            let workload = smoke();
            for seed in SEEDS {
                let config = IvfFlatConfig {
                    partitions: 32,
                    probes: 6,
                    iterations: 12,
                    seed,
                };
                let left =
                    IvfFlatIndex::try_new(workload.dataset.clone(), Metric::Euclidean, config)
                        .unwrap();
                let right =
                    IvfFlatIndex::try_new(workload.dataset.clone(), Metric::Euclidean, config)
                        .unwrap();
                let left = search_all(&left, &workload);
                let right = search_all(&right, &workload);
                assert_eq!(left, right);
                assert_quality(&left, &workload, 0.05);
            }
        }

        #[test]
        #[ignore = "requires external SIFT1M data through SIFT1M_DIR"]
        fn sift_nsw_smoke() {
            let workload = smoke();
            let index = NswIndex::try_new(
                workload.dataset.clone(),
                Metric::Euclidean,
                NswConfig {
                    max_connections: 12,
                    ef_construction: 64,
                    ef_search: 40,
                },
            )
            .unwrap();
            let results = search_all(&index, &workload);
            assert_quality(&results, &workload, 0.05);
        }

        #[test]
        #[ignore = "requires external SIFT1M data through SIFT1M_DIR"]
        fn sift_hnsw_smoke() {
            let workload = smoke();
            for seed in SEEDS {
                let config = HnswConfig {
                    max_connections: 12,
                    ef_construction: 64,
                    ef_search: 40,
                    max_level: 12,
                    seed,
                };
                let left = HnswIndex::try_new(workload.dataset.clone(), Metric::Euclidean, config)
                    .unwrap();
                let right = HnswIndex::try_new(workload.dataset.clone(), Metric::Euclidean, config)
                    .unwrap();
                let left = search_all(&left, &workload);
                let right = search_all(&right, &workload);
                assert_eq!(left, right);
                assert_quality(&left, &workload, 0.05);
            }
        }

        #[test]
        #[ignore = "requires external SIFT1M data through SIFT1M_DIR"]
        fn sift_ivf_pq_smoke() {
            let workload = smoke();
            for seed in SEEDS {
                let config = IvfPqConfig {
                    partitions: 32,
                    probes: 6,
                    iterations: 12,
                    subquantizers: 4,
                    codebook_size: 16,
                    rerank: 100,
                    seed,
                };
                let left = IvfPqIndex::try_new(workload.dataset.clone(), Metric::Euclidean, config)
                    .unwrap();
                let right =
                    IvfPqIndex::try_new(workload.dataset.clone(), Metric::Euclidean, config)
                        .unwrap();
                let left = search_all(&left, &workload);
                let right = search_all(&right, &workload);
                assert_eq!(left, right);
                assert_quality(&left, &workload, 0.05);
            }
        }
    }
}
