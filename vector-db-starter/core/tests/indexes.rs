use std::mem::size_of;

use vector_core_starter::{
    Dataset, FlatIndex, HnswConfig, HnswIndex, IndexConfig, IvfFlatConfig, IvfFlatIndex,
    IvfPqConfig, IvfPqIndex, Metric, Neighbor, NswConfig, NswIndex, VectorError, VectorIndex,
    recall_at_k,
};

fn line_dataset(size: usize) -> Dataset {
    Dataset::try_new(
        (0..size)
            .map(|value| vec![value as f32, (value % 7) as f32 + 1.0])
            .collect(),
    )
    .unwrap()
}

mod day_01 {
    mod checkpoint_1 {
        use super::super::*;

        #[test]
        fn flat_search_is_deterministic_and_validates_queries() {
            assert_eq!(
                Dataset::try_new(vec![]).unwrap_err(),
                VectorError::EmptyDataset
            );
            assert_eq!(
                Dataset::try_new(vec![vec![]]).unwrap_err(),
                VectorError::EmptyVector
            );
            assert_eq!(
                Dataset::try_new(vec![vec![1.0, 0.0], vec![1.0]]).unwrap_err(),
                VectorError::DimensionMismatch {
                    expected: 2,
                    actual: 1,
                }
            );
            assert_eq!(
                Dataset::try_new(vec![vec![1.0, 0.0], vec![2.0, f32::INFINITY]]).unwrap_err(),
                VectorError::NonFiniteValue {
                    vector: 1,
                    dimension: 1,
                }
            );

            let dataset =
                Dataset::try_new(vec![vec![1.0, 0.0], vec![-1.0, 0.0], vec![0.0, 1.0]]).unwrap();
            let index = FlatIndex::try_new(dataset, Metric::Euclidean).unwrap();

            let result = index.search(&[0.0, 0.0], 2).unwrap();
            assert_eq!(
                result.iter().map(|item| item.row).collect::<Vec<_>>(),
                [0, 1]
            );
            assert!(index.search(&[0.0], 1).is_err());
            assert!(index.search(&[f32::NAN, 0.0], 1).is_err());
        }

        #[test]
        fn cosine_rejects_zero_norm_vectors() {
            let dataset = Dataset::try_new(vec![vec![1.0, 0.0], vec![0.0, 0.0]]).unwrap();
            assert!(FlatIndex::try_new(dataset, Metric::Cosine).is_err());

            let dataset = Dataset::try_new(vec![vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
            let index = FlatIndex::try_new(dataset, Metric::Cosine).unwrap();
            assert_eq!(
                index.search(&[0.0, 0.0], 1).unwrap_err(),
                VectorError::ZeroNorm { vector: 2 }
            );
        }
    }
}

mod day_02 {
    mod checkpoint_1 {
        use super::super::*;

        #[test]
        fn recall_reports_result_overlap() {
            let expected = (0..4)
                .map(|row| Neighbor {
                    row,
                    distance: row as f32,
                })
                .collect::<Vec<_>>();
            let actual = [
                expected[0],
                expected[2],
                Neighbor {
                    row: 9,
                    distance: 3.0,
                },
            ];
            assert_eq!(recall_at_k(&expected, &actual, 3), 2.0 / 3.0);
            assert_eq!(recall_at_k(&expected[..2], &actual, 10), 0.5);
            assert_eq!(recall_at_k(&[], &actual, 10), 1.0);

            let duplicate_actual = [expected[0], expected[0], expected[2]];
            let duplicate_recall = recall_at_k(&expected[..2], &duplicate_actual, 3);
            assert_eq!(duplicate_recall, 0.5);
            assert!((0.0..=1.0).contains(&duplicate_recall));
        }
    }
    mod checkpoint_2 {
        use super::super::*;

        #[test]
        fn ivf_rejects_invalid_build_configuration() {
            let invalid_configs = [
                IvfFlatConfig {
                    partitions: 0,
                    probes: 1,
                    iterations: 1,
                    seed: 7,
                },
                IvfFlatConfig {
                    partitions: 81,
                    probes: 1,
                    iterations: 1,
                    seed: 7,
                },
                IvfFlatConfig {
                    partitions: 8,
                    probes: 0,
                    iterations: 1,
                    seed: 7,
                },
                IvfFlatConfig {
                    partitions: 8,
                    probes: 9,
                    iterations: 1,
                    seed: 7,
                },
                IvfFlatConfig {
                    partitions: 8,
                    probes: 1,
                    iterations: 0,
                    seed: 7,
                },
            ];

            for config in invalid_configs {
                let error = IndexConfig::IvfFlat(config)
                    .build(line_dataset(80), Metric::Euclidean)
                    .unwrap_err();
                assert!(matches!(error, VectorError::InvalidConfig(_)));
            }

            let config = IvfFlatConfig {
                partitions: 1,
                probes: 1,
                iterations: 1,
                seed: 7,
            };
            let zero_norm_dataset = Dataset::try_new(vec![vec![1.0, 0.0], vec![0.0, 0.0]]).unwrap();
            assert_eq!(
                IvfFlatIndex::try_new(zero_norm_dataset, Metric::Cosine, config).unwrap_err(),
                VectorError::ZeroNorm { vector: 1 }
            );
        }
    }
    mod checkpoint_3 {
        use super::super::*;

        #[test]
        fn ivf_build_is_seeded() {
            let config = IvfFlatConfig {
                partitions: 6,
                probes: 2,
                iterations: 8,
                seed: 42,
            };
            let left = IvfFlatIndex::try_new(line_dataset(60), Metric::Euclidean, config).unwrap();
            let right = IvfFlatIndex::try_new(line_dataset(60), Metric::Euclidean, config).unwrap();
            assert_eq!(left.centroids(), right.centroids());
            assert_eq!(left.list_sizes(), right.list_sizes());
        }

        #[test]
        fn ivf_cosine_recovers_from_a_zero_mean_cluster() {
            let dataset = Dataset::try_new(vec![
                vec![1.0, 0.0],
                vec![-1.0, 0.0],
                vec![0.0, 1.0],
                vec![0.0, -1.0],
            ])
            .unwrap();
            let index = IvfFlatIndex::try_new(
                dataset,
                Metric::Cosine,
                IvfFlatConfig {
                    partitions: 1,
                    probes: 1,
                    iterations: 3,
                    seed: 1,
                },
            )
            .unwrap();
            assert!(index.centroids()[0].iter().all(|value| value.is_finite()));
        }
    }
    mod checkpoint_4 {
        use super::super::*;

        #[test]
        fn ivf_scanning_every_partition_matches_exact_search() {
            let dataset = line_dataset(80);
            let exact = FlatIndex::try_new(dataset.clone(), Metric::Cosine).unwrap();
            let ivf = IvfFlatIndex::try_new(
                dataset,
                Metric::Cosine,
                IvfFlatConfig {
                    partitions: 8,
                    probes: 2,
                    iterations: 10,
                    seed: 7,
                },
            )
            .unwrap();
            let query = [31.2, 4.0];

            let expected = exact.search(&query, 80).unwrap();
            let actual = ivf.search_with_probes(&query, 80, 8).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(actual.len(), 80);
            assert!(actual.windows(2).all(|pair| pair[0] <= pair[1]));
            assert_eq!(ivf.list_sizes().iter().sum::<usize>(), 80);
            assert_eq!(
                ivf.search_with_probes(&[31.2], 80, 8).unwrap_err(),
                VectorError::DimensionMismatch {
                    expected: 2,
                    actual: 1,
                }
            );
            assert_eq!(
                ivf.search_with_probes(&[31.2, f32::INFINITY], 80, 8)
                    .unwrap_err(),
                VectorError::NonFiniteValue {
                    vector: 80,
                    dimension: 1,
                }
            );
            assert_eq!(
                ivf.search_with_probes(&[0.0, 0.0], 80, 8).unwrap_err(),
                VectorError::ZeroNorm { vector: 80 }
            );
            assert!(matches!(
                ivf.search_with_probes(&query, 80, 0),
                Err(VectorError::InvalidConfig(_))
            ));
            assert!(matches!(
                ivf.search_with_probes(&query, 80, 9),
                Err(VectorError::InvalidConfig(_))
            ));
        }
    }
}

mod day_03 {
    mod checkpoint_3 {
        use super::super::*;

        #[test]
        fn nsw_rejects_invalid_build_configuration_and_builds_a_bounded_reciprocal_graph() {
            let invalid_configs = [
                NswConfig {
                    max_connections: 0,
                    ef_construction: 8,
                    ef_search: 8,
                },
                NswConfig {
                    max_connections: 4,
                    ef_construction: 3,
                    ef_search: 8,
                },
                NswConfig {
                    max_connections: 4,
                    ef_construction: 8,
                    ef_search: 0,
                },
            ];
            for config in invalid_configs {
                assert!(matches!(
                    NswIndex::try_new(line_dataset(24), Metric::Euclidean, config),
                    Err(VectorError::InvalidConfig(_))
                ));
            }

            let zero_norm_dataset = Dataset::try_new(vec![vec![1.0, 0.0], vec![0.0, 0.0]]).unwrap();
            assert_eq!(
                NswIndex::try_new(
                    zero_norm_dataset,
                    Metric::Cosine,
                    NswConfig {
                        max_connections: 2,
                        ef_construction: 4,
                        ef_search: 4,
                    },
                )
                .unwrap_err(),
                VectorError::ZeroNorm { vector: 1 }
            );

            let config = NswConfig {
                max_connections: 4,
                ef_construction: 12,
                ef_search: 8,
            };
            let left = NswIndex::try_new(line_dataset(32), Metric::Euclidean, config).unwrap();
            let right = NswIndex::try_new(line_dataset(32), Metric::Euclidean, config).unwrap();
            assert_eq!(left.adjacency(), right.adjacency());

            for (row, neighbors) in left.adjacency().iter().enumerate() {
                assert!(neighbors.len() <= config.max_connections);
                let unique = neighbors
                    .iter()
                    .copied()
                    .collect::<std::collections::HashSet<_>>();
                assert_eq!(unique.len(), neighbors.len());
                assert!(!neighbors.contains(&row));
                for neighbor in neighbors {
                    assert!(left.adjacency()[*neighbor].contains(&row));
                }
            }
        }
    }
    mod checkpoint_4 {
        use super::super::*;

        #[test]
        fn nsw_search_validates_widens_and_matches_exact_on_connected_fixture() {
            let dataset = line_dataset(64);
            let exact = FlatIndex::try_new(dataset.clone(), Metric::Euclidean).unwrap();
            let nsw = NswIndex::try_new(
                dataset,
                Metric::Euclidean,
                NswConfig {
                    max_connections: 8,
                    ef_construction: 32,
                    ef_search: 32,
                },
            )
            .unwrap();
            let query = [17.4, 4.0];

            assert_eq!(
                nsw.search_with_ef(&[17.4], 5, 8).unwrap_err(),
                VectorError::DimensionMismatch {
                    expected: 2,
                    actual: 1,
                }
            );
            assert_eq!(
                nsw.search_with_ef(&[17.4, f32::NAN], 5, 8).unwrap_err(),
                VectorError::NonFiniteValue {
                    vector: 64,
                    dimension: 1,
                }
            );
            assert!(matches!(
                nsw.search_with_ef(&query, 5, 0),
                Err(VectorError::InvalidConfig(_))
            ));

            let widened = nsw.search_with_ef(&query, 5, 1).unwrap();
            assert_eq!(widened.len(), 5);
            assert!(widened.windows(2).all(|pair| pair[0] < pair[1]));

            let expected = exact.search(&query, 10).unwrap();
            let actual = nsw.search_with_ef(&query, 10, 64).unwrap();
            assert_eq!(actual, expected);

            let cosine = NswIndex::try_new(
                Dataset::try_new(vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]]).unwrap(),
                Metric::Cosine,
                NswConfig {
                    max_connections: 2,
                    ef_construction: 4,
                    ef_search: 4,
                },
            )
            .unwrap();
            assert_eq!(
                cosine.search_with_ef(&[0.0, 0.0], 1, 4).unwrap_err(),
                VectorError::ZeroNorm { vector: 3 }
            );
        }
    }
}

mod day_04 {
    mod checkpoint_2 {
        use super::super::*;

        #[test]
        fn hnsw_rejects_invalid_configuration_and_builds_seeded_nested_layers() {
            let invalid_configs = [
                HnswConfig {
                    max_connections: 0,
                    ef_construction: 8,
                    ef_search: 8,
                    max_level: 8,
                    seed: 99,
                },
                HnswConfig {
                    max_connections: 8,
                    ef_construction: 7,
                    ef_search: 8,
                    max_level: 8,
                    seed: 99,
                },
                HnswConfig {
                    max_connections: 8,
                    ef_construction: 8,
                    ef_search: 0,
                    max_level: 8,
                    seed: 99,
                },
                HnswConfig {
                    max_connections: 8,
                    ef_construction: 8,
                    ef_search: 8,
                    max_level: 0,
                    seed: 99,
                },
            ];
            for config in invalid_configs {
                assert!(matches!(
                    HnswIndex::try_new(line_dataset(24), Metric::Euclidean, config),
                    Err(VectorError::InvalidConfig(_))
                ));
            }

            let zero_norm_dataset = Dataset::try_new(vec![vec![1.0, 0.0], vec![0.0, 0.0]]).unwrap();
            assert_eq!(
                HnswIndex::try_new(
                    zero_norm_dataset,
                    Metric::Cosine,
                    HnswConfig {
                        max_connections: 1,
                        ef_construction: 2,
                        ef_search: 1,
                        max_level: 1,
                        seed: 99,
                    },
                )
                .unwrap_err(),
                VectorError::ZeroNorm { vector: 1 }
            );

            let config = HnswConfig {
                max_connections: 8,
                ef_construction: 40,
                ef_search: 32,
                max_level: 8,
                seed: 99,
            };
            let dataset = line_dataset(96);
            let left = HnswIndex::try_new(dataset.clone(), Metric::Euclidean, config).unwrap();
            let right = HnswIndex::try_new(dataset, Metric::Euclidean, config).unwrap();

            assert_eq!(left.levels(), right.levels());
            assert_eq!(left.top_level(), *left.levels().iter().max().unwrap());
            for level in 0..=left.top_level() {
                let adjacency = left.layer(level).unwrap();
                for (row, neighbors) in adjacency.iter().enumerate() {
                    assert!(neighbors.len() <= config.max_connections);
                    if left.levels()[row] < level {
                        assert!(neighbors.is_empty());
                        continue;
                    }
                    let unique = neighbors
                        .iter()
                        .copied()
                        .collect::<std::collections::HashSet<_>>();
                    assert_eq!(unique.len(), neighbors.len());
                    assert!(!neighbors.contains(&row));
                    for neighbor in neighbors {
                        assert!(left.levels()[*neighbor] >= level);
                        assert!(adjacency[*neighbor].contains(&row));
                    }
                }
            }
        }
    }
    mod checkpoint_3 {
        use super::super::*;

        #[test]
        fn hnsw_search_validates_widens_and_recovers_neighbors() {
            let config = HnswConfig {
                max_connections: 8,
                ef_construction: 40,
                ef_search: 32,
                max_level: 8,
                seed: 99,
            };
            let dataset = line_dataset(96);
            let exact = FlatIndex::try_new(dataset.clone(), Metric::Euclidean).unwrap();
            let index = HnswIndex::try_new(dataset, Metric::Euclidean, config).unwrap();
            let query = [72.6, 3.0];

            assert_eq!(
                index.search_with_ef(&[72.6], 5, 8).unwrap_err(),
                VectorError::DimensionMismatch {
                    expected: 2,
                    actual: 1,
                }
            );
            assert_eq!(
                index.search_with_ef(&[72.6, f32::NAN], 5, 8).unwrap_err(),
                VectorError::NonFiniteValue {
                    vector: 96,
                    dimension: 1,
                }
            );
            assert!(matches!(
                index.search_with_ef(&query, 5, 0),
                Err(VectorError::InvalidConfig(_))
            ));

            let widened = index.search_with_ef(&query, 5, 1).unwrap();
            assert_eq!(widened.len(), 5);
            assert!(widened.windows(2).all(|pair| pair[0] < pair[1]));

            let expected = exact.search(&query, 10).unwrap();
            let actual = index.search_with_ef(&query, 10, 96).unwrap();
            assert_eq!(recall_at_k(&expected, &actual, 10), 1.0);

            let descent_dataset = Dataset::try_new(
                (0..24)
                    .map(|value| vec![value as f32, (value % 3) as f32 + 1.0])
                    .collect(),
            )
            .unwrap();
            let descent_index = HnswIndex::try_new(
                descent_dataset,
                Metric::Euclidean,
                HnswConfig {
                    max_connections: 2,
                    ef_construction: 8,
                    ef_search: 1,
                    max_level: 8,
                    seed: 1,
                },
            )
            .unwrap();
            assert!(descent_index.top_level() > 0);
            assert_eq!(
                descent_index.search_with_ef(&[2.1, 3.25], 1, 1).unwrap()[0].row,
                2
            );
        }
    }
}

mod day_05 {
    mod checkpoint_1 {
        use super::super::*;

        #[test]
        fn ivf_pq_validates_its_euclidean_code_layout() {
            let dataset = line_dataset(32);
            let config = IvfPqConfig {
                partitions: 4,
                probes: 2,
                iterations: 4,
                subquantizers: 2,
                codebook_size: 8,
                rerank: 16,
                seed: 7,
            };
            assert!(IvfPqIndex::try_new(dataset.clone(), Metric::Cosine, config).is_err());
            assert!(
                IvfPqIndex::try_new(
                    dataset,
                    Metric::Euclidean,
                    IvfPqConfig {
                        subquantizers: 3,
                        ..config
                    },
                )
                .is_err()
            );
        }
    }
    mod checkpoint_2 {
        use super::super::*;

        #[test]
        fn ivf_pq_build_is_seeded_and_codes_each_row() {
            let config = IvfPqConfig {
                partitions: 6,
                probes: 2,
                iterations: 8,
                subquantizers: 2,
                codebook_size: 8,
                rerank: 20,
                seed: 42,
            };
            let left = IvfPqIndex::try_new(line_dataset(60), Metric::Euclidean, config).unwrap();
            let right = IvfPqIndex::try_new(line_dataset(60), Metric::Euclidean, config).unwrap();

            assert_eq!(left.centroids(), right.centroids());
            assert_eq!(left.codebooks(), right.codebooks());
            assert_eq!(left.list_sizes(), right.list_sizes());
            assert!(
                left.codebooks()
                    .iter()
                    .flatten()
                    .flatten()
                    .all(|value| value.is_finite())
            );
            assert_eq!(left.list_sizes().iter().sum::<usize>(), 60);
            assert_eq!(left.encoded_bytes(), 60 * config.subquantizers);
            assert_eq!(left.full_precision_bytes(), 60 * 2 * size_of::<f32>());
            assert!(left.encoded_bytes() + left.codebook_bytes() < left.full_precision_bytes());
        }
    }
    mod checkpoint_3 {
        use super::super::*;

        #[test]
        fn ivf_pq_rejects_non_finite_training_residuals() {
            let max = f32::MAX;
            let dataset = Dataset::try_new(vec![vec![max], vec![max], vec![-max]]).unwrap();
            let config = IvfPqConfig {
                partitions: 1,
                probes: 1,
                iterations: 3,
                subquantizers: 1,
                codebook_size: 2,
                rerank: 1,
                seed: 1,
            };

            let error = IvfPqIndex::try_new(dataset, Metric::Euclidean, config).unwrap_err();

            assert_eq!(
                error,
                VectorError::InvalidConfig("IVF-PQ training residuals must remain finite")
            );
        }

        #[test]
        fn ivf_pq_query_scoring_preserves_large_finite_ordering() {
            let dataset = Dataset::try_new(vec![vec![3e20], vec![2e20], vec![1e20]]).unwrap();
            let exact = FlatIndex::try_new(dataset.clone(), Metric::Euclidean).unwrap();
            let index = IvfPqIndex::try_new(
                dataset,
                Metric::Euclidean,
                IvfPqConfig {
                    partitions: 1,
                    probes: 1,
                    iterations: 3,
                    subquantizers: 1,
                    codebook_size: 3,
                    rerank: 1,
                    seed: 1,
                },
            )
            .unwrap();

            let expected = exact.search(&[0.0], 1).unwrap();
            let actual = index.search(&[0.0], 1).unwrap();

            assert_eq!(expected[0].row, 2);
            assert_eq!(actual, expected);
        }

        #[test]
        fn ivf_pq_coarse_selection_preserves_large_finite_ordering() {
            let dataset = Dataset::try_new(vec![
                vec![2e38, 2e38],
                vec![2.5e38, 2.5e38],
                vec![3e38, 3e38],
                vec![3.3e38, 3.3e38],
            ])
            .unwrap();
            let exact = FlatIndex::try_new(dataset.clone(), Metric::Euclidean).unwrap();
            let index = IvfPqIndex::try_new(
                dataset,
                Metric::Euclidean,
                IvfPqConfig {
                    partitions: 2,
                    probes: 1,
                    iterations: 3,
                    subquantizers: 2,
                    codebook_size: 4,
                    rerank: 1,
                    seed: 0,
                },
            )
            .unwrap();

            let expected = exact.search(&[0.0, 0.0], 1).unwrap();
            let actual = index.search(&[0.0, 0.0], 1).unwrap();

            assert_eq!(expected[0].row, 0);
            assert!(expected[0].distance.is_finite());
            assert_eq!(actual, expected);
        }

        #[test]
        fn ivf_pq_rerank_rejects_unrepresentable_result_distances() {
            let dataset = Dataset::try_new(vec![vec![3e38, 3e38], vec![2.5e38, 2.5e38]]).unwrap();
            let index = IvfPqIndex::try_new(
                dataset,
                Metric::Euclidean,
                IvfPqConfig {
                    partitions: 1,
                    probes: 1,
                    iterations: 3,
                    subquantizers: 2,
                    codebook_size: 2,
                    rerank: 2,
                    seed: 7,
                },
            )
            .unwrap();

            let error = index.search(&[0.0, 0.0], 1).unwrap_err();

            assert_eq!(
                error,
                VectorError::InvalidConfig(
                    "IVF-PQ result distances must remain representable as finite f32"
                )
            );
        }

        #[test]
        fn ivf_pq_rerank_uses_public_neighbor_ordering() {
            let dataset = Dataset::try_new(vec![vec![1.0, f32::EPSILON], vec![1.0, 0.0]]).unwrap();
            let exact = FlatIndex::try_new(dataset.clone(), Metric::Euclidean).unwrap();
            let index = IvfPqIndex::try_new(
                dataset,
                Metric::Euclidean,
                IvfPqConfig {
                    partitions: 1,
                    probes: 1,
                    iterations: 3,
                    subquantizers: 2,
                    codebook_size: 2,
                    rerank: 2,
                    seed: 7,
                },
            )
            .unwrap();

            let expected = exact.search(&[0.0, 0.0], 1).unwrap();
            let actual = index.search(&[0.0, 0.0], 1).unwrap();

            assert_eq!(expected[0].row, 0);
            assert_eq!(actual, expected);
        }

        #[test]
        fn ivf_pq_rerank_ignores_unrepresentable_unselected_candidates() {
            let dataset = Dataset::try_new(vec![vec![1.0, 0.0], vec![3e38, 3e38]]).unwrap();
            let index = IvfPqIndex::try_new(
                dataset,
                Metric::Euclidean,
                IvfPqConfig {
                    partitions: 1,
                    probes: 1,
                    iterations: 3,
                    subquantizers: 2,
                    codebook_size: 2,
                    rerank: 2,
                    seed: 7,
                },
            )
            .unwrap();

            let actual = index.search(&[0.0, 0.0], 1).unwrap();

            assert_eq!(actual.len(), 1);
            assert_eq!(actual[0].row, 0);
            assert_eq!(actual[0].distance, 1.0);
        }

        #[test]
        fn ivf_pq_full_scan_and_rerank_matches_exact_search() {
            let dataset = line_dataset(80);
            let exact = FlatIndex::try_new(dataset.clone(), Metric::Euclidean).unwrap();
            let index = IvfPqIndex::try_new(
                dataset,
                Metric::Euclidean,
                IvfPqConfig {
                    partitions: 8,
                    probes: 2,
                    iterations: 8,
                    subquantizers: 2,
                    codebook_size: 8,
                    rerank: 24,
                    seed: 7,
                },
            )
            .unwrap();
            let query = [31.2, 4.0];

            let expected = exact.search(&query, 12).unwrap();
            let actual = index.search_with_probes(&query, 12, 8, 80).unwrap();
            assert_eq!(actual, expected);
        }
    }
}

mod day_06 {
    mod checkpoint_1 {
        use super::super::*;

        #[test]
        fn randomized_indexes_preserve_invariants_across_seed_trajectories() {
            let seeds = [7, 0x5eed, 0x9e37_79b9_7f4a_7c15];
            let dataset = line_dataset(96);
            let exact = FlatIndex::try_new(dataset.clone(), Metric::Euclidean).unwrap();
            let query = [41.2, 4.0];
            let expected = exact.search(&query, 12).unwrap();

            for seed in seeds {
                let ivf = IvfFlatIndex::try_new(
                    dataset.clone(),
                    Metric::Euclidean,
                    IvfFlatConfig {
                        partitions: 8,
                        probes: 8,
                        iterations: 8,
                        seed,
                    },
                )
                .unwrap();
                assert_eq!(ivf.list_sizes().iter().sum::<usize>(), dataset.len());
                assert_eq!(ivf.search_with_probes(&query, 12, 8).unwrap(), expected);

                let hnsw = HnswIndex::try_new(
                    dataset.clone(),
                    Metric::Euclidean,
                    HnswConfig {
                        max_connections: 8,
                        ef_construction: 40,
                        ef_search: 32,
                        max_level: 8,
                        seed,
                    },
                )
                .unwrap();
                assert_eq!(
                    hnsw.search_with_ef(&query, 12, dataset.len()).unwrap(),
                    expected
                );

                let pq = IvfPqIndex::try_new(
                    dataset.clone(),
                    Metric::Euclidean,
                    IvfPqConfig {
                        partitions: 8,
                        probes: 8,
                        iterations: 8,
                        subquantizers: 2,
                        codebook_size: 8,
                        rerank: dataset.len(),
                        seed,
                    },
                )
                .unwrap();
                assert_eq!(
                    pq.search_with_probes(&query, 12, 8, dataset.len()).unwrap(),
                    expected
                );
            }
        }
    }
}
