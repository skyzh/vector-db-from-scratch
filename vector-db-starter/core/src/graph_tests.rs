mod day_03 {
    use crate::graph::{prune_neighbors, search_layer};
    use crate::{Dataset, Metric};

    fn rows(neighbors: &[crate::Neighbor]) -> Vec<usize> {
        neighbors.iter().map(|neighbor| neighbor.row).collect()
    }

    mod checkpoint_1 {
        use super::*;

        #[test]
        fn search_layer_respects_bounds_and_expands_equal_frontier() {
            let dataset = Dataset::try_new(vec![
                vec![0.0],
                vec![1.0],
                vec![2.0],
                vec![8.0],
                vec![9.0],
            ])
            .unwrap();
            let adjacency = vec![vec![1], vec![0, 2], vec![1], vec![4], vec![3]];

            let bounded = search_layer(
                &dataset,
                Metric::Euclidean,
                &[0.0],
                &adjacency,
                &[2, 2, 99],
                10,
                3,
            );
            assert_eq!(rows(&bounded), [0, 1, 2]);
            assert!(bounded.windows(2).all(|pair| pair[0] < pair[1]));

            let reachable =
                search_layer(&dataset, Metric::Euclidean, &[0.0], &adjacency, &[2], 10, 5);
            assert_eq!(rows(&reachable), [0, 1, 2]);

            let narrow =
                search_layer(&dataset, Metric::Euclidean, &[0.0], &adjacency, &[2], 1, 3);
            assert_eq!(rows(&narrow), [0]);
        }
    }

    mod checkpoint_2 {
        use super::*;

        #[test]
        fn prune_neighbors_is_deterministic_and_bounded() {
            let dataset =
                Dataset::try_new(vec![vec![0.0], vec![-1.0], vec![1.0], vec![4.0]]).unwrap();
            let mut neighbors = vec![2, 1, 1, 3];

            prune_neighbors(&dataset, Metric::Euclidean, 0, &mut neighbors, 2);

            assert_eq!(neighbors, [1, 2]);
        }
    }
}

mod day_04 {
    use crate::graph::greedy_search;
    use crate::{Dataset, Metric};

    mod checkpoint_1 {
        use super::*;

        #[test]
        fn greedy_search_moves_on_public_tie_order_and_respects_bounds() {
            let dataset =
                Dataset::try_new(vec![vec![8.0], vec![-1.0], vec![1.0], vec![0.0]]).unwrap();
            let adjacency = vec![vec![], vec![2, 3], vec![1, 3], vec![1, 2]];

            let bounded = greedy_search(&dataset, Metric::Euclidean, &[0.0], &adjacency, 2, 3);
            assert_eq!(bounded, 1);

            let unbounded = greedy_search(&dataset, Metric::Euclidean, &[0.0], &adjacency, 2, 4);
            assert_eq!(unbounded, 3);
        }
    }
}
