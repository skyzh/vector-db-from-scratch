use std::collections::HashSet;
use std::error::Error;
use std::time::{Duration, Instant};

use vector_benchmark_support::{
    Cli, Mode, RankRecall, TimedRun, Truth, load_sift1m, parse_cli, percentile, rank_recall,
    run_balanced,
};
use vector_core_starter::{
    Dataset, FlatIndex, HnswConfig, HnswIndex, IvfFlatConfig, IvfFlatIndex, IvfPqConfig,
    IvfPqIndex, Metric, Neighbor, NswConfig, NswIndex, VectorIndex,
};

const K: usize = 100;
const INDEX_COUNT: usize = 5;
const WARM_QUERY_COUNT: usize = 20;
const INDEX_NAMES: [&str; INDEX_COUNT] = ["flat", "ivf_flat", "nsw", "hnsw", "ivf_pq"];
const INDEX_CONFIGS: [&str; INDEX_COUNT] = [
    "exact",
    "partitions=32,probes=6,iterations=12,seed=7",
    "max_connections=12,ef_construction=64,ef_search=40",
    "max_connections=12,ef_construction=64,ef_search=40,max_level=12,seed=7",
    "partitions=32,probes=6,iterations=12,subquantizers=4,codebook_size=16,rerank=100,seed=7",
];

struct Workload {
    mode: Mode,
    truth: Truth,
    dataset: Dataset,
    queries: Vec<Vec<f32>>,
    exact_first: Vec<usize>,
}

struct BuiltIndex {
    name: &'static str,
    config: &'static str,
    index: Box<dyn VectorIndex>,
    build_time: Duration,
}

#[derive(Debug, Clone, Copy)]
struct Measurement {
    search_time: Duration,
    recall: RankRecall,
    p50: Duration,
    p99: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PqAccounting {
    codes_bytes: u64,
    codebooks_bytes: u64,
    full_vectors_bytes: u64,
}

impl PqAccounting {
    fn from_index(index: &IvfPqIndex) -> Self {
        Self {
            codes_bytes: u64::try_from(index.encoded_bytes()).expect("codes fit in u64"),
            codebooks_bytes: u64::try_from(index.codebook_bytes()).expect("codebooks fit in u64"),
            full_vectors_bytes: u64::try_from(index.full_precision_bytes())
                .expect("vectors fit in u64"),
        }
    }

    fn search_bytes(self) -> u64 {
        self.codes_bytes
            .checked_add(self.codebooks_bytes)
            .expect("IVF-PQ search representation byte count overflow")
    }

    fn compression(self) -> f64 {
        self.full_vectors_bytes as f64 / self.search_bytes() as f64
    }
}

fn main() {
    let cli = match parse_cli(std::env::args_os().skip(1)) {
        Ok(cli) => cli,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(error.exit_code());
        }
    };
    if let Err(error) = run(cli) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn Error>> {
    let sift = load_sift1m(&cli)?;
    let mode = sift.mode;
    let supplied_first = sift
        .supplied_ground_truth
        .iter()
        .map(|row| row[0])
        .collect::<Vec<_>>();
    let dataset = Dataset::try_new(sift.base)?;
    let queries = sift.queries;

    let dataset_for_flat = dataset.clone();
    let started = Instant::now();
    let flat = FlatIndex::try_new(dataset_for_flat, Metric::Euclidean)?;
    let flat_build = started.elapsed();
    let (truth, exact_first) = select_exact_truth(mode, supplied_first, || {
        queries
            .iter()
            .map(|query| flat.search(query, K).map(|rows| rows[0].row))
            .collect::<vector_core_starter::Result<Vec<_>>>()
    })?;

    let dataset_for_ivf_flat = dataset.clone();
    let ivf_flat_config = ivf_flat_config();
    let started = Instant::now();
    let ivf_flat = IvfFlatIndex::try_new(dataset_for_ivf_flat, Metric::Euclidean, ivf_flat_config)?;
    let ivf_flat_build = started.elapsed();

    let dataset_for_nsw = dataset.clone();
    let nsw_config = nsw_config();
    let started = Instant::now();
    let nsw = build_nsw(dataset_for_nsw, Metric::Euclidean, nsw_config)?;
    let nsw_build = started.elapsed();

    let dataset_for_hnsw = dataset.clone();
    let hnsw_config = hnsw_config();
    let started = Instant::now();
    let hnsw = build_hnsw(dataset_for_hnsw, Metric::Euclidean, hnsw_config)?;
    let hnsw_build = started.elapsed();

    let dataset_for_ivf_pq = dataset.clone();
    let ivf_pq_config = ivf_pq_config();
    let started = Instant::now();
    let ivf_pq = build_ivf_pq(dataset_for_ivf_pq, Metric::Euclidean, ivf_pq_config)?;
    let ivf_pq_build = started.elapsed();
    let accounting = PqAccounting::from_index(&ivf_pq);

    let indexes = vec![
        BuiltIndex {
            name: INDEX_NAMES[0],
            config: INDEX_CONFIGS[0],
            index: Box::new(flat),
            build_time: flat_build,
        },
        BuiltIndex {
            name: INDEX_NAMES[1],
            config: INDEX_CONFIGS[1],
            index: Box::new(ivf_flat),
            build_time: ivf_flat_build,
        },
        BuiltIndex {
            name: INDEX_NAMES[2],
            config: INDEX_CONFIGS[2],
            index: Box::new(nsw),
            build_time: nsw_build,
        },
        BuiltIndex {
            name: INDEX_NAMES[3],
            config: INDEX_CONFIGS[3],
            index: Box::new(hnsw),
            build_time: hnsw_build,
        },
        BuiltIndex {
            name: INDEX_NAMES[4],
            config: INDEX_CONFIGS[4],
            index: Box::new(ivf_pq),
            build_time: ivf_pq_build,
        },
    ];
    let runs = run_balanced(
        &queries,
        INDEX_COUNT,
        WARM_QUERY_COUNT.min(queries.len()),
        |index, query| indexes[index].index.search(query, K),
    )?;
    let workload = Workload {
        mode,
        truth,
        dataset,
        queries,
        exact_first,
    };
    for line in format_report(&workload, &indexes, &runs, accounting)? {
        println!("{line}");
    }
    Ok(())
}

fn select_exact_truth<E>(
    mode: Mode,
    supplied_first: Vec<usize>,
    recompute: impl FnOnce() -> Result<Vec<usize>, E>,
) -> Result<(Truth, Vec<usize>), E> {
    match mode {
        Mode::Full => Ok((Truth::SuppliedSift1m, supplied_first)),
        Mode::Smoke => Ok((Truth::RecomputedFlatSelectedBase, recompute()?)),
    }
}

fn ivf_flat_config() -> IvfFlatConfig {
    IvfFlatConfig {
        partitions: 32,
        probes: 6,
        iterations: 12,
        seed: 7,
    }
}

fn nsw_config() -> NswConfig {
    NswConfig {
        max_connections: 12,
        ef_construction: 64,
        ef_search: 40,
    }
}

fn hnsw_config() -> HnswConfig {
    HnswConfig {
        max_connections: 12,
        ef_construction: 64,
        ef_search: 40,
        max_level: 12,
        seed: 7,
    }
}

fn ivf_pq_config() -> IvfPqConfig {
    IvfPqConfig {
        partitions: 32,
        probes: 6,
        iterations: 12,
        subquantizers: 4,
        codebook_size: 16,
        rerank: 100,
        seed: 7,
    }
}

fn build_nsw(
    _dataset: Dataset,
    _metric: Metric,
    _config: NswConfig,
) -> vector_core_starter::Result<NswIndex> {
    todo!("Day 6: build NSW with the benchmark configuration")
}

fn build_hnsw(
    _dataset: Dataset,
    _metric: Metric,
    _config: HnswConfig,
) -> vector_core_starter::Result<HnswIndex> {
    todo!("Day 6: build HNSW with the benchmark configuration")
}

fn build_ivf_pq(
    _dataset: Dataset,
    _metric: Metric,
    _config: IvfPqConfig,
) -> vector_core_starter::Result<IvfPqIndex> {
    todo!("Day 6: build IVF-PQ with the benchmark configuration")
}

fn summarize(
    run: &TimedRun<Vec<Neighbor>>,
    exact_first: &[usize],
    base_rows: usize,
) -> Result<Measurement, Box<dyn Error>> {
    if run.results.len() != exact_first.len() || run.latencies.len() != exact_first.len() {
        return Err("search result count does not match query count".into());
    }
    let mut total = RankRecall {
        r1: 0.0,
        r10: 0.0,
        r100: 0.0,
    };
    for (neighbors, exact) in run.results.iter().zip(exact_first) {
        validate_neighbors(neighbors, base_rows, K)?;
        let rows = neighbors
            .iter()
            .map(|neighbor| neighbor.row)
            .collect::<Vec<_>>();
        let recall = rank_recall(&rows, *exact);
        total.r1 += recall.r1;
        total.r10 += recall.r10;
        total.r100 += recall.r100;
    }
    let queries = exact_first.len() as f64;
    let recall = RankRecall {
        r1: total.r1 / queries,
        r10: total.r10 / queries,
        r100: total.r100 / queries,
    };
    if !(0.0..=recall.r10).contains(&recall.r1) || !(recall.r10..=1.0).contains(&recall.r100) {
        return Err("rank recall is not finite and monotonic".into());
    }
    let mut latencies = run.latencies.clone();
    latencies.sort_unstable();
    let (p50, p99) = report_percentiles(&latencies);
    Ok(Measurement {
        search_time: run.latencies.iter().sum(),
        recall,
        p50,
        p99,
    })
}

fn report_percentiles(_sorted: &[Duration]) -> (Duration, Duration) {
    let _helper = percentile;
    todo!("Day 6: use the supplied nearest-rank percentile helper")
}

fn validate_neighbors(
    neighbors: &[Neighbor],
    base_rows: usize,
    k: usize,
) -> Result<(), Box<dyn Error>> {
    if neighbors.len() != k.min(base_rows) {
        return Err("search returned the wrong result count".into());
    }
    if neighbors
        .iter()
        .any(|neighbor| neighbor.row >= base_rows || !neighbor.distance.is_finite())
    {
        return Err("search returned an invalid row or distance".into());
    }
    if neighbors.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err("search results are not in public Neighbor order".into());
    }
    let unique = neighbors
        .iter()
        .map(|neighbor| neighbor.row)
        .collect::<HashSet<_>>();
    if unique.len() != neighbors.len() {
        return Err("search returned a duplicate row".into());
    }
    Ok(())
}

fn format_report(
    workload: &Workload,
    indexes: &[BuiltIndex],
    runs: &[TimedRun<Vec<Neighbor>>],
    accounting: PqAccounting,
) -> Result<Vec<String>, Box<dyn Error>> {
    if indexes.len() != INDEX_COUNT || runs.len() != INDEX_COUNT {
        return Err("benchmark index inventory is incomplete".into());
    }
    let mut lines = vec![format!(
        "workload: mode={}, parity={}, rows={}, dimensions={}, queries={}, metric=euclidean, k={K}, truth={}",
        workload.mode.mode_label(),
        workload.mode.parity_label(),
        workload.dataset.len(),
        workload.dataset.dimension(),
        workload.queries.len(),
        workload.truth.label(),
    )];
    for (ordinal, (index, run)) in indexes.iter().zip(runs).enumerate() {
        if index.name != INDEX_NAMES[ordinal]
            || index.config != INDEX_CONFIGS[ordinal]
            || index.index.kind() != index.name
            || index.index.dataset().vectors() != workload.dataset.vectors()
        {
            return Err("benchmark index inventory drifted".into());
        }
        let measurement = summarize(run, &workload.exact_first, workload.dataset.len())?;
        if ordinal == 0
            && measurement.recall
                != (RankRecall {
                    r1: 1.0,
                    r10: 1.0,
                    r100: 1.0,
                })
        {
            return Err("Flat disagrees with the selected exact truth".into());
        }
        lines.push(format_row(index, workload.queries.len(), measurement));
    }
    if workload.mode == Mode::Full
        && accounting
            != (PqAccounting {
                codes_bytes: 4_000_000,
                codebooks_bytes: 8_192,
                full_vectors_bytes: 512_000_000,
            })
    {
        return Err("full SIFT1M IVF-PQ accounting drifted".into());
    }
    lines.push(format_accounting(accounting));
    Ok(lines)
}

fn format_row(index: &BuiltIndex, query_count: usize, measurement: Measurement) -> String {
    let search_seconds = measurement.search_time.as_secs_f64();
    let qps = query_count as f64 / search_seconds;
    format!(
        "{}: config={}, build_s={:.3}, search_s={:.3}, qps={:.1}, r@1={:.4}, r@10={:.4}, r@100={:.4}, p50_ms={:.3}, p99_ms={:.3}",
        index.name,
        index.config,
        index.build_time.as_secs_f64(),
        search_seconds,
        qps,
        measurement.recall.r1,
        measurement.recall.r10,
        measurement.recall.r100,
        measurement.p50.as_secs_f64() * 1_000.0,
        measurement.p99.as_secs_f64() * 1_000.0,
    )
}

fn format_accounting(accounting: PqAccounting) -> String {
    format!(
        "ivf_pq search representation: codes_bytes={}, codebooks_bytes={}, search_bytes={}, full_vectors_bytes={}, compression={:.1}x",
        accounting.codes_bytes,
        accounting.codebooks_bytes,
        accounting.search_bytes(),
        accounting.full_vectors_bytes,
        accounting.compression(),
    )
}

#[cfg(test)]
mod day_06 {
    mod checkpoint_1 {
        use super::super::*;

        #[test]
        fn inventory_and_configs_match_the_frozen_matrix() {
            assert_eq!(INDEX_NAMES, ["flat", "ivf_flat", "nsw", "hnsw", "ivf_pq"]);
            assert_eq!(ivf_flat_config().seed, 7);
            assert_eq!(nsw_config().ef_search, 40);
            assert_eq!((hnsw_config().max_level, hnsw_config().seed), (12, 7));
            let pq = ivf_pq_config();
            assert_eq!(
                (pq.subquantizers, pq.codebook_size, pq.rerank, pq.seed),
                (4, 16, 100, 7)
            );

            let dataset = Dataset::try_new(vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.0],
                vec![1.0, 1.0, 0.0, 0.0],
                vec![0.0, 1.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0, 1.0],
                vec![1.0, 0.0, 0.0, 1.0],
            ])
            .unwrap();

            let nsw_config = NswConfig {
                max_connections: 2,
                ef_construction: 3,
                ef_search: 4,
            };
            let nsw = build_nsw(dataset.clone(), Metric::Dot, nsw_config).unwrap();
            assert_eq!(nsw.kind(), "nsw");
            assert_eq!(nsw.dataset().vectors(), dataset.vectors());
            assert_eq!(nsw.metric(), Metric::Dot);
            assert!(
                build_nsw(
                    dataset.clone(),
                    Metric::Dot,
                    NswConfig {
                        max_connections: 0,
                        ..nsw_config
                    },
                )
                .is_err()
            );

            let hnsw_config = HnswConfig {
                max_connections: 2,
                ef_construction: 3,
                ef_search: 4,
                max_level: 3,
                seed: 19,
            };
            let hnsw = build_hnsw(dataset.clone(), Metric::Cosine, hnsw_config).unwrap();
            assert_eq!(hnsw.kind(), "hnsw");
            assert_eq!(hnsw.dataset().vectors(), dataset.vectors());
            assert_eq!(hnsw.metric(), Metric::Cosine);
            assert!(
                build_hnsw(
                    dataset.clone(),
                    Metric::Cosine,
                    HnswConfig {
                        ef_search: 0,
                        ..hnsw_config
                    },
                )
                .is_err()
            );

            let ivf_pq_config = IvfPqConfig {
                partitions: 2,
                probes: 1,
                iterations: 2,
                subquantizers: 2,
                codebook_size: 2,
                rerank: 3,
                seed: 23,
            };
            let ivf_pq = build_ivf_pq(dataset.clone(), Metric::Euclidean, ivf_pq_config).unwrap();
            assert_eq!(ivf_pq.kind(), "ivf_pq");
            assert_eq!(ivf_pq.dataset().vectors(), dataset.vectors());
            assert_eq!(ivf_pq.metric(), Metric::Euclidean);
            assert!(
                build_ivf_pq(
                    dataset.clone(),
                    Metric::Euclidean,
                    IvfPqConfig {
                        subquantizers: 3,
                        ..ivf_pq_config
                    },
                )
                .is_err()
            );
            assert!(build_ivf_pq(dataset, Metric::Dot, ivf_pq_config).is_err());
        }
    }
    mod checkpoint_2 {
        use super::super::*;

        #[test]
        fn smoke_truth_is_recomputed_on_the_selected_base() {
            let (truth, exact) =
                select_exact_truth(Mode::Smoke, vec![99], || Ok::<_, &'static str>(vec![7]))
                    .unwrap();
            assert_eq!(truth, Truth::RecomputedFlatSelectedBase);
            assert_eq!(exact, [7]);

            let (truth, exact) = select_exact_truth(Mode::Full, vec![99], || {
                Err::<Vec<usize>, _>("full mode must not recompute")
            })
            .unwrap();
            assert_eq!(truth, Truth::SuppliedSift1m);
            assert_eq!(exact, [99]);
        }

        #[test]
        fn result_validation_requires_complete_unique_public_order() {
            let valid = (0..100)
                .map(|row| Neighbor {
                    row,
                    distance: row as f32,
                })
                .collect::<Vec<_>>();
            assert!(validate_neighbors(&valid, 10_000, 100).is_ok());
            assert!(validate_neighbors(&valid[..99], 10_000, 100).is_err());
        }

        #[test]
        fn summary_uses_rank_prefixes_and_validates_query_shape() {
            let neighbors = (0..100)
                .map(|row| Neighbor {
                    row,
                    distance: row as f32,
                })
                .collect::<Vec<_>>();
            let run = TimedRun {
                latencies: vec![Duration::from_millis(1), Duration::from_millis(2)],
                results: vec![neighbors.clone(), neighbors],
            };
            let measurement = summarize(&run, &[0, 5], 10_000).unwrap();
            assert_eq!(measurement.search_time, Duration::from_millis(3));
            assert_eq!(
                measurement.recall,
                RankRecall {
                    r1: 0.5,
                    r10: 1.0,
                    r100: 1.0,
                }
            );
            assert_eq!(measurement.p50, Duration::from_millis(1));
            assert_eq!(measurement.p99, Duration::from_millis(2));
            assert!(summarize(&run, &[0], 10_000).is_err());
        }

        #[test]
        fn full_accounting_matches_the_fixed_search_representation() {
            let accounting = PqAccounting {
                codes_bytes: 4_000_000,
                codebooks_bytes: 8_192,
                full_vectors_bytes: 512_000_000,
            };
            assert_eq!(accounting.search_bytes(), 4_008_192);
            assert_eq!(format!("{:.1}", accounting.compression()), "127.7");
        }
    }
}
