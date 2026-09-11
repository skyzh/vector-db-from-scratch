use std::collections::HashSet;
use std::error::Error;
use std::io::IsTerminal;
use std::time::{Duration, Instant};

use vector_benchmark_support::{
    Cli, FirstHit, Mode, Progress, TimedRun, Truth, first_hit, load_sift1m_with_progress,
    overlap_at_100, parse_cli, percentile, run_balanced_with_progress,
};
use vector_core::{
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
    "max_connections=12,ef_construction=64,ef_search_configured=40,ef_search_effective=100",
    "max_connections=12,ef_construction=64,ef_search_configured=40,ef_search_effective=100,max_level=12,seed=7",
    "partitions=32,probes=6,iterations=12,subquantizers=4,codebook_size=16,rerank=100,seed=7",
];

struct Workload {
    mode: Mode,
    truth: Truth,
    dataset: Dataset,
    queries: Vec<Vec<f32>>,
    exact_top_100: Vec<Vec<usize>>,
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
    first_hit: FirstHit,
    overlap_at_100: f64,
    returned_min: usize,
    returned_avg: f64,
    returned_max: usize,
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
    let stderr = std::io::stderr();
    let terminal = stderr.is_terminal();
    let mut progress = Progress::new(stderr, terminal);
    let sift = load_sift1m_with_progress(&cli, &mut progress)?;
    let mode = sift.mode;
    let supplied_top_100 = sift.supplied_ground_truth;
    let dataset = Dataset::try_new(sift.base)?;
    let queries = sift.queries;

    let dataset_for_flat = dataset.clone();
    progress.event("build flat", "start");
    let started = Instant::now();
    let flat = FlatIndex::try_new(dataset_for_flat, Metric::Euclidean)?;
    let flat_build = started.elapsed();
    progress.event("build flat", "complete");
    let (truth, exact_top_100) = select_exact_truth(mode, supplied_top_100, || {
        progress.begin("recompute smoke truth", "queries", queries.len(), None);
        let mut exact = Vec::with_capacity(queries.len());
        for (ordinal, query) in queries.iter().enumerate() {
            exact.push(
                flat.search(query, K)?
                    .into_iter()
                    .map(|neighbor| neighbor.row)
                    .collect(),
            );
            progress.advance(ordinal + 1);
        }
        progress.finish();
        Ok::<_, vector_core::VectorError>(exact)
    })?;

    let dataset_for_ivf_flat = dataset.clone();
    let ivf_flat_config = ivf_flat_config();
    progress.event("build ivf_flat", "start");
    let started = Instant::now();
    let ivf_flat = IvfFlatIndex::try_new(dataset_for_ivf_flat, Metric::Euclidean, ivf_flat_config)?;
    let ivf_flat_build = started.elapsed();
    progress.event("build ivf_flat", "complete");

    let dataset_for_nsw = dataset.clone();
    let nsw_config = nsw_config();
    progress.event("build nsw", "start");
    let started = Instant::now();
    let nsw = NswIndex::try_new(dataset_for_nsw, Metric::Euclidean, nsw_config)?;
    let nsw_build = started.elapsed();
    progress.event("build nsw", "complete");

    let dataset_for_hnsw = dataset.clone();
    let hnsw_config = hnsw_config();
    progress.event("build hnsw", "start");
    let started = Instant::now();
    let hnsw = HnswIndex::try_new(dataset_for_hnsw, Metric::Euclidean, hnsw_config)?;
    let hnsw_build = started.elapsed();
    progress.event("build hnsw", "complete");

    let dataset_for_ivf_pq = dataset.clone();
    let ivf_pq_config = ivf_pq_config();
    progress.event("build ivf_pq", "start");
    let started = Instant::now();
    let ivf_pq = IvfPqIndex::try_new(dataset_for_ivf_pq, Metric::Euclidean, ivf_pq_config)?;
    let ivf_pq_build = started.elapsed();
    progress.event("build ivf_pq", "complete");
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
    let runs = run_balanced_with_progress(
        &queries,
        INDEX_COUNT,
        WARM_QUERY_COUNT.min(queries.len()),
        &mut progress,
        |index, query| indexes[index].index.search(query, K),
    )?;
    let workload = Workload {
        mode,
        truth,
        dataset,
        queries,
        exact_top_100,
    };
    for line in format_report(&workload, &indexes, &runs, accounting)? {
        println!("{line}");
    }
    Ok(())
}

fn select_exact_truth<E>(
    mode: Mode,
    supplied_top_100: Vec<Vec<usize>>,
    recompute: impl FnOnce() -> Result<Vec<Vec<usize>>, E>,
) -> Result<(Truth, Vec<Vec<usize>>), E> {
    match mode {
        Mode::Full => Ok((Truth::SuppliedSift1m, supplied_top_100)),
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

fn summarize(
    run: &TimedRun<Vec<Neighbor>>,
    exact_top_100: &[Vec<usize>],
    base_rows: usize,
) -> Result<Measurement, Box<dyn Error>> {
    validate_run(run, exact_top_100, base_rows)?;
    let mut total = FirstHit {
        r1: 0.0,
        r10: 0.0,
        r100: 0.0,
    };
    let mut total_overlap = 0.0;
    let mut returned_min = usize::MAX;
    let mut returned_total = 0;
    let mut returned_max = 0;
    for (neighbors, exact) in run.results.iter().zip(exact_top_100) {
        let rows = neighbors
            .iter()
            .map(|neighbor| neighbor.row)
            .collect::<Vec<_>>();
        let hit = first_hit(&rows, exact[0]);
        total.r1 += hit.r1;
        total.r10 += hit.r10;
        total.r100 += hit.r100;
        total_overlap += overlap_at_100(&rows, exact);
        returned_min = returned_min.min(rows.len());
        returned_total += rows.len();
        returned_max = returned_max.max(rows.len());
    }
    let queries = exact_top_100.len() as f64;
    let first_hit = FirstHit {
        r1: total.r1 / queries,
        r10: total.r10 / queries,
        r100: total.r100 / queries,
    };
    if !(0.0..=first_hit.r10).contains(&first_hit.r1)
        || !(first_hit.r10..=1.0).contains(&first_hit.r100)
    {
        return Err("first-neighbor hit rate is not finite and monotonic".into());
    }
    let mut latencies = run.latencies.clone();
    latencies.sort_unstable();
    let (p50, p99) = report_percentiles(&latencies);
    Ok(Measurement {
        search_time: run.latencies.iter().sum(),
        first_hit,
        overlap_at_100: total_overlap / queries,
        returned_min,
        returned_avg: returned_total as f64 / queries,
        returned_max,
        p50,
        p99,
    })
}

fn validate_run(
    run: &TimedRun<Vec<Neighbor>>,
    exact_top_100: &[Vec<usize>],
    base_rows: usize,
) -> Result<(), Box<dyn Error>> {
    if exact_top_100.is_empty()
        || run.results.len() != exact_top_100.len()
        || run.latencies.len() != exact_top_100.len()
        || exact_top_100.iter().any(|truth| truth.len() != K)
    {
        return Err("search result count does not match query count".into());
    }
    for neighbors in &run.results {
        validate_neighbors(neighbors, base_rows, K)?;
    }
    Ok(())
}

fn report_percentiles(sorted: &[Duration]) -> (Duration, Duration) {
    (percentile(sorted, 50), percentile(sorted, 99))
}

fn validate_neighbors(
    neighbors: &[Neighbor],
    base_rows: usize,
    k: usize,
) -> Result<(), Box<dyn Error>> {
    if neighbors.len() > k.min(base_rows) {
        return Err("search returned more than k results".into());
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
    for run in runs {
        validate_run(run, &workload.exact_top_100, workload.dataset.len())?;
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
        let measurement = summarize(run, &workload.exact_top_100, workload.dataset.len())?;
        if ordinal == 0
            && (measurement.first_hit
                != (FirstHit {
                    r1: 1.0,
                    r10: 1.0,
                    r100: 1.0,
                })
                || measurement.overlap_at_100 != 1.0
                || measurement.returned_min != K
                || measurement.returned_max != K)
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
        "{}: config={}, build_s={:.3}, search_s={:.3}, qps={:.1}, first_hit@1={:.4}, first_hit@10={:.4}, first_hit@100={:.4}, overlap@100={:.4}, returned_min={}, returned_avg={:.1}, returned_max={}, p50_ms={:.3}, p99_ms={:.3}",
        index.name,
        index.config,
        index.build_time.as_secs_f64(),
        search_seconds,
        qps,
        measurement.first_hit.r1,
        measurement.first_hit.r10,
        measurement.first_hit.r100,
        measurement.overlap_at_100,
        measurement.returned_min,
        measurement.returned_avg,
        measurement.returned_max,
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
mod tests {
    use super::*;

    #[test]
    fn inventory_and_configs_match_the_frozen_matrix() {
        assert_eq!(INDEX_NAMES, ["flat", "ivf_flat", "nsw", "hnsw", "ivf_pq"]);
        assert_eq!(ivf_flat_config().seed, 7);
        assert_eq!(nsw_config().ef_search, 40);
        assert!(INDEX_CONFIGS[2].contains("ef_search_configured=40,ef_search_effective=100"));
        assert_eq!((hnsw_config().max_level, hnsw_config().seed), (12, 7));
        assert_eq!(
            INDEX_CONFIGS[3],
            "max_connections=12,ef_construction=64,ef_search_configured=40,ef_search_effective=100,max_level=12,seed=7"
        );
        let pq = ivf_pq_config();
        assert_eq!(
            (pq.subquantizers, pq.codebook_size, pq.rerank, pq.seed),
            (4, 16, 100, 7)
        );
    }

    #[test]
    fn smoke_truth_is_recomputed_on_the_selected_base() {
        let supplied = vec![(0..100).collect::<Vec<_>>()];
        let recomputed = vec![(100..200).collect::<Vec<_>>()];
        let (truth, exact) = select_exact_truth(Mode::Smoke, supplied.clone(), || {
            Ok::<_, &'static str>(recomputed.clone())
        })
        .unwrap();
        assert_eq!(truth, Truth::RecomputedFlatSelectedBase);
        assert_eq!(exact, recomputed);

        let (truth, exact) = select_exact_truth(Mode::Full, supplied.clone(), || {
            Err::<Vec<Vec<usize>>, _>("full mode must not recompute")
        })
        .unwrap();
        assert_eq!(truth, Truth::SuppliedSift1m);
        assert_eq!(exact, supplied);
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
        assert!(format_accounting(accounting).contains("search representation"));
    }

    #[test]
    fn result_validation_accepts_short_results_but_preserves_structural_errors() {
        let valid = (0..100)
            .map(|row| Neighbor {
                row,
                distance: row as f32,
            })
            .collect::<Vec<_>>();
        assert!(validate_neighbors(&valid, 10_000, 100).is_ok());
        assert!(validate_neighbors(&valid[..50], 10_000, 100).is_ok());
        assert!(validate_neighbors(&[], 10_000, 100).is_ok());
        let mut over_k = valid.clone();
        over_k.push(Neighbor {
            row: 100,
            distance: 100.0,
        });
        assert!(validate_neighbors(&over_k, 10_000, 100).is_err());
        let mut duplicate = valid.clone();
        duplicate[99].row = 98;
        duplicate[99].distance = 98.0;
        assert!(validate_neighbors(&duplicate, 10_000, 100).is_err());
        let unordered = [
            Neighbor {
                row: 1,
                distance: 1.0,
            },
            Neighbor {
                row: 0,
                distance: 0.0,
            },
        ];
        assert!(validate_neighbors(&unordered, 10_000, 100).is_err());
        let unordered_tie = [
            Neighbor {
                row: 1,
                distance: 0.0,
            },
            Neighbor {
                row: 0,
                distance: 0.0,
            },
        ];
        assert!(validate_neighbors(&unordered_tie, 10_000, 100).is_err());
        assert!(
            validate_neighbors(
                &[Neighbor {
                    row: 10_000,
                    distance: 0.0
                }],
                10_000,
                100
            )
            .is_err()
        );
        assert!(
            validate_neighbors(
                &[Neighbor {
                    row: 0,
                    distance: f32::NAN
                }],
                10_000,
                100
            )
            .is_err()
        );
    }

    #[test]
    fn summary_reports_first_hit_overlap_and_returned_counts() {
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
        let mut second_truth = (0..100).collect::<Vec<_>>();
        second_truth.swap(0, 5);
        let truth = vec![(0..100).collect::<Vec<_>>(), second_truth];
        let measurement = summarize(&run, &truth, 10_000).unwrap();
        assert_eq!(measurement.search_time, Duration::from_millis(3));
        assert_eq!(
            measurement.first_hit,
            FirstHit {
                r1: 0.5,
                r10: 1.0,
                r100: 1.0,
            }
        );
        assert_eq!(measurement.overlap_at_100, 1.0);
        assert_eq!(
            (
                measurement.returned_min,
                measurement.returned_avg,
                measurement.returned_max
            ),
            (100, 100.0, 100)
        );
        assert_eq!(measurement.p50, Duration::from_millis(1));
        assert_eq!(measurement.p99, Duration::from_millis(2));
        assert!(summarize(&run, &truth[..1], 10_000).is_err());
    }

    #[test]
    fn one_exact_first_and_fifty_true_rows_use_the_fixed_truth_denominator() {
        let truth = vec![(0..100).collect::<Vec<_>>()];
        let one_overlap = std::iter::once(0)
            .chain(100..199)
            .map(|row| Neighbor {
                row,
                distance: row as f32,
            })
            .collect::<Vec<_>>();
        let measurement = summarize(
            &TimedRun {
                latencies: vec![Duration::from_millis(1)],
                results: vec![one_overlap],
            },
            &truth,
            10_000,
        )
        .unwrap();
        assert_eq!(
            measurement.first_hit,
            FirstHit {
                r1: 1.0,
                r10: 1.0,
                r100: 1.0
            }
        );
        assert_eq!(measurement.overlap_at_100, 0.01);

        let fifty = (0..50)
            .map(|row| Neighbor {
                row,
                distance: row as f32,
            })
            .collect::<Vec<_>>();
        let measurement = summarize(
            &TimedRun {
                latencies: vec![Duration::from_millis(1)],
                results: vec![fifty],
            },
            &truth,
            10_000,
        )
        .unwrap();
        assert_eq!(measurement.overlap_at_100, 0.5);
        assert_eq!(
            (
                measurement.returned_min,
                measurement.returned_avg,
                measurement.returned_max
            ),
            (50, 50.0, 50)
        );
    }

    #[derive(Debug)]
    struct ReportIndex {
        name: &'static str,
        dataset: Dataset,
    }

    impl VectorIndex for ReportIndex {
        fn kind(&self) -> &'static str {
            self.name
        }

        fn dataset(&self) -> &Dataset {
            &self.dataset
        }

        fn metric(&self) -> Metric {
            Metric::Euclidean
        }

        fn search(&self, _query: &[f32], _k: usize) -> vector_core::Result<Vec<Neighbor>> {
            unreachable!("report formatting does not search")
        }
    }

    #[test]
    fn a_short_index_does_not_abort_the_five_index_report() {
        let dataset = Dataset::try_new((0..100).map(|row| vec![row as f32]).collect()).unwrap();
        let workload = Workload {
            mode: Mode::Smoke,
            truth: Truth::RecomputedFlatSelectedBase,
            dataset: dataset.clone(),
            queries: vec![vec![0.0]],
            exact_top_100: vec![(0..100).collect()],
        };
        let indexes = INDEX_NAMES
            .iter()
            .copied()
            .zip(INDEX_CONFIGS)
            .map(|(name, config)| BuiltIndex {
                name,
                config,
                index: Box::new(ReportIndex {
                    name,
                    dataset: dataset.clone(),
                }),
                build_time: Duration::from_millis(1),
            })
            .collect::<Vec<_>>();
        let runs = (0..INDEX_COUNT)
            .map(|ordinal| {
                let returned = if ordinal == 3 { 50 } else { 100 };
                TimedRun {
                    latencies: vec![Duration::from_millis(1)],
                    results: vec![
                        (0..returned)
                            .map(|row| Neighbor {
                                row,
                                distance: row as f32,
                            })
                            .collect(),
                    ],
                }
            })
            .collect::<Vec<_>>();
        let lines = format_report(
            &workload,
            &indexes,
            &runs,
            PqAccounting {
                codes_bytes: 1,
                codebooks_bytes: 1,
                full_vectors_bytes: 100,
            },
        )
        .unwrap();
        assert_eq!(lines.len(), 7);
        assert!(lines[4].contains("hnsw:"));
        assert!(lines[4].contains("overlap@100=0.5000"));
        assert!(lines[4].contains("returned_min=50, returned_avg=50.0, returned_max=50"));
    }

    #[test]
    fn a_broken_fifth_index_aborts_the_report_transactionally() {
        let dataset = Dataset::try_new((0..100).map(|row| vec![row as f32]).collect()).unwrap();
        let workload = Workload {
            mode: Mode::Smoke,
            truth: Truth::RecomputedFlatSelectedBase,
            dataset: dataset.clone(),
            queries: vec![vec![0.0]],
            exact_top_100: vec![(0..100).collect()],
        };
        let indexes = INDEX_NAMES
            .iter()
            .copied()
            .zip(INDEX_CONFIGS)
            .map(|(name, config)| BuiltIndex {
                name,
                config,
                index: Box::new(ReportIndex {
                    name,
                    dataset: dataset.clone(),
                }),
                build_time: Duration::from_millis(1),
            })
            .collect::<Vec<_>>();
        let runs = (0..INDEX_COUNT)
            .map(|ordinal| {
                let mut neighbors = (0..100)
                    .map(|row| Neighbor {
                        row,
                        distance: row as f32,
                    })
                    .collect::<Vec<_>>();
                if ordinal == INDEX_COUNT - 1 {
                    neighbors[99] = neighbors[98];
                }
                TimedRun {
                    latencies: vec![Duration::from_millis(1)],
                    results: vec![neighbors],
                }
            })
            .collect::<Vec<_>>();

        assert!(
            format_report(
                &workload,
                &indexes,
                &runs,
                PqAccounting {
                    codes_bytes: 1,
                    codebooks_bytes: 1,
                    full_vectors_bytes: 100,
                },
            )
            .is_err()
        );
    }

    #[test]
    fn report_percentiles_use_supplied_nearest_rank_helper() {
        let samples = (1..=100).map(Duration::from_micros).collect::<Vec<_>>();
        assert_eq!(
            report_percentiles(&samples),
            (Duration::from_micros(50), Duration::from_micros(99))
        );
    }
}
