use std::collections::HashSet;
use std::ffi::OsString;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const DIMENSIONS: usize = 128;
const BASE_ROWS: usize = 1_000_000;
const QUERY_ROWS: usize = 10_000;
const GROUND_TRUTH_WIDTH: usize = 100;
const BASE_BYTES: u64 = 516_000_000;
const QUERY_BYTES: u64 = 5_160_000;
const GROUND_TRUTH_BYTES: u64 = 4_040_000;
const SMOKE_BASE_ROWS: usize = 10_000;
const SMOKE_QUERY_ROWS: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Full,
    Smoke,
}

impl Mode {
    pub fn base_rows(self) -> usize {
        match self {
            Self::Full => BASE_ROWS,
            Self::Smoke => SMOKE_BASE_ROWS,
        }
    }

    pub fn query_rows(self) -> usize {
        match self {
            Self::Full => QUERY_ROWS,
            Self::Smoke => SMOKE_QUERY_ROWS,
        }
    }

    pub fn mode_label(self) -> &'static str {
        match self {
            Self::Full => "sift1m-full",
            Self::Smoke => "sift1m-smoke",
        }
    }

    pub fn parity_label(self) -> &'static str {
        match self {
            Self::Full => "bustub-sift1m",
            Self::Smoke => "non-parity",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cli {
    pub mode: Mode,
    pub data_dir: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Truth {
    SuppliedSift1m,
    RecomputedFlatSelectedBase,
}

impl Truth {
    pub fn label(self) -> &'static str {
        match self {
            Self::SuppliedSift1m => "supplied-sift1m-top-100",
            Self::RecomputedFlatSelectedBase => "recomputed-flat-selected-base-top-100",
        }
    }
}

#[derive(Debug)]
pub struct SiftWorkload {
    pub mode: Mode,
    pub base: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
    pub supplied_ground_truth: Vec<Vec<usize>>,
}

#[derive(Debug)]
pub struct TimedRun<T> {
    pub latencies: Vec<Duration>,
    pub results: Vec<T>,
}

#[derive(Debug)]
struct ProgressStage {
    label: String,
    unit: &'static str,
    total: usize,
    searches_per_unit: Option<usize>,
    next_percent: usize,
}

/// Best-effort benchmark progress written independently from the report.
#[derive(Debug)]
pub struct Progress<W: Write> {
    writer: W,
    terminal: bool,
    failed: bool,
    stage: Option<ProgressStage>,
}

impl<W: Write> Progress<W> {
    pub fn new(writer: W, terminal: bool) -> Self {
        Self {
            writer,
            terminal,
            failed: false,
            stage: None,
        }
    }

    pub fn begin(
        &mut self,
        label: impl Into<String>,
        unit: &'static str,
        total: usize,
        searches_per_unit: Option<usize>,
    ) {
        assert!(total > 0);
        self.stage = Some(ProgressStage {
            label: label.into(),
            unit,
            total,
            searches_per_unit,
            next_percent: 25,
        });
        self.write_milestone(0);
    }

    pub fn advance(&mut self, completed: usize) {
        let Some(stage) = &self.stage else {
            return;
        };
        let total = stage.total;
        let mut next_percent = stage.next_percent;
        while next_percent < 100
            && completed.saturating_mul(100) >= next_percent.saturating_mul(total)
        {
            self.write_milestone(next_percent);
            next_percent += 25;
        }
        if let Some(stage) = &mut self.stage {
            stage.next_percent = next_percent;
        }
    }

    pub fn finish(&mut self) {
        if self.stage.is_some() {
            self.write_milestone(100);
            self.stage = None;
        }
    }

    pub fn event(&mut self, label: &str, state: &str) {
        self.write_message(format!("progress: {label} {state}\n"));
    }

    pub fn failed(&self) -> bool {
        self.failed
    }

    pub fn into_inner(self) -> W {
        self.writer
    }

    fn write_milestone(&mut self, percent: usize) {
        let Some(stage) = &self.stage else {
            return;
        };
        let label = stage.label.clone();
        let unit = stage.unit;
        let total = stage.total;
        let searches_per_unit = stage.searches_per_unit;
        let completed = total.saturating_mul(percent).div_ceil(100);
        let searches = searches_per_unit.map(|per_unit| {
            (
                completed.saturating_mul(per_unit),
                total.saturating_mul(per_unit),
            )
        });
        let counts = match searches {
            Some((completed_searches, total_searches)) => format!(
                "{}={completed}/{}, searches={completed_searches}/{total_searches}",
                unit, total
            ),
            None => format!("{unit}={completed}/{total}"),
        };
        let prefix = if self.terminal { "\r" } else { "" };
        let suffix = if self.terminal && percent < 100 {
            ""
        } else {
            "\n"
        };
        self.write_message(format!(
            "{prefix}progress: {label} {percent}% ({counts}){suffix}"
        ));
    }

    fn write_message(&mut self, message: String) {
        if self.failed {
            return;
        }
        if self.writer.write_all(message.as_bytes()).is_err() || self.writer.flush().is_err() {
            self.failed = true;
        }
    }
}

impl Progress<io::Sink> {
    pub fn silent() -> Self {
        Self::new(io::sink(), false)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FirstHit {
    pub r1: f64,
    pub r10: f64,
    pub r100: f64,
}

#[derive(Debug)]
pub enum SiftError {
    Usage,
    Missing(PathBuf),
    Io {
        file: String,
        source: io::Error,
    },
    Size {
        file: String,
        expected: u64,
        actual: u64,
    },
    TruncatedHeader {
        file: String,
        record: usize,
    },
    Dimension {
        file: String,
        record: usize,
        expected: usize,
        actual: i32,
    },
    TruncatedPayload {
        file: String,
        record: usize,
        expected: usize,
        actual: usize,
    },
    NonFinite {
        file: String,
        record: usize,
        component: usize,
    },
    NegativeId {
        record: usize,
        id: usize,
        value: i32,
    },
    OutOfRangeId {
        record: usize,
        id: usize,
        value: usize,
        upper_bound: usize,
    },
    DuplicateId {
        record: usize,
        id: usize,
        value: usize,
    },
    Trailing {
        file: String,
        bytes: usize,
        records: usize,
    },
}

impl SiftError {
    pub fn exit_code(&self) -> i32 {
        if matches!(self, Self::Usage) { 2 } else { 1 }
    }
}

impl Display for SiftError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Usage => write!(formatter, "usage: recall [--smoke] <sift1m-dir>"),
            Self::Missing(path) => write!(formatter, "sift1m: missing {}", path.display()),
            Self::Io { file, source } => write!(formatter, "sift1m: {file}: {source}"),
            Self::Size {
                file,
                expected,
                actual,
            } => write!(
                formatter,
                "sift1m: {file}: expected {expected} bytes, got {actual}"
            ),
            Self::TruncatedHeader { file, record } => write!(
                formatter,
                "sift1m: {file} record {record}: truncated dimension header"
            ),
            Self::Dimension {
                file,
                record,
                expected,
                actual,
            } => write!(
                formatter,
                "sift1m: {file} record {record}: expected dimension {expected}, got {actual}"
            ),
            Self::TruncatedPayload {
                file,
                record,
                expected,
                actual,
            } => write!(
                formatter,
                "sift1m: {file} record {record}: truncated payload: expected {expected} bytes, got {actual}"
            ),
            Self::NonFinite {
                file,
                record,
                component,
            } => write!(
                formatter,
                "sift1m: {file} record {record} component {component}: non-finite value"
            ),
            Self::NegativeId { record, id, value } => write!(
                formatter,
                "sift1m: sift_groundtruth.ivecs record {record} id {id}: negative row id {value}"
            ),
            Self::OutOfRangeId {
                record,
                id,
                value,
                upper_bound,
            } => write!(
                formatter,
                "sift1m: sift_groundtruth.ivecs record {record} id {id}: row id {value} is outside 0..{upper_bound}"
            ),
            Self::DuplicateId { record, id, value } => write!(
                formatter,
                "sift1m: sift_groundtruth.ivecs record {record} id {id}: duplicate row id {value}"
            ),
            Self::Trailing {
                file,
                bytes,
                records,
            } => write!(
                formatter,
                "sift1m: {file}: trailing {bytes} bytes after {records} records"
            ),
        }
    }
}

impl std::error::Error for SiftError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

pub fn parse_cli<I, T>(args: I) -> Result<Cli, SiftError>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString>,
{
    let args = args.into_iter().map(Into::into).collect::<Vec<_>>();
    match args.as_slice() {
        [directory] if directory != "--smoke" && !directory.to_string_lossy().starts_with('-') => {
            Ok(Cli {
                mode: Mode::Full,
                data_dir: PathBuf::from(directory),
            })
        }
        [flag, directory] if flag == "--smoke" && !directory.to_string_lossy().starts_with('-') => {
            Ok(Cli {
                mode: Mode::Smoke,
                data_dir: PathBuf::from(directory),
            })
        }
        _ => Err(SiftError::Usage),
    }
}

pub fn load_sift1m(cli: &Cli) -> Result<SiftWorkload, SiftError> {
    let mut progress = Progress::silent();
    load_sift1m_with_progress(cli, &mut progress)
}

pub fn load_sift1m_with_progress<W: Write>(
    cli: &Cli,
    progress: &mut Progress<W>,
) -> Result<SiftWorkload, SiftError> {
    let base_path = cli.data_dir.join("sift_base.fvecs");
    let query_path = cli.data_dir.join("sift_query.fvecs");
    let ground_truth_path = cli.data_dir.join("sift_groundtruth.ivecs");

    let base = load_fvecs(
        &base_path,
        "sift_base.fvecs",
        BASE_BYTES,
        BASE_ROWS,
        DIMENSIONS,
        cli.mode.base_rows(),
        progress,
    )?;
    let queries = load_fvecs(
        &query_path,
        "sift_query.fvecs",
        QUERY_BYTES,
        QUERY_ROWS,
        DIMENSIONS,
        cli.mode.query_rows(),
        progress,
    )?;
    let supplied_ground_truth = load_ivecs(
        &ground_truth_path,
        GROUND_TRUTH_BYTES,
        QUERY_ROWS,
        GROUND_TRUTH_WIDTH,
        cli.mode.query_rows(),
        BASE_ROWS,
        progress,
    )?;

    Ok(SiftWorkload {
        mode: cli.mode,
        base,
        queries,
        supplied_ground_truth,
    })
}

pub fn run_balanced<T, E>(
    queries: &[Vec<f32>],
    index_count: usize,
    warm_query_count: usize,
    search: impl FnMut(usize, &[f32]) -> Result<T, E>,
) -> Result<Vec<TimedRun<T>>, E> {
    let mut progress = Progress::silent();
    run_balanced_with_progress(
        queries,
        index_count,
        warm_query_count,
        &mut progress,
        search,
    )
}

pub fn run_balanced_with_progress<T, E, W: Write>(
    queries: &[Vec<f32>],
    index_count: usize,
    warm_query_count: usize,
    progress: &mut Progress<W>,
    mut search: impl FnMut(usize, &[f32]) -> Result<T, E>,
) -> Result<Vec<TimedRun<T>>, E> {
    assert!(index_count > 0);
    if warm_query_count > 0 {
        progress.begin("warm search", "rounds", warm_query_count, Some(index_count));
    }
    for (query_ordinal, query) in queries.iter().take(warm_query_count).enumerate() {
        for offset in 0..index_count {
            let index_ordinal = (query_ordinal + offset) % index_count;
            std::hint::black_box(search(index_ordinal, query)?);
        }
        progress.advance(query_ordinal + 1);
    }
    if warm_query_count > 0 {
        progress.finish();
    }

    let mut runs = (0..index_count)
        .map(|_| TimedRun {
            latencies: Vec::with_capacity(queries.len()),
            results: Vec::with_capacity(queries.len()),
        })
        .collect::<Vec<_>>();
    if !queries.is_empty() {
        progress.begin("timed search", "rounds", queries.len(), Some(index_count));
    }
    for (query_ordinal, query) in queries.iter().enumerate() {
        for offset in 0..index_count {
            let index_ordinal = (query_ordinal + offset) % index_count;
            let started = Instant::now();
            let result = search(index_ordinal, query)?;
            let elapsed = started.elapsed();
            runs[index_ordinal].latencies.push(elapsed);
            runs[index_ordinal]
                .results
                .push(std::hint::black_box(result));
        }
        progress.advance(query_ordinal + 1);
    }
    if !queries.is_empty() {
        progress.finish();
    }
    Ok(runs)
}

pub fn first_hit(results: &[usize], exact_first: usize) -> FirstHit {
    let contains = |rank: usize| results.iter().take(rank).any(|row| *row == exact_first);
    FirstHit {
        r1: f64::from(contains(1)),
        r10: f64::from(contains(10)),
        r100: f64::from(contains(100)),
    }
}

pub fn overlap_at_100(results: &[usize], exact_top_100: &[usize]) -> f64 {
    assert_eq!(exact_top_100.len(), GROUND_TRUTH_WIDTH);
    let returned = results.iter().copied().collect::<HashSet<_>>();
    exact_top_100
        .iter()
        .filter(|row| returned.contains(row))
        .count() as f64
        / GROUND_TRUTH_WIDTH as f64
}

pub fn percentile(sorted: &[Duration], percent: usize) -> Duration {
    assert!(!sorted.is_empty());
    assert!(percent <= 100);
    let rank = (percent * sorted.len()).div_ceil(100).saturating_sub(1);
    sorted[rank.min(sorted.len() - 1)]
}

fn open_checked(path: &Path, file: &str, expected_size: u64) -> Result<BufReader<File>, SiftError> {
    let handle = match File::open(path) {
        Ok(handle) => handle,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Err(SiftError::Missing(path.to_owned()));
        }
        Err(source) => {
            return Err(SiftError::Io {
                file: file.to_owned(),
                source,
            });
        }
    };
    let actual = handle
        .metadata()
        .map_err(|source| SiftError::Io {
            file: file.to_owned(),
            source,
        })?
        .len();
    if actual != expected_size {
        return Err(SiftError::Size {
            file: file.to_owned(),
            expected: expected_size,
            actual,
        });
    }
    Ok(BufReader::new(handle))
}

fn load_fvecs<W: Write>(
    path: &Path,
    file: &str,
    expected_size: u64,
    records: usize,
    dimensions: usize,
    retain: usize,
    progress: &mut Progress<W>,
) -> Result<Vec<Vec<f32>>, SiftError> {
    let mut reader = open_checked(path, file, expected_size)?;
    read_fvecs_with_progress(&mut reader, file, records, dimensions, retain, progress)
}

fn load_ivecs<W: Write>(
    path: &Path,
    expected_size: u64,
    records: usize,
    dimensions: usize,
    retain: usize,
    upper_bound: usize,
    progress: &mut Progress<W>,
) -> Result<Vec<Vec<usize>>, SiftError> {
    let mut reader = open_checked(path, "sift_groundtruth.ivecs", expected_size)?;
    read_ivecs_with_progress(
        &mut reader,
        "sift_groundtruth.ivecs",
        records,
        dimensions,
        retain,
        upper_bound,
        progress,
    )
}

fn fill<R: Read>(reader: &mut R, file: &str, bytes: &mut [u8]) -> Result<usize, SiftError> {
    let mut filled = 0;
    while filled < bytes.len() {
        match reader.read(&mut bytes[filled..]) {
            Ok(0) => break,
            Ok(count) => filled += count,
            Err(source) => {
                return Err(SiftError::Io {
                    file: file.to_owned(),
                    source,
                });
            }
        }
    }
    Ok(filled)
}

fn read_header<R: Read>(reader: &mut R, file: &str, record: usize) -> Result<i32, SiftError> {
    let mut header = [0_u8; 4];
    if fill(reader, file, &mut header)? != header.len() {
        return Err(SiftError::TruncatedHeader {
            file: file.to_owned(),
            record,
        });
    }
    Ok(i32::from_le_bytes(header))
}

fn check_dimension(
    file: &str,
    record: usize,
    expected: usize,
    actual: i32,
) -> Result<(), SiftError> {
    if actual <= 0 || usize::try_from(actual).ok() != Some(expected) {
        return Err(SiftError::Dimension {
            file: file.to_owned(),
            record,
            expected,
            actual,
        });
    }
    Ok(())
}

fn check_trailing<R: Read>(reader: &mut R, file: &str, records: usize) -> Result<(), SiftError> {
    let mut trailing = Vec::new();
    reader
        .read_to_end(&mut trailing)
        .map_err(|source| SiftError::Io {
            file: file.to_owned(),
            source,
        })?;
    if !trailing.is_empty() {
        return Err(SiftError::Trailing {
            file: file.to_owned(),
            bytes: trailing.len(),
            records,
        });
    }
    Ok(())
}

#[cfg(test)]
fn read_fvecs<R: Read>(
    reader: &mut R,
    file: &str,
    records: usize,
    dimensions: usize,
    retain: usize,
) -> Result<Vec<Vec<f32>>, SiftError> {
    let mut progress = Progress::silent();
    read_fvecs_with_progress(reader, file, records, dimensions, retain, &mut progress)
}

fn read_fvecs_with_progress<R: Read, W: Write>(
    reader: &mut R,
    file: &str,
    records: usize,
    dimensions: usize,
    retain: usize,
    progress: &mut Progress<W>,
) -> Result<Vec<Vec<f32>>, SiftError> {
    assert!(retain <= records);
    progress.begin(format!("load {file}"), "rows", records, None);
    let payload_bytes = dimensions.checked_mul(4).expect("vector width overflow");
    let mut payload = vec![0_u8; payload_bytes];
    let mut retained = Vec::with_capacity(retain);
    for record in 0..records {
        let actual = read_header(reader, file, record)?;
        check_dimension(file, record, dimensions, actual)?;
        let actual_bytes = fill(reader, file, &mut payload)?;
        if actual_bytes != payload_bytes {
            return Err(SiftError::TruncatedPayload {
                file: file.to_owned(),
                record,
                expected: payload_bytes,
                actual: actual_bytes,
            });
        }
        let keep = record < retain;
        let mut values = keep.then(|| Vec::with_capacity(dimensions));
        for (component, bytes) in payload.as_chunks::<4>().0.iter().enumerate() {
            let value = f32::from_bits(u32::from_le_bytes(*bytes));
            if !value.is_finite() {
                return Err(SiftError::NonFinite {
                    file: file.to_owned(),
                    record,
                    component,
                });
            }
            if let Some(values) = &mut values {
                values.push(value);
            }
        }
        if let Some(values) = values {
            retained.push(values);
        }
        progress.advance(record + 1);
    }
    check_trailing(reader, file, records)?;
    progress.finish();
    Ok(retained)
}

#[cfg(test)]
fn read_ivecs<R: Read>(
    reader: &mut R,
    file: &str,
    records: usize,
    dimensions: usize,
    retain: usize,
    upper_bound: usize,
) -> Result<Vec<Vec<usize>>, SiftError> {
    let mut progress = Progress::silent();
    read_ivecs_with_progress(
        reader,
        file,
        records,
        dimensions,
        retain,
        upper_bound,
        &mut progress,
    )
}

fn read_ivecs_with_progress<R: Read, W: Write>(
    reader: &mut R,
    file: &str,
    records: usize,
    dimensions: usize,
    retain: usize,
    upper_bound: usize,
    progress: &mut Progress<W>,
) -> Result<Vec<Vec<usize>>, SiftError> {
    assert!(retain <= records);
    progress.begin(format!("load {file}"), "rows", records, None);
    let payload_bytes = dimensions.checked_mul(4).expect("vector width overflow");
    let mut payload = vec![0_u8; payload_bytes];
    let mut retained = Vec::with_capacity(retain);
    for record in 0..records {
        let actual = read_header(reader, file, record)?;
        check_dimension(file, record, dimensions, actual)?;
        let actual_bytes = fill(reader, file, &mut payload)?;
        if actual_bytes != payload_bytes {
            return Err(SiftError::TruncatedPayload {
                file: file.to_owned(),
                record,
                expected: payload_bytes,
                actual: actual_bytes,
            });
        }
        let mut seen = HashSet::with_capacity(dimensions);
        let keep = record < retain;
        let mut values = keep.then(|| Vec::with_capacity(dimensions));
        for (id, bytes) in payload.as_chunks::<4>().0.iter().enumerate() {
            let signed = i32::from_le_bytes(*bytes);
            let value = usize::try_from(signed).map_err(|_| SiftError::NegativeId {
                record,
                id,
                value: signed,
            })?;
            if value >= upper_bound {
                return Err(SiftError::OutOfRangeId {
                    record,
                    id,
                    value,
                    upper_bound,
                });
            }
            if !seen.insert(value) {
                return Err(SiftError::DuplicateId { record, id, value });
            }
            if let Some(values) = &mut values {
                values.push(value);
            }
        }
        if let Some(values) = values {
            retained.push(values);
        }
        progress.advance(record + 1);
    }
    check_trailing(reader, file, records)?;
    progress.finish();
    Ok(retained)
}

#[cfg(test)]
mod day_06 {
    use std::cell::RefCell;
    use std::fs;
    use std::io::{Cursor, Write};
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    mod checkpoint_1 {
        use super::*;

        fn fvecs(records: &[&[f32]]) -> Vec<u8> {
            let mut bytes = Vec::new();
            for record in records {
                bytes.extend_from_slice(&i32::try_from(record.len()).unwrap().to_le_bytes());
                for value in *record {
                    bytes.extend_from_slice(&value.to_bits().to_le_bytes());
                }
            }
            bytes
        }

        fn ivecs(records: &[&[i32]]) -> Vec<u8> {
            let mut bytes = Vec::new();
            for record in records {
                bytes.extend_from_slice(&i32::try_from(record.len()).unwrap().to_le_bytes());
                for value in *record {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
            bytes
        }

        struct FailingWriter;

        impl Write for FailingWriter {
            fn write(&mut self, _buffer: &[u8]) -> io::Result<usize> {
                Err(io::Error::other("simulated closed stderr"))
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        struct SlowWriter {
            delay: Duration,
        }

        impl Write for SlowWriter {
            fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
                thread::sleep(self.delay);
                Ok(buffer.len())
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        #[test]
        fn tiny_le_fixtures_decode_exact_values_and_ids() {
            let mut vectors = Cursor::new(include_bytes!("../tests/fixtures/tiny.fvecs"));
            let parsed = read_fvecs(&mut vectors, "tiny.fvecs", 2, 2, 2).unwrap();
            assert_eq!(parsed, [vec![1.0, 2.0], vec![3.5, -4.0]]);

            let mut ids = Cursor::new(include_bytes!("../tests/fixtures/tiny.ivecs"));
            let parsed = read_ivecs(&mut ids, "sift_groundtruth.ivecs", 2, 3, 2, 4).unwrap();
            assert_eq!(parsed, [vec![0, 2, 1], vec![3, 1, 0]]);
        }

        #[test]
        fn cli_requires_an_explicit_full_or_fixed_smoke_run() {
            assert_eq!(
                parse_cli(["/data/sift1M"]).unwrap(),
                Cli {
                    mode: Mode::Full,
                    data_dir: PathBuf::from("/data/sift1M"),
                }
            );
            assert_eq!(
                parse_cli(["--smoke", "/data/sift1M"]).unwrap().mode,
                Mode::Smoke
            );
            for invalid in [vec![], vec!["--other"], vec!["/data", "--smoke"]] {
                assert!(matches!(parse_cli(invalid), Err(SiftError::Usage)));
            }
            assert_eq!(SiftError::Usage.exit_code(), 2);
        }

        #[test]
        fn file_boundary_reports_missing_and_exact_size_errors_before_decoding() {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let directory = std::env::temp_dir().join(format!(
                "vector-benchmark-support-{}-{nonce}",
                std::process::id()
            ));
            fs::create_dir(&directory).unwrap();
            let missing = directory.join("missing.fvecs");
            assert!(matches!(
                open_checked(&missing, "missing.fvecs", 12),
                Err(SiftError::Missing(path)) if path == missing
            ));

            let short = directory.join("short.fvecs");
            fs::write(&short, [0_u8; 8]).unwrap();
            let error = open_checked(&short, "short.fvecs", 12).unwrap_err();
            assert!(matches!(
                error,
                SiftError::Size {
                    expected: 12,
                    actual: 8,
                    ..
                }
            ));
            assert_eq!(error.exit_code(), 1);
            fs::remove_dir_all(directory).unwrap();
        }

        #[test]
        fn decoder_rejects_each_structural_and_value_corruption() {
            let good = fvecs(&[&[1.0, 2.0], &[3.0, 4.0]]);

            let mut wrong_dimension = good.clone();
            wrong_dimension[..4].copy_from_slice(&3_i32.to_le_bytes());
            assert!(matches!(
                read_fvecs(&mut Cursor::new(wrong_dimension), "tiny.fvecs", 2, 2, 2),
                Err(SiftError::Dimension { record: 0, .. })
            ));

            assert!(matches!(
                read_fvecs(&mut Cursor::new(&good[..1]), "tiny.fvecs", 2, 2, 2),
                Err(SiftError::TruncatedHeader { record: 0, .. })
            ));
            assert!(matches!(
                read_fvecs(&mut Cursor::new(&good[..10]), "tiny.fvecs", 2, 2, 2),
                Err(SiftError::TruncatedPayload { record: 0, .. })
            ));

            let mut non_finite = good.clone();
            non_finite[4..8].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
            assert!(matches!(
                read_fvecs(&mut Cursor::new(non_finite), "tiny.fvecs", 2, 2, 2),
                Err(SiftError::NonFinite {
                    record: 0,
                    component: 0,
                    ..
                })
            ));

            let mut trailing = good;
            trailing.push(1);
            assert!(matches!(
                read_fvecs(&mut Cursor::new(trailing), "tiny.fvecs", 2, 2, 2),
                Err(SiftError::Trailing { bytes: 1, .. })
            ));
        }

        #[test]
        fn ground_truth_rejects_negative_out_of_range_and_duplicate_ids() {
            let cases = [
                (ivecs(&[&[-1, 0]]), "negative"),
                (ivecs(&[&[0, 4]]), "outside"),
                (ivecs(&[&[1, 1]]), "duplicate"),
            ];
            for (bytes, expected) in cases {
                let error = read_ivecs(
                    &mut Cursor::new(bytes),
                    "sift_groundtruth.ivecs",
                    1,
                    2,
                    1,
                    4,
                )
                .unwrap_err();
                assert!(error.to_string().contains(expected));
            }
        }

        #[test]
        fn first_hit_uses_first_neighbor_prefixes() {
            assert_eq!(
                first_hit(&[7, 8, 9], 7),
                FirstHit {
                    r1: 1.0,
                    r10: 1.0,
                    r100: 1.0,
                }
            );
            let at_six = first_hit(&[0, 1, 2, 3, 4, 7], 7);
            assert_eq!((at_six.r1, at_six.r10, at_six.r100), (0.0, 1.0, 1.0));
            let at_fifty_one = first_hit(&(0..=50).collect::<Vec<_>>(), 50);
            assert_eq!(
                (at_fifty_one.r1, at_fifty_one.r10, at_fifty_one.r100),
                (0.0, 0.0, 1.0)
            );
            assert_eq!(first_hit(&[0, 1, 2], 9).r100, 0.0);
        }

        #[test]
        fn overlap_uses_the_fixed_top_100_denominator_and_ignores_truth_order() {
            let exact = (0..100).collect::<Vec<_>>();
            let one_overlap = std::iter::once(0).chain(100..199).collect::<Vec<_>>();
            assert_eq!(
                first_hit(&one_overlap, 0),
                FirstHit {
                    r1: 1.0,
                    r10: 1.0,
                    r100: 1.0
                }
            );
            assert_eq!(overlap_at_100(&one_overlap, &exact), 0.01);

            let fifty = (0..50).collect::<Vec<_>>();
            assert_eq!(overlap_at_100(&fifty, &exact), 0.5);

            let mut permuted = exact.clone();
            permuted.rotate_left(37);
            assert_eq!(overlap_at_100(&one_overlap, &permuted), 0.01);
        }

        #[test]
        fn balanced_runner_warms_twenty_then_times_every_query_cyclically() {
            let queries = (0..23).map(|query| vec![query as f32]).collect::<Vec<_>>();
            let trace = RefCell::new(Vec::new());
            let runs = run_balanced(&queries, 5, 20, |index, query| {
                trace.borrow_mut().push((index, query[0] as usize));
                Ok::<_, ()>((index, query[0] as usize))
            })
            .unwrap();
            let trace = trace.into_inner();
            assert_eq!(trace.len(), (20 + 23) * 5);
            for (phase_queries, phase) in [(20, &trace[..100]), (23, &trace[100..])] {
                for (query, calls) in phase.as_chunks::<5>().0.iter().enumerate() {
                    assert!(query < phase_queries);
                    for (offset, (index, observed)) in calls.iter().enumerate() {
                        assert_eq!(*index, (query + offset) % 5);
                        assert_eq!(*observed, query);
                    }
                }
            }
            assert!(
                runs.iter()
                    .all(|run| run.results.len() == 23 && run.latencies.len() == 23)
            );
        }

        #[test]
        fn balanced_runner_propagates_search_errors() {
            let queries = vec![vec![0.0], vec![1.0]];
            let result = run_balanced(&queries, 2, 0, |index, query| {
                if index == 1 && query[0] == 0.0 {
                    Err("search failed")
                } else {
                    Ok(())
                }
            });
            assert_eq!(result.unwrap_err(), "search failed");
        }

        #[test]
        fn non_terminal_and_terminal_progress_have_exact_bounded_milestones() {
            let mut non_terminal = Progress::new(Vec::new(), false);
            non_terminal.begin("timed search", "rounds", 4, Some(5));
            for completed in 1..=4 {
                non_terminal.advance(completed);
            }
            non_terminal.finish();
            assert_eq!(
                String::from_utf8(non_terminal.into_inner()).unwrap(),
                concat!(
                    "progress: timed search 0% (rounds=0/4, searches=0/20)\n",
                    "progress: timed search 25% (rounds=1/4, searches=5/20)\n",
                    "progress: timed search 50% (rounds=2/4, searches=10/20)\n",
                    "progress: timed search 75% (rounds=3/4, searches=15/20)\n",
                    "progress: timed search 100% (rounds=4/4, searches=20/20)\n",
                )
            );

            let mut terminal = Progress::new(Vec::new(), true);
            terminal.begin("timed search", "rounds", 4, Some(5));
            for completed in 1..=4 {
                terminal.advance(completed);
            }
            terminal.finish();
            assert_eq!(
                String::from_utf8(terminal.into_inner()).unwrap(),
                concat!(
                    "\rprogress: timed search 0% (rounds=0/4, searches=0/20)",
                    "\rprogress: timed search 25% (rounds=1/4, searches=5/20)",
                    "\rprogress: timed search 50% (rounds=2/4, searches=10/20)",
                    "\rprogress: timed search 75% (rounds=3/4, searches=15/20)",
                    "\rprogress: timed search 100% (rounds=4/4, searches=20/20)\n",
                )
            );
        }

        #[test]
        fn sift_progress_uses_physical_and_balanced_search_denominators() {
            let mut progress = Progress::new(Vec::new(), false);
            for (label, unit, total, searches_per_unit) in [
                ("load sift_base.fvecs", "rows", 1_000_000, None),
                ("load sift_query.fvecs", "rows", 10_000, None),
                ("load sift_groundtruth.ivecs", "rows", 10_000, None),
                ("recompute smoke truth", "queries", 100, None),
                ("warm search", "rounds", 20, Some(5)),
                ("timed smoke search", "rounds", 100, Some(5)),
                ("timed full search", "rounds", 10_000, Some(5)),
            ] {
                progress.begin(label, unit, total, searches_per_unit);
                progress.advance(total / 4);
                progress.advance(total / 2);
                progress.advance(total * 3 / 4);
                progress.advance(total);
                progress.finish();
            }
            let output = String::from_utf8(progress.into_inner()).unwrap();
            assert!(output.contains("load sift_base.fvecs 100% (rows=1000000/1000000)"));
            assert!(output.contains("load sift_query.fvecs 100% (rows=10000/10000)"));
            assert!(output.contains("load sift_groundtruth.ivecs 100% (rows=10000/10000)"));
            assert!(output.contains("recompute smoke truth 100% (queries=100/100)"));
            assert!(output.contains("warm search 100% (rounds=20/20, searches=100/100)"));
            assert!(output.contains("timed smoke search 100% (rounds=100/100, searches=500/500)"));
            assert!(
                output
                    .contains("timed full search 100% (rounds=10000/10000, searches=50000/50000)")
            );
            assert_eq!(output.lines().count(), 35);
        }

        #[test]
        fn file_progress_counts_physical_rows_while_retaining_a_subset() {
            let bytes = fvecs(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0], &[7.0, 8.0]]);
            let mut progress = Progress::new(Vec::new(), false);
            let retained = read_fvecs_with_progress(
                &mut Cursor::new(bytes),
                "tiny.fvecs",
                4,
                2,
                1,
                &mut progress,
            )
            .unwrap();
            assert_eq!(retained, [vec![1.0, 2.0]]);
            let output = String::from_utf8(progress.into_inner()).unwrap();
            assert_eq!(output.lines().count(), 5);
            assert!(output.contains("0% (rows=0/4)"));
            assert!(output.contains("100% (rows=4/4)"));
        }

        #[test]
        fn silent_and_progress_aware_runners_preserve_results_and_report_output() {
            let queries = (0..4).map(|query| vec![query as f32]).collect::<Vec<_>>();
            let silent = run_balanced(&queries, 2, 2, |index, query| {
                Ok::<_, ()>((index, query[0] as usize))
            })
            .unwrap();
            let report_stdout = vec!["workload: unchanged".to_owned()];
            let mut progress = Progress::new(Vec::new(), false);
            let visible =
                run_balanced_with_progress(&queries, 2, 2, &mut progress, |index, query| {
                    Ok::<_, ()>((index, query[0] as usize))
                })
                .unwrap();
            assert_eq!(
                silent.iter().map(|run| &run.results).collect::<Vec<_>>(),
                visible.iter().map(|run| &run.results).collect::<Vec<_>>()
            );
            assert_eq!(report_stdout, ["workload: unchanged"]);
            assert!(!progress.into_inner().is_empty());
        }

        #[test]
        fn failed_progress_writer_is_nonfatal_and_progress_time_is_not_sampled() {
            let mut failed = Progress::new(FailingWriter, false);
            failed.begin("load tiny", "rows", 4, None);
            failed.advance(4);
            failed.finish();
            failed.event("build flat", "start");
            assert!(failed.failed());

            let queries = (0..4).map(|query| vec![query as f32]).collect::<Vec<_>>();
            let mut slow = Progress::new(
                SlowWriter {
                    delay: Duration::from_millis(40),
                },
                false,
            );
            let runs = run_balanced_with_progress(&queries, 1, 0, &mut slow, |_index, query| {
                Ok::<_, ()>(query[0])
            })
            .unwrap();
            assert_eq!(runs[0].latencies.len(), 4);
            assert!(
                runs[0]
                    .latencies
                    .iter()
                    .all(|latency| *latency < Duration::from_millis(20))
            );
        }

        #[test]
        fn nearest_rank_percentile_covers_round_and_non_round_counts() {
            let one = [Duration::from_micros(9)];
            let two = [Duration::from_micros(2), Duration::from_micros(5)];
            let hundred = (1..=100).map(Duration::from_micros).collect::<Vec<_>>();
            let six = (1..=6).map(Duration::from_micros).collect::<Vec<_>>();
            assert_eq!(percentile(&one, 99), one[0]);
            assert_eq!(percentile(&two, 50), two[0]);
            assert_eq!(percentile(&hundred, 99), Duration::from_micros(99));
            assert_eq!(percentile(&six, 34), Duration::from_micros(3));
        }

        #[test]
        fn mode_labels_and_fixed_counts_cannot_claim_smoke_parity() {
            assert_eq!(Mode::Full.base_rows(), 1_000_000);
            assert_eq!(Mode::Full.query_rows(), 10_000);
            assert_eq!(Mode::Full.parity_label(), "bustub-sift1m");
            assert_eq!(Mode::Smoke.base_rows(), 10_000);
            assert_eq!(Mode::Smoke.query_rows(), 100);
            assert_eq!(Mode::Smoke.parity_label(), "non-parity");
        }
    }
}
