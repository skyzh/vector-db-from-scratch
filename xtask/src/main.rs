use std::env;
use std::path::Path;
use std::process::{Command, ExitCode};

const FIRST_DAY: u8 = 1;
const LAST_DAY: u8 = 6;
const PACKAGES: [&str; 3] = [
    "vector-db-from-scratch-benchmark-support",
    "vector-db-from-scratch-core-starter",
    "vector-db-from-scratch-datafusion-starter",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Selection {
    day: u8,
    checkpoint: Option<u8>,
}

impl Selection {
    fn filter(self) -> String {
        match self.checkpoint {
            Some(checkpoint) => format!("day_{:02}::checkpoint_{checkpoint}::", self.day),
            None => format!("day_{:02}::", self.day),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TestMode {
    Focused(Selection),
    Through(u8),
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let command = args.next().ok_or_else(usage)?;
    let selector = args.next().ok_or_else(usage)?;
    if args.next().is_some() {
        return Err(usage());
    }

    let mode = match command.as_str() {
        "test" => TestMode::Focused(parse_selection(&selector)?),
        "test-through" => {
            if selector.starts_with("day_") {
                let selection = parse_selection(&selector)?;
                if selection.checkpoint.is_some() {
                    return Err("test-through accepts a whole day, not a checkpoint".to_owned());
                }
                TestMode::Through(selection.day)
            } else {
                TestMode::Through(parse_legacy_day(&selector)?)
            }
        }
        // Keep the original numeric spelling as an undocumented compatibility alias.
        "test-day" => TestMode::Focused(Selection {
            day: parse_legacy_day(&selector)?,
            checkpoint: None,
        }),
        _ => return Err(usage()),
    };

    run_tests(mode)
}

fn parse_selection(value: &str) -> Result<Selection, String> {
    let mut parts = value.split("::");
    let day = parse_day_component(parts.next().unwrap_or_default())?;
    let checkpoint = match parts.next() {
        Some(value) => Some(parse_checkpoint_component(day, value)?),
        None => None,
    };
    if parts.next().is_some() {
        return Err(format!("invalid test selector '{value}'"));
    }
    Ok(Selection { day, checkpoint })
}

fn parse_day_component(value: &str) -> Result<u8, String> {
    let digits = value
        .strip_prefix("day_")
        .filter(|digits| digits.len() == 2)
        .ok_or_else(|| format!("invalid day selector '{value}'; expected day_01 through day_06"))?;
    let day = digits
        .parse::<u8>()
        .map_err(|_| format!("invalid day selector '{value}'; expected day_01 through day_06"))?;
    if !(FIRST_DAY..=LAST_DAY).contains(&day) {
        return Err(format!(
            "invalid day selector '{value}'; expected day_01 through day_06"
        ));
    }
    Ok(day)
}

fn parse_checkpoint_component(day: u8, value: &str) -> Result<u8, String> {
    value
        .strip_prefix("checkpoint_")
        .and_then(|digits| digits.parse::<u8>().ok())
        .filter(|checkpoint| has_tests(day, *checkpoint))
        .ok_or_else(|| format!("{value} has no supplied Rust tests on day_{day:02}"))
}

fn parse_legacy_day(value: &str) -> Result<u8, String> {
    let day = value
        .parse::<u8>()
        .map_err(|_| format!("day must be an integer from {FIRST_DAY} through {LAST_DAY}"))?;
    if !(FIRST_DAY..=LAST_DAY).contains(&day) {
        return Err(format!(
            "day must be an integer from {FIRST_DAY} through {LAST_DAY}"
        ));
    }
    Ok(day)
}

fn has_tests(day: u8, checkpoint: u8) -> bool {
    match day {
        1 => matches!(checkpoint, 1 | 3 | 4 | 5),
        2 => matches!(checkpoint, 1..=5),
        3 => matches!(checkpoint, 1..=4),
        4 => matches!(checkpoint, 1..=3),
        5 => matches!(checkpoint, 1..=4),
        6 => matches!(checkpoint, 1..=2),
        _ => false,
    }
}

fn run_tests(mode: TestMode) -> Result<(), String> {
    let (label, filter, skipped_days) = match mode {
        TestMode::Focused(selection) => {
            let filter = selection.filter();
            (filter.clone(), filter, Vec::new())
        }
        TestMode::Through(day) => (
            format!("day_01 through day_{day:02}"),
            "day_".to_owned(),
            ((day + 1)..=LAST_DAY).collect(),
        ),
    };

    println!("== Rust learner tests: {label} ==");
    let mut command = Command::new("cargo");
    command
        .current_dir(workspace_root())
        .arg("test")
        .arg("--locked");
    for package in PACKAGES {
        command.arg("-p").arg(package);
    }
    command.arg("--all-targets").arg(filter);

    let list_only = env::var_os("XTASK_LIST_ONLY").is_some();
    if list_only || !skipped_days.is_empty() {
        command.arg("--");
    }
    if list_only {
        command.arg("--list");
    }
    for day in skipped_days {
        command.arg("--skip").arg(format!("day_{day:02}::"));
    }

    println!("+ {command:?}");
    let status = command
        .status()
        .map_err(|error| format!("failed to start Cargo for {label}: {error}"))?;
    if !status.success() {
        return Err(format!("Rust learner tests failed: {label}"));
    }
    Ok(())
}

fn workspace_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask must live directly under the workspace root")
}

fn usage() -> String {
    "usage: cargo xtask test day_NN[::checkpoint_N] | cargo xtask test-through day_NN".to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_whole_day_and_test_bearing_checkpoint_selectors() {
        assert_eq!(
            parse_selection("day_02").unwrap(),
            Selection {
                day: 2,
                checkpoint: None,
            }
        );
        assert_eq!(
            parse_selection("day_02::checkpoint_1").unwrap(),
            Selection {
                day: 2,
                checkpoint: Some(1),
            }
        );
    }

    #[test]
    fn rejects_invalid_or_empty_selectors() {
        for invalid in [
            "",
            "day_2",
            "day_00",
            "day_07",
            "day_02::checkpoint_0",
            "day_02::checkpoint_6",
            "day_01::checkpoint_2",
            "day_02::checkpoint_1::extra",
        ] {
            assert!(parse_selection(invalid).is_err(), "accepted {invalid}");
        }
    }

    #[test]
    fn every_declared_checkpoint_has_a_namespace_filter() {
        for day in FIRST_DAY..=LAST_DAY {
            for checkpoint in 1..=6 {
                if has_tests(day, checkpoint) {
                    let selection = Selection {
                        day,
                        checkpoint: Some(checkpoint),
                    };
                    assert_eq!(
                        selection.filter(),
                        format!("day_{day:02}::checkpoint_{checkpoint}::")
                    );
                }
            }
        }
    }

    #[test]
    fn cumulative_selection_skips_every_future_day() {
        let requested = 2;
        let skipped = ((requested + 1)..=LAST_DAY)
            .map(|day| format!("day_{day:02}::"))
            .collect::<Vec<_>>();
        assert_eq!(skipped, ["day_03::", "day_04::", "day_05::", "day_06::"]);
    }
}
