use datafusion::arrow::array::StringArray;
use datafusion::arrow::util::display::array_value_to_string;
use datafusion::prelude::SessionContext;
use datafusion_cli::exec::exec_from_files;
use datafusion_cli::print_format::PrintFormat;
use datafusion_cli::print_options::{MaxRows, PrintOptions};
use std::path::PathBuf;
use vector_core::{IndexConfig, IvfFlatConfig, Metric};
use vector_datafusion::{VectorSqlOutput, VectorSqlSession};

fn session() -> VectorSqlSession {
    VectorSqlSession::new(
        Metric::Cosine,
        IndexConfig::IvfFlat(IvfFlatConfig {
            partitions: 2,
            probes: 2,
            iterations: 8,
            seed: 7,
        }),
    )
}

#[tokio::test]
async fn public_file_runner_returns_missing_file_errors() {
    let missing = PathBuf::from(format!(
        "target/definitely-missing-cli-input-{}-{}.sql",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    let options = PrintOptions {
        format: PrintFormat::Table,
        quiet: true,
        maxrows: MaxRows::Unlimited,
        color: false,
    };
    let error = exec_from_files(
        &SessionContext::new(),
        vec![missing.to_string_lossy().into_owned()],
        &options,
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("No such file"), "{error}");
}

#[test]
fn raw_cli_boundary_rejects_modifiers_that_logical_plans_erase() {
    let session = session();
    for sql in [
        "CREATE INDEX points_idx ON points USING ivfflat (embedding DESC)",
        "CREATE INDEX points_idx ON points USING ivfflat (embedding) INCLUDE (id)",
        "CREATE INDEX points_idx ON points USING ivfflat (embedding) WHERE id > 0",
    ] {
        let error = session.validate_cli_sql(sql).unwrap_err().to_string();
        assert!(error.contains("plain"), "{sql}: {error}");
    }
    session
        .validate_cli_sql("CREATE INDEX points_idx ON points USING ivfflat (embedding)")
        .unwrap();
}

#[tokio::test]
async fn logical_cli_boundary_rejects_sort_options_and_preserves_focused_behavior() {
    let mut session = session();
    for sql in [
        "CREATE TABLE points (id BIGINT NOT NULL, embedding REAL[3] NOT NULL)",
        "INSERT INTO points VALUES (1, [1.0, 0.0, 0.0]), (2, [0.0, 1.0, 0.0])",
    ] {
        session.execute(sql).await.unwrap();
    }

    let descending = session
        .cli_session_context()
        .state()
        .create_logical_plan("CREATE INDEX points_idx ON points USING ivfflat (embedding DESC)")
        .await
        .unwrap();
    let error = session
        .execute_cli_plan(descending)
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("plain CREATE INDEX"), "{error}");

    let wrong_kind = session
        .cli_session_context()
        .state()
        .create_logical_plan("CREATE INDEX points_idx ON points USING flat (embedding)")
        .await
        .unwrap();
    let error = session
        .execute_cli_plan(wrong_kind)
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("requires USING ivfflat"), "{error}");

    let canonical_sql = "CREATE INDEX points_idx ON points USING ivfflat (embedding)";
    session.validate_cli_sql(canonical_sql).unwrap();
    let canonical = session
        .cli_session_context()
        .state()
        .create_logical_plan(canonical_sql)
        .await
        .unwrap();
    let created = session
        .execute_cli_plan(canonical)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let result = created[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(result.value(0), "created vector index points_idx");

    let stale = session
        .cli_session_context()
        .state()
        .create_logical_plan("INSERT INTO points VALUES (3, [0.0, 0.0, 1.0])")
        .await
        .unwrap();
    let error = session
        .execute_cli_plan(stale)
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("would make its vector index stale"),
        "{error}"
    );

    assert_eq!(
        array_value_to_string(created[0].column(0).as_ref(), 0).unwrap(),
        "created vector index points_idx"
    );
    assert!(matches!(
        session.execute("SELECT id FROM points").await.unwrap(),
        VectorSqlOutput::Query(_)
    ));
}
