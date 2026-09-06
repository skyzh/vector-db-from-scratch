use std::path::Path;
use std::time::Duration;

use async_trait::async_trait;
use datafusion::arrow::datatypes::{DataType, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::display::array_value_to_string;
use datafusion::common::DataFusionError;
use sqllogictest::{AsyncDB, DBOutput, DefaultColumnType, Runner};
use vector_core::{HnswConfig, IndexConfig, IvfFlatConfig, IvfPqConfig, Metric, NswConfig};
use vector_datafusion_starter::{VectorSqlOutput, VectorSqlSession};

struct DataFusionDb {
    session: VectorSqlSession,
}

#[async_trait]
impl AsyncDB for DataFusionDb {
    type Error = DataFusionError;
    type ColumnType = DefaultColumnType;

    async fn run(&mut self, sql: &str) -> Result<DBOutput<Self::ColumnType>, Self::Error> {
        match self.session.execute(sql).await? {
            VectorSqlOutput::Query(batches) if is_explain(sql) => Ok(DBOutput::Rows {
                types: vec![DefaultColumnType::Text],
                rows: vector_plan_rows(&batches)?,
            }),
            VectorSqlOutput::Query(batches) => {
                let types = batches
                    .first()
                    .map(RecordBatch::schema)
                    .map(|schema| column_types(&schema))
                    .unwrap_or_default();
                Ok(DBOutput::Rows {
                    types,
                    rows: string_rows(&batches)?,
                })
            }
            VectorSqlOutput::StatementComplete(count) => Ok(DBOutput::StatementComplete(count)),
            VectorSqlOutput::CreatedIndex(_) => Ok(DBOutput::StatementComplete(0)),
        }
    }

    async fn shutdown(&mut self) {}

    fn engine_name(&self) -> &str {
        "DataFusion"
    }

    async fn sleep(duration: Duration) {
        tokio::time::sleep(duration).await;
    }
}

fn is_explain(sql: &str) -> bool {
    sql.trim_start()
        .get(..7)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("EXPLAIN"))
}

fn vector_plan_rows(batches: &[RecordBatch]) -> Result<Vec<Vec<String>>, DataFusionError> {
    let rows = string_rows(batches)?;
    let physical_plan = rows
        .iter()
        .find(|row| row.first().is_some_and(|value| value == "physical_plan"))
        .and_then(|row| row.get(1))
        .ok_or_else(|| DataFusionError::Execution("EXPLAIN omitted the physical plan".into()))?;
    let operators = physical_plan
        .lines()
        .map(str::trim)
        .filter(|line| {
            line.starts_with("SortExec")
                || line.starts_with("VectorIndexScanExec")
                || line.starts_with("FilterExec")
                || line.starts_with("DataSourceExec")
        })
        .map(|line| vec![line.to_owned()])
        .collect::<Vec<_>>();
    if operators.is_empty() {
        return Err(DataFusionError::Execution(
            "EXPLAIN omitted the vector plan operators".into(),
        ));
    }
    Ok(operators)
}

fn column_types(schema: &SchemaRef) -> Vec<DefaultColumnType> {
    schema
        .fields()
        .iter()
        .map(|field| match field.data_type() {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => DefaultColumnType::Integer,
            DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal32(_, _)
            | DataType::Decimal64(_, _)
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _) => DefaultColumnType::FloatingPoint,
            _ => DefaultColumnType::Text,
        })
        .collect()
}

fn string_rows(batches: &[RecordBatch]) -> Result<Vec<Vec<String>>, DataFusionError> {
    let mut rows = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            rows.push(
                batch
                    .columns()
                    .iter()
                    .map(|column| array_value_to_string(column.as_ref(), row))
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
    }
    Ok(rows)
}

async fn database(config: IndexConfig) -> Result<DataFusionDb, DataFusionError> {
    Ok(DataFusionDb {
        session: VectorSqlSession::new(Metric::Euclidean, config),
    })
}

async fn run_case(filename: &str, config: IndexConfig) {
    let mut runner = Runner::new(move || {
        let config = config.clone();
        async move { database(config).await }
    });
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/slt")
        .join(filename);
    if let Err(error) = runner.run_file_async(&path).await {
        panic!("{}", error.display(false));
    }
    runner.shutdown_async().await;
}

mod day_01 {
    mod checkpoint_5 {
        use super::super::*;

        #[tokio::test]
        async fn table_and_optimizer_sql() {
            run_case("vector.01-index-match.slt", IndexConfig::Flat).await;
        }
    }
}

mod day_02 {
    mod checkpoint_5 {
        use super::super::*;

        #[tokio::test]
        async fn ivfflat_sql() {
            run_case(
                "vector.02-ivfflat.slt",
                IndexConfig::IvfFlat(IvfFlatConfig {
                    partitions: 3,
                    probes: 3,
                    iterations: 8,
                    seed: 7,
                }),
            )
            .await;
        }
    }
}

mod day_03 {
    mod checkpoint_4 {
        use super::super::*;

        #[tokio::test]
        async fn nsw_sql() {
            run_case(
                "vector.03-nsw.slt",
                IndexConfig::Nsw(NswConfig {
                    max_connections: 4,
                    ef_construction: 8,
                    ef_search: 8,
                }),
            )
            .await;
        }
    }
}

mod day_04 {
    mod checkpoint_3 {
        use super::super::*;

        #[tokio::test]
        async fn hnsw_sql() {
            run_case(
                "vector.04-hnsw.slt",
                IndexConfig::Hnsw(HnswConfig {
                    max_connections: 4,
                    ef_construction: 8,
                    ef_search: 8,
                    max_level: 4,
                    seed: 7,
                }),
            )
            .await;
        }
    }
}

mod day_05 {
    mod checkpoint_4 {
        use super::super::*;

        #[tokio::test]
        async fn ivf_pq_sql() {
            run_case(
                "vector.05-ivfpq.slt",
                IndexConfig::IvfPq(IvfPqConfig {
                    partitions: 2,
                    probes: 2,
                    iterations: 4,
                    subquantizers: 1,
                    codebook_size: 4,
                    rerank: 8,
                    seed: 7,
                }),
            )
            .await;
        }
    }
}
