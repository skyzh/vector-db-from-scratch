use std::sync::Arc;

use datafusion::arrow::array::{
    ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, Int32Array,
    StringArray, UInt32Array, UInt64Array,
};
use datafusion::arrow::buffer::NullBuffer;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::common::Result as DataFusionResult;
use datafusion::datasource::MemTable;
use datafusion::execution::config::SessionConfig;
use datafusion::execution::context::SessionContext;
use vector_core::{HnswConfig, IndexConfig, IvfPqConfig, Metric};
use vector_datafusion_starter::{
    VectorIndexAttachment, VectorRow, vector_mem_table, with_vector_indexes,
    with_vector_search_options,
};

fn rows() -> Vec<VectorRow> {
    vec![
        VectorRow::new(10, vec![1.0, 0.0, 0.0], "east"),
        VectorRow::new(20, vec![0.9, 0.1, 0.0], "east"),
        VectorRow::new(30, vec![0.0, 1.0, 0.0], "north"),
        VectorRow::new(40, vec![-1.0, 0.0, 0.0], "west"),
        VectorRow::new(50, vec![0.0, 0.0, 1.0], "up"),
    ]
}

async fn context_with_batches(
    table_name: &str,
    batches: Vec<RecordBatch>,
    vector_column: &str,
    metric: Metric,
    config: IndexConfig,
) -> SessionContext {
    let base = SessionContext::new_with_config(with_vector_search_options(SessionConfig::new()));
    let table = Arc::new(MemTable::try_new(batches[0].schema(), vec![batches]).unwrap());
    base.register_table(table_name, table.clone()).unwrap();
    let attachment =
        VectorIndexAttachment::try_new(&base, table_name, &table, vector_column, metric, config)
            .await
            .unwrap();
    with_vector_indexes(&base, vec![attachment])
}

async fn attachment_result(
    batches: Vec<RecordBatch>,
    vector_column: &str,
) -> DataFusionResult<VectorIndexAttachment> {
    let base = SessionContext::new_with_config(with_vector_search_options(SessionConfig::new()));
    let table = Arc::new(MemTable::try_new(batches[0].schema(), vec![batches])?);
    base.register_table("documents", table.clone())?;
    VectorIndexAttachment::try_new(
        &base,
        "documents",
        &table,
        vector_column,
        Metric::Euclidean,
        IndexConfig::Flat,
    )
    .await
}

async fn context(metric: Metric, config: IndexConfig) -> SessionContext {
    let base = SessionContext::new_with_config(with_vector_search_options(SessionConfig::new()));
    let table = vector_mem_table(rows()).unwrap();
    base.register_table("points", table.clone()).unwrap();
    let attachment =
        VectorIndexAttachment::try_new(&base, "points", &table, "embedding", metric, config)
            .await
            .unwrap();
    with_vector_indexes(&base, vec![attachment])
}

fn vector_array<const N: usize>(vectors: &[[f32; N]]) -> ArrayRef {
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    Arc::new(
        FixedSizeListArray::try_new(
            item,
            i32::try_from(N).unwrap(),
            Arc::new(Float32Array::from(
                vectors
                    .iter()
                    .flat_map(|vector| vector.iter().copied())
                    .collect::<Vec<_>>(),
            )),
            None,
        )
        .unwrap(),
    )
}

fn rich_schema_batches() -> Vec<RecordBatch> {
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let schema = Arc::new(Schema::new(vec![
        Field::new("doc_key", DataType::Utf8, false),
        Field::new("tenant_id", DataType::UInt32, false),
        Field::new("price", DataType::Float64, false),
        Field::new("inventory", DataType::Int32, false),
        Field::new(
            "text_embedding",
            DataType::FixedSizeList(Arc::clone(&item), 3),
            false,
        ),
        Field::new("image_embedding", DataType::FixedSizeList(item, 3), false),
        Field::new("active", DataType::Boolean, false),
    ]));
    vec![
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["alpha", "beta", "gamma", "delta"])),
                Arc::new(UInt32Array::from(vec![7, 8, 9, 10])),
                Arc::new(Float64Array::from(vec![10.5, 20.25, 30.75, 40.0])),
                Arc::new(Int32Array::from(vec![4, 3, 2, 1])),
                vector_array(&[
                    [1.0, 0.0, 0.0],
                    [0.9, 0.1, 0.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ]),
                vector_array(&[
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.9, 0.1, 0.0],
                    [1.0, 0.0, 0.0],
                ]),
                Arc::new(BooleanArray::from(vec![true, false, true, false])),
            ],
        )
        .unwrap(),
    ]
}

async fn explain(context: &SessionContext, sql: &str) -> String {
    let batches = context
        .sql(&format!("EXPLAIN {sql}"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    pretty_format_batches(&batches).unwrap().to_string()
}

mod day_01 {
    mod checkpoint_3 {
        use super::super::*;

        #[tokio::test]
        async fn rich_schema_matches_only_the_configured_vector_column() {
            let context = context_with_batches(
                "documents",
                rich_schema_batches(),
                "text_embedding",
                Metric::Euclidean,
                IndexConfig::Flat,
            )
            .await;
            let text_sql = "SELECT price, doc_key, active, tenant_id, inventory FROM documents \
                            ORDER BY array_distance(text_embedding, [1.0, 0.0, 0.0]) LIMIT 2";
            let text_plan = explain(&context, text_sql).await;
            assert!(text_plan.contains("VectorIndexScanExec"), "{text_plan}");
            assert!(text_plan.contains("ordered=false"), "{text_plan}");
            assert!(text_plan.contains("SortExec: TopK(fetch=2)"), "{text_plan}");

            let text = context
                .sql(text_sql)
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
            assert_eq!(
                text[0]
                    .schema()
                    .fields()
                    .iter()
                    .map(|field| field.name().as_str())
                    .collect::<Vec<_>>(),
                ["price", "doc_key", "active", "tenant_id", "inventory"]
            );
            assert_eq!(
                text[0]
                    .column(1)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>(),
                [Some("alpha"), Some("beta")]
            );

            let image_sql = "SELECT price, doc_key, active, tenant_id, inventory FROM documents \
                             ORDER BY array_distance(image_embedding, [1.0, 0.0, 0.0]) LIMIT 2";
            let image_plan = explain(&context, image_sql).await;
            assert!(image_plan.contains("DataSourceExec"), "{image_plan}");
            assert!(
                image_plan.contains("SortExec: TopK(fetch=2)"),
                "{image_plan}"
            );
            assert!(!image_plan.contains("VectorIndexScanExec"), "{image_plan}");
            let image = context
                .sql(image_sql)
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
            assert_eq!(
                image[0]
                    .column(1)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>(),
                [Some("delta"), Some("gamma")]
            );
        }

        #[tokio::test]
        async fn rich_schema_rejects_a_missing_selected_column() {
            let error = attachment_result(rich_schema_batches(), "missing_embedding")
                .await
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("vector column 'missing_embedding' does not exist or is ambiguous")
            );
        }

        #[tokio::test]
        async fn rich_schema_rejects_a_scalar_selected_column() {
            let error = attachment_result(rich_schema_batches(), "price")
                .await
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("vector column 'price' must be FixedSizeList<Float32>")
            );
        }

        #[tokio::test]
        async fn rich_schema_rejects_a_zero_width_selected_column() {
            let item = Arc::new(Field::new("item", DataType::Float32, false));
            let schema = Arc::new(Schema::new(vec![Field::new(
                "text_embedding",
                DataType::FixedSizeList(Arc::clone(&item), 0),
                false,
            )]));
            let batch = RecordBatch::try_new(
                schema,
                vec![Arc::new(FixedSizeListArray::new_null(item, 0, 0))],
            )
            .unwrap();
            let error = attachment_result(vec![batch], "text_embedding")
                .await
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("vector column 'text_embedding' dimension must be greater than zero")
            );
        }

        #[tokio::test]
        async fn rich_schema_rejects_a_null_selected_value() {
            let item = Arc::new(Field::new("item", DataType::Float32, false));
            let schema = Arc::new(Schema::new(vec![Field::new(
                "text_embedding",
                DataType::FixedSizeList(Arc::clone(&item), 3),
                true,
            )]));
            let vectors = FixedSizeListArray::try_new(
                item,
                3,
                Arc::new(Float32Array::from(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0])),
                Some(NullBuffer::from(vec![true, false])),
            )
            .unwrap();
            let batch = RecordBatch::try_new(schema, vec![Arc::new(vectors)]).unwrap();
            let error = attachment_result(vec![batch], "text_embedding")
                .await
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("vector column 'text_embedding' contains null at row 1")
            );
        }
    }
    mod checkpoint_4 {
        use super::super::*;

        #[tokio::test]
        async fn compatible_top_k_uses_vector_index_scan_and_keeps_sort() {
            let context = context(Metric::Cosine, IndexConfig::Flat).await;
            let sql = "SELECT id, payload FROM points \
                       ORDER BY cosine_distance(embedding, [1.0, 0.0, 0.0]) LIMIT 2";
            let plan = explain(&context, sql).await;
            assert!(plan.contains("VectorIndexScanExec"), "{plan}");
            assert!(plan.contains("ordered=false"), "{plan}");
            assert!(plan.contains("SortExec: TopK(fetch=2)"), "{plan}");
            assert!(plan.contains("fetch=Some(2)"), "{plan}");

            let batches = context.sql(sql).await.unwrap().collect().await.unwrap();
            let ids = batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            assert_eq!(ids.values(), &[10, 20]);
        }

        #[tokio::test]
        async fn filter_keeps_datafusion_exact_fallback() {
            let context = context(Metric::Cosine, IndexConfig::Flat).await;
            let sql = "SELECT id FROM points WHERE payload = 'north' \
                       ORDER BY cosine_distance(embedding, [1.0, 0.0, 0.0]) LIMIT 1";
            let plan = explain(&context, sql).await;
            assert!(plan.contains("SortExec"), "{plan}");
            assert!(plan.contains("DataSourceExec"), "{plan}");
            assert!(!plan.contains("VectorIndexScanExec"), "{plan}");
        }

        #[tokio::test]
        async fn unsafe_sort_shapes_are_not_lowered() {
            let context = context(Metric::Cosine, IndexConfig::Flat).await;
            for sql in [
                "SELECT id FROM points ORDER BY array_distance(embedding, [1.0, 0.0, 0.0]) LIMIT 2",
                "SELECT id FROM points ORDER BY cosine_distance(embedding, [1.0, 0.0, 0.0]) DESC LIMIT 2",
                "SELECT id FROM points ORDER BY cosine_distance(embedding, [1.0, 0.0, 0.0]), id LIMIT 2",
                "SELECT id FROM points ORDER BY cosine_distance(embedding, embedding) LIMIT 2",
            ] {
                let plan = explain(&context, sql).await;
                assert!(plan.contains("SortExec"), "{plan}");
                assert!(!plan.contains("VectorIndexScanExec"), "{plan}");
            }
        }

        #[tokio::test]
        async fn dot_product_requires_descending_order() {
            let context = context(Metric::Dot, IndexConfig::Flat).await;
            let sql = "SELECT id FROM points \
                       ORDER BY inner_product(embedding, [1.0, 0.0, 0.0]) DESC LIMIT 2";
            let plan = explain(&context, sql).await;
            assert!(plan.contains("VectorIndexScanExec"), "{plan}");
            let batches = context.sql(sql).await.unwrap().collect().await.unwrap();
            let ids = batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            assert_eq!(ids.values(), &[10, 20]);
        }
    }
    mod checkpoint_5 {
        use super::super::*;

        #[tokio::test]
        async fn ordered_session_mode_allows_sort_elision() {
            let context = context(Metric::Cosine, IndexConfig::Flat).await;
            context
                .sql("SET vector_search.ordered = true")
                .await
                .unwrap();
            let sql = "SELECT id, payload FROM points \
                       ORDER BY cosine_distance(embedding, [1.0, 0.0, 0.0]) LIMIT 2";
            let plan = explain(&context, sql).await;
            assert!(plan.contains("VectorIndexScanExec"), "{plan}");
            assert!(plan.contains("ordered=true"), "{plan}");
            assert!(!plan.contains("SortExec"), "{plan}");
        }
    }
}

mod day_04 {
    mod checkpoint_3 {
        use super::super::*;

        #[tokio::test]
        async fn hnsw_is_visible_in_explain() {
            let context = context(
                Metric::Cosine,
                IndexConfig::Hnsw(HnswConfig {
                    max_connections: 3,
                    ef_construction: 5,
                    ef_search: 5,
                    max_level: 4,
                    seed: 7,
                }),
            )
            .await;
            let plan = explain(
                &context,
                "SELECT id FROM points ORDER BY cosine_distance(embedding, [1.0, 0.0, 0.0]) LIMIT 2",
            )
            .await;
            assert!(plan.contains("VectorIndexScanExec"), "{plan}");
            assert!(plan.contains("index=hnsw"), "{plan}");
            assert!(plan.contains("fetch=Some(2)"), "{plan}");
        }
    }
}

mod day_05 {
    mod checkpoint_3 {
        use super::super::*;

        #[tokio::test]
        async fn ivf_pq_is_visible_in_explain() {
            let context = context(
                Metric::Euclidean,
                IndexConfig::IvfPq(IvfPqConfig {
                    partitions: 2,
                    probes: 2,
                    iterations: 4,
                    subquantizers: 1,
                    codebook_size: 4,
                    rerank: 5,
                    seed: 7,
                }),
            )
            .await;
            let plan = explain(
                &context,
                "SELECT id FROM points ORDER BY array_distance(embedding, [1.0, 0.0, 0.0]) LIMIT 2",
            )
            .await;
            assert!(plan.contains("VectorIndexScanExec"), "{plan}");
            assert!(plan.contains("index=ivf_pq"), "{plan}");
            assert!(plan.contains("fetch=Some(2)"), "{plan}");
        }
    }
}
