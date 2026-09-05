use std::io::{self, IsTerminal, Read};
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::common::{DataFusionError, Result};
use datafusion::dataframe::DataFrame;
use datafusion::execution::{TaskContext, context::SessionState};
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::SessionContext;
use datafusion_cli::DATAFUSION_CLI_VERSION;
use datafusion_cli::cli_context::CliSessionContext;
use datafusion_cli::exec::{exec_from_commands, exec_from_repl};
use datafusion_cli::print_format::PrintFormat;
use datafusion_cli::print_options::{MaxRows, PrintOptions};
use tokio::sync::Mutex;
use vector_core::{IndexConfig, IvfFlatConfig, Metric};
use vector_datafusion::VectorSqlSession;

struct VectorCliContext {
    planner: SessionContext,
    session: Mutex<VectorSqlSession>,
}

impl VectorCliContext {
    fn new(session: VectorSqlSession) -> Self {
        Self {
            planner: session.cli_session_context(),
            session: Mutex::new(session),
        }
    }
}

#[async_trait]
impl CliSessionContext for VectorCliContext {
    fn task_ctx(&self) -> Arc<TaskContext> {
        CliSessionContext::task_ctx(&self.planner)
    }

    fn session_state(&self) -> SessionState {
        CliSessionContext::session_state(&self.planner)
    }

    async fn validate_sql(&self, sql: &str) -> Result<()> {
        self.session.lock().await.validate_cli_sql(sql)
    }

    async fn execute_logical_plan(&self, plan: LogicalPlan) -> Result<DataFrame> {
        self.session.lock().await.execute_cli_plan(plan).await
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let session = VectorSqlSession::new(
        Metric::Cosine,
        IndexConfig::IvfFlat(IvfFlatConfig {
            partitions: 2,
            probes: 2,
            iterations: 8,
            seed: 7,
        }),
    );
    let context = VectorCliContext::new(session);
    let interactive = io::stdin().is_terminal();
    let mut print_options = PrintOptions {
        format: PrintFormat::Automatic,
        quiet: false,
        maxrows: MaxRows::Unlimited,
        color: io::stdout().is_terminal(),
    };

    println!("DataFusion CLI v{DATAFUSION_CLI_VERSION}");
    if interactive {
        exec_from_repl(&context, &mut print_options)
            .await
            .map_err(|error| DataFusionError::External(Box::new(error)))
    } else {
        let mut sql = String::new();
        io::stdin()
            .read_to_string(&mut sql)
            .map_err(DataFusionError::IoError)?;
        exec_from_commands(&context, vec![sql], &print_options).await
    }
}
