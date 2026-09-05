// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// This file has been modified by the Vector Database from Scratch project.

//! Execution functions

use crate::cli_context::CliSessionContext;
use crate::helper::{split_complete_from_semicolon, split_from_semicolon};
use crate::print_format::PrintFormat;
use crate::{
    command::{Command, OutputFormat},
    helper::CliHelper,
    print_options::{MaxRows, PrintOptions},
};
use datafusion::common::instant::Instant;
use datafusion::common::{plan_datafusion_err, plan_err};
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::memory_pool::MemoryConsumer;
use datafusion::logical_expr::LogicalPlan;
use datafusion::physical_plan::execution_plan::EmissionType;
use datafusion::physical_plan::spill::get_record_batch_memory_size;
use datafusion::physical_plan::{ExecutionPlanProperties, execute_stream};
use datafusion::sql::parser::{DFParser, Statement};
use datafusion::sql::sqlparser;
use datafusion::sql::sqlparser::dialect::dialect_from_str;
use futures::StreamExt;
use rustyline::Editor;
use rustyline::error::ReadlineError;
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use tokio::signal;

/// run and execute SQL statements and commands, against a context with the given print options
pub async fn exec_from_commands(
    ctx: &dyn CliSessionContext,
    commands: Vec<String>,
    print_options: &PrintOptions,
) -> Result<()> {
    for sql in commands {
        exec_and_print(ctx, print_options, sql).await?;
    }

    Ok(())
}

/// run and execute SQL statements and commands from a file, against a context with the given print options
pub async fn exec_from_lines(
    ctx: &dyn CliSessionContext,
    reader: &mut BufReader<File>,
    print_options: &PrintOptions,
) -> Result<()> {
    let mut query = "".to_owned();
    let task_ctx = ctx.task_ctx();
    let dialect_name = &task_ctx.session_config().options().sql_parser.dialect;
    let dialect = dialect_from_str(dialect_name)
        .ok_or_else(|| plan_datafusion_err!("Unsupported SQL dialect: {dialect_name}"))?;

    for (line_number, line) in reader.lines().enumerate() {
        match line {
            Ok(line) if line_number == 0 && line.starts_with("#!") => {
                continue;
            }
            Ok(line) => {
                query.push_str(&line);
                query.push('\n');
                let (statements, remainder) =
                    split_complete_from_semicolon(&query, dialect.as_ref());
                query = remainder;
                for statement in statements {
                    match exec_and_print(ctx, print_options, statement).await {
                        Ok(_) => {}
                        Err(err) => eprintln!("{err}"),
                    }
                }
            }
            Err(error) => return Err(DataFusionError::IoError(error)),
        }
    }

    // run the left over query if the last statement doesn't contain ‘;’
    // ignore if it only consists of '\n'
    if query.contains(|c| c != '\n') {
        exec_and_print(ctx, print_options, query).await?;
    }

    Ok(())
}

pub async fn exec_from_files(
    ctx: &dyn CliSessionContext,
    files: Vec<String>,
    print_options: &PrintOptions,
) -> Result<()> {
    for file_path in files {
        let file = File::open(file_path).map_err(DataFusionError::IoError)?;
        let mut reader = BufReader::new(file);
        exec_from_lines(ctx, &mut reader, print_options).await?;
    }

    Ok(())
}

/// run and execute SQL statements and commands against a context with the given print options
pub async fn exec_from_repl(
    ctx: &dyn CliSessionContext,
    print_options: &mut PrintOptions,
) -> rustyline::Result<()> {
    let mut rl = Editor::new()?;
    rl.set_helper(Some(CliHelper::new(
        &ctx.task_ctx().session_config().options().sql_parser.dialect,
        print_options.color,
    )));
    rl.load_history(".history").ok();

    loop {
        match rl.readline("> ") {
            Ok(line) if line.starts_with('\\') => {
                rl.add_history_entry(line.trim_end())?;
                let command = line.split_whitespace().collect::<Vec<_>>().join(" ");
                if let Ok(cmd) = &command[1..].parse::<Command>() {
                    match cmd {
                        Command::Quit => break,
                        Command::OutputFormat(subcommand) => {
                            if let Some(subcommand) = subcommand {
                                if let Ok(command) = subcommand.parse::<OutputFormat>() {
                                    if let Err(e) = command.execute(print_options).await {
                                        eprintln!("{e}")
                                    }
                                } else {
                                    eprintln!(
                                        "'\\{}' is not a valid command, you can use '\\?' to see all commands",
                                        &line[1..]
                                    );
                                }
                            } else {
                                println!("Output format is {:?}.", print_options.format);
                            }
                        }
                        _ => {
                            if let Err(e) = cmd.execute(ctx, print_options).await {
                                eprintln!("{e}")
                            }
                        }
                    }
                } else {
                    eprintln!(
                        "'\\{}' is not a valid command, you can use '\\?' to see all commands",
                        &line[1..]
                    );
                }
            }
            Ok(line) if line == "quit" => {
                rl.add_history_entry(line)?;
                break;
            }
            Ok(line) => {
                let task_ctx = ctx.task_ctx();
                let dialect_name = &task_ctx.session_config().options().sql_parser.dialect;
                let lines = dialect_from_str(dialect_name)
                    .map(|dialect| split_from_semicolon(&line, dialect.as_ref()))
                    .unwrap_or_else(|| vec![line]);
                for line in lines {
                    rl.add_history_entry(line.trim_end())?;
                    tokio::select! {
                        res = exec_and_print(ctx, print_options, line) => match res {
                            Ok(_) => {}
                            Err(err) => eprintln!("{err}"),
                        },
                        _ = signal::ctrl_c() => {
                            println!("^C");
                            continue
                        },
                    }
                    // dialect might have changed
                    rl.helper_mut()
                        .unwrap()
                        .set_dialect(&ctx.task_ctx().session_config().options().sql_parser.dialect);
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                rl.helper().unwrap().reset_hint();
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!("\\q");
                break;
            }
            Err(err) => {
                eprintln!("Unknown error happened {err:?}");
                break;
            }
        }
    }

    rl.save_history(".history")
}

pub(super) async fn exec_and_print(
    ctx: &dyn CliSessionContext,
    print_options: &PrintOptions,
    sql: String,
) -> Result<()> {
    ctx.validate_sql(&sql).await?;
    let task_ctx = ctx.task_ctx();
    let options = task_ctx.session_config().options();
    let dialect = &options.sql_parser.dialect;
    let dialect = dialect_from_str(dialect).ok_or_else(|| {
        plan_datafusion_err!(
            "Unsupported SQL dialect: {dialect}. Available dialects: \
                 Generic, MySQL, PostgreSQL, Hive, SQLite, Snowflake, Redshift, \
                 MsSQL, ClickHouse, BigQuery, Ansi, DuckDB, Databricks."
        )
    })?;

    let statements = DFParser::parse_sql_with_dialect(&sql, dialect.as_ref())?;
    for statement in statements {
        StatementExecutor::new(statement)
            .execute(ctx, print_options)
            .await?;
    }

    Ok(())
}

struct StatementExecutor {
    statement: Statement,
}

impl StatementExecutor {
    fn new(statement: Statement) -> Self {
        Self { statement }
    }

    async fn execute(
        self,
        ctx: &dyn CliSessionContext,
        print_options: &PrintOptions,
    ) -> Result<()> {
        let now = Instant::now();
        let (df, adjusted) = self
            .create_and_execute_logical_plan(ctx, print_options)
            .await?;
        let physical_plan = df.create_physical_plan().await?;
        let task_ctx = ctx.task_ctx();
        let options = task_ctx.session_config().options();

        // Track memory usage for the query result if it's bounded
        let reservation = MemoryConsumer::new("DataFusion-Cli").register(task_ctx.memory_pool());

        if physical_plan.boundedness().is_unbounded() {
            if physical_plan.pipeline_behavior() == EmissionType::Final {
                return plan_err!(
                    "The given query can generate a valid result only once \
                    the source finishes, but the source is unbounded"
                );
            }
            // As the input stream comes, we can generate results.
            // However, memory safety is not guaranteed.
            let stream = execute_stream(physical_plan, task_ctx.clone())?;
            print_options
                .print_stream(stream, now, &options.format)
                .await?;
        } else {
            // Bounded stream; collected results size is limited by the maxrows option
            let schema = physical_plan.schema();
            let mut stream = execute_stream(physical_plan, task_ctx.clone())?;
            let mut results = vec![];
            let mut row_count = 0_usize;
            let max_rows = match print_options.maxrows {
                MaxRows::Unlimited => usize::MAX,
                MaxRows::Limited(n) => n,
            };
            while let Some(batch) = stream.next().await {
                let batch = batch?;
                let curr_num_rows = batch.num_rows();
                // Stop collecting results if the number of rows exceeds the limit
                // results batch should include the last batch that exceeds the limit
                if row_count < max_rows.saturating_add(curr_num_rows) {
                    // Try to grow the reservation to accommodate the batch in memory
                    reservation.try_grow(get_record_batch_memory_size(&batch))?;
                    results.push(batch);
                }
                row_count += curr_num_rows;
            }
            adjusted.into_inner().print_batches(
                schema,
                &results,
                now,
                row_count,
                &options.format,
            )?;
            reservation.free();
        }

        Ok(())
    }

    async fn create_and_execute_logical_plan(
        self,
        ctx: &dyn CliSessionContext,
        print_options: &PrintOptions,
    ) -> Result<(datafusion::dataframe::DataFrame, AdjustedPrintOptions)> {
        let adjusted =
            AdjustedPrintOptions::new(print_options.clone()).with_statement(&self.statement);

        let plan = create_plan(ctx, self.statement).await?;
        let adjusted = adjusted.with_plan(&plan);
        let df = ctx.execute_logical_plan(plan).await?;

        Ok((df, adjusted))
    }
}

/// Track adjustments to the print options based on the plan / statement being executed
#[derive(Debug)]
struct AdjustedPrintOptions {
    inner: PrintOptions,
}

impl AdjustedPrintOptions {
    fn new(inner: PrintOptions) -> Self {
        Self { inner }
    }
    /// Adjust print options based on any statement specific requirements
    fn with_statement(mut self, statement: &Statement) -> Self {
        if let Statement::Statement(sql_stmt) = statement {
            // SHOW / SHOW ALL
            if let sqlparser::ast::Statement::ShowVariable { .. } = sql_stmt.as_ref() {
                self.inner.maxrows = MaxRows::Unlimited
            }
        }
        self
    }

    /// Adjust print options based on any plan specific requirements
    fn with_plan(mut self, plan: &LogicalPlan) -> Self {
        // For plans like `Explain` ignore `MaxRows` option and always display
        // all rows
        if matches!(
            plan,
            LogicalPlan::Explain(_) | LogicalPlan::DescribeTable(_) | LogicalPlan::Analyze(_)
        ) {
            self.inner.maxrows = MaxRows::Unlimited;
        }
        self
    }

    /// Finalize and return the inner `PrintOptions`
    fn into_inner(mut self) -> PrintOptions {
        if self.inner.format == PrintFormat::Automatic {
            self.inner.format = PrintFormat::Table;
        }

        self.inner
    }
}

async fn create_plan(
    ctx: &dyn CliSessionContext,
    statement: Statement,
) -> Result<LogicalPlan, DataFusionError> {
    ctx.session_state().statement_to_plan(statement).await
}
