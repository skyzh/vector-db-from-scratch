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

//! Print format variants

use std::str::FromStr;

use crate::print_options::MaxRows;

use datafusion::arrow::csv::writer::WriterBuilder;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::json::{ArrayWriter, LineDelimitedWriter};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches_with_options;
use datafusion::config::FormatOptions;
use datafusion::error::Result;

/// Allow records to be printed in different formats
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum PrintFormat {
    Csv,
    Tsv,
    Table,
    Json,
    NdJson,
    Automatic,
}

impl FromStr for PrintFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "csv" => Ok(Self::Csv),
            "tsv" => Ok(Self::Tsv),
            "table" => Ok(Self::Table),
            "json" => Ok(Self::Json),
            "nd-json" | "ndjson" => Ok(Self::NdJson),
            "automatic" => Ok(Self::Automatic),
            _ => Err(format!("unknown print format: {s}")),
        }
    }
}

macro_rules! batches_to_json {
    ($WRITER: ident, $writer: expr, $batches: expr) => {{
        {
            if !$batches.is_empty() {
                let mut json_writer = $WRITER::new(&mut *$writer);
                for batch in $batches {
                    json_writer.write(batch)?;
                }
                json_writer.finish()?;
                json_finish!($WRITER, $writer);
            }
        }
        Ok(()) as Result<()>
    }};
}

macro_rules! json_finish {
    (ArrayWriter, $writer: expr) => {{
        writeln!($writer)?;
    }};
    (LineDelimitedWriter, $writer: expr) => {{}};
}

fn print_batches_with_sep<W: std::io::Write>(
    writer: &mut W,
    batches: &[RecordBatch],
    delimiter: u8,
    with_header: bool,
) -> Result<()> {
    let builder = WriterBuilder::new()
        .with_header(with_header)
        .with_delimiter(delimiter);
    let mut csv_writer = builder.build(writer);

    for batch in batches {
        csv_writer.write(batch)?;
    }

    Ok(())
}

fn keep_only_maxrows(s: &str, maxrows: usize) -> String {
    let lines: Vec<String> = s.lines().map(String::from).collect();

    assert!(lines.len() >= maxrows + 4); // 4 lines for top and bottom border

    let last_line = &lines[lines.len() - 1]; // bottom border line

    let spaces = last_line.len().saturating_sub(4);
    let dotted_line = format!("| .{}|", " ".repeat(spaces));

    let mut result = lines[0..(maxrows + 3)].to_vec(); // Keep top border and `maxrows` lines
    result.extend(vec![dotted_line; 3]); // Append ... lines
    result.push(last_line.clone());

    result.join("\n")
}

fn format_batches_with_maxrows<W: std::io::Write>(
    writer: &mut W,
    batches: &[RecordBatch],
    maxrows: MaxRows,
    format_options: &FormatOptions,
) -> Result<()> {
    let options: datafusion::arrow::util::display::FormatOptions = format_options.try_into()?;

    match maxrows {
        MaxRows::Limited(maxrows) => {
            // Filter batches to meet the maxrows condition
            let mut filtered_batches = Vec::new();
            let mut row_count: usize = 0;
            let mut over_limit = false;
            for batch in batches {
                if row_count + batch.num_rows() > maxrows {
                    // If adding this batch exceeds maxrows, slice the batch
                    let limit = maxrows - row_count;
                    let sliced_batch = batch.slice(0, limit);
                    filtered_batches.push(sliced_batch);
                    over_limit = true;
                    break;
                } else {
                    filtered_batches.push(batch.clone());
                    row_count += batch.num_rows();
                }
            }

            let formatted = pretty_format_batches_with_options(&filtered_batches, &options)?;
            if over_limit {
                let mut formatted_str = format!("{formatted}");
                formatted_str = keep_only_maxrows(&formatted_str, maxrows);
                writeln!(writer, "{formatted_str}")?;
            } else {
                writeln!(writer, "{formatted}")?;
            }
        }
        MaxRows::Unlimited => {
            let formatted = pretty_format_batches_with_options(batches, &options)?;
            writeln!(writer, "{formatted}")?;
        }
    }

    Ok(())
}

impl PrintFormat {
    pub const fn variants() -> &'static [&'static str] {
        &["csv", "tsv", "table", "json", "nd-json", "automatic"]
    }

    /// Print the batches to a writer using the specified format
    pub fn print_batches<W: std::io::Write>(
        &self,
        writer: &mut W,
        schema: SchemaRef,
        batches: &[RecordBatch],
        maxrows: MaxRows,
        with_header: bool,
        format_options: &FormatOptions,
    ) -> Result<()> {
        // filter out any empty batches
        let batches: Vec<_> = batches
            .iter()
            .filter(|b| b.num_rows() > 0)
            .cloned()
            .collect();
        if batches.is_empty() {
            return self.print_empty(writer, schema, format_options);
        }

        match self {
            Self::Csv | Self::Automatic => {
                print_batches_with_sep(writer, &batches, b',', with_header)
            }
            Self::Tsv => print_batches_with_sep(writer, &batches, b'\t', with_header),
            Self::Table => {
                if maxrows == MaxRows::Limited(0) {
                    return Ok(());
                }
                format_batches_with_maxrows(writer, &batches, maxrows, format_options)
            }
            Self::Json => batches_to_json!(ArrayWriter, writer, &batches),
            Self::NdJson => batches_to_json!(LineDelimitedWriter, writer, &batches),
        }
    }

    /// Print when the result batches contain no rows
    fn print_empty<W: std::io::Write>(
        &self,
        writer: &mut W,
        schema: SchemaRef,
        format_options: &FormatOptions,
    ) -> Result<()> {
        match self {
            // Print column headers for Table format
            Self::Table if !schema.fields().is_empty() => {
                let format_options: datafusion::arrow::util::display::FormatOptions =
                    format_options.try_into()?;

                let empty_batch = RecordBatch::new_empty(schema);
                let formatted =
                    pretty_format_batches_with_options(&[empty_batch], &format_options)?;
                writeln!(writer, "{formatted}")?;
            }
            _ => {}
        }
        Ok(())
    }
}
