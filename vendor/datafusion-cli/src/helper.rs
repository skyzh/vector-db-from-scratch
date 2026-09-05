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

//! Helper that helps with interactive editing, including multi-line parsing and validation,
//! and auto-completion for file name during creating external table.

use std::borrow::Cow;
use std::cell::Cell;

use crate::highlighter::{Color, NoSyntaxHighlighter, SyntaxHighlighter};

use datafusion::common::config::Dialect;
use datafusion::sql::parser::{DFParser, Statement};
use datafusion::sql::sqlparser::dialect::{Dialect as SqlDialect, dialect_from_str};
use datafusion::sql::sqlparser::tokenizer::{Token, Tokenizer};

use rustyline::completion::{Completer, FilenameCompleter, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::{CmdKind, Highlighter};
use rustyline::hint::Hinter;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{Context, Helper, Result};

/// Default suggestion shown when the input line is empty.
const DEFAULT_HINT_SUGGESTION: &str = " \\? for help, \\q to quit";

pub struct CliHelper {
    completer: FilenameCompleter,
    dialect: Dialect,
    highlighter: Box<dyn Highlighter>,
    /// Tracks whether to show the default hint. Set to `false` once the user
    /// types anything, so the hint doesn't reappear after deleting back to
    /// an empty line. Reset to `true` when the line is submitted.
    show_hint: Cell<bool>,
}

impl CliHelper {
    pub fn new(dialect: &Dialect, color: bool) -> Self {
        let highlighter: Box<dyn Highlighter> = if !color {
            Box::new(NoSyntaxHighlighter {})
        } else {
            Box::new(SyntaxHighlighter::new(dialect))
        };
        Self {
            completer: FilenameCompleter::new(),
            dialect: *dialect,
            highlighter,
            show_hint: Cell::new(true),
        }
    }

    pub fn set_dialect(&mut self, dialect: &Dialect) {
        if *dialect != self.dialect {
            self.dialect = *dialect;
        }
    }

    /// Re-enable the default hint for the next prompt.
    pub fn reset_hint(&self) {
        self.show_hint.set(true);
    }

    fn validate_input(&self, input: &str) -> Result<ValidationResult> {
        if input == "quit" {
            Ok(ValidationResult::Valid(None))
        } else if input.starts_with('\\') {
            // command
            Ok(ValidationResult::Valid(None))
        } else {
            let dialect = match dialect_from_str(self.dialect) {
                Some(dialect) => dialect,
                None => {
                    return Ok(ValidationResult::Invalid(Some(format!(
                        "  🤔 Invalid dialect: {}",
                        self.dialect
                    ))));
                }
            };
            let (lines, remainder) = split_complete_from_semicolon(input, dialect.as_ref());
            if lines.is_empty()
                || (!remainder.trim().is_empty()
                    && !is_only_closed_comments_and_whitespace(&remainder, dialect.as_ref()))
            {
                return Ok(ValidationResult::Incomplete);
            }
            for line in lines {
                match DFParser::parse_sql_with_dialect(&line, dialect.as_ref()) {
                    Ok(statements) if statements.is_empty() => {
                        return Ok(ValidationResult::Invalid(Some(
                            "  🤔 You entered an empty statement".to_string(),
                        )));
                    }
                    Ok(_statements) => {}
                    Err(err) => {
                        return Ok(ValidationResult::Invalid(Some(format!(
                            "  🤔 Invalid statement: {err}",
                        ))));
                    }
                }
            }
            Ok(ValidationResult::Valid(None))
        }
    }
}

fn is_only_closed_comments_and_whitespace(sql: &str, dialect: &dyn SqlDialect) -> bool {
    Tokenizer::new(dialect, sql).tokenize().is_ok_and(|tokens| {
        tokens
            .iter()
            .all(|token| matches!(token, Token::Whitespace(_)))
    })
}

impl Default for CliHelper {
    fn default() -> Self {
        Self::new(&Dialect::Generic, false)
    }
}

impl Highlighter for CliHelper {
    fn highlight<'l>(&self, line: &'l str, pos: usize) -> Cow<'l, str> {
        self.highlighter.highlight(line, pos)
    }

    fn highlight_char(&self, line: &str, pos: usize, kind: CmdKind) -> bool {
        self.highlighter.highlight_char(line, pos, kind)
    }

    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        Color::gray(hint).into()
    }
}

impl Hinter for CliHelper {
    type Hint = String;

    fn hint(&self, line: &str, _pos: usize, _ctx: &Context<'_>) -> Option<String> {
        if !line.is_empty() {
            self.show_hint.set(false);
        }
        (self.show_hint.get() && line.trim().is_empty()).then(|| DEFAULT_HINT_SUGGESTION.to_owned())
    }
}

/// returns true if the current position is after the open quote for
/// creating an external table.
fn is_open_quote_for_location(line: &str, pos: usize) -> bool {
    let mut sql = line[..pos].to_string();
    sql.push('\'');
    DFParser::parse_sql(&sql)
        .is_ok_and(|stmts| matches!(stmts.back(), Some(Statement::CreateExternalTable(_))))
}

impl Completer for CliHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        ctx: &Context<'_>,
    ) -> std::result::Result<(usize, Vec<Pair>), ReadlineError> {
        if is_open_quote_for_location(line, pos) {
            self.completer.complete(line, pos, ctx)
        } else {
            Ok((0, Vec::with_capacity(0)))
        }
    }
}

impl Validator for CliHelper {
    fn validate(&self, ctx: &mut ValidationContext<'_>) -> Result<ValidationResult> {
        let input = ctx.input().trim_end();
        let result = self.validate_input(input);
        self.reset_hint();
        result
    }
}

impl Helper for CliHelper {}

/// Splits a string which consists of multiple queries.
pub(crate) fn split_from_semicolon(sql: &str, dialect: &dyn SqlDialect) -> Vec<String> {
    let (mut commands, remainder) = split_complete_from_semicolon(sql, dialect);
    if !remainder.trim().is_empty() {
        commands.push(format!("{};", remainder.trim()));
    }
    commands
}

/// Splits complete semicolon-terminated queries and returns any trailing fragment.
pub(crate) fn split_complete_from_semicolon(
    sql: &str,
    dialect: &dyn SqlDialect,
) -> (Vec<String>, String) {
    let mut commands = Vec::new();
    let mut tokens = Vec::new();
    let mut tokenizer = Tokenizer::new(dialect, sql);
    // Incomplete quoted strings and comments are expected while collecting a statement. The
    // tokenizer leaves every complete token in `tokens`, so top-level semicolons before that
    // trailing fragment are still safe boundaries.
    let _ = tokenizer.tokenize_with_location_into_buf(&mut tokens);
    let mut statement_start = 0;

    for token in tokens {
        if token.token == Token::SemiColon {
            let statement_end = byte_offset(sql, token.span.end.line, token.span.end.column);
            let statement = sql[statement_start..statement_end].trim();
            if !statement.is_empty() {
                commands.push(statement.to_owned());
            }
            statement_start = statement_end;
        }
    }

    (commands, sql[statement_start..].to_owned())
}

fn byte_offset(sql: &str, target_line: u64, target_column: u64) -> usize {
    let mut line = 1;
    let mut column = 1;

    for (offset, character) in sql.char_indices() {
        if line == target_line && column == target_column {
            return offset;
        }
        if character == '\n' {
            line += 1;
            column = 1;
        } else {
            column += 1;
        }
    }

    sql.len()
}
