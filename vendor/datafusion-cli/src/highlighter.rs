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

//! The syntax highlighter.

use std::{
    borrow::Cow::{self, Borrowed},
    fmt::Display,
};

use datafusion::common::config;
use datafusion::sql::sqlparser::{
    dialect::{Dialect, GenericDialect, dialect_from_str},
    keywords::Keyword,
    tokenizer::{Token, Tokenizer},
};
use rustyline::highlight::{CmdKind, Highlighter};

/// The syntax highlighter.
#[derive(Debug)]
pub struct SyntaxHighlighter {
    dialect: Box<dyn Dialect>,
}

impl SyntaxHighlighter {
    pub fn new(dialect: &config::Dialect) -> Self {
        let dialect = dialect_from_str(dialect).unwrap_or_else(|| Box::new(GenericDialect {}));
        Self { dialect }
    }
}

pub struct NoSyntaxHighlighter {}

impl Highlighter for NoSyntaxHighlighter {}

impl Highlighter for SyntaxHighlighter {
    fn highlight<'l>(&self, line: &'l str, _: usize) -> Cow<'l, str> {
        let mut out_line = String::new();

        // `with_unescape(false)` since we want to rebuild the original string.
        let mut tokenizer = Tokenizer::new(self.dialect.as_ref(), line).with_unescape(false);
        let tokens = tokenizer.tokenize();
        match tokens {
            Ok(tokens) => {
                for token in tokens.iter() {
                    match token {
                        Token::Word(w) if w.keyword != Keyword::NoKeyword => {
                            out_line.push_str(&Color::red(token));
                        }
                        Token::SingleQuotedString(_) => {
                            out_line.push_str(&Color::green(token));
                        }
                        other => out_line.push_str(&format!("{other}")),
                    }
                }
                out_line.into()
            }
            Err(_) => Borrowed(line),
        }
    }

    fn highlight_char(&self, line: &str, _pos: usize, _cmd: CmdKind) -> bool {
        !line.is_empty()
    }
}

/// Convenient utility to return strings with [ANSI color](https://gist.github.com/JBlond/2fea43a3049b38287e5e9cefc87b2124).
pub(crate) struct Color {}

impl Color {
    pub(crate) fn green(s: impl Display) -> String {
        format!("\x1b[92m{s}\x1b[0m")
    }

    pub(crate) fn red(s: impl Display) -> String {
        format!("\x1b[91m{s}\x1b[0m")
    }

    pub(crate) fn gray(s: impl Display) -> String {
        format!("\x1b[90m{s}\x1b[0m")
    }
}
