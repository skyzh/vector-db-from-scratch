# Upstream provenance

This directory vendors the library portion of Apache DataFusion CLI release `54.1.0`, commit
`0d1f2ebe2cc97c91b736bc0a160b5b73cf40437a` from
<https://github.com/apache/datafusion/tree/54.1.0/datafusion-cli>.

The upstream `LICENSE.txt`, `NOTICE.txt`, `README.md`, Cargo manifest license header, and retained
Rust source headers are preserved beside the code. The crate is not published independently.

## Course-scoped changes

The vendored library keeps the session abstraction, commands, Rustyline editing and highlighting,
statement execution, function help, and table/CSV/TSV/JSON/NDJSON printing used by the course. The
course adds a `validate_sql` hook to the session abstraction so raw `CREATE INDEX` syntax is checked
before DataFusion planning.

The upstream binary, examples, tests, catalog setup, memory-pool selection, object-store setup and
profiling, cloud dependencies, Parquet metadata table function, and unrelated CLI-only dependencies
are omitted. These omissions deliberately make this a course-scoped fork rather than a drop-in
replacement for every upstream CLI feature.

## Updating

1. Check out the next upstream release and record its exact tag and commit here.
2. Diff the retained files against that release before copying any changes.
3. Reapply the bounded `validate_sql` hook and the documented removals; do not copy the upstream
   binary, cloud/object-store setup, tests, or unrelated dependencies by default.
4. Preserve `LICENSE.txt`, `NOTICE.txt`, `README.md`, and all applicable ASF source headers.
5. Run the course-owned terminal behavior suite plus the locked workspace build, tests, examples,
   doctests, and strict Clippy gates before accepting the update.
