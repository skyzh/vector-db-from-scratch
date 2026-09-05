#!/usr/bin/env python3
"""Public real-terminal checks for the course SQL example."""

import os
import pty
import re
import select
import subprocess
import sys
import tempfile
import time


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def normalized(transcript: bytes) -> str:
    return ANSI.sub("", transcript.decode("utf-8", errors="replace")).replace("\r", "")


class Terminal:
    def __init__(self, executable: str) -> None:
        master, slave = pty.openpty()
        environment = os.environ.copy()
        environment["TERM"] = "dumb"
        self.master = master
        self.output = bytearray()
        self.process = subprocess.Popen(
            [executable],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            close_fds=True,
            env=environment,
        )
        os.close(slave)

    def send(self, text: str) -> None:
        os.write(self.master, text.encode())

    def wait_for_prompts(self, count: int, timeout: float = 15.0) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            text = normalized(self.output)
            if text.count("> ") >= count:
                return text
            ready, _, _ = select.select([self.master], [], [], 0.1)
            if ready:
                try:
                    self.output.extend(os.read(self.master, 65536))
                except OSError:
                    break
            if self.process.poll() is not None:
                break
        raise AssertionError(
            f"timed out waiting for prompt {count}; transcript:\n{normalized(self.output)}"
        )

    def finish(self, command: str = "\\q") -> str:
        self.send(f"{command}\n")
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise AssertionError(f"CLI did not exit; transcript:\n{normalized(self.output)}")
        while True:
            ready, _, _ = select.select([self.master], [], [], 0)
            if not ready:
                break
            try:
                chunk = os.read(self.master, 65536)
            except OSError:
                break
            if not chunk:
                break
            self.output.extend(chunk)
        os.close(self.master)
        self.master = -1
        if self.process.returncode != 0:
            raise AssertionError(
                f"CLI exited with {self.process.returncode}; transcript:\n"
                f"{normalized(self.output)}"
            )
        return normalized(self.output)


def assert_json_value(transcript: str, name: str, value: int) -> None:
    pattern = rf'\[\s*\{{\s*"{re.escape(name)}"\s*:\s*{value}\s*\}}\s*\]'
    if not re.search(pattern, transcript, re.DOTALL):
        raise AssertionError(f"missing JSON result for {name}; transcript:\n{transcript}")


def assert_json_string(transcript: str, name: str, value: str) -> None:
    pattern = rf'\[\s*\{{\s*"{re.escape(name)}"\s*:\s*"{re.escape(value)}"\s*\}}\s*\]'
    if not re.search(pattern, transcript, re.DOTALL):
        raise AssertionError(f"missing JSON result for {name}; transcript:\n{transcript}")


def assert_redirected_sql(executable: str, sql: str, marker: str, value: int) -> None:
    result = subprocess.run(
        [executable], input=sql, text=True, capture_output=True, check=False
    )
    output = result.stdout + result.stderr
    if result.returncode != 0 or marker not in output or str(value) not in output:
        raise AssertionError(
            f"redirected SQL control failed for {marker}; output:\n{output}"
        )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} PATH_TO_SQL_EXAMPLE")
    executable = os.path.abspath(sys.argv[1])
    if not os.path.isfile(executable):
        raise SystemExit(f"SQL example does not exist: {executable}")

    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as script:
        script.write(
            "SELECT * FROM definitely_missing_table; "
            "SELECT 42 AS continued_after_error;\n"
        )
        script.write("SELECT 51 AS before_block;\n")
        script.write("/* a multiline block comment\n")
        script.write("contains an internal semicolon; without ending the statement */\n")
        script.write("SELECT 52 AS after_block;\n")
        script_path = script.name
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as nested_script:
        nested_script.write("SELECT 53 AS before_nested;\n")
        nested_script.write("/* outer block\n")
        nested_script.write("   /* inner block; */\n")
        nested_script.write("   still inside outer;\n")
        nested_script.write("*/\n")
        nested_script.write("SELECT 54 AS after_nested;\n")
        nested_script_path = nested_script.name
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as inline_script:
        inline_script.write(
            "SELECT 55 AS before_inline; -- comment; "
            "SELECT 999 AS commented_inline;\n"
        )
        inline_script.write("SELECT 56 AS after_inline;\n")
        inline_script_path = inline_script.name
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as dash_close_script:
        dash_close_script.write("SELECT 57 AS before_dash_close;\n")
        dash_close_script.write("/* open block comment\n")
        dash_close_script.write("-- */\n")
        dash_close_script.write("SELECT 58 AS after_dash_close;\n")
        dash_close_script_path = dash_close_script.name
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as dollar_script:
        dollar_script.write("SELECT 59 AS before_dollar;\n")
        dollar_script.write("SELECT $$kept;as;text$$ AS dollar_quoted;\n")
        dollar_script.write("SELECT 60 AS after_dollar;\n")
        dollar_script_path = dollar_script.name
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as shebang_script:
        shebang_script.write("#!/usr/bin/env datafusion-cli\n")
        shebang_script.write("SELECT 61 AS after_first_line_shebang;\n")
        shebang_script.write("/* open block comment\n")
        shebang_script.write("#! */\n")
        shebang_script.write("SELECT 62 AS after_shebang_marker_close;\n")
        shebang_script_path = shebang_script.name
    preserved_literal_sql = (
        "SELECT char_length('ab  \n"
        "cd\t\n"
        "efgh') AS included_literal_length;\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as literal_script:
        literal_script.write(preserved_literal_sql)
        literal_script_path = literal_script.name
    with tempfile.NamedTemporaryFile("wb", suffix=".sql", delete=False) as invalid_script:
        invalid_script.write(b"SELECT 1;\xff\n")
        invalid_script_path = invalid_script.name
    missing_script_path = f"{script_path}.missing"
    interactive_multiline_sql = (
        "/* interactive multiline comment;\n"
        "still inside the comment;\n"
        "*/\n"
        "SELECT 63 AS after_interactive_multiline;\n"
    )
    interactive_nested_sql = (
        "/* interactive outer comment\n"
        "/* interactive inner comment; */\n"
        "still inside the outer comment;\n"
        "*/\n"
        "SELECT 64 AS after_interactive_nested;\n"
    )
    trailing_line_comment_sql = (
        "SELECT 65 AS before_trailing_line_comment; -- closed trailing comment\n"
    )
    trailing_block_comment_sql = (
        "SELECT 66 AS before_trailing_block_comment; "
        "/* closed trailing block; comment */\n"
    )

    assert_redirected_sql(
        executable, interactive_multiline_sql, "after_interactive_multiline", 63
    )
    assert_redirected_sql(
        executable, interactive_nested_sql, "after_interactive_nested", 64
    )
    assert_redirected_sql(
        executable, preserved_literal_sql, "included_literal_length", 13
    )
    assert_redirected_sql(
        executable,
        trailing_line_comment_sql,
        "before_trailing_line_comment",
        65,
    )
    assert_redirected_sql(
        executable,
        trailing_block_comment_sql,
        "before_trailing_block_comment",
        66,
    )

    terminal = Terminal(executable)
    try:
        terminal.wait_for_prompts(1)
        terminal.send("\\pset format json\n")
        terminal.wait_for_prompts(2)
        terminal.send("SELECT 1 AS format_probe;\n")
        terminal.wait_for_prompts(3)
        terminal.send(f"\\i {script_path}\n")
        terminal.wait_for_prompts(4)
        terminal.send(f"\\i {nested_script_path}\n")
        terminal.wait_for_prompts(5)
        terminal.send(f"\\i {inline_script_path}\n")
        terminal.wait_for_prompts(6)
        terminal.send(f"\\i {dash_close_script_path}\n")
        terminal.wait_for_prompts(7)
        terminal.send(f"\\i {dollar_script_path}\n")
        terminal.wait_for_prompts(8)
        terminal.send(f"\\i {shebang_script_path}\n")
        terminal.wait_for_prompts(9)
        terminal.send(f"\\i {literal_script_path}\n")
        terminal.wait_for_prompts(10)
        terminal.send(interactive_multiline_sql)
        terminal.wait_for_prompts(11)
        terminal.send(interactive_nested_sql)
        terminal.wait_for_prompts(12)
        terminal.send(trailing_line_comment_sql)
        terminal.wait_for_prompts(13)
        terminal.send(trailing_block_comment_sql)
        terminal.wait_for_prompts(14)
        terminal.send(f"\\i {invalid_script_path}\n")
        terminal.wait_for_prompts(15)
        terminal.send("SELECT 43 AS continued_after_decode_error;\n")
        terminal.wait_for_prompts(16)
        terminal.send(f"\\i {missing_script_path}\n")
        terminal.wait_for_prompts(17)
        terminal.send("SELECT 44 AS continued_after_open_error;\n")
        terminal.wait_for_prompts(18)
        transcript = terminal.finish()

        if "Output format is Json." not in transcript:
            raise AssertionError(f"output format did not change; transcript:\n{transcript}")
        assert_json_value(transcript, "format_probe", 1)
        if "definitely_missing_table" not in transcript:
            raise AssertionError(f"included error was not reported; transcript:\n{transcript}")
        assert_json_value(transcript, "continued_after_error", 42)
        assert_json_value(transcript, "before_block", 51)
        assert_json_value(transcript, "after_block", 52)
        assert_json_value(transcript, "before_nested", 53)
        assert_json_value(transcript, "after_nested", 54)
        assert_json_value(transcript, "before_inline", 55)
        assert_json_value(transcript, "after_inline", 56)
        if "commented_inline" in transcript:
            raise AssertionError(f"line-comment SQL was executed; transcript:\n{transcript}")
        assert_json_value(transcript, "before_dash_close", 57)
        assert_json_value(transcript, "after_dash_close", 58)
        assert_json_value(transcript, "before_dollar", 59)
        assert_json_string(transcript, "dollar_quoted", "kept;as;text")
        assert_json_value(transcript, "after_dollar", 60)
        assert_json_value(transcript, "after_first_line_shebang", 61)
        assert_json_value(transcript, "after_shebang_marker_close", 62)
        assert_json_value(transcript, "included_literal_length", 13)
        assert_json_value(transcript, "after_interactive_multiline", 63)
        assert_json_value(transcript, "after_interactive_nested", 64)
        assert_json_value(transcript, "before_trailing_line_comment", 65)
        assert_json_value(transcript, "before_trailing_block_comment", 66)
        if "ParserError" in transcript or "TokenizerError" in transcript:
            raise AssertionError(f"block comment was split as SQL; transcript:\n{transcript}")
        if "stream did not contain valid UTF-8" not in transcript:
            raise AssertionError(f"decode error was not reported; transcript:\n{transcript}")
        assert_json_value(transcript, "continued_after_decode_error", 43)
        if missing_script_path not in transcript:
            raise AssertionError(f"open error was not reported; transcript:\n{transcript}")
        assert_json_value(transcript, "continued_after_open_error", 44)

        quit_terminal = Terminal(executable)
        quit_terminal.wait_for_prompts(1)
        quit_transcript = quit_terminal.finish("quit")
        if "ParserError" in quit_transcript:
            raise AssertionError(f"bare quit was parsed as SQL; transcript:\n{quit_transcript}")
    finally:
        if terminal.process.poll() is None:
            terminal.process.kill()
            terminal.process.wait()
        if terminal.master >= 0:
            os.close(terminal.master)
        os.unlink(script_path)
        os.unlink(nested_script_path)
        os.unlink(inline_script_path)
        os.unlink(dash_close_script_path)
        os.unlink(dollar_script_path)
        os.unlink(shebang_script_path)
        os.unlink(literal_script_path)
        os.unlink(invalid_script_path)


if __name__ == "__main__":
    main()
