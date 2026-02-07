"""Tests for ContextTools."""

import pytest

from major_tom.tools.context_tools import ContextTools


class TestContextTools:
    def test_read_active_file_found(self, tmp_path):
        # Create a file in the monitor_path (passed as search root)
        test_file = tmp_path / "hello.py"
        test_file.write_text("print('hello')")

        # ContextTools checks monitor_path as a search root and walks it
        result = ContextTools.read_active_file("hello.py - VS Code", tmp_path)
        assert result is not None
        assert "print('hello')" in result
        assert "[FILE_READ]" in result

    def test_read_active_file_not_found(self, tmp_path):
        result = ContextTools.read_active_file("Editing missing.py - VS Code", tmp_path)
        assert result is None

    def test_read_active_file_no_extension(self, tmp_path):
        result = ContextTools.read_active_file("Safari - Google", tmp_path)
        assert result is None

    def test_read_active_file_encoding_error(self, tmp_path):
        test_file = tmp_path / "binary.py"
        test_file.write_bytes(b"\xff\xfe" + b"\x00" * 100)
        # Should not crash, errors='ignore' in implementation
        try:
            result = ContextTools.read_active_file("binary.py", tmp_path)
        except Exception:
            pytest.fail("read_active_file should not raise on encoding errors")

    def test_read_active_file_nested(self, tmp_path):
        subdir = tmp_path / "project" / "src"
        subdir.mkdir(parents=True)
        test_file = subdir / "main.py"
        test_file.write_text("import os")

        result = ContextTools.read_active_file("main.py - VS Code", tmp_path)
        assert result is not None
        assert "import os" in result

    def test_read_clipboard_returns_string_or_none(self):
        # Just test it doesn't crash; actual clipboard access is platform-dependent
        result = ContextTools.read_clipboard()
        assert result is None or isinstance(result, str)
