"""Unit Tests for Interactive Database Chat REPL Client on the Dev Branch.

Purpose:
    Contains unit tests for the functions in db_chat_repl.py, focusing on database counts,
    system prompt fallbacks, row truncation, and dynamic metadata column formatting.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open
from typing import Dict, List, Tuple, Any

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db_chat_repl
from db_chat_repl import (
    load_system_prompt,
    get_total_photos_count,
    execute_sql
)


class TestDBChatRepl(unittest.TestCase):
    """Test suite for db_chat_repl.py functionality.

    Attributes:
        None
    """

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="Mock prompt {current_time} {total_photos}")
    def test_load_system_prompt_success(self, mock_file: MagicMock, mock_exists: MagicMock) -> None:
        """Tests load_system_prompt retrieves the external prompt content successfully.

        Args:
            mock_file: Mocked builtins.open.
            mock_exists: Mocked os.path.exists.

        Returns:
            None
        """
        mock_exists.return_value = True
        prompt = load_system_prompt("mock_prompt.txt")
        self.assertEqual(prompt, "Mock prompt {current_time} {total_photos}")

    @patch("os.path.exists")
    def test_load_system_prompt_fallback(self, mock_exists: MagicMock) -> None:
        """Tests load_system_prompt uses fallback string when prompt file is missing.

        Args:
            mock_exists: Mocked os.path.exists.

        Returns:
            None
        """
        mock_exists.return_value = False
        prompt = load_system_prompt("nonexistent_prompt.txt")
        self.assertIn("Total photo records currently cataloged", prompt)

    @patch("os.path.exists")
    @patch("sqlite3.connect")
    def test_get_total_photos_count(self, mock_connect: MagicMock, mock_exists: MagicMock) -> None:
        """Tests get_total_photos_count retrieves correct count from photos table.

        Args:
            mock_connect: Mocked sqlite3 connection.
            mock_exists: Mocked os.path.exists.

        Returns:
            None
        """
        mock_exists.return_value = True
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (100,)
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        count = get_total_photos_count()
        self.assertEqual(count, 100)

    @patch("os.path.exists")
    @patch("sqlite3.connect")
    def test_execute_sql_bullets_and_truncation(self, mock_connect: MagicMock, mock_exists: MagicMock) -> None:
        """Tests execute_sql formats bullet lists with truncated metadata, and truncates VLM cells.

        Args:
            mock_connect: Mocked sqlite3 connection.
            mock_exists: Mocked os.path.exists.

        Returns:
            None
        """
        mock_exists.return_value = True
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.description = [("full_path", None), ("primary_subject", None)]
        
        long_desc = "A black cat " * 200  # 2400 chars
        mock_cursor.fetchall.return_value = [
            ("D:\\Pictures\\cat.jpg", long_desc)
        ] * 11
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        raw_markdown, term_display, paths = execute_sql("SELECT full_path, primary_subject FROM photos")

        # Verify that terminal display bullets append the description (formatted dynamically)
        self.assertIn("D:\\Pictures\\cat.jpg", term_display)
        self.assertIn("primary_subject: ", term_display)
        
        # Verify that the full description is present (not truncated in console)
        self.assertIn("A black cat", term_display)
        
        # Verify subsequent lines are indented
        self.assertIn("\n    ", term_display)

        # Verify raw markdown sent to VLM truncates cells to 2000 chars as well (including 3 for '...')
        self.assertIn("...", raw_markdown)
        self.assertEqual(paths, ["D:\\Pictures\\cat.jpg"] * 11)

    @patch("os.path.exists")
    @patch("sqlite3.connect")
    def test_execute_sql_300_rows_truncation(self, mock_connect: MagicMock, mock_exists: MagicMock) -> None:
        """Tests row truncation caps VLM markdown at 100 rows and terminal display at 300 rows.

        Args:
            mock_connect: Mocked sqlite3 connection.
            mock_exists: Mocked os.path.exists.

        Returns:
            None
        """
        mock_exists.return_value = True
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.description = [("full_path", None), ("primary_subject", None)]
        
        # Mock 350 rows
        mock_cursor.fetchall.return_value = [
            (f"D:\\Pictures\\cat_{i}.jpg", "A black cat") for i in range(350)
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        raw_markdown, term_display, paths = execute_sql("SELECT full_path, primary_subject FROM photos")

        # VLM raw_markdown table should have prefix note (2 lines), header (1 line), divider (1 line), and exactly 5 rows.
        markdown_lines = raw_markdown.splitlines()
        self.assertEqual(len(markdown_lines), 9)
        self.assertIn("Returned 300 rows.", raw_markdown)
        self.assertIn("Only the first 5 rows are shown below as a sample", raw_markdown)

        # Terminal display should have exactly 300 bullet records in paths
        self.assertEqual(len(paths), 300)
        self.assertIn("[300] D:\\Pictures\\cat_299.jpg", term_display)

    @patch("os.path.exists")
    @patch("sqlite3.connect")
    def test_execute_sql_multi_column_formatting(self, mock_connect: MagicMock, mock_exists: MagicMock) -> None:
        """Tests execute_sql dynamically formats all selected metadata columns.

        Args:
            mock_connect: Mocked sqlite3 connection.
            mock_exists: Mocked os.path.exists.

        Returns:
            None
        """
        mock_exists.return_value = True
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ("full_path", None),
            ("primary_subject", None),
            ("environment", None),
            ("suggested_tags", None)
        ]
        mock_cursor.fetchall.return_value = [
            ("D:\\Pictures\\cat.jpg", "A black cat", "living room", "[\"cat\", \"indoor\"]")
        ] * 11
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        raw_markdown, term_display, paths = execute_sql("SELECT * FROM photos")

        lines = term_display.splitlines()
        self.assertEqual(lines[0], "[1] D:\\Pictures\\cat.jpg")
        self.assertEqual(lines[1], "    primary_subject: A black cat")
        self.assertEqual(lines[2], "    environment: living room")
        self.assertEqual(lines[3], "    suggested_tags: [\"cat\", \"indoor\"]")

    @patch("builtins.input")
    @patch("threading.Thread")
    @patch("db_chat_repl.get_total_photos_count")
    @patch("db_chat_repl.load_system_prompt")
    def test_run_repl_catalog_command(self, mock_load_prompt: MagicMock, mock_count: MagicMock, mock_thread: MagicMock, mock_input: MagicMock) -> None:
        """Tests that run_repl correctly parses and launches the cataloger command from prompt.

        Args:
            mock_load_prompt: Mocked load_system_prompt.
            mock_count: Mocked get_total_photos_count.
            mock_thread: Mocked threading.Thread.
            mock_input: Mocked builtins.input.

        Returns:
            None
        """
        # Mock input sequence: run catalog command, then exit
        mock_input.side_effect = ["/catalog --max-photos 5", "exit"]
        mock_count.return_value = 100
        mock_load_prompt.return_value = "Mock system prompt"

        # Run REPL in remote mode so it does not boot local wsl server
        from db_chat_repl import run_repl
        run_repl(remote=True)

        # Verify that threading.Thread was called to launch the cataloger in background
        mock_thread.assert_called_once()
        kwargs = mock_thread.call_args[1]
        self.assertEqual(kwargs["name"], "CatalogerRun")
        self.assertEqual(kwargs["args"], (["--max-photos", "5"],))

    @patch("builtins.input")
    @patch("db_chat_repl.get_total_photos_count")
    @patch("db_chat_repl.load_system_prompt")
    @patch("sqlite3.connect")
    def test_run_repl_direct_sql(self, mock_connect: MagicMock, mock_load_prompt: MagicMock, mock_count: MagicMock, mock_input: MagicMock) -> None:
        """Tests that run_repl executes direct SQL input queries without using VLM.

        Args:
            mock_connect: Mocked sqlite3 connection.
            mock_load_prompt: Mocked load_system_prompt.
            mock_count: Mocked get_total_photos_count.
            mock_input: Mocked builtins.input.

        Returns:
            None
        """
        # Mock input sequence: direct SELECT query, then exit
        mock_input.side_effect = ["SELECT * FROM photos LIMIT 5", "exit"]
        mock_count.return_value = 100
        mock_load_prompt.return_value = "Mock system prompt"

        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.description = [("full_path", None)]
        mock_cursor.fetchall.return_value = [("D:\\Pictures\\photo.jpg",)]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        from db_chat_repl import run_repl
        run_repl(remote=True)

        # Verify that execute_sql was called with the direct SQL input query
        mock_cursor.execute.assert_called_with("SELECT * FROM photos LIMIT 5")

    def test_type_hints(self) -> None:
        """Verifies type hints exist for all key functions in db_chat_repl.py."""
        import typing
        from db_chat_repl import load_system_prompt, get_total_photos_count, execute_sql, run_repl
        
        for func in [load_system_prompt, get_total_photos_count, execute_sql, run_repl]:
            hints = typing.get_type_hints(func)
            self.assertTrue(len(hints) > 0, f"Function {func.__name__} has no type hints.")

    @patch("builtins.input")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    @patch("db_chat_repl.get_total_photos_count")
    @patch("db_chat_repl.load_system_prompt")
    def test_run_repl_playlist_command(self, mock_load_prompt: MagicMock, mock_count: MagicMock, mock_makedirs: MagicMock, mock_file: MagicMock, mock_input: MagicMock) -> None:
        """Tests that run_repl processes the /playlist command and writes an M3U file."""
        import db_chat_repl
        # Inject mock paths into last_query_paths
        db_chat_repl.last_query_paths = [
            r"D:\Users\steven\Music\Track1.flac",
            r"D:\Users\steven\Music\Track2.mp3",
            r"D:\Users\steven\Pictures\Photo.jpg"  # Non-audio file
        ]

        # Input "/playlist test_list", then exit
        mock_input.side_effect = ["/playlist test_list", "exit"]
        mock_count.return_value = 100
        mock_load_prompt.return_value = "Mock system prompt"

        from db_chat_repl import run_repl
        run_repl(remote=True)

        # Assert directory creation was attempted
        mock_makedirs.assert_called_with(r"D:\Users\steven\Music\Playlists", exist_ok=True)
        
        # Assert file writing was triggered for only the 2 audio files
        mock_file.assert_called_with(r"D:\Users\steven\Music\Playlists\test_list.m3u", "w", encoding="utf-8")
        
        # Verify tracks written
        handle = mock_file()
        calls = [c[0][0] for c in handle.write.call_args_list]
        self.assertIn("D:\\Users\\steven\\Music\\Track1.flac\n", calls)
        self.assertIn("D:\\Users\\steven\\Music\\Track2.mp3\n", calls)
        self.assertNotIn("D:\\Users\\steven\\Pictures\\Photo.jpg\n", calls)

    @patch("builtins.input")
    @patch("requests.get")
    @patch("db_chat_repl.get_total_photos_count")
    @patch("db_chat_repl.load_system_prompt")
    def test_run_repl_play_command(self, mock_load_prompt: MagicMock, mock_count: MagicMock, mock_get: MagicMock, mock_input: MagicMock) -> None:
        """Tests that run_repl processes the /play command and sends MCWS HTTP requests."""
        import db_chat_repl
        # Inject mock paths into last_query_paths
        db_chat_repl.last_query_paths = [
            r"D:\Users\steven\Music\Track1.flac",
            r"D:\Users\steven\Music\Track2.mp3"
        ]

        # Input "/play", then exit
        mock_input.side_effect = ["/play", "exit"]
        mock_count.return_value = 100
        mock_load_prompt.return_value = "Mock system prompt"

        # Mock requests.get response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        from db_chat_repl import run_repl
        run_repl(remote=True)

        # Verify ClearPlaylist was called first
        mock_get.assert_any_call("http://127.0.0.1:52198/MCWS/v1/Playback/ClearPlaylist?Zone=0&ZoneType=ID", timeout=5)
        
        # Verify PlayByFilename was called for each track
        import urllib.parse
        encoded1 = urllib.parse.quote(r"D:\Users\steven\Music\Track1.flac")
        encoded2 = urllib.parse.quote(r"D:\Users\steven\Music\Track2.mp3")
        
        mock_get.assert_any_call(f"http://127.0.0.1:52198/MCWS/v1/Playback/PlayByFilename?Filenames={encoded1}&Location=End&Zone=0&ZoneType=ID", timeout=5)
        mock_get.assert_any_call(f"http://127.0.0.1:52198/MCWS/v1/Playback/PlayByFilename?Filenames={encoded2}&Location=End&Zone=0&ZoneType=ID", timeout=5)
        
        # Verify Play was called at the end
        mock_get.assert_any_call("http://127.0.0.1:52198/MCWS/v1/Playback/Play?Zone=0&ZoneType=ID", timeout=5)

    @patch("builtins.input")
    @patch("requests.get")
    @patch("db_chat_repl.get_total_photos_count")
    @patch("db_chat_repl.load_system_prompt")
    def test_run_repl_queue_command(self, mock_load_prompt: MagicMock, mock_count: MagicMock, mock_get: MagicMock, mock_input: MagicMock) -> None:
        """Tests that run_repl processes the /queue / /add command and sends MCWS HTTP requests without clearing."""
        import db_chat_repl
        # Inject mock paths into last_query_paths
        db_chat_repl.last_query_paths = [
            r"D:\Users\steven\Music\Track1.flac",
            r"D:\Users\steven\Music\Track2.mp3"
        ]

        # Input "/queue 2", then exit
        mock_input.side_effect = ["/queue 2", "exit"]
        mock_count.return_value = 100
        mock_load_prompt.return_value = "Mock system prompt"

        # Mock requests.get response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        from db_chat_repl import run_repl
        run_repl(remote=True)

        # Assert ClearPlaylist was NOT called
        for call_arg in mock_get.call_args_list:
            url = call_arg[0][0]
            self.assertNotIn("ClearPlaylist", url)
            self.assertNotIn("Playback/Play?", url)  # Play should also not be called for queues

        # Verify PlayByFilename was called for track 2 only
        import urllib.parse
        encoded2 = urllib.parse.quote(r"D:\Users\steven\Music\Track2.mp3")
        mock_get.assert_any_call(f"http://127.0.0.1:52198/MCWS/v1/Playback/PlayByFilename?Filenames={encoded2}&Location=End&Zone=0&ZoneType=ID", timeout=5)


if __name__ == "__main__":
    unittest.main()
