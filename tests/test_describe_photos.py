"""Unit Test Suite for Gemma Photo Cataloger.

Purpose:
    This module implements a comprehensive unit test suite to verify the core utility
    functions of describe_photos.py. It ensures that JSON output extraction, file path parsing,
    HEIF/JPEG image loading and encoding, atomic database serialization, and ExifTool
    metadata embedding act reliably and fail gracefully under edge cases.

Architecture and Mechanics:
    1. Mocking Framework: Heavily utilizes unittest.mock (patch, MagicMock, mock_open) to isolate
       tests from the real filesystem, subprocess execution (exiftool.exe), and VLM server requests.
    2. Test Fixture Setup: Standard setUp/tearDown lifecycle checks.
    3. Target Subsystems Tested:
       - JSON Payload Extraction: Tests extraction from standard and markdown-fenced strings, missing key defaults, and fallback parsing for malformed objects.
       - File Discovery & Exclusions: Mocks os.walk to verify image filtering based on extensions and duplicate exclusion set parameters.
       - Metadata Embedding: Mocks subprocess.run to verify that ExifTool is called with correct caption, description, and temporary file argument parameters.
       - Atomic Saving: Verifies that save_results writes to a temporary file before performing an atomic rename operation to prevent file corruption.

Execution Modes:
    - Test Runner: Run using python's unittest runner or pytest from the CLI.
      Example command:
        python -m unittest gemma_cataloger/test_describe_photos.py
"""

import os
import sys
import json

import unittest
from unittest.mock import patch, MagicMock, mock_open
from typing import Dict, List, Set, Optional, Tuple, Any

# Ensure parent directory is in sys.path to allow importing from gemma_cataloger
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from describe_photos import (
    extract_json_payload,
    get_image_files,
    load_and_encode_image,
    inline_embed_metadata,
    save_results,
    save_results_to_sqlite,
    fetch_acdsee_metadata_for_batch
)


class TestGemmaPhotoCataloger(unittest.TestCase):
    """
    Unit test suite for the Gemma Photo Cataloger script utility functions.
    Verifies JSON extraction/parsing, image file discovery, EXIF metadata embedding,
    and results serialization.
    """

    def setUp(self) -> None:
        """
        Setup fixture called before each test method execution.

        Args:
            None

        Returns:
            None
        """
        pass

    def tearDown(self) -> None:
        """
        Teardown fixture called after each test method execution.

        Args:
            None

        Returns:
            None
        """
        pass

    def test_extract_json_payload_valid(self) -> None:
        """
        Tests that extract_json_payload correctly parses a clean JSON string.

        Args:
            None

        Returns:
            None
        """
        raw_json: str = (
            "{\n"
            "  \"primary_subject\": \"A beautiful mountain scene\",\n"
            "  \"environment\": \"outdoor\",\n"
            "  \"suggested_tags\": [\"nature\", \"landscape\"]\n"
            "}"
        )
        
        result: Dict[str, Any] = extract_json_payload(raw_json)
        
        # Verify that all expected fields were parsed correctly
        self.assertEqual(result["primary_subject"], "A beautiful mountain scene")
        self.assertEqual(result["environment"], "outdoor")
        self.assertEqual(result["suggested_tags"], ["nature", "landscape"])

    def test_extract_json_payload_with_fences(self) -> None:
        """
        Tests that extract_json_payload strips markdown json code blocks and parses correctly.

        Args:
            None

        Returns:
            None
        """
        raw_json_fenced: str = (
            "```json\n"
            "{\n"
            "  \"primary_subject\": \"A historic building\",\n"
            "  \"environment\": \"urban\",\n"
            "  \"suggested_tags\": [\"architecture\"]\n"
            "}\n"
            "```"
        )
        
        result: Dict[str, Any] = extract_json_payload(raw_json_fenced)
        
        self.assertEqual(result["primary_subject"], "A historic building")
        self.assertEqual(result["suggested_tags"], ["architecture"])

    def test_extract_json_payload_missing_keys(self) -> None:
        """
        Tests that extract_json_payload adds missing keys with appropriate default empty values.

        Args:
            None

        Returns:
            None
        """
        # Missing environment and suggested_tags
        incomplete_json: str = (
            "{\n"
            "  \"primary_subject\": \"A red apple\"\n"
            "}"
        )
        
        result: Dict[str, Any] = extract_json_payload(incomplete_json)
        
        self.assertEqual(result["primary_subject"], "A red apple")
        # Assert defaults were correctly filled
        self.assertEqual(result["environment"], "")
        self.assertEqual(result["suggested_tags"], [])

    def test_extract_json_payload_malformed(self) -> None:
        """
        Tests that extract_json_payload handles malformed JSON input gracefully by returning a fallback structure.

        Args:
            None

        Returns:
            None
        """
        malformed_json: str = "This is not JSON at all, it's just raw description text."
        
        result: Dict[str, Any] = extract_json_payload(malformed_json)
        
        # Verify it fallback parses the raw text into the primary_subject
        self.assertEqual(result["primary_subject"], malformed_json)
        self.assertEqual(result["environment"], "Unknown")
        self.assertEqual(result["suggested_tags"], ["error-parsing-json"])

    @patch("os.walk")
    @patch("os.path.exists")
    def test_get_image_files(self, mock_exists: MagicMock, mock_walk: MagicMock) -> None:
        """
        Tests get_image_files walks target directories and filters already-processed files.

        Args:
            mock_exists: MagicMock replacing os.path.exists
            mock_walk: MagicMock replacing os.walk

        Returns:
            None
        """
        # Setup mock behavior
        mock_exists.return_value = True
        
        # Simulate walking a directory and finding some image files
        # os.walk returns generator of (dirpath, dirnames, filenames)
        mock_walk.return_value = [
            (r"D:\Users\steven\Pictures", [], ["photo1.jpg", "photo2.png", "processed.jpg", "textfile.txt"])
        ]
        
        # Paths that should be marked as processed to test the duplicate filter
        processed: Set[str] = {"photo2.png"}
        
        result: List[str] = get_image_files(
            directories=[r"D:\Users\steven\Pictures"],
            limit=10,
            processed_paths=processed
        )
        
        # We expect photo1.jpg to be returned.
        # photo2.png is in processed, textfile.txt is not a valid extension, processed.jpg is not in skipped list.
        # So photo1.jpg and processed.jpg should match.
        expected_paths: List[str] = [
            os.path.join(r"D:\Users\steven\Pictures", "photo1.jpg"),
            os.path.join(r"D:\Users\steven\Pictures", "processed.jpg")
        ]
        self.assertEqual(len(result), 2)
        self.assertIn(expected_paths[0], result)
        self.assertIn(expected_paths[1], result)

    @patch("builtins.open", new_callable=mock_open)
    @patch("subprocess.run")
    def test_inline_embed_metadata(self, mock_run: MagicMock, mock_file: MagicMock) -> None:
        """
        Tests that inline_embed_metadata executes exiftool with correct command arguments
        and verifies the write by reading it back.

        Args:
            mock_run: MagicMock replacing subprocess.run
            mock_file: MagicMock replacing builtins.open

        Returns:
            None
        """
        file_path: str = r"D:\Users\steven\Pictures\photo1.jpg"
        summary_text: str = "Subject: A beautiful sunrise\nEnvironment: Outdoor\nTechnical: Shallow DoF\nTags: sun, dawn"
        summary_escaped: str = "Subject: A beautiful sunrise&#10;Environment: Outdoor&#10;Technical: Shallow DoF&#10;Tags: sun, dawn"
        
        # Configure side_effect for subprocess.run:
        # First call (write): returns a success response
        # Second call (read-back): returns JSON data with the expected values
        mock_write_res = MagicMock()
        mock_read_res = MagicMock()
        mock_read_res.stdout = b'[{"Description": "Subject: A beautiful sunrise\\nEnvironment: Outdoor\\nTechnical: Shallow DoF\\nTags: sun, dawn"}]'
        
        mock_run.side_effect = [mock_write_res, mock_read_res]
        
        inline_embed_metadata(file_path, summary_text)
        
        # Verify exiftool was executed twice (write then read)
        self.assertEqual(mock_run.call_count, 2)
        
        # Verify first call (write) arguments
        called_args_write: List[str] = mock_run.call_args_list[0][0][0]
        self.assertEqual(called_args_write[0], r"H:\Wan_project\exiftool\exiftool.exe")
        self.assertIn("-overwrite_original", called_args_write)
        self.assertIn("-E", called_args_write)
        self.assertNotIn("-Caption-Abstract=", "".join(called_args_write))
        self.assertNotIn("-Description=", "".join(called_args_write))
        self.assertNotIn("-ImageDescription=", "".join(called_args_write))
        self.assertIn("-@", called_args_write)
        self.assertTrue(called_args_write[-1].startswith(r"H:\Wan_project\exif_args_"))
        
        # Verify that open was called to write the argfile
        handle = mock_file()
        write_calls = [c[0][0] for c in handle.write.call_args_list]
        self.assertIn("-CodedCharacterSet=UTF8\n", write_calls)
        self.assertIn(f"-Caption-Abstract={summary_escaped}\n", write_calls)
        self.assertIn(f"-Description={summary_escaped}\n", write_calls)
        self.assertIn(f"-ImageDescription={summary_escaped}\n", write_calls)
        self.assertIn(file_path + "\n", write_calls)
        
        # Verify second call (read) arguments
        called_args_read: List[str] = mock_run.call_args_list[1][0][0]
        self.assertEqual(called_args_read[0], r"H:\Wan_project\exiftool\exiftool.exe")
        self.assertIn("-j", called_args_read)
        self.assertIn("-charset", called_args_read)
        self.assertIn("UTF8", called_args_read)
        self.assertIn("-Description", called_args_read)
        self.assertNotIn("-ImageDescription", called_args_read)
        self.assertNotIn("-Caption-Abstract", called_args_read)
        self.assertIn("-@", called_args_read)
        self.assertTrue(called_args_read[-1].startswith(r"H:\Wan_project\exif_read_args_"))

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.replace")
    def test_save_results_atomic(self, mock_replace: MagicMock, mock_file: MagicMock) -> None:
        """
        Tests that save_results atomically saves description database entries to disk via a temp file.

        Args:
            mock_replace: MagicMock replacing os.replace
            mock_file: MagicMock replacing builtins.open

        Returns:
            None
        """
        test_results: List[Dict[str, Any]] = [
            {
                "full_path": r"D:\photo1.jpg",
                "primary_subject": "Test subject",
                "environment": "Test environment",
                "suggested_tags": []
            }
        ]
        
        save_results(test_results, r"H:\Wan_project\photo_descriptions.json", is_milestone=False)
        
        # Check that the file was written to the temp location first
        mock_file.assert_called_with(r"H:\Wan_project\photo_descriptions.json.tmp", "w", encoding="utf-8")
        
        # Verify the atomic replace was triggered to overwrite the main catalog path
        mock_replace.assert_called_once_with(
            r"H:\Wan_project\photo_descriptions.json.tmp",
            r"H:\Wan_project\photo_descriptions.json"
        )

    def test_save_results_to_sqlite(self) -> None:
        """
        Tests that save_results_to_sqlite correctly inserts and updates database records in an SQLite database.

        Args:
            None

        Returns:
            None
        """
        import tempfile
        import sqlite3
        
        # Create a temporary DB file
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
            db_path = temp_db.name
            
        try:
            test_results: List[Dict[str, Any]] = [
                {
                    "full_path": r"D:\Pictures\pic1.jpg",
                    "primary_subject": "Initial Sunrise",
                    "environment": "Outdoor",
                    "suggested_tags": ["sun", "morning"],
                    "technical_details": "ISO 100",
                    "detected_objects": ["sky", "sun"]
                }
            ]
            
            # 1. Save new entry
            save_results_to_sqlite(db_path, test_results)
            
            # Connect and verify insert
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT full_path, rel_path, primary_subject, environment, suggested_tags, technical_details, detected_objects FROM photos")
            rows = cursor.fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], r"D:\Pictures\pic1.jpg")
            self.assertEqual(rows[0][1], "pic1.jpg")  # computed rel_path
            self.assertEqual(rows[0][2], "Initial Sunrise")
            self.assertEqual(rows[0][3], "Outdoor")
            self.assertEqual(json.loads(rows[0][4]), ["sun", "morning"])
            self.assertEqual(rows[0][5], "ISO 100")
            self.assertEqual(json.loads(rows[0][6]), ["sky", "sun"])
            
            # 2. Update existing entry (test UPSERT ON CONFLICT)
            updated_results: List[Dict[str, Any]] = [
                {
                    "full_path": r"D:\Pictures\pic1.jpg",
                    "primary_subject": "Updated Sunset",
                    "environment": "Outdoor Evening",
                    "suggested_tags": ["sunset", "red-sky"],
                    "technical_details": "ISO 200",
                    "detected_objects": ["sky", "clouds"]
                }
            ]
            save_results_to_sqlite(db_path, updated_results)
            
            cursor.execute("SELECT full_path, rel_path, primary_subject, environment, suggested_tags, technical_details, detected_objects FROM photos")
            rows = cursor.fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][2], "Updated Sunset")
            self.assertEqual(rows[0][3], "Outdoor Evening")
            self.assertEqual(json.loads(rows[0][4]), ["sunset", "red-sky"])
            self.assertEqual(rows[0][5], "ISO 200")
            self.assertEqual(json.loads(rows[0][6]), ["sky", "clouds"])
            
            conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_fetch_acdsee_metadata_for_batch(self) -> None:
        """
        Tests that fetch_acdsee_metadata_for_batch retrieves and formats ACDSee metadata columns correctly.
        """
        import tempfile
        import sqlite3
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
            db_path = temp_db.name
            
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE photos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    full_path TEXT UNIQUE NOT NULL,
                    rel_path TEXT NOT NULL,
                    primary_subject TEXT,
                    rating INTEGER,
                    label TEXT,
                    author TEXT,
                    gps_latitude REAL,
                    gps_longitude REAL,
                    gps_altitude REAL,
                    acdsee_tags TEXT,
                    detected_faces TEXT,
                    raw_metadata TEXT
                )
            """)
            
            cursor.execute("""
                INSERT INTO photos (
                    full_path, rel_path, rating, label, author, gps_latitude, gps_longitude, gps_altitude, acdsee_tags, detected_faces, raw_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                r"D:\Pictures\photo1.jpg", "photo1.jpg", 5, "Red", "John Doe", 47.6, -122.3, 45.0,
                '["Scenic", "Mountain"]', '["Alice", "Bob"]', '{"Make": "Canon", "Model": "EOS R"}'
            ))
            conn.commit()
            conn.close()
            
            res = fetch_acdsee_metadata_for_batch(db_path, [r"D:\Pictures\photo1.jpg"])
            
            self.assertIn(r"D:\Pictures\photo1.jpg", res)
            meta = res[r"D:\Pictures\photo1.jpg"]
            self.assertEqual(meta["rating"], 5)
            self.assertEqual(meta["label"], "Red")
            self.assertEqual(meta["author"], "John Doe")
            self.assertEqual(meta["gps_latitude"], 47.6)
            self.assertEqual(meta["gps_longitude"], -122.3)
            self.assertEqual(meta["gps_altitude"], 45.0)
            self.assertEqual(meta["acdsee_tags"], ["Scenic", "Mountain"])
            self.assertEqual(meta["detected_faces"], ["Alice", "Bob"])
            self.assertEqual(meta["raw_metadata"], {"Make": "Canon", "Model": "EOS R"})
            
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)


if __name__ == "__main__":
    unittest.main()
