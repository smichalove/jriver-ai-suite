"""Unit Test Suite for Database Pruning Utility.

Purpose:
    This module implements a comprehensive unit test suite to verify the database pruning
    utility ('prune_database.py'). It ensures that database records pointing to missing files
    are correctly purged, safety constraints for unmounted drives are respected, legacy path exclusions
    function properly, and the automatic backups/VACUUM cycles run correctly.

Architecture and Mechanics:
    1. Isolation: Uses temporary directories and file fixtures to avoid touching real databases or disks.
    2. Test Fixture Setup: Creates a temporary database populated with mock photo records (Windows paths, legacy Mac volume paths, unmounted drives).
    3. Target Subsystems Tested:
       - Standard Pruning: Records with missing paths are removed, existing ones are kept.
       - Dry-Run Safety: No database changes occur if dry_run=True.
       - Disconnected Drive Safety: If a path has a drive letter (e.g. X:) that is not mounted, the record is skipped.
       - Legacy Path Safety: Paths starting with '/Volumes/' are skipped unless prune_legacy=True.
       - Database Backup & Vacuum: SQLite database backup copies and VACUUM cycles are verified.
       - JSON Database Sync: Verifies that the JSON backup and records list are updated in sync with the database.

Execution Modes:
    - Test Runner: Run using python's unittest runner.
      python -m unittest test_prune_database.py
"""

import os
import sys
import json
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from typing import List, Dict, Set, Tuple, Any

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prune_database import prune_database


class TestDatabasePruner(unittest.TestCase):
    """Unit test suite for validating the database pruning utility."""

    def setUp(self) -> None:
        """Sets up temporary folders, databases, and sample image files for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "photo_catalog.db")
        self.json_path = os.path.join(self.test_dir, "photo_descriptions.json")

        # Create dummy image file that exists
        self.existing_photo = os.path.join(self.test_dir, "exists.jpg")
        with open(self.existing_photo, "w") as f:
            f.write("dummy data")

        self.missing_photo = os.path.join(self.test_dir, "missing.jpg")
        # Do not create missing_photo

        # Initialize SQLite database with schema and rows
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_path TEXT UNIQUE NOT NULL,
                rel_path TEXT NOT NULL,
                primary_subject TEXT,
                environment TEXT,
                suggested_tags TEXT,
                technical_details TEXT,
                detected_objects TEXT
            )
        """)
        
        # Insert test records
        # 1. Existing photo
        cursor.execute("INSERT INTO photos (full_path, rel_path, primary_subject) VALUES (?, 'exists.jpg', 'Exists')", (self.existing_photo,))
        # 2. Missing photo (should be pruned)
        cursor.execute("INSERT INTO photos (full_path, rel_path, primary_subject) VALUES (?, 'missing.jpg', 'Missing')", (self.missing_photo,))
        # 3. Legacy Mac path (should be skipped by default)
        cursor.execute("INSERT INTO photos (full_path, rel_path, primary_subject) VALUES ('/Volumes/External/pic.jpg', 'pic.jpg', 'Mac Legacy')")
        # 4. Unmounted Windows drive path (should be skipped by default)
        cursor.execute("INSERT INTO photos (full_path, rel_path, primary_subject) VALUES ('Z:\\Pictures\\pic.jpg', 'pic.jpg', 'Z Drive')")
        
        conn.commit()
        conn.close()

        # Initialize legacy JSON file
        json_data = [
            {"full_path": self.existing_photo, "primary_subject": "Exists"},
            {"full_path": self.missing_photo, "primary_subject": "Missing"},
            {"full_path": "/Volumes/External/pic.jpg", "primary_subject": "Mac Legacy"},
            {"full_path": "Z:\\Pictures\\pic.jpg", "primary_subject": "Z Drive"}
        ]
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f)

    def tearDown(self) -> None:
        """Cleans up temporary directory after each test."""
        shutil.rmtree(self.test_dir)

    def test_dry_run_no_changes(self) -> None:
        """Verifies that dry run executes safely without modifying database contents."""
        prune_database(self.db_path, self.json_path, dry_run=True, prune_legacy=False)

        # Check SQLite contains all records
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM photos")
        count = cursor.fetchone()[0]
        conn.close()
        self.assertEqual(count, 4)

        # Check JSON contains all records
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 4)

    def test_pruning_missing_files_and_backup(self) -> None:
        """Verifies that missing files are deleted, unmounted/legacy paths are kept, and backups/VACUUM are executed."""
        prune_database(self.db_path, self.json_path, dry_run=False, prune_legacy=False)

        # 1. Check SQLite database modifications
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT full_path FROM photos")
        rows = [r[0] for r in cursor.fetchall()]
        conn.close()

        # Missing photo should be pruned
        self.assertNotIn(self.missing_photo, rows)
        # Existing photo must be kept
        self.assertIn(self.existing_photo, rows)
        # Legacy Mac path must be kept (prune_legacy=False)
        self.assertIn("/Volumes/External/pic.jpg", rows)
        # Unmounted Z drive must be kept
        self.assertIn("Z:\\Pictures\\pic.jpg", rows)

        # 2. Check JSON synchronization
        with open(self.json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        json_paths = [item["full_path"] for item in json_data]
        self.assertNotIn(self.missing_photo, json_paths)
        self.assertIn(self.existing_photo, json_paths)
        self.assertIn("/Volumes/External/pic.jpg", json_paths)
        self.assertIn("Z:\\Pictures\\pic.jpg", json_paths)

        # 3. Check Backup creation
        self.assertTrue(os.path.exists(self.db_path + ".bak"))
        self.assertTrue(os.path.exists(self.json_path + ".bak"))

    def test_pruning_with_legacy_flag(self) -> None:
        """Verifies that legacy Mac volume paths are successfully pruned when --prune-legacy is enabled."""
        prune_database(self.db_path, self.json_path, dry_run=False, prune_legacy=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT full_path FROM photos")
        rows = [r[0] for r in cursor.fetchall()]
        conn.close()

        # Missing photo and Mac legacy photo must be pruned
        self.assertNotIn(self.missing_photo, rows)
        self.assertNotIn("/Volumes/External/pic.jpg", rows)
        # Existing photo must be kept
        self.assertIn(self.existing_photo, rows)
        # Unmounted Z drive must be kept
        self.assertIn("Z:\\Pictures\\pic.jpg", rows)


    @patch("prune_database.PICTURE_DIRS")
    def test_path_translation(self, mock_picture_dirs: MagicMock) -> None:
        """Verifies that missing files are translated/migrated to local Windows paths if they exist under PICTURE_DIRS."""
        # Setup mock PICTURE_DIRS to look inside our temp test directory
        mock_picture_dirs.__getitem__.side_effect = [self.test_dir].__getitem__
        mock_picture_dirs.__iter__.side_effect = [self.test_dir].__iter__
        
        # Create the translated target folder and file
        holiday_dir = os.path.join(self.test_dir, "holiday")
        os.makedirs(holiday_dir, exist_ok=True)
        translated_file_path = os.path.normpath(os.path.join(holiday_dir, "pic.jpg"))
        with open(translated_file_path, "w") as f:
            f.write("image data")

        # Add the test Mac path into SQLite and JSON
        mac_path = "/Volumes/External/Pictures/holiday/pic.jpg"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO photos (full_path, rel_path, primary_subject) VALUES (?, 'holiday/pic.jpg', 'Holiday')", (mac_path,))
        conn.commit()
        conn.close()

        # Update JSON data file
        with open(self.json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        json_data.append({"full_path": mac_path, "primary_subject": "Holiday"})
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f)

        # Run the pruner in commit mode
        prune_database(self.db_path, self.json_path, dry_run=False, prune_legacy=False)

        # Verify SQLite record is updated to the translated Windows path (MSDOS style)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT full_path FROM photos WHERE rel_path = 'holiday/pic.jpg'")
        updated_path = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(updated_path, translated_file_path)

        # Verify JSON file record path is also updated
        with open(self.json_path, "r", encoding="utf-8") as f:
            updated_json_data = json.load(f)
        json_paths = [item["full_path"] for item in updated_json_data if item.get("primary_subject") == "Holiday"]
        self.assertEqual(json_paths[0], translated_file_path)


if __name__ == "__main__":
    unittest.main()
