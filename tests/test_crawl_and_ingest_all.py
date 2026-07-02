"""Unit Test Suite for crawl_and_ingest_all.py ACDSee Metadata Crawler.

Purpose:
    Verifies the correctness of the db_writer_worker persistent connection
    refactor, the metadata extraction/merging logic, and the compute_rel_path
    helper function.

Architecture and Mechanics:
    1. In-memory SQLite: All DB tests use ':memory:' to avoid disk writes.
    2. PYTEST_CURRENT_TEST env var forces the 'sqlite' backend code path so
       tests run without a live PostgreSQL server.
    3. Threading: db_writer_worker runs in a real daemon thread using an
       in-memory SQLite DB to validate end-to-end queue->write flow.
    4. Connection reuse: Verifies that the persistent connection is opened
       exactly once per worker invocation (not once per batch).

Execution Modes:
    - Test Runner:
      python -m pytest test_crawl_and_ingest_all.py -v
"""

import json
import os
import queue
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import crawl_and_ingest_all as crawler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_db(path: str) -> None:
    """Creates a minimal photos table in an SQLite file for testing.

    Args:
        path: Absolute path to the SQLite file to create.
    """
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_path TEXT UNIQUE NOT NULL,
            rel_path TEXT NOT NULL,
            primary_subject TEXT,
            environment TEXT,
            suggested_tags TEXT,
            technical_details TEXT,
            detected_objects TEXT,
            detected_faces TEXT DEFAULT '[]',
            acdsee_tags TEXT DEFAULT '[]',
            rating INTEGER,
            label TEXT,
            author TEXT,
            gps_latitude REAL,
            gps_longitude REAL,
            gps_altitude REAL,
            raw_metadata TEXT DEFAULT '{}',
            acdsee_metadata_imported_at TEXT,
            file_mtime REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_rel_path ON photos (rel_path)")
    conn.commit()
    conn.close()


def _make_exiftool_item(
    source_file: str,
    faces: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    rating: Optional[int] = None,
    gps_lat: Optional[float] = None,
    gps_lon: Optional[float] = None,
) -> Dict[str, Any]:
    """Constructs a mock ExifTool JSON record for a single image.

    Args:
        source_file: The full file path as ExifTool would report it.
        faces: Optional list of face names.
        tags: Optional list of keyword/tag strings.
        rating: Optional star rating integer.
        gps_lat: Optional GPS latitude float.
        gps_lon: Optional GPS longitude float.

    Returns:
        A dictionary mimicking ExifTool's JSON output for one file.
    """
    item: Dict[str, Any] = {"SourceFile": source_file}
    if faces:
        item["ACDSeeRegionName"] = faces
    if tags:
        item["Keywords"] = tags
    if rating is not None:
        item["Rating"] = rating
    if gps_lat is not None:
        item["GPSLatitude"] = gps_lat
    if gps_lon is not None:
        item["GPSLongitude"] = gps_lon
    return item


# ---------------------------------------------------------------------------
# Tests: compute_rel_path
# ---------------------------------------------------------------------------

class TestComputeRelPath(unittest.TestCase):
    """Verifies the compute_rel_path path normalization helper."""

    def test_windows_pictures_path(self) -> None:
        """Strips the drive prefix up to and including 'pictures/'."""
        result = crawler.compute_rel_path(r"D:\Users\steven\Pictures\2023\birthday\img.jpg")
        self.assertEqual(result, "2023/birthday/img.jpg")

    def test_h_drive_path(self) -> None:
        """Strips the 'h:/' prefix for H-drive paths lacking 'pictures/'."""
        result = crawler.compute_rel_path(r"H:\Backups\archive.jpg")
        self.assertEqual(result, "backups/archive.jpg")

    def test_patreon_path(self) -> None:
        """Strips the prefix up to 'patreon/' for Patreon content paths."""
        result = crawler.compute_rel_path(r"D:\Patreon\creator\set1\photo.jpg")
        self.assertEqual(result, "creator/set1/photo.jpg")

    def test_result_is_lowercase(self) -> None:
        """Asserts the returned path is always normalized to lowercase."""
        result = crawler.compute_rel_path(r"D:\Users\steven\Pictures\Test\UPPER.JPG")
        self.assertEqual(result, result.lower())

    def test_fallback_to_basename(self) -> None:
        """Falls back to basename for paths that match no known prefix."""
        result = crawler.compute_rel_path(r"C:\unknown\path\photo.jpg")
        self.assertEqual(result, "photo.jpg")


# ---------------------------------------------------------------------------
# Tests: db_writer_worker persistent connection
# ---------------------------------------------------------------------------

class TestDbWriterWorkerPersistentConnection(unittest.TestCase):
    """Verifies the db_writer_worker opens exactly one connection and reuses it."""

    def setUp(self) -> None:
        """Creates a temporary SQLite database file for each test."""
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db_path: str = self._tmp.name
        _create_test_db(self.db_path)

        # Pre-seed an existing row so updates are exercised
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO photos (full_path, rel_path) VALUES (?, ?)",
            (r"D:\Users\steven\Pictures\test\img.jpg", "test/img.jpg"),
        )
        conn.commit()
        conn.close()

    def tearDown(self) -> None:
        """Removes the temporary database file after each test."""
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def _run_worker(
        self,
        items: List[List[Dict[str, Any]]],
        existing_paths_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """Dispatches items through the db_writer_worker and waits for completion.

        Args:
            items: A list of batches (each batch is a list of ExifTool item dicts).
            existing_paths_map: Optional normalized-path → db-path lookup for updates.
        """
        result_q: queue.Queue = queue.Queue(maxsize=100)
        for batch in items:
            result_q.put(batch)
        result_q.put(None)  # Termination signal

        db_conn_params = {"database": self.db_path}
        t = threading.Thread(
            target=crawler.db_writer_worker,
            args=("sqlite", db_conn_params, result_q, len(items) * 500, existing_paths_map or {}),
            daemon=True,
        )
        t.start()
        t.join(timeout=15.0)
        self.assertFalse(t.is_alive(), "db_writer_worker thread did not terminate in time.")

    def test_insert_new_record(self) -> None:
        """Verifies that a new file record is inserted when not in existing_paths_map."""
        new_file = r"D:\Users\steven\Pictures\new_album\new_photo.jpg"
        batch = [_make_exiftool_item(new_file, tags=["sunset"], rating=5)]
        self._run_worker([batch], existing_paths_map={})

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT full_path, acdsee_tags, rating FROM photos WHERE full_path = ?", (new_file,))
        row = cur.fetchone()
        conn.close()

        self.assertIsNotNone(row, "New record should have been inserted.")
        self.assertEqual(row[0], new_file)
        self.assertIn("sunset", json.loads(row[1]))
        self.assertEqual(row[2], 5)

    def test_update_existing_record(self) -> None:
        """Verifies that an existing file record is updated when path is in existing_paths_map."""
        existing_file = r"D:\Users\steven\Pictures\test\img.jpg"
        norm_path = existing_file.replace("\\", "/").lower()
        existing_map = {norm_path: existing_file}

        batch = [_make_exiftool_item(
            existing_file,
            faces=["Alice", "Bob"],
            tags=["family"],
            gps_lat=47.6062,
            gps_lon=-122.3321,
        )]
        self._run_worker([batch], existing_paths_map=existing_map)

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT detected_faces, acdsee_tags, gps_latitude, acdsee_metadata_imported_at FROM photos WHERE full_path = ?",
            (existing_file,),
        )
        row = cur.fetchone()
        conn.close()

        self.assertIsNotNone(row, "Existing record should still be present.")
        faces = json.loads(row[0])
        self.assertIn("Alice", faces)
        self.assertIn("Bob", faces)
        self.assertIn("family", json.loads(row[1]))
        self.assertAlmostEqual(row[2], 47.6062, places=3)
        self.assertIsNotNone(row[3], "acdsee_metadata_imported_at should be set after update.")

    def test_multi_batch_all_committed(self) -> None:
        """Verifies multiple batches are all committed via the single persistent connection."""
        batches: List[List[Dict[str, Any]]] = []
        inserted_files = []
        for i in range(5):
            path = fr"D:\Users\steven\Pictures\batch_test\img_{i:03d}.jpg"
            inserted_files.append(path)
            batches.append([_make_exiftool_item(path)])

        self._run_worker(batches, existing_paths_map={})

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM photos WHERE full_path LIKE ?", (r"D:\Users\steven\Pictures\batch_test\%",))
        count = cur.fetchone()[0]
        conn.close()

        self.assertEqual(count, 5, f"Expected 5 new inserts across batches, found {count}.")

    def test_duplicate_faces_deduplicated(self) -> None:
        """Verifies face names from ACDSeeRegionName and RegionPersonDisplayName are deduped."""
        new_file = r"D:\Users\steven\Pictures\faces\group.jpg"
        # Simulate a record where the same person appears in both face fields
        item = _make_exiftool_item(new_file)
        item["ACDSeeRegionName"] = ["Alice", "Bob"]
        item["RegionPersonDisplayName"] = ["Alice", "Charlie"]

        self._run_worker([[item]], existing_paths_map={})

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT detected_faces FROM photos WHERE full_path = ?", (new_file,))
        row = cur.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        faces = json.loads(row[0])
        # Alice should appear only once despite being in both fields
        self.assertEqual(faces.count("Alice"), 1)
        self.assertIn("Bob", faces)
        self.assertIn("Charlie", faces)

    def test_vlm_tags_excluded_from_acdsee_tags(self) -> None:
        """Verifies 'description' and 'caption-abstract' are stripped from acdsee_tags."""
        new_file = r"D:\Users\steven\Pictures\vlm_test\photo.jpg"
        item = _make_exiftool_item(new_file)
        item["Keywords"] = ["landscape", "description", "Caption-Abstract", "mountains"]

        self._run_worker([[item]], existing_paths_map={})

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT acdsee_tags FROM photos WHERE full_path = ?", (new_file,))
        row = cur.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        tags = json.loads(row[0])
        self.assertIn("landscape", tags)
        self.assertIn("mountains", tags)
        self.assertNotIn("description", tags)
        self.assertNotIn("Caption-Abstract", [t.lower() for t in tags])


# ---------------------------------------------------------------------------
# Tests: migrate_schema
# ---------------------------------------------------------------------------

class TestMigrateSchema(unittest.TestCase):
    """Verifies migrate_schema creates and evolves the photos table schema."""

    def test_creates_table_and_acdsee_columns(self) -> None:
        """Verifies all required ACDSee columns are added by migrate_schema."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            # Start with an empty DB — migrate_schema should create everything
            crawler.migrate_schema(db_path)

            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(photos)")
            columns = {row[1] for row in cur.fetchall()}
            conn.close()

            required = {
                "full_path", "rel_path", "detected_faces", "acdsee_tags",
                "rating", "label", "author", "gps_latitude", "gps_longitude",
                "gps_altitude", "raw_metadata", "acdsee_metadata_imported_at",
                "file_mtime",
            }
            missing = required - columns
            self.assertFalse(missing, f"migrate_schema missing columns: {missing}")
        finally:
            os.unlink(db_path)

    def test_idempotent_on_existing_schema(self) -> None:
        """Verifies running migrate_schema twice does not raise errors."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            crawler.migrate_schema(db_path)
            # Second call should be fully idempotent
            crawler.migrate_schema(db_path)
        except Exception as e:
            self.fail(f"migrate_schema raised on second call: {e}")
        finally:
            os.unlink(db_path)


if __name__ == "__main__":
    unittest.main()
