"""Unit Test Suite for add_locations.py Offline Geocoding Utility.

Purpose:
    Verifies database connection resolving, database schema migrations,
    coordinate queries, offline geocoding format mappings, and batch update logic.

Architecture and Mechanics:
    1. Isolation: Uses in-memory SQLite databases (":memory:") for DB tests to avoid disk side effects.
    2. Mocking: Mocks the 'reverse_geocoder.search' method to run tests instantly without
       loading the heavy GeoNames database into memory or executing network requests.
    3. Target Subsystems Tested:
       - Connection Resolution: Checks that argument overrides for '--db', '--root',
         and '--backend' are correctly resolved.
       - Schema Migration: Verifies that 'location_name' is added if missing.
       - Coordinate Querying: Confirms only rows with GPS coordinates and NULL location_names are selected.
       - Geocoding Mappings: Asserts that GeoNames search dicts map to "City, State, Country Code".
       - Batch Database Updates: Ensures update queries are batched and committed correctly.
"""

import os
import sys
import sqlite3
import unittest
from unittest.mock import patch, MagicMock
from typing import List, Tuple, Dict, Union, Optional

# Support importing from either the workspace root or the folder directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import add_locations


class TestGeocodingUtility(unittest.TestCase):
    """Unit test suite for validating the offline geocoding utility."""

    def setUp(self) -> None:
        """Sets up an in-memory SQLite database and populates schema for testing."""
        self.conn: sqlite3.Connection = sqlite3.connect(":memory:")
        self.cursor: sqlite3.Cursor = self.conn.cursor()
        
        # Create core table schema
        self.cursor.execute("""
            CREATE TABLE photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_path TEXT UNIQUE NOT NULL,
                rel_path TEXT NOT NULL,
                primary_subject TEXT,
                gps_latitude REAL,
                gps_longitude REAL,
                gps_altitude REAL
            )
        """)
        self.conn.commit()

    def tearDown(self) -> None:
        """Closes the connection to clean up the in-memory database."""
        self.conn.close()

    def test_migrate_database_schema(self) -> None:
        """Checks that the schema migration adds 'location_name' if missing."""
        # Check that location_name does not exist initially
        self.cursor.execute("PRAGMA table_info(photos)")
        cols: List[str] = [col[1] for col in self.cursor.fetchall()]
        self.assertNotIn("location_name", cols)

        # Run migration
        add_locations.migrate_database_schema(self.conn, "sqlite")

        # Verify column now exists
        self.cursor.execute("PRAGMA table_info(photos)")
        cols_after: List[str] = [col[1] for col in self.cursor.fetchall()]
        self.assertIn("location_name", cols_after)

    def test_get_unresolved_gps_coordinates(self) -> None:
        """Ensures get_unresolved_gps_coordinates retrieves only photos needing resolution."""
        add_locations.migrate_database_schema(self.conn, "sqlite")

        # Insert test records:
        # 1. Has coordinates, needs resolution
        self.cursor.execute(
            "INSERT INTO photos (full_path, rel_path, gps_latitude, gps_longitude) VALUES (?, ?, ?, ?)",
            ("D:\\pic1.jpg", "pic1.jpg", 47.57, -122.01)
        )
        # 2. Has coordinates, already resolved
        self.cursor.execute(
            "INSERT INTO photos (full_path, rel_path, gps_latitude, gps_longitude, location_name) VALUES (?, ?, ?, ?, ?)",
            ("D:\\pic2.jpg", "pic2.jpg", 48.85, 2.35, "Paris, FR")
        )
        # 3. Missing coordinates (latitude or longitude is NULL)
        self.cursor.execute(
            "INSERT INTO photos (full_path, rel_path) VALUES (?, ?)",
            ("D:\\pic3.jpg", "pic3.jpg")
        )
        self.conn.commit()

        coords: List[Tuple[float, float]] = add_locations.get_unresolved_gps_coordinates(self.conn)
        
        # Should only contain (47.57, -122.01)
        self.assertEqual(len(coords), 1)
        self.assertAlmostEqual(coords[0][0], 47.57)
        self.assertAlmostEqual(coords[0][1], -122.01)

    @patch("reverse_geocoder.search")
    def test_perform_bulk_reverse_geocoding(self, mock_search: MagicMock) -> None:
        """Verifies that coordinates are mapped correctly from GeoNames query results."""
        # Mock reverse_geocoder.search return structure
        mock_search.return_value = [
            {"name": "Issaquah", "admin1": "Washington", "cc": "US"},
            {"name": "Paris", "admin1": "Île-de-France", "cc": "FR"}
        ]

        input_coords: List[Tuple[float, float]] = [(47.57, -122.01), (48.85, 2.35)]
        resolved: Dict[Tuple[float, float], str] = add_locations.perform_bulk_reverse_geocoding(input_coords)

        # Assert correct formatting
        self.assertEqual(resolved[(47.57, -122.01)], "Issaquah, Washington, US")
        self.assertEqual(resolved[(48.85, 2.35)], "Paris, Île-de-France, FR")
        
        # Verify search was called with coordinates
        mock_search.assert_called_once_with(input_coords)

    def test_update_photos_in_batches(self) -> None:
        """Verifies that geocoded results are updated back to the database in batches."""
        add_locations.migrate_database_schema(self.conn, "sqlite")

        # Insert 3 photos requiring updates
        self.cursor.execute(
            "INSERT INTO photos (full_path, rel_path, gps_latitude, gps_longitude) VALUES (?, ?, ?, ?)",
            ("D:\\p1.jpg", "p1.jpg", 47.5, -122.0)
        )
        self.cursor.execute(
            "INSERT INTO photos (full_path, rel_path, gps_latitude, gps_longitude) VALUES (?, ?, ?, ?)",
            ("D:\\p2.jpg", "p2.jpg", 48.0, 2.0)
        )
        self.cursor.execute(
            "INSERT INTO photos (full_path, rel_path, gps_latitude, gps_longitude) VALUES (?, ?, ?, ?)",
            ("D:\\p3.jpg", "p3.jpg", 45.0, 1.0)
        )
        self.conn.commit()

        resolved_map: Dict[Tuple[float, float], str] = {
            (47.5, -122.0): "City A, State A, US",
            (48.0, 2.0): "City B, State B, FR",
            (45.0, 1.0): "City C, State C, IT"
        }

        # Override the batch update size to test batch split logic with smaller numbers
        with patch.object(add_locations, "BATCH_UPDATE_SIZE", 2), \
             patch("time.sleep") as mock_sleep:
            add_locations.update_photos_in_batches(self.conn, "sqlite", resolved_map)
            # Batch size is 2, so it should sleep exactly once (at the end of the first batch)
            mock_sleep.assert_called_once_with(0.05)

        # Check DB updates committed correctly
        self.cursor.execute("SELECT full_path, location_name FROM photos ORDER BY full_path")
        rows = self.cursor.fetchall()
        
        self.assertEqual(rows[0], ("D:\\p1.jpg", "City A, State A, US"))
        self.assertEqual(rows[1], ("D:\\p2.jpg", "City B, State B, FR"))
        self.assertEqual(rows[2], ("D:\\p3.jpg", "City C, State C, IT"))

    @patch("psycopg2.connect")
    @patch("sqlite3.connect")
    def test_get_db_conn_args(self, mock_sqlite_conn: MagicMock, mock_pg_conn: MagicMock) -> None:
        """Ensures that get_db_conn resolves parameters, overrides, and environment vars correctly."""
        # 1. Override backend to postgresql
        conn, backend = add_locations.get_db_conn(db_backend="postgresql")
        self.assertEqual(backend, "postgresql")
        mock_pg_conn.assert_called_once()
        
        # 2. Override backend to sqlite with a custom path
        mock_sqlite_conn.reset_mock()
        conn, backend = add_locations.get_db_conn(db_backend="sqlite", db_path="E:\\custom.db")
        self.assertEqual(backend, "sqlite")
        mock_sqlite_conn.assert_called_once_with("E:\\custom.db", timeout=60.0)

        # 3. Override backend to sqlite using --root folder mapping
        mock_sqlite_conn.reset_mock()
        conn, backend = add_locations.get_db_conn(db_backend="sqlite", root_dir="D:\\test_folder")
        self.assertEqual(backend, "sqlite")
        expected_path = os.path.normpath("D:\\test_folder\\photo_catalog.db")
        # Extract path argument from call
        actual_path = os.path.normpath(mock_sqlite_conn.call_args[0][0])
        self.assertEqual(actual_path, expected_path)


if __name__ == "__main__":
    unittest.main()
