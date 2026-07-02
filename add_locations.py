"""Offline Reverse Geocoding Utility for Photo Catalog Database.

This script migrates the database to support location names, resolves GPS
coordinates to nearest cities/states/countries offline using `reverse_geocoder`,
and writes the results back to the database in deferred transactions.
Supports command-line overrides for database paths and backends.
"""

import sqlite3
import time
import sys
import os
import argparse
import psycopg2
import psycopg2.extensions
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Union, Optional
import reverse_geocoder as rg

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Load workspace environment variables
PROJECT_DIR: str = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(PROJECT_DIR, "auth", ".env")):
    load_dotenv(os.path.join(PROJECT_DIR, "auth", ".env"))
elif os.path.exists("auth/.env"):
    load_dotenv("auth/.env")
else:
    load_dotenv()

DEFAULT_DB_PATH: str = os.path.join(PROJECT_DIR, "photo_catalog.db")
BATCH_UPDATE_SIZE: int = 100
YIELD_SLEEP_SECONDS: float = 0.05

# DB Connection Type
DatabaseConnection = Union[sqlite3.Connection, psycopg2.extensions.connection]


def get_db_conn(
    db_backend: Optional[str] = None,
    db_path: Optional[str] = None,
    root_dir: Optional[str] = None
) -> Tuple[DatabaseConnection, str]:
    """Returns a tuple of (connection, backend_type) depending on configuration.

    Args:
        db_backend: Override database backend choice ('sqlite' or 'postgresql').
        db_path: Override path to SQLite database.
        root_dir: Override root directory (looks for photo_catalog.db inside it if db_path is empty).

    Returns:
        A tuple of (database connection object, database backend type string).

    Raises:
        ValueError: If the resolved database backend type is unsupported.
    """
    is_testing: bool = "pytest" in sys.argv[0] or "unittest" in sys.argv[0] or "PYTEST_CURRENT_TEST" in os.environ
    backend: str = (db_backend or ("sqlite" if is_testing else os.getenv("DB_BACKEND", "postgresql"))).lower()
    
    if backend == "postgresql":
        db_conn_params: Dict[str, Union[str, int]] = {
            "dbname": os.getenv("DB_NAME", "photo_catalog"),
            "user": os.getenv("DB_USER", "postgres"),
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
        }
        pwd_path: str = os.path.join(PROJECT_DIR, "auth", "db_password.txt")
        if os.path.exists(pwd_path):
            with open(pwd_path, "r", encoding="utf-8") as f:
                db_conn_params["password"] = f.read().strip()
        conn: psycopg2.extensions.connection = psycopg2.connect(**db_conn_params)
        return conn, "postgresql"
    elif backend == "sqlite":
        # Resolve SQLite database file path hierarchy
        sqlite_path: str = ""
        if db_path:
            sqlite_path = db_path
        elif root_dir:
            sqlite_path = os.path.join(root_dir, "photo_catalog.db")
        else:
            sqlite_path = os.getenv("DB_PATH") or DEFAULT_DB_PATH
            
        print(f"Connecting to SQLite database at: {sqlite_path}")
        conn_sqlite: sqlite3.Connection = sqlite3.connect(sqlite_path, timeout=60.0)
        return conn_sqlite, "sqlite"
    else:
        raise ValueError(f"Unsupported database backend configured: {backend}")


def migrate_database_schema(conn: DatabaseConnection, backend: str) -> None:
    """Creates the location_name column if it does not exist in photos table.

    Args:
        conn: The database connection object.
        backend: The active backend type ('sqlite' or 'postgresql').
    """
    cursor = conn.cursor()
    columns: List[str] = []
    if backend == "sqlite":
        cursor.execute("PRAGMA table_info(photos)")
        columns = [col[1] for col in cursor.fetchall()]
    else:
        cursor.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'photos';
        """)
        columns = [row[0] for row in cursor.fetchall()]
    
    if "location_name" not in columns:
        print("Schema Migration: Adding 'location_name' column to the 'photos' table...")
        cursor.execute("ALTER TABLE photos ADD COLUMN location_name TEXT")
        conn.commit()
        print("Schema Migration Completed: Added 'location_name' column.")
    else:
        print("Schema Migration Check: 'location_name' column already exists.")


def get_unresolved_gps_coordinates(conn: DatabaseConnection) -> List[Tuple[float, float]]:
    """Retrieves all unique unresolved coordinates in the database.

    Args:
        conn: The database connection object.

    Returns:
        A list of coordinate tuples (latitude, longitude).
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT gps_latitude, gps_longitude FROM photos
        WHERE gps_latitude IS NOT NULL
          AND gps_longitude IS NOT NULL
          AND location_name IS NULL
    """)
    return [(float(row[0]), float(row[1])) for row in cursor.fetchall()]


def perform_bulk_reverse_geocoding(coords: List[Tuple[float, float]]) -> Dict[Tuple[float, float], str]:
    """Resolves coordinates offline using the GeoNames database.

    Args:
        coords: List of latitude/longitude tuples.

    Returns:
        A dictionary mapping (lat, lon) to a formatted location string.
    """
    if not coords:
        return {}
    
    print(f"Resolving {len(coords)} unique coordinate pairs using offline database...")
    start_time: float = time.time()
    
    # reverse_geocoder search retrieves nearest city details
    results = rg.search(coords)
    
    resolved_map: Dict[Tuple[float, float], str] = {}
    for coord, res in zip(coords, results):
        name: str = res.get('name', '')
        admin1: str = res.get('admin1', '')
        cc: str = res.get('cc', '')
        
        parts: List[str] = [x for x in [name, admin1, cc] if x]
        resolved_map[coord] = ", ".join(parts)
        
    duration: float = time.time() - start_time
    print(f"Geocoding complete in {duration:.4f} seconds ({len(coords) / duration:.2f} lookups/sec).")
    return resolved_map


def update_photos_in_batches(conn: DatabaseConnection, backend: str, resolved_map: Dict[Tuple[float, float], str]) -> None:
    """Updates the resolved locations back to the photos table in safe batches.

    Args:
        conn: The database connection object.
        backend: The database backend string ('sqlite' or 'postgresql').
        resolved_map: Mapping of coordinates to resolved location strings.
    """
    if not resolved_map:
        print("No records require location updates.")
        return
        
    cursor = conn.cursor()
    
    # Retrieve all photos needing updates to match them by coordinate pairs
    cursor.execute("""
        SELECT id, gps_latitude, gps_longitude FROM photos
        WHERE gps_latitude IS NOT NULL
          AND gps_longitude IS NOT NULL
          AND location_name IS NULL
    """)
    photos_to_update = cursor.fetchall()
    total_photos: int = len(photos_to_update)
    print(f"Found {total_photos} photo records in database requiring location_name updates.")
    
    batch_updates: List[Tuple[str, int]] = []
    updated_count: int = 0
    query: str = "UPDATE photos SET location_name = %s WHERE id = %s" if backend == "postgresql" else "UPDATE photos SET location_name = ? WHERE id = ?"
    
    for row in photos_to_update:
        row_id: int = row[0]
        lat: float = float(row[1])
        lon: float = float(row[2])
        
        loc_str: Optional[str] = resolved_map.get((lat, lon))
        if loc_str:
            batch_updates.append((loc_str, row_id))
            
        # Write to database in small batches
        if len(batch_updates) >= BATCH_UPDATE_SIZE:
            if backend == "postgresql":
                execute_batch(cursor, query, batch_updates)
            else:
                cursor.executemany(query, batch_updates)
            conn.commit()
            updated_count += len(batch_updates)
            print(f"Committed: {updated_count}/{total_photos} location updates...")
            batch_updates.clear()
            
            # Defer and yield write locks to keep database responsive
            time.sleep(YIELD_SLEEP_SECONDS)
            
    # Commit final leftover batch
    if batch_updates:
        if backend == "postgresql":
            execute_batch(cursor, query, batch_updates)
        else:
            cursor.executemany(query, batch_updates)
        conn.commit()
        updated_count += len(batch_updates)
        print(f"Completed: {updated_count}/{total_photos} location updates committed.")


def main() -> None:
    """Main execution orchestrator."""
    parser = argparse.ArgumentParser(description="Offline Reverse Geocoding & Schema Migration")
    parser.add_argument("--db", type=str, default=None, help="Path to SQLite catalog database file")
    parser.add_argument("--root", type=str, default=None, help="Root folder containing the database file (resolves to root/photo_catalog.db)")
    parser.add_argument("--backend", type=str, default=None, choices=["sqlite", "postgresql"], help="Override active database backend")
    # Preprocess arguments to handle space after "--" (e.g. "-- root" -> "--root")
    cleaned_args: List[str] = []
    i = 0
    args_list = sys.argv[1:]
    while i < len(args_list):
        arg = args_list[i]
        if arg == "--" and i + 1 < len(args_list) and args_list[i + 1] in ("root", "db", "backend"):
            cleaned_args.append(f"--{args_list[i + 1]}")
            i += 2
        else:
            cleaned_args.append(arg)
            i += 1

    args = parser.parse_args(cleaned_args)

    print("==================================================")
    print("  Offline Reverse Geocoding & Schema Migration  ")
    print("==================================================")
    
    try:
        conn, backend = get_db_conn(db_backend=args.backend, db_path=args.db, root_dir=args.root)
        if backend == "sqlite":
            conn.execute("PRAGMA busy_timeout = 60000;")
        
        # 1. Run Schema Migration
        migrate_database_schema(conn, backend)
        
        # 2. Query coordinates needing resolution
        coords = get_unresolved_gps_coordinates(conn)
        if not coords:
            print("All photo coordinates are already geocoded. No work to do!")
            conn.close()
            return
            
        # 3. Perform offline geocoding
        resolved_map = perform_bulk_reverse_geocoding(coords)
        
        # 4. Perform database batch updates
        update_photos_in_batches(conn, backend, resolved_map)
        
        conn.close()
        print("Database offline geocoding process finished successfully!")
        
    except (sqlite3.Error, psycopg2.Error) as e:
        print(f"Database Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
