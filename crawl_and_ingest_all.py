"""ACDSee XMP Metadata Crawler and Ingester.

Purpose:
    This script crawls the photo directory, extracts embedded ACDSee/XMP metadata
    (ratings, labels, keywords, face regions, and GPS coordinates) in parallel
    threads using ExifTool, and ingests them into the SQLite database.

Architecture and Mechanics:
    1. Resumability: Inspects the database and loads already processed file paths.
       Files with a populated 'acdsee_metadata_imported_at' timestamp are skipped,
       preventing duplicate scans and redundant I/O operations.
    2. Batching (Argfiles): Processes only outstanding files by writing lists of
       paths to temporary UTF-8 argument files ('-@') for ExifTool, bypassing
       Windows command-line length limits.
    3. Multithreaded Producer-Consumer:
       - Producers (4-6 workers): Read file batches, run ExifTool in parallel,
         parse metadata, and put results on a thread-safe queue.
       - Consumer (1 worker): Pulls results from the queue and updates the SQLite
         database in sequential batches.
    4. Concurrency Guardrails: Enforce WAL mode and a 60-second database timeout.
       Includes a database write retry loop with exponential backoff to tolerate
       concurrent reads/writes from the VLM cataloger or REPL.

Execution Modes:
    - Command Line:
      python crawl_and_ingest_all.py --root <folder> [--limit-photos <N>] [--workers <W>]
"""

import os
import sys
import json
import time
import queue
import random
import logging
import sqlite3
import argparse
import tempfile
import datetime
import subprocess
import psycopg2
from psycopg2.extras import execute_batch
from sql_loader import get_sql
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple, Set, Optional

# Reconfigure console streams for UTF-8 on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

PROJECT_DIR: str = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH: str = os.path.join(PROJECT_DIR, "photo_catalog.db")
DEFAULT_EXIFTOOL_PATH: str = r"H:\Wan_project\exiftool\exiftool.exe"
LOG_FILE: str = os.path.join(PROJECT_DIR, "gemma_cataloger.log")

# Configure root logger to output to both console and gemma_cataloger.log
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

if root_logger.hasHandlers():
    root_logger.handlers.clear()

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)


def compute_rel_path(full_path: str) -> str:
    """Computes a normalized relative path matching the cataloger's indexing rules.

    Args:
        full_path: The absolute path of the image.

    Returns:
        The normalized relative path string in lowercase.
    """
    logger.debug(f"compute_rel_path: entering compute_rel_path for {full_path}")
    path_norm: str = full_path.replace("\\", "/").lower()
    if "vi ko\u0142odko/" in path_norm:
        rel_path = "vi ko\u0142odko/" + path_norm.split("vi ko\u0142odko/", 1)[1]
    elif "pictures/" in path_norm:
        rel_path = path_norm.split("pictures/", 1)[1]
    elif "patreon/" in path_norm:
        rel_path = path_norm.split("patreon/", 1)[1]
    elif "h:/" in path_norm:
        rel_path = path_norm.replace("h:/", "")
    else:
        rel_path = os.path.basename(path_norm)
    return rel_path.lower()





def migrate_schema(db_backend: str, db_conn_params: Optional[Dict[str, Any]] = None) -> None:
    """Ensures all metadata columns exist in the database, migrating if necessary.

    Args:
        db_backend: The active database backend ('sqlite' or 'postgresql') or path to SQLite file.
        db_conn_params: Connection parameters for the backend.
    """
    if db_conn_params is None:
        db_path = db_backend
        db_backend = "sqlite"
        db_conn_params = {"database": db_path}

    logger.info(f"migrate_schema: Checking database schema migration state for {db_backend}")
    if db_backend == "postgresql":
        conn = psycopg2.connect(**db_conn_params)
        conn.set_client_encoding('UTF8')
    else:
        conn = sqlite3.connect(db_conn_params["database"], timeout=60.0)
    try:
        cursor = conn.cursor()
        
        if db_backend == "sqlite":
            # Enable WAL mode for high-concurrency read-write transactions
            cursor.execute("PRAGMA journal_mode=WAL")
            # Ensure the photos table exists before checking columns
            cursor.execute(get_sql("queries/check_photos_table_exists.sql", db_backend))
            table_exists = bool(cursor.fetchone())
        else:
            # Check if table exists in PostgreSQL
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'photos'
                );
            """)
            table_exists = cursor.fetchone()[0]
            
        if not table_exists:
            logger.info("migrate_schema: Creating 'photos' table as it does not exist...")
            if db_backend == "sqlite":
                cursor.execute(get_sql("schema/create_photos_table.sql", db_backend))
                for stmt in get_sql("schema/create_indexes.sql", db_backend).split(";"):
                    if stmt.strip():
                        cursor.execute(stmt)
            else:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS photos (
                        id SERIAL PRIMARY KEY,
                        full_path TEXT UNIQUE NOT NULL,
                        rel_path TEXT NOT NULL,
                        primary_subject TEXT,
                        environment TEXT,
                        suggested_tags TEXT,
                        technical_details TEXT,
                        detected_objects TEXT
                    );
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_photos_rel_path ON photos (rel_path);
                """)
        
        if db_backend == "sqlite":
            cursor.execute("PRAGMA table_info(photos)")
            columns = {row[1] for row in cursor.fetchall()}
        else:
            cursor.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'photos';
            """)
            columns = {row[0] for row in cursor.fetchall()}

        new_cols: List[Tuple[str, str]] = [
            ("detected_faces", "TEXT DEFAULT '[]'"),
            ("acdsee_tags", "TEXT DEFAULT '[]'"),
            ("rating", "INTEGER"),
            ("label", "TEXT"),
            ("author", "TEXT"),
            ("gps_latitude", "REAL" if db_backend == "sqlite" else "double precision"),
            ("gps_longitude", "REAL" if db_backend == "sqlite" else "double precision"),
            ("gps_altitude", "REAL" if db_backend == "sqlite" else "double precision"),
            ("raw_metadata", "TEXT DEFAULT '{}'"),
            ("acdsee_metadata_imported_at", "TEXT"),
            ("file_mtime", "REAL" if db_backend == "sqlite" else "double precision")
        ]

        migrated = False
        for col_name, col_def in new_cols:
            if col_name not in columns:
                logger.info(f"migrate_schema: Applying schema migration: Adding '{col_name}' column...")
                cursor.execute(f"ALTER TABLE photos ADD COLUMN {col_name} {col_def}")
                migrated = True
        
        if migrated or not table_exists:
            conn.commit()
    finally:
        conn.close()


def get_already_ingested_paths(db_backend: str, db_conn_params: Optional[Dict[str, Any]] = None) -> Set[str]:
    """Retrieves paths of photos that have already been crawled and ingested.

    Args:
        db_backend: The active database backend ('sqlite' or 'postgresql') or path to SQLite file.
        db_conn_params: Connection parameters for the backend.

    Returns:
        A set of lowercase, slash-normalized file paths representing processed photos.
    """
    if db_conn_params is None:
        db_path = db_backend
        db_backend = "sqlite"
        db_conn_params = {"database": db_path}
    """Retrieves paths of photos that have already been crawled and ingested.

    Args:
        db_backend: The active database backend ('sqlite' or 'postgresql').
        db_conn_params: Connection parameters for the backend.

    Returns:
        A set of lowercase, slash-normalized file paths representing processed photos.
    """
    logger.info("get_already_ingested_paths: Querying already ingested metadata skip markers...")
    if db_backend == "postgresql":
        conn = psycopg2.connect(**db_conn_params)
        conn.set_client_encoding('UTF8')
    else:
        conn = sqlite3.connect(db_conn_params["database"], timeout=60.0)
    try:
        cursor = conn.cursor()
        cursor.execute(get_sql("queries/get_already_ingested.sql", db_backend))
        # Normalize slashes and casing for exact in-memory comparison
        return {row[0].replace("\\", "/").lower() for row in cursor.fetchall()}
    except (sqlite3.OperationalError, psycopg2.ProgrammingError, psycopg2.OperationalError):
        # Table might not exist or be unmigrated yet, which migrate_schema will address
        return set()
    finally:
        conn.close()


def get_existing_paths_map(db_backend: str, db_conn_params: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Builds a mapping of slash-normalized lowercase paths to original DB paths.

    Args:
        db_backend: The active database backend ('sqlite' or 'postgresql') or path to SQLite file.
        db_conn_params: Connection parameters for the backend.

    Returns:
        A dictionary mapping 'normalized_path' to the exact 'full_path' string in the DB.
    """
    if db_conn_params is None:
        db_path = db_backend
        db_backend = "sqlite"
        db_conn_params = {"database": db_path}
    """Builds a mapping of slash-normalized lowercase paths to original DB paths.

    Args:
        db_backend: The active database backend ('sqlite' or 'postgresql').
        db_conn_params: Connection parameters for the backend.

    Returns:
        A dictionary mapping 'normalized_path' to the exact 'full_path' string in the DB.
    """
    logger.info("get_existing_paths_map: Fetching existing path mapping to differentiate insert vs update...")
    if db_backend == "postgresql":
        conn = psycopg2.connect(**db_conn_params)
        conn.set_client_encoding('UTF8')
    else:
        conn = sqlite3.connect(db_conn_params["database"], timeout=60.0)
    try:
        cursor = conn.cursor()
        cursor.execute(get_sql("queries/get_existing_paths.sql", db_backend))
        return {row[0].replace("\\", "/").lower(): row[0] for row in cursor.fetchall()}
    except (sqlite3.OperationalError, psycopg2.ProgrammingError, psycopg2.OperationalError):
        return {}
    finally:
        conn.close()


def preload_batch(batch_paths: List[str]) -> None:
    """Preload only the metadata headers of a batch of image files into the OS cache.

    Reads the first 256 KB of each file sequentially to warm the Windows page
    cache, preventing ExifTool from hitting the physical disk for metadata
    headers while avoiding the I/O bottleneck of reading redundant raw pixel bytes.

    Args:
        batch_paths: List of absolute file paths to preload.
    """
    header_read_size: int = 256 * 1024  # 256 KB
    for path in batch_paths:
        try:
            with open(path, "rb") as f:
                _ = f.read(header_read_size)
        except OSError:
            pass


def exiftool_worker(
    exiftool_path: str, 
    batch_paths: List[str], 
    result_queue: queue.Queue
) -> None:
    """Subprocess thread worker to extract metadata from a batch of images.

    Args:
        exiftool_path: Path to the ExifTool executable.
        batch_paths: List of absolute file paths to scan.
        result_queue: The queue to receive output metadata dictionaries.
    """
    logger.info(f"exiftool_worker: Parsing metadata batch of size {len(batch_paths)} using ExifTool...")
    if not batch_paths:
        return

    # Write target file paths to a temporary UTF-8 argfile to prevent CLI character mapping errors
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as f:
        for path in batch_paths:
            f.write(f"{path}\n")
        argfile_path = f.name

    try:
        cmd: List[str] = [
            exiftool_path,
            "-json",
            "-n",  # Output numerical coordinates and values
            "-charset", "filename=utf8",
            "-ACDSeeRegionName",
            "-RegionPersonDisplayName",
            "-Keywords",
            "-Subject",
            "-HierarchicalSubject",
            "-Categories",
            "-Rating",
            "-Label",
            "-Creator",
            "-Artist",
            "-By-line",
            "-GPSLatitude",
            "-GPSLongitude",
            "-GPSAltitude",
            "-ext", "jpg", "-ext", "jpeg", "-ext", "png", "-ext", "webp", "-ext", "heic", "-ext", "tif", "-ext", "tiff", "-ext", "bmp", "-ext", "xmp",
            "-@", argfile_path
        ]
        
        # Execute ExifTool with a 300-second timeout to prevent hangs
        # We use check=False because minor warnings on files make ExifTool return exit code 1
        # while still successfully outputting valid JSON metadata on stdout.
        res = subprocess.run(cmd, capture_output=True, timeout=300.0, check=False)
        if res.stdout:
            extracted_data = json.loads(res.stdout)
            if isinstance(extracted_data, list):
                result_queue.put(extracted_data)
        if res.returncode != 0 and not res.stdout:
            # Only print warning if no data was returned at all
            logger.warning(f"ExifTool returned non-zero exit code {res.returncode}. Stderr: {res.stderr.decode('utf-8', errors='replace')}")
    except Exception as e:
        logger.error(f"ExifTool worker failed for batch of size {len(batch_paths)}: {e}")
    finally:
        try:
            os.remove(argfile_path)
        except OSError:
            pass


def db_writer_worker(
    db_backend: str, 
    db_conn_params: Any, 
    result_queue: Optional[queue.Queue] = None, 
    total_expected: int = 0,
    existing_paths_map: Optional[Dict[str, str]] = None
) -> None:
    """Consumer thread worker that batch updates metadata in the SQLite or PostgreSQL database.

    Opens a single persistent database connection at startup and reuses it
    across all batch writes, eliminating the TCP handshake and authentication
    overhead that occurs when a new psycopg2 connection is created per batch.
    On write failure, performs a rollback and attempts to reconnect before
    retrying with exponential backoff.

    Args:
        db_backend: The active database backend ('sqlite' or 'postgresql') or path to SQLite file.
        db_conn_params: Connection parameters for the backend, or result_queue (if old signature).
        result_queue: Queue of parsed ExifTool metadata items.
        total_expected: Total expected number of photos to ingest for progress reporting.
        existing_paths_map: Mapping of normalized paths to database paths.
    """
    # Check for backward-compatible call signature:
    # db_writer_worker(db_path, result_queue, total_expected, existing_paths_map)
    if isinstance(db_conn_params, queue.Queue):
        db_path = db_backend
        actual_queue = db_conn_params
        actual_total = result_queue  # was 3rd arg (total_expected)
        actual_paths_map = total_expected  # was 4th arg (existing_paths_map)
        
        db_backend = "sqlite"
        db_conn_params = {"database": db_path}
        result_queue = actual_queue
        total_expected = actual_total
        existing_paths_map = actual_paths_map

    if existing_paths_map is None:
        existing_paths_map = {}

    logger.info(f"db_writer_worker: Starting DB writer thread. Target: {db_backend}, total expected: {total_expected}")

    def _open_conn() -> Any:
        """Opens and configures a fresh database connection for this worker."""
        if db_backend == "postgresql":
            c = psycopg2.connect(**db_conn_params)
            c.set_client_encoding('UTF8')
            return c
        else:
            c = sqlite3.connect(db_conn_params["database"], timeout=30.0)
            c.execute("PRAGMA journal_mode=WAL;")
            c.execute("PRAGMA synchronous=NORMAL;")
            return c

    total_processed: int = 0
    total_updates: int = 0
    total_inserts: int = 0
    batches_completed: int = 0
    t_start = time.time()

    # Open one persistent connection for the lifetime of this worker thread.
    # Reusing the connection eliminates per-batch TCP handshake and PostgreSQL
    # authentication overhead — the primary write-speed bottleneck when connecting
    # fresh on every 500-file batch commit.
    conn = _open_conn()

    try:
        while True:
            try:
                # Block waiting for items from producers
                batch_data = result_queue.get(timeout=10.0)
                if batch_data is None:  # Termination signal received
                    break

                update_records: List[Tuple[Any, ...]] = []
                insert_records: List[Tuple[Any, ...]] = []
                imported_at: str = datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"

                for item in batch_data:
                    full_path = item.get("SourceFile")
                    if not full_path: continue
                    
                    # Extract and merge metadata fields per reference specifications
                    
                    # 1. Faces Extraction (Merge ACDSeeRegionName and RegionPersonDisplayName)
                    faces: List[str] = []
                    for face_key in ["ACDSeeRegionName", "RegionPersonDisplayName"]:
                        val = item.get(face_key)
                        if val:
                            if isinstance(val, list):
                                faces.extend(str(x) for x in val)
                            else:
                                faces.append(str(val))
                    detected_faces_json = json.dumps(list(dict.fromkeys(faces)))

                    # 2. Tags/Keywords Extraction (Merge Keywords, Subject, HierarchicalSubject, and Categories)
                    tags: List[str] = []
                    for tag_key in ["Keywords", "Subject", "HierarchicalSubject", "Categories"]:
                        val = item.get(tag_key)
                        if val:
                            if isinstance(val, list):
                                tags.extend(str(x) for x in val)
                            else:
                                tags.append(str(val))
                    # Clean/exclude description fields to protect VLM description data
                    tags = [t for t in tags if t.lower() not in ("description", "caption-abstract", "imagedescription")]
                    acdsee_tags_json = json.dumps(list(dict.fromkeys(tags)))

                    rating = item.get("Rating")
                    label = item.get("Label")
                    author = item.get("Creator") or item.get("Artist") or item.get("By-line")
                    gps_latitude = item.get("GPSLatitude")
                    gps_longitude = item.get("GPSLongitude")
                    gps_altitude = item.get("GPSAltitude")
                    raw_metadata_json = json.dumps(item)
                    rel_path = compute_rel_path(full_path)
                    
                    # Fetch modification time safely
                    try:
                        file_mtime = os.path.getmtime(full_path)
                    except OSError:
                        file_mtime = 0.0

                    # Normalized matching to resolve slash issues
                    full_path_norm = full_path.replace("\\", "/").lower()
                    if full_path_norm in existing_paths_map:
                        db_full_path = existing_paths_map[full_path_norm]
                        update_records.append((
                            detected_faces_json, acdsee_tags_json, rating, label, author,
                            gps_latitude, gps_longitude, gps_altitude, raw_metadata_json, 
                            imported_at, file_mtime, db_full_path
                        ))
                    else:
                        insert_records.append((
                            full_path, rel_path, detected_faces_json, acdsee_tags_json, rating, label, author,
                            gps_latitude, gps_longitude, gps_altitude, raw_metadata_json, imported_at, file_mtime
                        ))

                # Database write retry loop — reuses the persistent connection.
                # Only reconnects on transient failures to avoid connection churn.
                write_success = False
                for attempt in range(10):
                    try:
                        cursor = conn.cursor()
                        if db_backend == "postgresql":
                            if update_records:
                                execute_batch(cursor, get_sql("queries/update_photo.sql", db_backend), update_records)
                            if insert_records:
                                execute_batch(cursor, get_sql("queries/insert_photo.sql", db_backend), insert_records)
                        else:
                            if update_records:
                                cursor.executemany(get_sql("queries/update_photo.sql", db_backend), update_records)
                            if insert_records:
                                cursor.executemany(get_sql("queries/insert_photo.sql", db_backend), insert_records)

                        conn.commit()
                        write_success = True
                        total_updates += len(update_records)
                        total_inserts += len(insert_records)
                        break
                    except (sqlite3.OperationalError, psycopg2.OperationalError, psycopg2.DatabaseError) as e:
                        is_locked = "locked" in str(e).lower() or "lock" in str(e).lower()
                        # Always rollback the failed transaction before retrying
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                        if not is_locked or attempt >= 9:
                            logger.error(f"Database write failed and exceeded retries: {e}")
                            break
                        # Reconnect the persistent connection on transient lock/connection errors
                        try:
                            conn.close()
                        except Exception:
                            pass
                        try:
                            conn = _open_conn()
                            logger.info(f"db_writer_worker: Reconnected to {db_backend} after transient error (attempt {attempt + 1}).")
                        except Exception as reconnect_err:
                            logger.error(f"db_writer_worker: Reconnection failed: {reconnect_err}")
                            break
                        time.sleep(random.uniform(2.0, 5.0))

                if write_success:
                    total_processed += len(batch_data)
                    batches_completed += 1
                    percent = (total_processed / total_expected) * 100 if total_expected > 0 else 100
                    
                    # Calculate elapsed time and rec/s speed
                    elapsed = time.time() - t_start
                    speed = total_processed / elapsed if elapsed > 0 else 0
                    
                    # Estimate remaining time (ETA)
                    remaining_files = total_expected - total_processed
                    eta_sec = remaining_files / speed if speed > 0 else 0
                    eta_str = str(datetime.timedelta(seconds=int(eta_sec))) if eta_sec > 0 else "0:00:00"
                    
                    logger.info(
                        f"Progress: {total_processed}/{total_expected} files ({percent:.1f}%) | "
                        f"Batch {batches_completed} | "
                        f"Speed: {speed:.1f} rec/s | "
                        f"ETA: {eta_str} | "
                        f"Updates: {total_updates} | Inserts: {total_inserts}"
                    )

                result_queue.task_done()
            except queue.Empty:
                # Occurs when queue is empty, loop and continue waiting
                continue
            except Exception as e:
                logger.error(f"DB Writer encountered an error: {e}")
                break
    finally:
        # Guarantee the persistent connection is always closed cleanly on exit,
        # even if the worker loop exits due to an unhandled exception.
        try:
            conn.close()
        except Exception:
            pass

    logger.info(f"Database writing finished! Total Ingested: {total_processed} (Updated: {total_updates}, Inserted: {total_inserts}).")


def main() -> None:
    """Core CLI execution wrapper."""
    parser = argparse.ArgumentParser(description="High-Performance ACDSee Metadata Crawler")
    parser.add_argument("--root", type=str, required=True, help="Root folder to scan recursively (e.g. D:\\Users\\steven\\Pictures)")
    parser.add_argument("--limit-photos", "--limit", dest="limit_photos", type=int, default=0, help="Maximum number of photos to ingest in this execution (0 for unlimited)")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent ExifTool parser worker threads (4-6 recommended for HDD)")
    parser.add_argument("--batch-size", "--batch", dest="batch_size", type=int, default=50, help="Number of files to pack in each ExifTool subprocess execution (default 50)")
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH, help="Path to SQLite catalog database")
    parser.add_argument("--exiftool", type=str, default=DEFAULT_EXIFTOOL_PATH, help="Path to ExifTool executable")
    parser.add_argument("--force", action="store_true", help="Force re-crawling and overwriting of already ingested metadata")
    # Preprocess arguments to gracefully filter out the word 'size' from '--batch size <N>' typos
    cleaned_args: List[str] = []
    skip_next: bool = False
    for i, arg in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        if arg in ("--batch", "--batch-size") and i + 1 < len(sys.argv[1:]) and sys.argv[1:][i+1].lower() == "size":
            cleaned_args.append(arg)
            skip_next = True
        else:
            cleaned_args.append(arg)

    args = parser.parse_args(cleaned_args)

    # Pre-execution environment checks
    if not os.path.exists(args.root):
        logger.critical(f"Target root folder does not exist: {args.root}")
        sys.exit(1)
    if not os.path.exists(args.exiftool):
        logger.critical(f"ExifTool executable not found at: {args.exiftool}")
        sys.exit(1)

    logger.info("==================================================")
    logger.info("  ACDSee Metadata High-Performance Crawler  ")
    logger.info("==================================================")
    logger.info(f"Scanning Directory:  {args.root}")
    logger.info(f"Workers Configured:  {args.workers} threads")
    logger.info(f"Limit Configured:    {args.limit_photos if args.limit_photos > 0 else 'Unlimited'} photos")

    # Detect active database backend from .env
    is_testing = "pytest" in sys.argv[0] or "unittest" in sys.argv[0] or "PYTEST_CURRENT_TEST" in os.environ
    db_backend = "sqlite" if is_testing else os.getenv("DB_BACKEND", "postgresql").lower()
    
    db_conn_params = {}
    if db_backend == "postgresql":
        db_conn_params = {
            "dbname": os.getenv("DB_NAME", "photo_catalog"),
            "user": os.getenv("DB_USER", "postgres"),
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
        }
        # Load database password from auth/db_password.txt
        pwd_path = os.path.join(PROJECT_DIR, "auth", "db_password.txt")
        if os.path.exists(pwd_path):
            with open(pwd_path, "r", encoding="utf-8") as f:
                db_conn_params["password"] = f.read().strip()
    else:
        db_conn_params = {"database": args.db}

    # 1. Apply schema migration changes
    migrate_schema(db_backend, db_conn_params)

    # 2. Query processed metadata markers to skip duplicates
    logger.info("Loading database cache for skip markers...")
    already_ingested: Set[str] = set() if args.force else get_already_ingested_paths(db_backend, db_conn_params)
    existing_paths_map: Dict[str, str] = get_existing_paths_map(db_backend, db_conn_params)
    logger.info(f"Loaded {len(already_ingested)} processed path records.")

    # 3. Scan directory files recursively
    logger.info("Traversing files on drive (Sequential directory list)...")
    files_to_scan: List[str] = []
    
    t_start = time.time()
    total_seen = 0
    
    for dirpath, _, filenames in os.walk(args.root):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in (".jpg", ".jpeg", ".png", ".webp", ".heic", ".tif", ".tiff", ".bmp"):
                total_seen += 1
                full_path = os.path.join(dirpath, filename)
                # Normalize slashes and casing for duplicate checking
                norm_path = full_path.replace("\\", "/").lower()
                
                if norm_path not in already_ingested:
                    files_to_scan.append(full_path)
                    
                    # Stop traversal immediately if we hit the user-configured processing limit
                    if args.limit_photos > 0 and len(files_to_scan) >= args.limit_photos:
                        break
        if args.limit_photos > 0 and len(files_to_scan) >= args.limit_photos:
            break

    logger.info(f"Drive traversal finished. Found {len(files_to_scan)} outstanding photos requiring metadata ingestion (Skipped {total_seen - len(files_to_scan)} already processed).")
    
    if not files_to_scan:
        logger.info("All files in the scanned directories are already up-to-date!")
        return

    # 4. Partition files into batch sizes
    batches: List[List[str]] = [
        files_to_scan[i : i + args.batch_size] 
        for i in range(0, len(files_to_scan), args.batch_size)
    ]
    logger.info(f"Partitioned files into {len(batches)} batches of size {args.batch_size}.")

    # 5. Initialize Queue and thread tasks
    result_queue: queue.Queue = queue.Queue(maxsize=100)
    
    # Spawn consumer thread first
    import threading
    db_writer = threading.Thread(
        target=db_writer_worker,
        args=(db_backend, db_conn_params, result_queue, len(files_to_scan), existing_paths_map),
        daemon=True
    )
    db_writer.start()

    logger.info("Starting extraction and ingestion pipeline...")
    # Spawn producer thread pool executor manually to allow clean immediate interrupt
    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = []
    try:
        for batch in batches:
            # Warm OS Page Cache sequentially on the main thread before submitting to workers
            preload_batch(batch)
            futures.append(executor.submit(exiftool_worker, args.exiftool, batch, result_queue))
        
        # Wait for all producer tasks to complete while remaining responsive to KeyboardInterrupt (Ctrl+C)
        while any(not f.done() for f in futures):
            time.sleep(0.1)
            
        executor.shutdown(wait=True)
    except KeyboardInterrupt:
        logger.warning("Shutdown requested. Cancelling pending tasks...")
        for f in futures:
            f.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        os._exit(1)
    
    # Put termination marker on the queue and join DB writer
    result_queue.put(None)
    
    # Safely join the database writer while remaining responsive to Ctrl+C
    while db_writer.is_alive():
        db_writer.join(timeout=0.1)

    duration = time.time() - t_start
    logger.info(f"Completed in {duration:.2f} seconds ({len(files_to_scan)/duration:.2f} files/sec).")


if __name__ == "__main__":
    main()
