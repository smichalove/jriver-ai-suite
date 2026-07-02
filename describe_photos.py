"""Local Gemma 4 VLM Photo Cataloger - Architectural & Operational Reference.

This module provides the core orchestrator pipeline for local, offline photo description
and EXIF metadata archiving. It leverages Google's Gemma 4 12B IT vision-language model,
quantized to 4-bit, to perform compliant, objective indexing of photo libraries.

Architecture Overview:
----------------------
1. Directory Crawler (os.walk):
   Scans target photo libraries recursively, filtering files by extension (.jpg, .png, etc.).
   Maintains O(1) deduplication lookup by loading existing entries from photo_descriptions.json
   and comparing both absolute and relative normalized paths. Invalid/failed runs (e.g., safety
   violations, formatting errors) are filtered out of the skip-list to allow automatic retries.

2. Quantized Model Configuration (BitsAndBytes):
   Loads google/gemma-4-12B-it using BitsAndBytes NF4 4-bit quantization, allowing the 12B weights
   to run comfortably under your GPU's 16GB VRAM (~13.4GB VRAM for batch_size=2, ~15.0GB VRAM for
   batch_size=4). Prevents model blindness by skipping quantization on key vision, audio, and head modules.
   Monkey-patches the LayerNorm uint8 casting bug inside transformers.models.gemma4_unified.

3. Assistant Prefix Prefilling (Steering):
   Forces compliant JSON formatting by prefilling the assistant prompt with '{\n  "primary_subject": "'.
   This steering bypasses the model's inner thoughts/preamble and guarantees compliant JSON output.

4. Asynchronous I/O Pipeline:
   Loads images in parallel CPU threads (ThreadPoolExecutor) while simultaneously executing batch
   GPU inference on the main thread, maximizing throughput (speeding up processing by 61%).

5. Windows-safe ExifTool Writer (Argfile):
   Bypasses Windows command-line Unicode character limitations (e.g., directory names with Polish characters
   like 'Vi Kołodko') by writing the target paths to a temporary UTF-8 text file and calling ExifTool
   using the `-@` and `-charset filename=utf8` flags.

6. Logging and Console Output:
   Logs progress both to stdout (console) and a persistent local project log file (gemma_cataloger.log).
   All warning/success messages are kept plain-text (avoiding non-ASCII emojis) to prevent crash issues on
   default Windows CP1252 terminal encoders.
"""

import os
import sys
import sqlite3
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

# Load workspace environment variables
PROJECT_DIR: str = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(PROJECT_DIR, "auth", ".env")):
    load_dotenv(os.path.join(PROJECT_DIR, "auth", ".env"))
elif os.path.exists("auth/.env"):
    load_dotenv("auth/.env")
else:
    load_dotenv()

# Reconfigure standard output streams to use UTF-8 on Windows to prevent encoding crashes (CP1252)
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

__version__: str = "0.0.1-dev"


import json
import time
import logging
import mimetypes
import concurrent.futures
import threading
import subprocess
import argparse
import uuid
import base64
import io
from typing import List, Dict, Set, Optional, Tuple, Any

from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None  # Allow massively high-res images to be processed
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow reading truncated or slightly corrupted files
import pillow_heif
pillow_heif.register_heif_opener()

import wsl_client

def get_db_backend(db_path: str = "") -> str:
    """Detects and returns the active database backend ('sqlite' or 'postgresql').
    
    Automatically falls back to 'sqlite' if running inside a unit test environment.
    """
    is_testing = "pytest" in sys.argv[0] or "unittest" in sys.argv[0] or "PYTEST_CURRENT_TEST" in os.environ
    if is_testing:
        return "sqlite"
    return os.getenv("DB_BACKEND", "postgresql").lower()

def get_db_conn(db_path: str) -> Tuple[Any, str]:
    """Returns a tuple of (connection, backend_type) depending on DB_BACKEND configuration.
    
    If DB_BACKEND is 'postgresql', connects to PostgreSQL and returns (conn, 'postgresql').
    Otherwise, connects to SQLite and returns (conn, 'sqlite').
    """
    backend = get_db_backend(db_path)
    if backend == "postgresql":
        db_conn_params = {
            "dbname": os.getenv("DB_NAME", "photo_catalog"),
            "user": os.getenv("DB_USER", "postgres"),
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
        }
        pwd_path = os.path.join(PROJECT_DIR, "auth", "db_password.txt")
        if os.path.exists(pwd_path):
            with open(pwd_path, "r", encoding="utf-8") as f:
                db_conn_params["password"] = f.read().strip()
        conn = psycopg2.connect(**db_conn_params)
        return conn, "postgresql"
    else:
        conn = sqlite3.connect(db_path, timeout=60.0)
        return conn, "sqlite"

def migrate_photos_schema(conn: Any, backend: str) -> None:
    """Ensures all metadata columns exist in the database, migrating if necessary.

    Args:
        conn: Connection object.
        backend: The active database backend ('sqlite' or 'postgresql').
    """
    cursor = conn.cursor()
    try:
        if backend == "sqlite":
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
            ("gps_latitude", "REAL" if backend == "sqlite" else "double precision"),
            ("gps_longitude", "REAL" if backend == "sqlite" else "double precision"),
            ("gps_altitude", "REAL" if backend == "sqlite" else "double precision"),
            ("raw_metadata", "TEXT DEFAULT '{}'"),
            ("acdsee_metadata_imported_at", "TEXT"),
            ("file_mtime", "REAL" if backend == "sqlite" else "double precision")
        ]

        for col_name, col_def in new_cols:
            if col_name not in columns:
                logger.info(f"migrate_photos_schema: Applying schema migration: Adding '{col_name}' column to {backend}...")
                cursor.execute(f"ALTER TABLE photos ADD COLUMN {col_name} {col_def}")
        conn.commit()
    except Exception as e:
        logger.warning(f"Failed to migrate database schema: {e}")
        conn.rollback()


# Setup logging to both console and a file in the project directory (gemma_cataloger)
LOG_FILE: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemma_cataloger.log")

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Create formatter
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Clear existing handlers if any to avoid duplicate logs
if root_logger.hasHandlers():
    root_logger.handlers.clear()

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

# File handler in the gemma_cataloger directory
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

# --- Configuration defaults ---
MODEL_ID: str = "google/gemma-4-12B-it"
PROJECT_DIR: str = r"H:\Wan_project"
PICTURE_DIRS: List[str] = [
    r"D:\Users\steven\Pictures",
    r"D:\Users\steven\Patreon\gallery-dl\patreon\Vi Kołodko"
]
OUTPUT_JSON: str = os.path.join(PROJECT_DIR, "photo_descriptions.json")
DB_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photo_catalog.db")
SUBMITTED_CACHE: str = os.path.join(PROJECT_DIR, "submitted_photos_cache.txt")
MAX_IMAGE_DIM: int = 1024

# Threading lock for saving the JSON file safely
json_lock = threading.Lock()
# Threading lock for saving the SQLite database safely
sqlite_lock = threading.Lock()


def compute_rel_path(full_path: str) -> str:
    """Computes a normalized relative path matching the cataloger's indexing rules.

    Args:
        full_path: The absolute path of the image.

    Returns:
        The normalized relative path string in lowercase.
    """
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


def is_model_refusal(subject: str) -> bool:
    """Checks if the VLM response represents a refusal or processing error rather than a valid description.

    Args:
        subject: The primary subject text returned by the model.

    Returns:
        True if the text is a refusal (starts with 'sorry', 'i cannot', 'error', etc.), False otherwise.
    """
    subject_lower = subject.lower().strip()
    return (
        not subject_lower or
        subject_lower.startswith("sorry") or
        subject_lower.startswith("i'm sorry") or
        subject_lower.startswith("i am sorry") or
        subject_lower.startswith("i cannot") or
        subject_lower.startswith("i can't") or
        "safety violation" in subject_lower or
        "please provide" in subject_lower or
        subject_lower.startswith("error")
    )


def has_embedded_metadata(file_path: str) -> bool:
    """Checks if the file already contains description metadata using ExifTool.

    Args:
        file_path: The absolute path to the image file.

    Returns:
        True if the file has a Description, Caption-Abstract, or ImageDescription, False otherwise.
    """
    try:
        cmd = [
            r'H:\Wan_project\exiftool\exiftool.exe',
            '-s', '-S',
            '-Description',
            '-ImageDescription',
            '-Caption-Abstract',
            file_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=5.0)
        out = res.stdout.decode('utf-8', errors='replace').strip()
        return len(out) > 0
    except Exception:
        return False


def get_image_files(directories: List[str], limit: Optional[int] = 20, processed_paths: Optional[Set[str]] = None, skip_existing_exif: bool = False) -> List[str]:
    """
    Walks the directories recursively to find all image files, skipping those already processed.

    Args:
        directories: List of directories to search.
        limit: Max number of images to return.
        processed_paths: Set of already processed file paths (lowercase).

    Returns:
        List of absolute paths to unprocessed image files.
    """
    if processed_paths is None:
        processed_paths = set()
    
    image_extensions: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff", ".bmp"}
    skipped_count: int = 0
    raw_images: List[str] = []
    
    for directory in directories:
        if not os.path.exists(directory):
            continue
        logger.info(f"Walking directory: {directory}")
        for root, dirs, files in os.walk(directory):
            if "venv" in root or ".git" in root or "$RECYCLE.BIN" in root or "System Volume Information" in root:
                continue
            for file in files:
                if file.startswith("."):
                    continue
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    raw_images.append(os.path.join(root, file))

    # First pass: filter by processed_paths (which is instant)
    candidates: List[str] = []
    for full_path in raw_images:
        rel_path = compute_rel_path(full_path)
        if rel_path in processed_paths or full_path.lower() in processed_paths:
            skipped_count += 1
            continue
        candidates.append(full_path)

    # Second pass: check embedded metadata in parallel if requested
    if skip_existing_exif and candidates:
        logger.info(f"Verifying metadata for {len(candidates)} candidate images in parallel (16 threads)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            meta_results = list(executor.map(has_embedded_metadata, candidates))
        
        filtered_candidates = []
        for path, has_meta in zip(candidates, meta_results):
            if has_meta:
                skipped_count += 1
            else:
                filtered_candidates.append(path)
        candidates = filtered_candidates

    images = candidates
    
    # Sort all found image paths alphabetically to ensure clean chronological progression globally
    images.sort()
    
    # Apply the photo limit after scanning the entire directory tree
    if limit:
        images = images[:limit]
    
    logger.info(f"Skipped {skipped_count} already processed images.")
    return images


def load_and_encode_image(img_path: str) -> Optional[str]:
    """Loads an image from disk, converts to RGB, compresses to JPEG (quality=90),
    and returns the Base64-encoded string representation.

    Args:
        img_path: Absolute path to the image file.

    Returns:
        The Base64-encoded string representation if successful, None otherwise.
    """
    img: Optional[Image.Image] = None
    buffered: Optional[io.BytesIO] = None
    try:
        if os.path.exists(img_path) and os.path.getsize(img_path) == 0:
            logger.warning(f"Skipping empty 0-byte image file: {img_path}")
            return None
        img = Image.open(img_path).convert("RGB")
        img.load()

        # Prevent CUDA Out-of-Memory crashes on extremely high-res images
        # by dynamically downscaling them to fit within a safe bounding box in-memory.
        # The original image file on disk remains completely unmodified.
        if max(img.size) > MAX_IMAGE_DIM:
            logger.info(f"Scaling down in-memory representation of {os.path.basename(img_path)} from {img.size} to max {MAX_IMAGE_DIM}px for VRAM safety.")
            img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.Resampling.LANCZOS)

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    except Exception as e:
        logger.warning(f"Skipping unreadable image {img_path}: {e}")
        return None
    finally:
        if img is not None:
            try:
                img.close()
            except Exception:
                pass
        if buffered is not None:
            try:
                buffered.close()
            except Exception:
                pass

def verify_embedded_metadata(file_path: str, expected_text: str) -> bool:
    """Reads back embedded metadata tags from the image file and verifies they match the expected text.

    Args:
        file_path: The absolute path to the image file.
        expected_text: The expected text summary that was written.

    Returns:
        True if all tags are successfully verified and match, False otherwise.

    Raises:
        None
    """
    # Create a unique temporary filename in the project directory for the argfile
    # This handles unicode file paths on Windows safely
    arg_file_path: str = os.path.join(PROJECT_DIR, f"exif_read_args_{uuid.uuid4().hex}.txt")
    try:
        with open(arg_file_path, "w", encoding="utf-8") as arg_f:
            arg_f.write(file_path + "\n")
            
        cmd = [
            r'H:\Wan_project\exiftool\exiftool.exe',
            '-j',
            '-charset', 'UTF8',
            '-charset', 'filename=utf8',
            '-Description',
            '-@', arg_file_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=5.0)
        out = res.stdout.decode('utf-8', errors='replace').strip()
        
        data = json.loads(out)
        if not data or not isinstance(data, list):
            logger.warning(f"No metadata returned by ExifTool for verification of {os.path.basename(file_path)}")
            return False
            
        file_data = data[0]
        if "Description" in file_data:
            found_val = file_data["Description"]
            # Normalize newlines and whitespace to avoid platform-specific discrepancies (\r\n vs \n)
            normalized_found = found_val.replace("\r\n", "\n").strip()
            normalized_expected = expected_text.replace("\r\n", "\n").strip()
            if len(normalized_found) != len(normalized_expected) or normalized_found != normalized_expected:
                logger.warning(
                    f"Metadata verification mismatch for {os.path.basename(file_path)} in tag 'Description'.\n"
                    f"Expected (len={len(normalized_expected)}):\n{normalized_expected}\n"
                    f"Found (len={len(normalized_found)}):\n{normalized_found}"
                )
                return False
            return True
        else:
            logger.warning(f"Expected metadata tag 'Description' was not found in {os.path.basename(file_path)} during read-back verification.")
            return False
    except Exception as e:
        logger.error(f"Failed to read back metadata for verification from {file_path}: {e}")
        return False
    finally:
        if os.path.exists(arg_file_path):
            try:
                os.remove(arg_file_path)
            except Exception:
                pass


def inline_embed_metadata(file_path: str, summary_text: str) -> None:
    """Runs exiftool to natively embed the summary description into the file.

    It creates a temporary UTF-8 arguments file on Windows to safely pass the image file
    path to ExifTool, bypassing Windows command-line Unicode character mapping limitations.
    After writing, it reads back the metadata to verify it matches the expected text,
    retrying on failure.

    Args:
        file_path: The absolute path to the image file.
        summary_text: The concatenated descriptive string to embed.

    Returns:
        None

    Raises:
        None (All errors, such as subprocess or file access errors, are handled internally).
    """
    # Create a unique temporary filename in the project directory for the argfile
    # This prevents collisions between concurrent threads and keeps the path local
    arg_file_path: str = os.path.join(PROJECT_DIR, f"exif_args_{uuid.uuid4().hex}.txt")
    
    try:
        # Escape newlines as HTML entities in the argfile to support multi-line metadata on Windows
        summary_escaped = summary_text.replace("\r\n", "&#10;").replace("\n", "&#10;").replace("\r", "&#10;")
        
        with open(arg_file_path, "w", encoding="utf-8") as arg_f:
            arg_f.write("-CodedCharacterSet=UTF8\n")
            arg_f.write(f"-Caption-Abstract={summary_escaped}\n")
            arg_f.write(f"-Description={summary_escaped}\n")
            arg_f.write(f"-ImageDescription={summary_escaped}\n")
            arg_f.write(file_path + "\n")
            # If a sidecar .xmp file exists, or the file is a WebP (which requires a sidecar for ACDSee), write the sidecar path too
            sidecar_path = file_path + ".xmp"
            if os.path.exists(sidecar_path) or file_path.lower().endswith(".webp"):
                arg_f.write(sidecar_path + "\n")
            
        cmd: List[str] = [
            r'H:\Wan_project\exiftool\exiftool.exe',
            '-m', 
            '-E',                        # Decode HTML character entities (like &#10;) in values
            '-charset', 'iptc=UTF8',
            '-charset', 'UTF8',      
            '-charset', 'filename=utf8', # Instruct Exiftool to decode the argfile names as UTF-8
            '-overwrite_original',
            '-@', arg_file_path          # Load the filename from the temporary argfile
        ]
        
        # Try up to 2 attempts to write and verify
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                # Execute ExifTool in a separate subprocess with a 15.0 second timeout to prevent infinite hangs
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15.0)
                
                # Read back and verify
                if verify_embedded_metadata(file_path, summary_text):
                    logger.info(f"EXIF embedded and verified successfully: {file_path}")
                    return
                else:
                    logger.warning(f"EXIF verification mismatch on attempt {attempt}/{max_attempts} for: {file_path}")
                    if attempt < max_attempts:
                        time.sleep(1.0)
            except subprocess.SubprocessError as e:
                # Safely decode stderr if it is a CalledProcessError with stderr output, otherwise convert exception to string
                err_msg: str = ""
                if isinstance(e, subprocess.CalledProcessError) and e.stderr:
                    err_msg = e.stderr.decode('utf-8', errors='replace').strip()
                else:
                    err_msg = str(e)
                
                # Recover from temporary lock files or photoshop IRB issues if present
                if "Bad Photoshop IRB resource" in err_msg:
                    try:
                        fallback_cmd: List[str] = cmd.copy()
                        fallback_cmd.insert(4, "-Photoshop:All=")
                        subprocess.run(fallback_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15.0)
                        if verify_embedded_metadata(file_path, summary_text):
                            logger.info(f"EXIF embedded and verified successfully (Photoshop IRB fallback): {file_path}")
                            return
                    except Exception:
                        pass
                elif "Temporary file already exists" in err_msg:
                    tmp_file: str = file_path + "_exiftool_tmp"
                    if os.path.exists(tmp_file):
                        try:
                            os.remove(tmp_file)
                            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15.0)
                            if verify_embedded_metadata(file_path, summary_text):
                                logger.info(f"EXIF embedded and verified successfully (temp file retry): {file_path}")
                                return
                        except Exception:
                            pass
                elif "Not a valid" in err_msg and "looks more like" in err_msg:
                    logger.warning(f"Warning: Skipped Exif (Invalid Ext): {file_path}")
                    return
                elif "Format error in file" in err_msg or "Bad format" in err_msg:
                    logger.warning(f"Warning: Skipped Exif (Corrupt): {file_path}")
                    return
                else:
                    if attempt < max_attempts:
                        time.sleep(1.0)
                    else:
                        logger.error(f"Failed to write EXIF after process error: {err_msg}")
                        
        logger.error(f"EXIF write verification failed after {max_attempts} attempts for: {file_path}")
        
    except Exception as e:
        if "[WinError 5]" in str(e) or "Access is denied" in str(e):
             logger.warning(f"Warning: Skipped Exif (Read Only): {file_path}")
        else:
             logger.error(f"Failed inline EXIF write for {file_path}: {e}")
    finally:
        # Guarantee removal of the temporary argfile to keep the project directory clean
        if os.path.exists(arg_file_path):
            try:
                os.remove(arg_file_path)
            except Exception:
                pass


def fetch_acdsee_metadata_for_batch(db_path: str, valid_paths: List[str]) -> Dict[str, Dict[str, Any]]:
    """Queries the SQLite database to fetch ACDSee metadata in bulk for a batch of image paths.

    Args:
        db_path: The absolute path to the SQLite database file.
        valid_paths: A list of absolute file paths to query.

    Returns:
        A dictionary mapping each file path to its corresponding ACDSee/XMP metadata fields.
    """
    metadata_map: Dict[str, Dict[str, Any]] = {}
    if not valid_paths:
        return metadata_map

    db_backend = get_db_backend(db_path)
    if db_backend != "postgresql" and not os.path.exists(db_path):
        return metadata_map

    conn = None
    try:
        conn, backend = get_db_conn(db_path)
        cursor = conn.cursor()
        
        # Check if table exists to avoid errors on unmigrated databases
        if backend == "sqlite":
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='photos'")
            table_exists = bool(cursor.fetchone())
        else:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'photos'
                );
            """)
            table_exists = cursor.fetchone()[0]

        if not table_exists:
            return metadata_map

        # Map target columns. We retrieve only metadata fields populated by crawl_and_ingest_all.py
        if backend == "postgresql":
            placeholders = ",".join(["%s"] * len(valid_paths))
        else:
            placeholders = ",".join(["?"] * len(valid_paths))
            
        query = f"""
            SELECT full_path, rating, label, author, gps_latitude, gps_longitude, gps_altitude, acdsee_tags, detected_faces, raw_metadata
            FROM photos WHERE full_path IN ({placeholders})
        """
        cursor.execute(query, valid_paths)
        
        for row in cursor.fetchall():
            f_path, rating, label, author, lat, lon, alt, tags_json, faces_json, raw_meta_json = row
            meta_dict: Dict[str, Any] = {}
            
            if rating is not None:
                meta_dict["rating"] = rating
            if label is not None:
                meta_dict["label"] = label
            if author is not None:
                meta_dict["author"] = author
            if lat is not None:
                meta_dict["gps_latitude"] = lat
            if lon is not None:
                meta_dict["gps_longitude"] = lon
            if alt is not None:
                meta_dict["gps_altitude"] = alt
                
            # Deserialize JSON fields
            if tags_json:
                try:
                    meta_dict["acdsee_tags"] = json.loads(tags_json)
                except Exception:
                    pass
            if faces_json:
                try:
                    meta_dict["detected_faces"] = json.loads(faces_json)
                except Exception:
                    pass
            if raw_meta_json:
                try:
                    meta_dict["raw_metadata"] = json.loads(raw_meta_json)
                except Exception:
                    pass
                    
            metadata_map[f_path] = meta_dict
            
    except Exception as e:
        logger.warning(f"Failed to fetch batch ACDSee metadata: {e}")
    finally:
        if conn:
            conn.close()
        
    return metadata_map


def save_results_to_sqlite(db_path: str, results_to_save: List[Dict[str, Any]]) -> None:
    """Saves/upserts cataloging results directly to the SQLite database.

    Args:
        db_path: The absolute path to the SQLite database file.
        results_to_save: A list of dicts containing image paths and cataloging metadata.

    Returns:
        None

    Raises:
        sqlite3.Error: If a database operation fails.
    """
    if not results_to_save:
        return

    with sqlite_lock:
        conn = None
        try:
            conn, backend = get_db_conn(db_path)
            if backend == "sqlite":
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
            cursor = conn.cursor()
            
            # Build schema if not exists
            if backend == "sqlite":
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS photos (
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
                cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_photos_full_path ON photos (full_path)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_rel_path ON photos (rel_path)")
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
                    )
                """)
                cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_photos_full_path ON photos (full_path)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_rel_path ON photos (rel_path)")
            
            insert_data: List[Tuple[str, str, str, str, str, str, str]] = []
            for item in results_to_save:
                full_path: str = item.get("full_path", "")
                if not full_path:
                    continue
                
                rel_path: str = compute_rel_path(full_path)
                primary_subject: str = item.get("primary_subject", "")
                environment: str = item.get("environment", "")
                
                # Serialize collections to JSON strings
                tags: List[str] = item.get("suggested_tags", [])
                suggested_tags: str = json.dumps(tags)
                
                technical_details: str = item.get("technical_details", "")
                
                objects: List[str] = item.get("detected_objects", [])
                detected_objects: str = json.dumps(objects)
                
                insert_data.append((
                    full_path,
                    rel_path,
                    primary_subject,
                    environment,
                    suggested_tags,
                    technical_details,
                    detected_objects
                ))

            if insert_data:
                if backend == "postgresql":
                    execute_batch(cursor, """
                        INSERT INTO photos (
                            full_path, rel_path, primary_subject, environment, suggested_tags, technical_details, detected_objects
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT(full_path) DO UPDATE SET
                            rel_path=excluded.rel_path,
                            primary_subject=excluded.primary_subject,
                            environment=excluded.environment,
                            suggested_tags=excluded.suggested_tags,
                            technical_details=excluded.technical_details,
                            detected_objects=excluded.detected_objects
                    """, insert_data)
                else:
                    cursor.executemany("""
                        INSERT INTO photos (
                            full_path, rel_path, primary_subject, environment, suggested_tags, technical_details, detected_objects
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(full_path) DO UPDATE SET
                            rel_path=excluded.rel_path,
                            primary_subject=excluded.primary_subject,
                            environment=excluded.environment,
                            suggested_tags=excluded.suggested_tags,
                            technical_details=excluded.technical_details,
                            detected_objects=excluded.detected_objects
                    """, insert_data)
                conn.commit()
                logger.info(f"Saved {len(insert_data)} records directly to {backend} database.")
        except (sqlite3.Error, Exception) as e:
            logger.error(f"Database save error: {e}")
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()


def save_results(results: List[Dict[str, Any]], output_json: str, is_milestone: bool = False) -> None:
    """
    Saves the list of dictionaries safely to the JSON file using an atomic replace.

    Args:
        results: The current list of image descriptions.
        output_json: The absolute path to write the output JSON file.
        is_milestone: Whether this save is a major milestone logging event.

    Returns:
        None
    """
    with json_lock:
        temp_filename = f"{output_json}.tmp"
        try:
            with open(temp_filename, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
            
            # Attempt atomic replace with retries to handle transient locks (e.g. from Google Drive sync)
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    os.replace(temp_filename, output_json)  # Atomic replace prevents corruption
                    logger.info(f"Saved JSON description database to disk ({len(results)} entries)")
                    break
                except PermissionError as pe:
                    if attempt < max_retries - 1:
                        logger.warning(f"Database file locked, retrying replace in 1.0s (attempt {attempt+1}/{max_retries})...")
                        time.sleep(1.0)
                    else:
                        raise pe
        except Exception as e:
            logger.error(f"Failed to save results to {output_json}: {e}")


def extract_json_payload(raw_text: str) -> Dict[str, Any]:
    """
    Cleans up markdown code fences and parses the JSON string into a Python dictionary.

    Args:
        raw_text: The raw output string from the model.

    Returns:
        A dictionary containing the parsed metadata, or a dictionary with error info if parsing fails.
    """
    cleaned: str = raw_text.strip()
    if '"primary_subject": ""' in cleaned:
        cleaned = cleaned.replace('"primary_subject": ""', '"primary_subject": "', 1)
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Pre-parse syntax fixes
    import re
    # Remove double/stray commas (e.g. , , or ,\s*,)
    cleaned = re.sub(r',\s*,', ',', cleaned)
    # Remove leading commas after open brace: { ,
    cleaned = re.sub(r'\{\s*,', '{', cleaned)
    # Remove trailing commas before close brace: , }
    cleaned = re.sub(r',\s*\}', '}', cleaned)

    try:
        parsed: Dict[str, Any] = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"Standard JSON parsing failed: {e}. Attempting regex fallback recovery...")
        # Fallback parser using regular expressions
        parsed = {}
        
        # 1. Extract primary_subject
        sub_m = re.search(r'"?primary_subject"?\s*:\s*"(.*?)"(?=\s*(?:,|\s*\}|\s*"))', cleaned, re.DOTALL | re.IGNORECASE)
        if sub_m:
            parsed["primary_subject"] = sub_m.group(1).replace('\\"', '"').replace('\\\\', '\\')
            
        # 2. Extract environment
        env_m = re.search(r'"?environment"?\s*:\s*"(.*?)"(?=\s*(?:,|\s*\}|\s*"))', cleaned, re.DOTALL | re.IGNORECASE)
        if env_m:
            parsed["environment"] = env_m.group(1).replace('\\"', '"').replace('\\\\', '\\')
            
        # 3. Extract suggested_tags
        tags_m = re.search(r'"?suggested_tags"?\s*:\s*\[(.*?)\]', cleaned, re.DOTALL | re.IGNORECASE)
        if tags_m:
            # Extract tags inside bracket list
            tags_content = tags_m.group(1)
            tags_list = re.findall(r'"([^"]*)"', tags_content)
            if not tags_list:
                # Fallback for unquoted tags
                tags_list = [t.strip() for t in tags_content.split(",") if t.strip()]
            parsed["suggested_tags"] = tags_list
            
        # If we couldn't extract anything, fallback to original raw_text
        if not parsed:
            logger.error("Regex fallback recovery failed completely.")
            return {
                "primary_subject": raw_text,
                "environment": "Unknown",
                "suggested_tags": ["error-parsing-json"]
            }
        logger.info("Successfully recovered JSON fields using regex fallback.")

    # Self-correcting key normalization (protects against VLM spelling/naming typos)
    normalized: Dict[str, Any] = {}
    for key, val in parsed.items():
        k_lower = key.lower()
        if "subject" in k_lower:
            normalized["primary_subject"] = val
        elif "env" in k_lower:
            normalized["environment"] = val
        elif "tag" in k_lower:
            normalized["suggested_tags"] = val
        else:
            normalized[key] = val

    # Ensure all expected keys exist with appropriate defaults
    required_keys = ["primary_subject", "environment", "suggested_tags"]
    for key in required_keys:
        if key not in normalized:
            normalized[key] = "" if key != "suggested_tags" else []
            
    # Fallback: if primary_subject is empty, stuff raw_text into it
    if not str(normalized.get("primary_subject", "")).strip() and raw_text.strip():
        normalized["primary_subject"] = raw_text.strip()
        
    return normalized


def add_or_update_result(results_list: List[Dict[str, Any]], path: str, metadata: Dict[str, Any]) -> None:
    """Inserts or updates a photo description entry in the results list to prevent duplicates.

    Args:
        results_list: The active list of photo descriptions.
        path: The absolute path of the image.
        metadata: The dictionary containing VLM tags and description.

    Returns:
        None
    """
    for item in results_list:
        if item.get("full_path", "").lower() == path.lower():
            # Update all keys from metadata in-place
            for key, val in metadata.items():
                item[key] = val
            return
    
    # If not found, append a new entry
    new_entry: Dict[str, Any] = {"full_path": path}
    new_entry.update(metadata)
    results_list.append(new_entry)
def process_batches(
    image_paths: List[str], 
    results: List[Dict[str, Any]], 
    prompt_path: str,
    active_servers: List[wsl_client.VLMServerConnection],
    max_workers: int,
    output_json: str,
    db_path: Optional[str] = None,
    embed_exif: bool = True,
    no_json_update: bool = False,
    temperature: float = 0.2,
    no_db: bool = False
) -> bool:
    """Processes images using active VLM servers in parallel.

    Dynamically loads and base64-encodes images in parallel, and distributes
    batches across all active servers. Gracefully handles server-specific
    failures by re-queueing images.

    Args:
        image_paths: List of absolute file paths to the images.
        results: The mutable list holding all descriptions.
        prompt_path: The absolute path to the prompt.txt file.
        active_servers: List of active VLMServerConnection instances.
        max_workers: Parallel thread count for disk loads and base64 encoding.
        output_json: The absolute path to write the output JSON file.
        db_path: Optional path to the SQLite database.
        embed_exif: If True, write metadata back to EXIF tags.
        no_json_update: If True, skip JSON database updates.
        temperature: Sampling temperature for model generation.
        no_db: If True, skip database writes entirely.

    Returns:
        True if the local VLM server was started inside the worker, False otherwise.
    """
    import queue
    
    local_started: List[bool] = [False]
    image_queue: queue.Queue[str] = queue.Queue()
    for path in image_paths:
        image_queue.put(path)
        
    exif_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    
    # Initialize prompt text from prompt_path
    current_prompt_text: str = ""
    try:
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                current_prompt_text = f.read().strip()
    except Exception as e:
        logger.warning(f"Initial load of prompt from {prompt_path} failed: {e}")

    def worker(server: wsl_client.VLMServerConnection) -> None:
        """Worker thread processing images for a specific VLM server.

        Args:
            server: The VLMServerConnection instance to send queries to.
        """
        nonlocal current_prompt_text
        
        # If this is the local server and it is not yet alive, boot it asynchronously in this thread
        if server.name == "Local RTX 5080" and not server.is_alive():
            logger.info("[Local RTX 5080] Local VLM server is offline. Starting local WSL VLM server in background worker thread...")
            local_started[0] = True
            if not wsl_client.start_wsl_server():
                logger.error("[Local RTX 5080] Failed to start local WSL VLM server in worker thread. Worker exiting.")
                return
            logger.info("[Local RTX 5080] Local VLM server started successfully.")
        
        while not image_queue.empty():
            # 1. Grab a batch for this server
            batch_paths: List[str] = []
            for _ in range(server.batch_size):
                try:
                    path = image_queue.get_nowait()
                    batch_paths.append(path)
                except queue.Empty:
                    break
            
            if not batch_paths:
                break
                
            # Dynamically reload prompt instructions
            try:
                if os.path.exists(prompt_path):
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        current_prompt_text = f.read().strip()
            except Exception as pe:
                logger.warning(f"Failed to dynamically reload prompt: {pe}")

            logger.info(f"[{server.name}] Pulling batch of size {len(batch_paths)} (Remaining: {image_queue.qsize()})")
            
            # Load and encode images in the batch (using max_workers thread pool internally to parallelize disk IO)
            batch_b64s: List[str] = []
            valid_paths: List[str] = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_workers // len(active_servers))) as loader:
                futures = {loader.submit(load_and_encode_image, path): path for path in batch_paths}
                for future in concurrent.futures.as_completed(futures):
                    path = futures[future]
                    img_b64 = future.result()
                    if img_b64 is not None:
                        batch_b64s.append(img_b64)
                        valid_paths.append(path)
                    else:
                        add_or_update_result(results, path, {
                            "primary_subject": "Error: Failed to open or encode image file.",
                            "environment": "Unknown",
                            "suggested_tags": ["error-loading-file"]
                        })
            
            if not batch_b64s:
                for _ in batch_paths:
                    image_queue.task_done()
                continue
                
            # Spawn background lookup of ACDSee metadata from SQLite while VLM prompt return is cooking.
            # Running this in a separate worker thread hides the database read latency under the model's 
            # inference latency (cooking time). Since SQLite read operations take under a millisecond, 
            # this database query finishes almost immediately, leaving the main thread unblocked to wait 
            # for the GPU server to finish model generation.
            acdsee_results: List[Dict[str, Dict[str, Any]]] = [{}]
            def db_lookup_worker() -> None:
                if db_path:
                    # Thread-safe database query to prevent locking or shared handle conflicts
                    acdsee_results[0] = fetch_acdsee_metadata_for_batch(db_path, valid_paths)
            
            db_thread = threading.Thread(target=db_lookup_worker, name=f"DBLookup-{server.name}")
            db_thread.start()
            
            try:
                # Query the specific server (blocking call while LLM is cooking)
                raw_responses = server.query(batch_b64s, current_prompt_text, temperature=temperature)
                
                # Join the background DB thread to ensure metadata lookup is complete.
                # Since the model takes several seconds/minutes to generate tokens, the DB thread is guaranteed
                # to have already exited, resulting in a zero-latency block here.
                db_thread.join()
                acdsee_map = acdsee_results[0]
                
                # Parse and save responses
                for path, raw_text in zip(valid_paths, raw_responses):
                    metadata: Dict[str, Any] = extract_json_payload(raw_text)
                    logger.info(f"[{server.name}] Processed: {path}")
                    
                    # Merge in the ACDSee metadata if available
                    if path in acdsee_map:
                        metadata.update(acdsee_map[path])
                    
                    # Print summary to screen
                    print(f"\n[{server.name}] Generated Description for: {os.path.basename(path)}", flush=True)
                    print(f"  Primary Subject: {metadata.get('primary_subject', '')}", flush=True)
                    print(f"  Environment:     {metadata.get('environment', '')}", flush=True)
                    print(f"  Suggested Tags:  {', '.join(metadata.get('suggested_tags', []))}\n", flush=True)
                    
                    add_or_update_result(results, path, metadata)
                    
                    if embed_exif:
                        summary_text: str = (
                            f"Subject: {metadata.get('primary_subject', '')}\n"
                            f"Environment: {metadata.get('environment', '')}\n"
                            f"Tags: {', '.join(metadata.get('suggested_tags', []))}"
                        )
                        try:
                            exif_executor.submit(inline_embed_metadata, path, summary_text)
                        except RuntimeError as re:
                            if "shutdown" in str(re).lower():
                                logger.info(f"[{server.name}] EXIF embedding skipped for {os.path.basename(path)} (shutdown in progress).")
                            else:
                                raise

                # Save results to SQLite or PostgreSQL database and JSON
                db_backend = get_db_backend(db_path)
                if (db_path or db_backend == "postgresql") and not no_db:
                    batch_results = [r for r in results if r.get("full_path") in valid_paths]
                    try:
                        save_results_to_sqlite(db_path, batch_results)
                    except Exception as sqle:
                        logger.error(f"{db_backend.capitalize()} database save failed: {sqle}")
                        raise
                
                if not no_json_update:
                    save_results(results, output_json, is_milestone=True)
                    
                # Mark successfully processed items as done
                for _ in batch_paths:
                    image_queue.task_done()
                    
            except Exception as e:
                logger.exception(f"[{server.name}] Batch processing failed: {e}. Re-queueing images...")
                # Re-add all paths in this batch back to the queue so the other server can process them!
                for path in batch_paths:
                    image_queue.put(path)
                    image_queue.task_done()
                time.sleep(5.0)

    # Spawn worker threads for each active server
    threads: List[threading.Thread] = []
    for srv in active_servers:
        t = threading.Thread(target=worker, args=(srv,), name=f"Worker-{srv.name}")
        t.start()
        threads.append(t)
        
    # Wait for all workers to complete
    for t in threads:
        t.join()

    # Wait for EXIF tasks to complete
    if embed_exif:
        exc_type, _, _ = sys.exc_info()
        if exc_type is not None:
            logger.warning("Aborting: Canceling pending EXIF embedding tasks due to error/interrupt...")
            exif_executor.shutdown(wait=True, cancel_futures=True)
        else:
            logger.info("Waiting for background EXIF metadata embedding to complete sequentially...")
            exif_executor.shutdown(wait=True)
            logger.info("All EXIF metadata embedding completed.")
    else:
        exif_executor.shutdown(wait=False)
        
    return local_started[0]


def cleanup_old_logs(project_dir: str) -> None:
    """Checks if two days (48 hours) have passed since the last log cleanup.

    If so, deletes or truncates log files (.log and .log.bak) in the project directory
    and updates the cleanup timestamp marker file.

    Args:
        project_dir: The directory containing the log files.

    Returns:
        None

    Raises:
        None
    """
    marker_path: str = os.path.join(project_dir, ".last_log_cleanup")
    two_days_seconds: float = 2.0 * 24.0 * 60.0 * 60.0  # 172800 seconds
    now: float = time.time()

    perform_cleanup: bool = False
    if not os.path.exists(marker_path):
        perform_cleanup = True
    else:
        try:
            with open(marker_path, "r", encoding="utf-8") as f:
                last_cleanup: float = float(f.read().strip())
            if now - last_cleanup >= two_days_seconds:
                perform_cleanup = True
        except Exception:
            # If reading the marker file fails or contains invalid data, run cleanup anyway to be safe
            perform_cleanup = True

    if not perform_cleanup:
        return

    logger.info("Executing log cleanup (every other day)...")
    for file in os.listdir(project_dir):
        if file.endswith(".log") or file.endswith(".log.bak"):
            file_path: str = os.path.join(project_dir, file)
            try:
                # Attempt to delete the log file directly
                os.remove(file_path)
                logger.info(f"Deleted log file: {file}")
            except PermissionError:
                # Handle active file lock (e.g. from running processes) by truncating the file to 0 bytes
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.truncate(0)
                    logger.info(f"Log file locked by active process. Truncated: {file}")
                except Exception as e:
                    logger.warning(f"Failed to truncate locked log file {file}: {e}")
            except Exception as e:
                logger.warning(f"Failed to delete log file {file}: {e}")

    try:
        with open(marker_path, "w", encoding="utf-8") as f:
            f.write(str(now))
    except Exception as e:
        logger.error(f"Failed to update log cleanup timestamp marker: {e}")


def main() -> None:
    """
    Main orchestrator function: loads arguments, reads prompt text,
    loads the database cache, loads Gemma 4 12B IT, and starts processing.

    Args:
        None

    Returns:
        None
    """
    logger.info(f"Starting Local Gemma 4 Photo Cataloger v{__version__}")
    
    # 1. Parse command line arguments to allow flexible runs
    parser = argparse.ArgumentParser(description="Modular Gemma 4 Photo Describer Tool.")
    parser.add_argument(
        "--max-photos", 
        type=int, 
        default=int(os.environ.get("MAX_PHOTOS", 100)),
        help="Maximum number of new images to describe."
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=2,
        help="Batch size for model evaluation."
    )
    parser.add_argument(
        "--max-workers", 
        type=int, 
        default=8,
        help="Number of background CPU worker threads for image loading."
    )
    parser.add_argument(
        "--dir",
        type=str,
        action="append",
        dest="dir",
        help="Directory to scan for images. Can be specified multiple times."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_JSON,
        help="Path to the output photo descriptions JSON database."
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DB_PATH,
        help="Path to the SQLite database file."
    )
    parser.add_argument(
        "--submitted-cache",
        type=str,
        default=SUBMITTED_CACHE,
        help="Path to the submitted photos cache file to avoid reprocessing."
    )
    parser.add_argument(
        "--embed-exif",
        action="store_true",
        default=True,
        help="Natively embed descriptions into image EXIF tags using exiftool (enabled by default)."
    )
    parser.add_argument(
        "--no-embed-exif",
        action="store_false",
        dest="embed_exif",
        help="Disable natively embedding descriptions into image EXIF tags."
    )
    parser.add_argument(
        "--file",
        type=str,
        help="A single image file path to process. Bypasses skip-cache and processes this file specifically."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation of all images in scanned directories by bypassing the skip cache."
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not update or save the generated descriptions to the JSON database."
    )
    parser.add_argument(
        "--skip-existing-exif",
        action="store_true",
        help="Skip images that already have description metadata embedded in their EXIF/XMP tags."
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Do not save the generated descriptions to the SQLite database."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for VLM generation. Higher values (e.g. 0.7) prevent context verbatim parroting."
    )
    args = parser.parse_args()

    if args.no_db:
        args.db = ""

    # Run automatic log cleanup check (every 2 days)
    current_project_dir: str = os.path.dirname(os.path.abspath(__file__))
    cleanup_old_logs(current_project_dir)

    # 2. Load the prompt dynamically from prompt.txt
    prompt_path: str = os.path.join(os.path.dirname(__file__), "prompt.txt")
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text: str = f.read().strip()
            logger.info("Loaded prompt instructions from prompt.txt successfully.")
    else:
        logger.error(f"Failed to find prompt configuration at: {prompt_path}")
        sys.exit(1)

    results: List[Dict[str, Any]] = []
    
    # Load existing database records (SQLite preferred, fallback to JSON)
    has_db = False
    db_backend = get_db_backend(args.db)
    
    if (db_backend == "postgresql" or (args.db and os.path.exists(args.db))) and not args.no_db:
        try:
            conn, backend = get_db_conn(args.db)
            cursor = conn.cursor()
            
            if backend == "sqlite":
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='photos'")
                table_exists = bool(cursor.fetchone())
            else:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'photos'
                    );
                """)
                table_exists = cursor.fetchone()[0]

            if table_exists:
                migrate_photos_schema(conn, backend)
                cursor.execute("SELECT full_path, rel_path, primary_subject, environment, suggested_tags, technical_details, detected_objects FROM photos")
                for row in cursor.fetchall():
                    full_path = row[0]
                    rel_path = row[1]
                    subject = row[2] or ""
                    env = row[3] or ""
                    try:
                        tags = json.loads(row[4] or "[]")
                    except Exception:
                        tags = []
                    tech = row[5] or ""
                    try:
                        obj = json.loads(row[6] or "[]")
                    except Exception:
                        obj = []
                    
                    item = {
                        "full_path": full_path,
                        "rel_path": rel_path,
                        "primary_subject": subject,
                        "environment": env,
                        "suggested_tags": tags,
                        "technical_details": tech,
                        "detected_objects": obj
                    }
                    
                    # Discard failed entries or model glitches so we can retry them
                    tags_lower = [str(t).lower() for t in tags]
                    if not subject or is_model_refusal(subject) or "error-parsing-json" in tags_lower:
                        continue
                    results.append(item)
                logger.info(f"Loaded {len(results)} valid descriptions from {backend} database.")
                has_db = True
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to load existing {db_backend} database: {e}. Trying JSON fallback.")

    if not has_db and os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    raw_results = json.loads(content)
                    for item in raw_results:
                        if isinstance(item, dict):
                            subject = item.get("primary_subject", "").lower()
                            tags = [str(t).lower() for t in item.get("suggested_tags", [])]
                            
                            # Discard failed entries or model glitches so we can retry them
                            if not subject or is_model_refusal(subject) or "error-parsing-json" in tags:
                                continue
                            results.append(item)
                    
            if 'raw_results' in locals():
                retry_count = len(raw_results) - len(results)
                logger.info(f"Loaded {len(results)} valid descriptions from {args.output}. Retrying {retry_count} previous failures.")
            else:
                logger.info("Loaded 0 existing descriptions.")
        except Exception as e:
            logger.warning(f"Failed to load existing JSON: {e}. Starting fresh.")
            
    # Extract processed paths to skip duplicates (bypass if --force is set)
    processed_paths: Set[str] = set()
    if not args.force:
        for item in results:
            full_path = item.get("full_path", "")
            processed_paths.add(full_path.lower())
            
            rel_path = compute_rel_path(full_path)
            processed_paths.add(rel_path)
            
        # Read Submitted Cache tracking file
        if args.submitted_cache and os.path.exists(args.submitted_cache):
            with open(args.submitted_cache, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        processed_paths.add(line.strip().replace("\\", "/").lower())

    if args.file:
        normalized_file = os.path.abspath(args.file)
        if os.path.isfile(normalized_file):
            images = [normalized_file]
            # Remove any existing description for this file to prevent duplicate entries in the database
            results = [item for item in results if item.get("full_path", "").lower() != normalized_file.lower()]
            logger.info(f"Forcing single file processing for: {normalized_file}")
        else:
            logger.error(f"Provided path is not a file: {normalized_file}")
            sys.exit(1)
    else:
        target_dirs: List[str] = args.dir if args.dir else PICTURE_DIRS
        logger.info(f"Scanning directories for up to {args.max_photos} unprocessed images...")
        images = get_image_files(target_dirs, limit=args.max_photos, processed_paths=processed_paths, skip_existing_exif=args.skip_existing_exif)
        # Sort image paths alphabetically to ensure clean chronological processing
        images.sort()
        logger.info(f"Found {len(images)} new images to process.")
    
    if not images:
        logger.info("No new images found. All done!")
        return

    # Probe and set up VLM server connections with safe batch sizes to prevent VRAM paging on display GPUs
    local_server = wsl_client.VLMServerConnection("Local RTX 5080", "http://127.0.0.1:8000", batch_size=args.batch_size)
    remote_server = wsl_client.VLMServerConnection("Remote RTX 4070 Ti SUPER", "http://192.168.8.113:8000", batch_size=args.batch_size)
    
    active_servers: List[wsl_client.VLMServerConnection] = []
    
    logger.info("Probing VLM servers...")
    
    # Check if remote server is online
    logger.info("Checking remote VLM server availability (RTX 4070 Ti SUPER)...")
    if remote_server.is_alive():
        logger.info("Remote VLM server is online and active.")
        active_servers.append(remote_server)
    else:
        logger.info("Remote VLM server is offline or unreachable.")
        
    # Register the local server. If it is offline, the worker thread will boot it asynchronously in the background.
    logger.info("Checking local VLM server availability (RTX 5080)...")
    if local_server.is_alive():
        logger.info("Local VLM server is online and active.")
        active_servers.append(local_server)
    else:
        logger.info("Local VLM server is offline. It will be started asynchronously in the background worker thread.")
        active_servers.append(local_server)
            
    if not active_servers:
        logger.error("No active VLM servers available! Exiting.")
        sys.exit(1)
        
    logger.info(f"Active VLM servers for this run: {[s.name for s in active_servers]}")

    local_started: bool = False
    try:
        # Process all image paths in batches using the active REST API servers in parallel
        logger.info("Commencing concurrent batch generation...")
        start_time: float = time.time()
        local_started = process_batches(
            images, 
            results, 
            prompt_path, 
            active_servers, 
            args.max_workers, 
            args.output,
            db_path=args.db,
            embed_exif=args.embed_exif,
            no_json_update=args.no_json,
            temperature=args.temperature,
            no_db=args.no_db
        )
        end_time: float = time.time()
        logger.info(f"PROCESSED: {len(images)} images in {end_time - start_time:.2f} seconds")
        if not args.no_json:
            logger.info(f"Completed processing. Operations successfully saved to JSON: {args.output}")
        if (args.db or db_backend == "postgresql") and not args.no_db:
            logger.info(f"Completed processing. Operations successfully saved to {db_backend} database.")
    except wsl_client.FatalVLMServerError as fse:
        logger.error(f"Fatal VLM Server Error: {fse}")
        logger.error("Aborting cataloging pipeline.")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt (Ctrl-C) detected. Aborting cataloging pipeline.")
    finally:
        if local_started:
            try:
                if not sys.stdin.isatty():
                    logger.info("Non-interactive session (background thread). Leaving local VLM model server active.")
                    wsl_client.leave_keep_alive_running()
                else:
                    print("\n" + "=" * 50)
                    user_choice: str = input("Do you want to shut down the local VLM model server? (y/n) [n]: ").strip().lower()
                    if user_choice in ("y", "yes"):
                        wsl_client.stop_wsl_server()
                    else:
                        logger.info("Leaving local VLM model server active.")
                        wsl_client.leave_keep_alive_running()
            except KeyboardInterrupt:
                logger.info("Exiting immediately. Leaving local VLM model server active.")
                wsl_client.leave_keep_alive_running()
            except Exception as e:
                logger.warning(f"Error checking server shutdown preference: {e}")



if __name__ == "__main__":
    main()
