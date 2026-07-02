"""JRiver Music Library Metadata Ingester.

Purpose:
    This script crawls the target music folder, parses JRiver XML sidecars (*_JRSidecar.xml)
    or reads audio file tags directly using mutagen, associates track metadata with
    folder-level cover art, and bulk-ingests them into the `music_tracks` table.
    It protects existing curated metadata using ON CONFLICT DO NOTHING.

Architecture and Mechanics:
    1. Schema Migration: Ensures `music_tracks` PostgreSQL/SQLite table exists.
    2. Parallel File Scan & Tag Parser: Uses ThreadPoolExecutor to walk and read metadata:
       - Uses parse_jr_sidecar for files with sidecar XMLs.
       - Uses mutagen for raw audio files (FLAC, WAV, M4A, etc.) without sidecars.
    3. Bulk Insert with Conflict Protection: Uses ON CONFLICT DO NOTHING to preserve
       already curated tracks in the database.
"""

import os
import sys
import xml.etree.ElementTree as ET
import sqlite3
import logging
import argparse
import datetime
import concurrent.futures
from typing import Dict, List, Any, Set, Tuple, Optional, Union
import psycopg2
from psycopg2.extras import execute_batch
import mutagen

PROJECT_DIR: str = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH: str = os.path.join(PROJECT_DIR, "photo_catalog.db")
LOG_FILE: str = os.path.join(PROJECT_DIR, "gemma_cataloger.log")

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

console = logging.StreamHandler(sys.stdout)
console.setFormatter(formatter)
logger.addHandler(console)

file_log = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_log.setFormatter(formatter)
logger.addHandler(file_log)


def migrate_music_schema(db_backend: str, db_conn_params: Dict[str, Any]) -> None:
    """Creates the target `music_tracks` table and index if they do not exist.

    Args:
        db_backend: Name of the database backend ('sqlite' or 'postgresql').
        db_conn_params: Connection parameters dictionary.
    """
    logger.info(f"Checking migration state for music tracks table on {db_backend}...")
    
    if db_backend == "postgresql":
        conn = psycopg2.connect(**db_conn_params)
        conn.set_client_encoding("UTF8")
    else:
        conn = sqlite3.connect(db_conn_params["database"], timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")

    try:
        cursor = conn.cursor()
        
        # Check if music_tracks table exists
        if db_backend == "sqlite":
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='music_tracks'")
            table_exists = bool(cursor.fetchone())
        else:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'music_tracks'
                );
            """)
            table_exists = cursor.fetchone()[0]

        if not table_exists:
            logger.info("Creating 'music_tracks' table...")
            if db_backend == "sqlite":
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS music_tracks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_path TEXT UNIQUE NOT NULL,
                        title TEXT,
                        artist TEXT,
                        album TEXT,
                        genre TEXT,
                        track_number INTEGER,
                        rating INTEGER,
                        album_art_path TEXT,
                        jriver_genre TEXT,
                        suggested_genre TEXT,
                        xml_metadata_path TEXT,
                        date_imported TEXT
                    );
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_music_file_path ON music_tracks (file_path)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_music_album ON music_tracks (album)")
            else:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS music_tracks (
                        id SERIAL PRIMARY KEY,
                        file_path TEXT UNIQUE NOT NULL,
                        title TEXT,
                        artist TEXT,
                        album TEXT,
                        genre TEXT,
                        track_number INTEGER,
                        rating INTEGER,
                        album_art_path TEXT,
                        jriver_genre TEXT,
                        suggested_genre TEXT,
                        xml_metadata_path TEXT,
                        date_imported TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_music_file_path ON music_tracks (file_path)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_music_album ON music_tracks (album)")
            conn.commit()
            logger.info("Table 'music_tracks' created successfully.")
        else:
            logger.info("Table 'music_tracks' already exists.")
    finally:
        conn.close()


def parse_jr_sidecar(xml_path: str) -> Dict[str, str]:
    """Parses JRiver XML sidecar files flatly.

    Args:
        xml_path: Absolute filesystem path to the XML sidecar file.

    Returns:
        Flat dictionary containing Field Name values.
    """
    metadata: Dict[str, str] = {}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for item in root.findall(".//Item"):
            for field in item.findall("Field"):
                name = field.get("Name")
                text = field.text
                if name:
                    metadata[name] = text if text is not None else ""
    except Exception as e:
        logger.error(f"Failed to parse JRiver sidecar XML at {xml_path}: {e}")
    return metadata


def parse_audio_mutagen(file_path: str) -> Dict[str, Any]:
    """Reads embedded tags from an audio file using Mutagen.

    Args:
        file_path: Absolute path to the audio file.

    Returns:
        A dictionary containing parsed metadata.
    """
    metadata: Dict[str, Any] = {
        "Title": os.path.splitext(os.path.basename(file_path))[0],
        "Artist": "Unknown Artist",
        "Album": "Unknown Album",
        "Genre": "Unknown Genre",
        "Rating": 0,
        "Track #": 0
    }
    try:
        audio = mutagen.File(file_path, easy=True)
        if audio is not None:
            def get_tag(key: str, default: str) -> str:
                val = audio.get(key)
                if val and isinstance(val, list):
                    return str(val[0])
                return default

            metadata["Title"] = get_tag("title", metadata["Title"])
            metadata["Artist"] = get_tag("artist", "Unknown Artist")
            metadata["Album"] = get_tag("album", "Unknown Album")
            metadata["Genre"] = get_tag("genre", "Unknown Genre")
            
            track_num_str = get_tag("tracknumber", "0")
            if "/" in track_num_str:
                track_num_str = track_num_str.split("/")[0]
            try:
                metadata["Track #"] = int(track_num_str)
            except ValueError:
                metadata["Track #"] = 0
                
            # Attempt to read rating from raw tags if accessible
            try:
                raw_audio = mutagen.File(file_path, easy=False)
                if raw_audio:
                    for k in raw_audio.keys():
                        if "rating" in k.lower() or "popm" in k.lower():
                            raw_val = raw_audio[k]
                            val_str = str(raw_val[0]) if isinstance(raw_val, list) else str(raw_val)
                            if val_str.isdigit():
                                metadata["Rating"] = int(val_str)
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"Mutagen failed to read {file_path}: {e}")
    return metadata


def extract_largest_image_from_pdf(pdf_path: str, output_dir: str) -> Optional[str]:
    """Inspects a PDF booklet, finds the largest embedded image, and extracts it.

    Args:
        pdf_path: Absolute path to the PDF file.
        output_dir: Directory where the extracted image should be written.

    Returns:
        Absolute path to the extracted image file, or None if extraction failed.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        largest_image_bytes = None
        largest_size = 0
        largest_ext = "jpg"
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)
            
            for img_info in image_list:
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                width = base_image["width"]
                height = base_image["height"]
                ext = base_image["ext"]
                
                resolution = width * height
                if resolution > largest_size:
                    largest_size = resolution
                    largest_image_bytes = image_bytes
                    largest_ext = ext
                    
        if largest_image_bytes and largest_size >= 150000:
            out_filename = f"Folder_extracted_from_pdf.{largest_ext}"
            out_path = os.path.join(output_dir, out_filename)
            with open(out_path, "wb") as f:
                f.write(largest_image_bytes)
            logger.info(f"Extracted cover art from PDF: {os.path.basename(pdf_path)} -> {out_filename}")
            return out_path
    except Exception as e:
        logger.warning(f"Failed to extract images from PDF booklet {pdf_path}: {e}")
    return None


def find_album_art(track_directory: str) -> Optional[str]:
    """Searches the track directory and standard subdirectories for cover art files.

    Args:
        track_directory: Path to the directory containing the track file.

    Returns:
        Absolute path to the largest cover art image, or None if not found.
    """
    standard_names: Set[str] = {"folder.jpg", "cover.jpg", "albumart.jpg", "albumartsmall.jpg", "folder.png", "cover.png", "front.jpg", "front.png"}
    candidates: List[str] = []
    
    dirs_to_search: List[str] = [track_directory]
    try:
        if os.path.exists(track_directory):
            for entry in os.scandir(track_directory):
                if entry.is_dir():
                    name_lower = entry.name.lower()
                    if name_lower in {"artwork", "covers", "scans", "art", "images", "covert"}:
                        dirs_to_search.append(entry.path)
    except OSError:
        pass

    for search_dir in dirs_to_search:
        try:
            for entry in os.scandir(search_dir):
                if entry.is_file():
                    name_lower = entry.name.lower()
                    if name_lower in standard_names:
                        candidates.append(entry.path)
                    elif any(k in name_lower for k in ["cover", "folder", "albumart", "front", "art"]):
                        if name_lower.endswith((".jpg", ".jpeg", ".png", ".webp")):
                            candidates.append(entry.path)
        except OSError:
            continue

    if not candidates:
        try:
            pdf_files = []
            for search_dir in dirs_to_search:
                for entry in os.scandir(search_dir):
                    if entry.is_file() and entry.name.lower().endswith(".pdf"):
                        pdf_files.append(entry.path)
            for pdf_path in pdf_files:
                extracted_path = extract_largest_image_from_pdf(pdf_path, track_directory)
                if extracted_path:
                    candidates.append(extracted_path)
        except OSError:
            pass

    if not candidates:
        return None

    try:
        candidates.sort(key=lambda x: os.path.getsize(x), reverse=True)
        return candidates[0]
    except OSError:
        return candidates[0]


def process_single_file(file_path: str, xml_metadata_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Helper worker task to parse a single audio track using mutagen or JRiver sidecar.

    Args:
        file_path: The absolute path to the audio file.
        xml_metadata_path: Optional JRiver XML sidecar path.

    Returns:
        A dictionary representing parsed track metadata, or None.
    """
    try:
        if not os.path.exists(file_path):
            return None

        # Determine metadata source
        if xml_metadata_path and os.path.exists(xml_metadata_path):
            xml_data = parse_jr_sidecar(xml_metadata_path)
            title = xml_data.get("Name", os.path.splitext(os.path.basename(file_path))[0])
            artist = xml_data.get("Artist", "Unknown Artist")
            album = xml_data.get("Album", "Unknown Album")
            genre = xml_data.get("Genre", "Unknown Genre")
            
            rating_str = xml_data.get("Rating", "0")
            try:
                rating = int(rating_str)
            except ValueError:
                rating = 0

            track_num_str = xml_data.get("Track #", "0")
            try:
                track_number = int(track_num_str)
            except ValueError:
                track_number = 0
        else:
            # Read embedded tags via Mutagen
            audio_data = parse_audio_mutagen(file_path)
            title = audio_data["Title"]
            artist = audio_data["Artist"]
            album = audio_data["Album"]
            genre = audio_data["Genre"]
            rating = audio_data["Rating"]
            track_number = audio_data["Track #"]

        track_dir = os.path.dirname(file_path)
        album_art = find_album_art(track_dir)

        return {
            "file_path": file_path,
            "title": title,
            "artist": artist,
            "album": album,
            "genre": genre,
            "track_number": track_number,
            "rating": rating,
            "album_art_path": album_art or "",
            "jriver_genre": genre,
            "xml_metadata_path": xml_metadata_path or ""
        }
    except Exception as ex:
        logger.error(f"Error parsing track {file_path}: {ex}")
        return None


def scan_and_package_tracks(root_dir: str, max_workers: int, db_backend: str, db_conn_params: Dict[str, Any], limit: Optional[int] = None) -> int:
    """Crawls root_dir recursively and parses all tracks in parallel, flushing in chunks of 1000.

    Args:
        root_dir: Root directory of the music library.
        max_workers: Number of threads to run in parallel.
        db_backend: Name of the database backend ('sqlite' or 'postgresql').
        db_conn_params: Connection parameters dictionary.
        limit: Limit number of tracks to scan.

    Returns:
        The total number of tracks successfully written/processed.
    """
    logger.info("Loading existing track paths from database...")
    existing_paths: Set[str] = set()
    try:
        if db_backend == "postgresql":
            conn = psycopg2.connect(**db_conn_params)
        else:
            conn = sqlite3.connect(db_conn_params["database"])
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM music_tracks")
        existing_paths = {row[0] for row in cursor.fetchall()}
        conn.close()
        logger.info(f"Loaded {len(existing_paths):,} existing track paths. These will be skipped during scanning.")
    except Exception as e:
        logger.warning(f"Could not load existing paths from database (performing full scan): {e}")

    logger.info(f"Crawling root music directory recursively: {root_dir}")
    audio_extensions = {".flac", ".wav", ".m4a", ".ape", ".flv", ".mp4", ".mkv", ".mov", ".mpg"}
    
    # 1. First Pass: Gather files
    audio_files: List[Tuple[str, Optional[str]]] = []
    
    for root, _, files in os.walk(root_dir):
        # Locate JRiver sidecar XML files in this folder
        folder_xmls: Dict[str, str] = {}
        for file in files:
            if file.endswith("_JRSidecar.xml"):
                base_name = file[:-14]
                folder_xmls[base_name.lower()] = os.path.join(root, file)

        # Locate audio files
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in audio_extensions:
                file_path = os.path.join(root, file)
                
                # Skip if file path is already in the database
                if file_path in existing_paths:
                    continue
                
                # Check if this track has a corresponding sidecar XML
                base_name = os.path.splitext(file)[0]
                xml_path = folder_xmls.get(base_name.lower())
                
                audio_files.append((file_path, xml_path))
                if limit and len(audio_files) >= limit:
                    break
        if limit and len(audio_files) >= limit:
            break

    logger.info(f"Found {len(audio_files)} audio tracks to process. Parsing in parallel utilizing {max_workers} threads...")
    
    packaged_records: List[Dict[str, Any]] = []
    completed = 0
    total = len(audio_files)
    total_written = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, path, xml): path for path, xml in audio_files}
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            if completed % 500 == 0 or completed == total:
                logger.info(f"  [PARSER PROGRESS] Parsed {completed}/{total} files...")
            res = future.result()
            if res:
                packaged_records.append(res)
                
            # Flush batch of 1000 to the database to free memory and guarantee progress
            if len(packaged_records) >= 1000:
                batch_ingest_tracks(db_backend, db_conn_params, packaged_records)
                total_written += len(packaged_records)
                packaged_records = []

    # Final flush of remaining records
    if packaged_records:
        batch_ingest_tracks(db_backend, db_conn_params, packaged_records)
        total_written += len(packaged_records)
        
    return total_written


def batch_ingest_tracks(db_backend: str, db_conn_params: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
    """Performs bulk transactional inserts using ON CONFLICT DO NOTHING to preserve curated entries.

    Args:
        db_backend: Name of the database backend ('sqlite' or 'postgresql').
        db_conn_params: Connection parameters dictionary.
        records: List of track metadata dictionaries.
    """
    logger.info(f"Starting batch database ingestion of {len(records)} tracks into {db_backend}...")
    
    if db_backend == "postgresql":
        conn = psycopg2.connect(**db_conn_params)
        conn.set_client_encoding("UTF8")
    else:
        conn = sqlite3.connect(db_conn_params["database"], timeout=60.0)
        conn.execute("PRAGMA journal_mode=WAL;")

    try:
        cursor = conn.cursor()
        
        insert_query = """
            INSERT INTO music_tracks (
                file_path, title, artist, album, genre, 
                track_number, rating, album_art_path, 
                jriver_genre, xml_metadata_path
            ) VALUES (
                %(file_path)s, %(title)s, %(artist)s, %(album)s, %(genre)s, 
                %(track_number)s, %(rating)s, %(album_art_path)s, 
                %(jriver_genre)s, %(xml_metadata_path)s
            ) ON CONFLICT (file_path) DO NOTHING
        """
        
        if db_backend == "sqlite":
            insert_query = insert_query.replace("%(", ":").replace(")s", "")
            cursor.executemany(insert_query, records)
        else:
            execute_batch(cursor, insert_query, records)
            
        conn.commit()
        logger.info(f"Ingestion transaction completed. {len(records)} tracks processed (already-curated items preserved).")
    except Exception as e:
        conn.rollback()
        logger.error(f"Ingestion failed: {e}")
        raise e
    finally:
        conn.close()


def main() -> None:
    """CLI execution entry parser."""
    parser = argparse.ArgumentParser(description="Ingest JRiver Media Library into Database.")
    parser.add_argument("--root", required=True, help="Root folder of the music library (D:\\Users\\steven\\Music)")
    parser.add_argument("--limit-tracks", type=int, default=None, help="Limit number of tracks to ingest (for testing)")
    parser.add_argument("--db-backend", default="sqlite", choices=["sqlite", "postgresql"], help="Database backend")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to SQLite database file")
    parser.add_argument("--max-workers", type=int, default=16, help="Number of worker threads (default: 16)")
    
    args = parser.parse_args()

    db_conn_params: Dict[str, Any] = {}
    if args.db_backend == "postgresql":
        from dotenv import load_dotenv
        load_dotenv("auth/.env")
        db_conn_params = {
            "dbname": os.getenv("DB_NAME", "photo_catalog"),
            "user": os.getenv("DB_USER", "postgres"),
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "password": os.getenv("DB_PASSWORD", "")
        }
    else:
        db_conn_params = {"database": args.db_path}

    migrate_music_schema(args.db_backend, db_conn_params)

    total_written = scan_and_package_tracks(
        args.root, args.max_workers, args.db_backend, db_conn_params, args.limit_tracks
    )
    logger.info(f"Ingestion process completed successfully. Total processed: {total_written} tracks.")


if __name__ == "__main__":
    main()
