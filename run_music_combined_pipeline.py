"""Unified Music Cataloger Orchestrator Pipeline.

Purpose:
    This script consolidates JRiver XML sidecar track parsing, cover art/PDF
    booklet extraction, VLM-based visual cover description, and LLM-based metadata
    curation (resolving Artist, Album, Genre, and Title gaps) into a single,
    unified Python execution flow.

Architecture and Mechanics:
    1. Directory Chunking: Groups music directories into batches of 20.
    2. For each chunk of 20 directories:
       a. Parse & Ingest Tracks: Parses track XMLs, skips MP3s, and bulk-saves tracks.
       b. Batched VLM Cover Scanning: Subprocesses the pristine production 'describe_photos.py'
          passing the 20 directories as arguments, analyzing new covers.
       c. LLM/VLM Curation: Synthesizes and corrects metadata gaps (Unknown Artist, etc.)
          for this chunk's tracks using the visual cover descriptions.
       d. Database Commit: Commits the curated metadata directly to PostgreSQL.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import psycopg2
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, Optional, Set, Union

# Reconfigure output encoding for Windows CP1252 compatibility
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

PROJECT_DIR: str = os.path.dirname(os.path.abspath(__file__))

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console = logging.StreamHandler(sys.stdout)
console.setFormatter(formatter)
logger.addHandler(console)

# Core imports from our modular components
import ingest_music_library
import clean_database_artists
import sql_loader


def extract_metadata_via_exiftool(file_path: str) -> Dict[str, str]:
    """Runs ExifTool to read embedded audio tags from a file.

    Args:
        file_path: The absolute path of the target audio file.

    Returns:
        A dictionary matching JRiver sidecar XML tag formats.
    """
    import uuid
    exiftool_path = r"H:\Wan_project\exiftool\exiftool.exe"
    arg_file_path = os.path.join(PROJECT_DIR, f"exif_read_args_{uuid.uuid4().hex}.txt")
    
    try:
        with open(arg_file_path, "w", encoding="utf-8") as arg_f:
            arg_f.write("-json\n")
            arg_f.write("-charset\n")
            arg_f.write("filename=utf8\n")
            arg_f.write("-Artist\n")
            arg_f.write("-Album\n")
            arg_f.write("-Title\n")
            arg_f.write("-Genre\n")
            arg_f.write("-TrackNumber\n")
            arg_f.write("-Rating\n")
            arg_f.write(file_path + "\n")
            
        cmd = [
            exiftool_path,
            "-@", arg_file_path
        ]
        res = subprocess.run(cmd, capture_output=True, check=True)
        data = json.loads(res.stdout)
        if data and isinstance(data, list):
            item = data[0]
            
            def clean_value(val: Any) -> str:
                if isinstance(val, list):
                    return ", ".join(str(v) for v in val if v)
                return str(val) if val is not None else ""
                
            artist = clean_value(item.get("Artist", "Unknown Artist"))
            album = clean_value(item.get("Album", "Unknown Album"))
            title = clean_value(item.get("Title", os.path.splitext(os.path.basename(file_path))[0]))
            genre = clean_value(item.get("Genre", "Unknown Genre"))
            rating_val = clean_value(item.get("Rating", "0"))
            track_num = clean_value(item.get("TrackNumber", "0"))
            
            return {
                "Artist": artist,
                "Album": album,
                "Name": title,
                "Genre": genre,
                "Rating": rating_val,
                "Track #": track_num
            }
    except Exception as e:
        logger.warning(f"ExifTool failed to extract tags from {file_path}: {e}")
    finally:
        if os.path.exists(arg_file_path):
            try:
                os.remove(arg_file_path)
            except Exception:
                pass
    return {}


def get_pg_conn_params() -> Dict[str, Any]:
    """Loads PostgreSQL connection parameters from env and db_password.txt.

    Args:
        None

    Returns:
        A dictionary containing database connection parameters (dbname, user,
        host, port, password).

    Raises:
        None
    """
    load_dotenv(os.path.join(PROJECT_DIR, "auth", ".env"))
    dbname = os.getenv("DB_NAME", "photo_catalog")
    user = os.getenv("DB_USER", "postgres")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    
    pwd = ""
    pwd_path = os.path.join(PROJECT_DIR, "auth", "db_password.txt")
    if os.path.exists(pwd_path):
        with open(pwd_path, "r", encoding="utf-8") as f:
            pwd = f.read().strip()
    else:
        pwd = os.getenv("DB_PASSWORD", "")

    return {
        "dbname": dbname,
        "user": user,
        "host": host,
        "port": int(port),
        "password": pwd
    }


def get_pg_conn() -> Any:
    """Gets an active PostgreSQL database connection.

    Args:
        None

    Returns:
        A psycopg2 connection object connected to the PostgreSQL database.

    Raises:
        psycopg2.OperationalError: If connection to the database fails.
    """
    return psycopg2.connect(**get_pg_conn_params())



def filter_unprocessed_covers(conn: Any, cover_paths: Set[str]) -> Set[str]:
    """Checks the photos database and returns only cover paths that lack descriptions.

    Args:
        conn: An active PostgreSQL connection object.
        cover_paths: A set of absolute cover image file paths to filter.

    Returns:
        A set of cover image paths that are not yet described in the database.

    Raises:
        psycopg2.DatabaseError: If query execution fails.
    """
    if not cover_paths:
        return set()
        
    cursor = conn.cursor()
    cursor.execute(sql_loader.get_sql("queries/check_covers_exist.sql"), (list(cover_paths),))
    existing = {row[0] for row in cursor.fetchall()}
    return cover_paths - existing


def run_llm_curation(conn: Any, chunk_dirs: Optional[List[str]] = None, dry_run: bool = False, limit_dirs: Optional[int] = None, limit_batches: Optional[int] = None) -> None:
    """Triggers the LLM-synthesis database cleaner stage for a specific chunk or globally.

    Args:
        conn: An active PostgreSQL connection object.
        chunk_dirs: Optional list of directories representing the active chunk. If
            provided, curation is filtered strictly to tracks residing in these
            folders. If None, queries and curates all tracks with gaps globally.
        dry_run: If True, prints planned metadata database updates without committing.
        limit_dirs: Optional limit on the number of directories to process.
        limit_batches: Optional limit on the total number of batches to curate.

    Returns:
        None

    Raises:
        psycopg2.DatabaseError: If database updates fail.
    """
    if chunk_dirs:
        logger.info("[CURATION STAGE] Starting database metadata curation loop for this chunk...")
    else:
        logger.info("[GLOBAL CURATION SWEEP] Starting global database sweep for remaining metadata gaps...")
    
    cursor = conn.cursor()
    
    # Query tracks that have gaps (curate if artist, genre, or album is unknown)
    query = sql_loader.get_sql("queries/get_music_gaps.sql")
    cursor.execute(query)
    rows = cursor.fetchall()
    
    # Filter rows to only those residing in our current chunk directories (if specified)
    if chunk_dirs:
        chunk_dirs_set = set(os.path.normpath(d) for d in chunk_dirs)
        chunk_rows = [row for row in rows if os.path.normpath(os.path.dirname(row[0])) in chunk_dirs_set]
    else:
        chunk_rows = rows
        
    if not chunk_rows:
        if chunk_dirs:
            logger.info("[CURATION STAGE] No tracks with metadata gaps in this chunk's directories.")
        else:
            logger.info("[GLOBAL CURATION SWEEP] No remaining tracks with metadata gaps found in the database.")
        return

    if chunk_dirs:
        logger.info(f"[CURATION STAGE] Found {len(chunk_rows)} tracks with gaps in this chunk. Grouping by directory...")
    else:
        logger.info(f"[GLOBAL CURATION SWEEP] Found {len(chunk_rows)} remaining tracks with gaps in database. Grouping by directory...")
    
    # Group tracks by parent directory
    groups: Dict[str, Dict[str, Any]] = {}
    for file_path, title, artist, album, genre, album_art_path in chunk_rows:
        dir_path = os.path.dirname(file_path)
        if dir_path not in groups:
            groups[dir_path] = {
                "album_art": album_art_path,
                "album": album or "Unknown Album",
                "genre": genre or "Unknown Genre",
                "artist_orig": artist or "Unknown Artist",
                "tracks": []
            }
        groups[dir_path]["tracks"].append((file_path, title))

    sorted_dir_paths = sorted(groups.keys())
    sorted_dir_paths = sorted(groups.keys())
    if limit_dirs is not None:
        logger.info(f"Limiting curation sweep to the first {limit_dirs} directories out of {len(sorted_dir_paths)} directories with gaps.")
        sorted_dir_paths = sorted_dir_paths[:limit_dirs]

    # Probe and find all active curation servers
    from clean_database_artists import get_active_curation_servers
    active_servers = get_active_curation_servers()
    if not active_servers:
        logger.error("No active curation servers found!")
        return
        
    logger.info(f"Active curation servers for this run: {[s.name for s in active_servers]}")
    
    # Build a thread-safe task queue of chunks
    import queue
    import threading
    import time
    
    job_queue = queue.Queue()
    
    # We will need a lock to coordinate printing log lines and checking heuristic fallbacks
    print_lock = threading.Lock()
    any_llm_updates_by_dir = {dpath: False for dpath in sorted_dir_paths}
    
    for dir_path in sorted_dir_paths:
        info = groups[dir_path]
        album_art_path = info["album_art"]
        album_name = info["album"]
        genre_name = info["genre"]
        artist_name = info["artist_orig"]
        tracks = info["tracks"]
        dir_base = os.path.basename(dir_path)
        
        visual_desc = "No cover art available"
        visual_tags = []
        if album_art_path:
            cursor.execute(
                sql_loader.get_sql("queries/get_photo_metadata.sql"),
                (album_art_path,)
            )
            row = cursor.fetchone()
            if row:
                visual_desc = row[0] or "No cover art available"
                try:
                    visual_tags = json.loads(row[1]) if row[1] else []
                except Exception:
                    pass
                    
        TRACK_CHUNK_SIZE = 10
        import math
        total_chunks = math.ceil(len(tracks) / TRACK_CHUNK_SIZE)
        
        for chunk_idx, t_start in enumerate(range(0, len(tracks), TRACK_CHUNK_SIZE)):
            t_chunk = tracks[t_start : t_start + TRACK_CHUNK_SIZE]
            job_queue.put((
                dir_path, album_name, genre_name, t_chunk, visual_desc, visual_tags,
                chunk_idx, total_chunks, artist_name, dir_base
            ))
            if limit_batches is not None and job_queue.qsize() >= limit_batches:
                break
        if limit_batches is not None and job_queue.qsize() >= limit_batches:
            break
            
    total_jobs = job_queue.qsize()
    logger.info(f"Created {total_jobs} curation tasks. Distributing across {len(active_servers)} active servers...")

    def is_placeholder(val: str) -> bool:
        """Helper to detect placeholder values that should not be written to files."""
        if not val:
            return True
        val_lower = val.lower()
        return any(p in val_lower for p in ("unknown", "unresolved", "n/a", "temp"))

    # Worker thread logic
    def curation_worker(server) -> None:
        # Each thread gets its own PG connection for thread safety
        thread_conn = get_pg_conn()
        thread_cursor = thread_conn.cursor()
        
        while True:
            try:
                job = job_queue.get_nowait()
            except queue.Empty:
                break
                
            (dir_path, album_name, genre_name, t_chunk, visual_desc, visual_tags,
             chunk_idx, total_chunks, artist_name, dir_base) = job
             
            try:
                logger.info(f"[{server.name}] [CURATION INPUTS] Folder: '{album_name} ({dir_base})' | Current DB Tags: Artist='{artist_name}', Album='{album_name}', Genre='{genre_name}'")
                c_artist, c_album, c_genre, c_cleaned, c_track_updates = clean_database_artists.extract_group_metadata_via_llm(
                    dir_path, album_name, genre_name, t_chunk, visual_desc, visual_tags, server, len(active_servers)
                )
                
                if c_artist is None:
                    logger.warning(f"[{server.name}] [LLM FAILED] Could not resolve metadata for folder '{album_name} ({dir_base})'. Running immediate offline fallback...")
                    h_artist, h_album, h_genre, h_cleaned, h_method = clean_database_artists.resolve_group_metadata_offline(dir_path, t_chunk)
                    
                    update_query = sql_loader.get_sql("queries/update_music_track.sql")
                    for fpath, orig_title in t_chunk:
                        fname = os.path.basename(fpath)
                        t_title = h_cleaned.get(fname, orig_title)
                        logger.info(f"  -> [{server.name}] [POSTGRESQL UPDATE] [FALLBACK HEURISTIC ({h_method})] artist='{h_artist}', album='{h_album}', genre='{h_genre}', title='{t_title}' for path: {fpath}")
                        thread_cursor.execute(update_query, (h_artist, h_album, h_genre, t_title, fpath))
                        
                        # Direct physical file tag write
                        if not dry_run and fpath.lower().endswith(".flac") and os.path.exists(fpath):
                            try:
                                from mutagen.flac import FLAC
                                audio = FLAC(fpath)
                                modified = False
                                if h_artist and not is_placeholder(h_artist) and audio.get("artist", [""])[0] != h_artist:
                                    audio["artist"] = h_artist
                                    modified = True
                                if h_album and not is_placeholder(h_album) and audio.get("album", [""])[0] != h_album:
                                    audio["album"] = h_album
                                    modified = True
                                if h_genre and not is_placeholder(h_genre) and audio.get("genre", [""])[0] != h_genre:
                                    audio["genre"] = h_genre
                                    modified = True
                                if t_title and not is_placeholder(t_title) and audio.get("title", [""])[0] != t_title:
                                    audio["title"] = t_title
                                    modified = True
                                if modified:
                                    audio.save()
                                    logger.info(f"  -> [{server.name}] [FLAC TAGS WRITTEN] Saved heuristic tags to file: {os.path.basename(fpath)}")
                            except Exception as e:
                                logger.error(f"  -> [{server.name}] [FLAC TAGS FAILED] Failed to write tags to {fpath}: {e}")
                    thread_conn.commit()
                    logger.info(f"[{server.name}] [SUCCESS] [HEURISTIC FALLBACK] Committed batch {chunk_idx + 1}/{total_chunks} for folder '{album_name} ({dir_base})' ({len(t_chunk)} tracks)")
                    
                    with print_lock:
                        any_llm_updates_by_dir[dir_path] = True
                    continue
                
                resolved_chunk_artist = c_artist if c_artist else artist_name
                resolved_chunk_album = c_album if c_album else album_name
                resolved_chunk_genre = c_genre if c_genre else genre_name
                
                if resolved_chunk_artist == "Unknown Artist" and resolved_chunk_genre and any(g in resolved_chunk_genre.lower() for g in ["soundtrack", "score", "various"]):
                    resolved_chunk_artist = "Various Artists"
                    
                final_artist = resolved_chunk_artist
                final_album = resolved_chunk_album
                final_genre = resolved_chunk_genre
                
                if final_artist != "Unknown Artist" or final_album != "Unknown Album" or final_genre != "Unknown Genre" or c_track_updates:
                    with print_lock:
                        any_llm_updates_by_dir[dir_path] = True
                        
                    if dry_run:
                        for fpath, orig_title in t_chunk:
                            fname = os.path.basename(fpath)
                            t_up = c_track_updates.get(fname, {})
                            t_artist = t_up.get("artist", final_artist)
                            t_album = t_up.get("album", final_album)
                            t_genre = t_up.get("genre", final_genre)
                            t_title = t_up.get("title", c_cleaned.get(fname, orig_title))
                            logger.info(f"  -> [{server.name}] [DRY RUN] Would update: artist='{t_artist}', album='{t_album}', genre='{t_genre}', title='{t_title}' for path: {fpath}")
                    else:
                        update_query = sql_loader.get_sql("queries/update_music_track.sql")
                        for fpath, orig_title in t_chunk:
                            fname = os.path.basename(fpath)
                            t_up = c_track_updates.get(fname, {})
                            t_artist = t_up.get("artist", final_artist)
                            t_album = t_up.get("album", final_album)
                            t_genre = t_up.get("genre", final_genre)
                            t_title = t_up.get("title", c_cleaned.get(fname, orig_title))
                            logger.info(f"  -> [{server.name}] [POSTGRESQL UPDATE] artist='{t_artist}', album='{t_album}', genre='{t_genre}', title='{t_title}' for path: {fpath}")
                            thread_cursor.execute(update_query, (t_artist, t_album, t_genre, t_title, fpath))
                            
                            # Direct physical file tag write
                            if not dry_run and fpath.lower().endswith(".flac") and os.path.exists(fpath):
                                try:
                                    from mutagen.flac import FLAC
                                    audio = FLAC(fpath)
                                    modified = False
                                    if t_artist and not is_placeholder(t_artist) and audio.get("artist", [""])[0] != t_artist:
                                        audio["artist"] = t_artist
                                        modified = True
                                    if t_album and not is_placeholder(t_album) and audio.get("album", [""])[0] != t_album:
                                        audio["album"] = t_album
                                        modified = True
                                    if t_genre and not is_placeholder(t_genre) and audio.get("genre", [""])[0] != t_genre:
                                        audio["genre"] = t_genre
                                        modified = True
                                    if t_title and not is_placeholder(t_title) and audio.get("title", [""])[0] != t_title:
                                        audio["title"] = t_title
                                        modified = True
                                    if modified:
                                        audio.save()
                                        logger.info(f"  -> [{server.name}] [FLAC TAGS WRITTEN] Saved curated tags to file: {os.path.basename(fpath)}")
                                except Exception as e:
                                    logger.error(f"  -> [{server.name}] [FLAC TAGS FAILED] Failed to write tags to {fpath}: {e}")
                        thread_conn.commit()
                        logger.info(f"[{server.name}] [SUCCESS] Committed batch {chunk_idx + 1}/{total_chunks} for folder '{album_name} ({dir_base})' ({len(t_chunk)} tracks)")
                else:
                    logger.info(f"[{server.name}] [UNRESOLVED] Batch {chunk_idx + 1}/{total_chunks} for folder '{album_name} ({dir_base})' could not be resolved.")
            except Exception as e:
                logger.error(f"[{server.name}] Error processing batch {chunk_idx + 1} for folder '{album_name} ({dir_base})': {e}")
            finally:
                job_queue.task_done()
                
        thread_conn.close()

    # Spawn curation worker threads for each active server
    worker_threads = []
    for srv in active_servers:
        t = threading.Thread(target=curation_worker, args=(srv,), name=f"CurationWorker-{srv.name}")
        t.daemon = True
        t.start()
        worker_threads.append(t)
        
    # Wait for the worker queue to be fully completed
    while not job_queue.empty() or any(t.is_alive() for t in worker_threads):
        time.sleep(0.1)

    # Perform offline heuristic fallbacks for specific album directories that could not be resolved by the LLM
    generic_mixed_dirs = {"old", "music", "home videos", "downloads", "itunes", "dvd", "video_ts"}
    for dir_path in sorted_dir_paths:
        dir_base = os.path.basename(dir_path)
        dir_base_lower = dir_base.lower()
        if not any_llm_updates_by_dir[dir_path] and dir_base_lower not in generic_mixed_dirs:
            logger.info(f"  - Folder '{dir_base}' was not resolved by LLM. Attempting offline heuristic split fallback...")
            info = groups[dir_path]
            tracks = info["tracks"]
            h_artist, h_album, h_genre, h_cleaned, h_method = clean_database_artists.resolve_group_metadata_offline(dir_path, tracks)
            if h_artist != "Unknown Artist":
                if dry_run:
                    for fpath, orig_title in tracks:
                        logger.info(f"  -> [DRY RUN] [HEURISTIC] Would update: artist='{h_artist}', album='{h_album}', genre='{h_genre}', title='{h_cleaned.get(os.path.basename(fpath), orig_title)}' for path: {fpath}")
                else:
                    update_query = sql_loader.get_sql("queries/update_music_track.sql")
                    for fpath, orig_title in tracks:
                        fname = os.path.basename(fpath)
                        t_title = h_cleaned.get(fname, orig_title)
                        logger.info(f"  -> [POSTGRESQL UPDATE] [HEURISTIC] artist='{h_artist}', album='{h_album}', genre='{h_genre}', title='{t_title}' for path: {fpath}")
                        cursor.execute(update_query, (h_artist, h_album, h_genre, t_title, fpath))
                    conn.commit()
                    logger.info(f"[SUCCESS] [HEURISTIC] Database successfully committed resolved folder: '{dir_base}' ({len(tracks)} tracks)")


def normalize_database_genres(conn: Any) -> None:
    """Sweeps the database to resolve inconsistent genre tags for tracks sharing the same album.
    
    Prefers specific genres over generic ones (like 'Music', 'Unknown Genre', or NULL)
    and unifies tags dynamically.
    """
    logger.info("Starting database genre normalization sweep...")
    cursor = conn.cursor()
    try:
        # Fetch albums that have multiple different genres
        query = sql_loader.get_sql("queries/get_mismatched_album_genres.sql")
        cursor.execute(query)
        albums_mismatched = cursor.fetchall()
        
        updated_count = 0
        for album, genres in albums_mismatched:
            # Filter out generic or unknown genres
            filtered_genres = [g for g in genres if g and g.lower() not in ("music", "unknown genre", "unknown", "unresolved", "n/a", "other")]
            
            if not filtered_genres:
                # If all genres were generic, pick the first one (e.g. 'Music') or keep unchanged
                continue
                
            # Prefer the most common or specific genre. If multiple specific ones exist, pick the first one
            preferred_genre = filtered_genres[0]
            
            logger.info(f"Normalizing album '{album}': unifying genres {genres} -> '{preferred_genre}'")
            update_query = sql_loader.get_sql("queries/update_album_genres.sql")
            cursor.execute(update_query, (preferred_genre, album))
            updated_count += cursor.rowcount
            
        conn.commit()
        logger.info(f"Database genre normalization sweep completed. Unified {updated_count} tracks.")
    except Exception as e:
        conn.rollback()
        logger.warning(f"Database genre normalization sweep failed: {e}")


def process_single_folder(
    folder: str,
    existing_paths: Set[str]
) -> Tuple[List[Dict[str, Union[str, int]]], Optional[str]]:
    """Crawls a single folder, parses JRiver XML or embedded metadata using ExifTool, and detects cover art.

    Args:
        folder: Absolute path to the directory.
        existing_paths: Set of track file paths already imported in PostgreSQL.

    Returns:
        A tuple containing a list of dictionaries representing parsed track metadata payloads, 
        and the path to the cover art image (or None if not found).

    Raises:
        None
    """
    cover_path = ingest_music_library.find_album_art(folder)
    
    try:
        dir_files = os.listdir(folder)
    except OSError as e:
        logger.warning(f"Failed to list files in folder {folder}: {e}")
        return [], None
        
    folder_tracks: List[Dict[str, Union[str, int]]] = []
    has_new_files_in_folder = False
    
    for file in dir_files:
        ext = os.path.splitext(file)[1].lower()
        if ext in {".flac", ".wav", ".m4a", ".ape", ".flv", ".mp4", ".mkv", ".mov", ".mpg"}:
            file_path = os.path.join(folder, file)
            
            # Skip MP3 tracks as requested by the user
            if ext == ".mp3":
                continue
                
            # Skip files already in the database
            if file_path in existing_paths:
                continue
                
            if not has_new_files_in_folder:
                logger.info(f"Scanning new tracks in: {folder}")
                has_new_files_in_folder = True
                
            # Check for JRiver sidecar XML naming variations
            base_name, ext_val = os.path.splitext(file)
            xml_name1 = f"{base_name}_{ext_val[1:]}_JRSidecar.xml"
            xml_name2 = f"{file}_JRSidecar.xml"
            xml_path = ""
            for xml_name in (xml_name1, xml_name2):
                chk_path = os.path.join(folder, xml_name)
                if os.path.exists(chk_path):
                    xml_path = chk_path
                    break
                    
            if xml_path:
                # Sidecar XML exists - parse JRiver tags
                xml_data = ingest_music_library.parse_jr_sidecar(xml_path)
                title = xml_data.get("Name", base_name)
                artist = xml_data.get("Artist", "Unknown Artist")
                album = xml_data.get("Album", "Unknown Album")
                genre = xml_data.get("Genre", "Unknown Genre")
                rating_str = xml_data.get("Rating", "0")
                track_num_str = xml_data.get("Track #", "0")
            else:
                # No sidecar XML exists - read embedded tags directly using ExifTool
                tags = extract_metadata_via_exiftool(file_path)
                title = tags.get("Name", base_name)
                artist = tags.get("Artist", "Unknown Artist")
                album = tags.get("Album", "Unknown Album")
                genre = tags.get("Genre", "Unknown Genre")
                rating_str = tags.get("Rating", "0")
                track_num_str = tags.get("Track #", "0")
                
            try:
                rating = int(rating_str)
            except ValueError:
                rating = 0
            try:
                track_number = int(track_num_str)
            except ValueError:
                track_number = 0
                
            logger.info(f"  [PARSED] '{file}' -> Title='{title}', Artist='{artist}', Album='{album}', Genre='{genre}' (Art: {os.path.basename(cover_path) if cover_path else 'None'})")
            folder_tracks.append({
                "file_path": file_path,
                "title": title,
                "artist": artist,
                "album": album,
                "genre": genre,
                "track_number": track_number,
                "rating": rating,
                "album_art_path": cover_path or "",
                "jriver_genre": genre,
                "xml_metadata_path": xml_path or ""
            })
            
    return folder_tracks, cover_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified JRiver Ingest & Curation Pipeline.")
    parser.add_argument("--dir", default="D:\\Users\\steven\\Music", help="Root folder of the music library")
    parser.add_argument("--limit-dirs", type=int, default=None, help="Limit number of subdirectories to scan")
    parser.add_argument("--limit-batches", type=int, default=None, help="Limit total number of curation batches to process")
    parser.add_argument("--batch-size", type=int, default=3, help="GPU batch size")
    parser.add_argument("--max-workers", type=int, default=20, help="CPU loader workers")
    parser.add_argument("--dry-run", action="store_true", help="Display curation without committing")
    parser.add_argument("--curate-only", action="store_true", help="Skip crawling and VLM, only run metadata curation and normalization")
    parser.add_argument("--pipeline-chunk-size", type=int, default=50, help="Number of folders to process end-to-end in each batch")
    args = parser.parse_args()

    logger.info("=================================================================")
    logger.info("Starting Unified Music Ingest, VLM Scanning, and Curation Loop")
    logger.info("=================================================================")

    conn = get_pg_conn()

    # Step 0: Auto-revert any invalid 'I Will Wait For You' curations from previous runs
    logger.info("Auto-reverting any invalid 'I Will Wait For You' track records in PostgreSQL...")
    cursor = conn.cursor()
    cursor.execute(sql_loader.get_sql("queries/revert_wait_for_you.sql"))
    conn.commit()
    logger.info(f"Reverted {cursor.rowcount} incorrect track records.")

    if args.curate_only:
        logger.info("Running in curation-only mode. Skipping directory scanning and VLM processing...")
        run_llm_curation(conn, None, args.dry_run, args.limit_dirs, args.limit_batches)
        if not args.dry_run:
            normalize_database_genres(conn)
        conn.close()
        logger.info("Unified Cataloging Pipeline completed successfully.")
        return

    # Step 1: Discover all directories containing sidecars or audio files
    logger.info(f"Scanning library directories in '{args.dir}'...")
    candidate_dirs: Set[str] = set()
    audio_extensions = {".flac", ".wav", ".m4a", ".ape", ".flv", ".mp4", ".mkv", ".mov", ".mpg"}
    
    file_count = 0
    dir_count = 0
    import time
    last_print_time = time.time()
    
    for root, _, files in os.walk(args.dir):
        dir_count += 1
        file_count += len(files)
        
        current_time = time.time()
        if current_time - last_print_time >= 10.0:
            logger.info(f"  [SCAN PROGRESS] Visited {dir_count:,} folders, scanned {file_count:,} files... Found {len(candidate_dirs)} candidate audio directories.")
            last_print_time = current_time
            
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in audio_extensions or file.endswith("_JRSidecar.xml"):
                candidate_dirs.add(root)
                break
                
    logger.info(f"Scan complete. Visited {dir_count:,} folders, scanned {file_count:,} files total.")

    sorted_dirs = sorted(list(candidate_dirs))
    if args.limit_dirs:
        logger.info(f"Limiting scan to the first {args.limit_dirs} directories out of {len(sorted_dirs)} found.")
        sorted_dirs = sorted_dirs[:args.limit_dirs]
    else:
        logger.info(f"Found {len(sorted_dirs)} folders to scan.")

    # Step 1b: Process directories in end-to-end batches to write & curate incrementally
    import concurrent.futures
    PIPELINE_CHUNK_SIZE = args.pipeline_chunk_size
    total_pipeline_chunks = (len(sorted_dirs) - 1) // PIPELINE_CHUNK_SIZE + 1
    
    for p_idx, i in enumerate(range(0, len(sorted_dirs), PIPELINE_CHUNK_SIZE)):
        chunk_dirs = sorted_dirs[i : i + PIPELINE_CHUNK_SIZE]
        logger.info(f"\n=================================================================")
        logger.info(f"PROCESSING PIPELINE BATCH {p_idx + 1}/{total_pipeline_chunks} ({len(chunk_dirs)} folders)")
        logger.info(f"=================================================================")

        # A. Ingest tracks for this batch in parallel
        # Reload existing_paths to include what was just ingested or resolved
        cursor = conn.cursor()
        cursor.execute(sql_loader.get_sql("queries/get_all_track_paths.sql"))
        existing_paths = {row[0] for row in cursor.fetchall()}
        
        chunk_covers: Set[str] = set()
        chunk_new_tracks: List[Dict[str, Union[str, int]]] = []
        
        logger.info(f"Scanning and parsing folder metadata in parallel using {args.max_workers} worker threads...")
        completed_folders = 0
        total_folders = len(chunk_dirs)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_folder = {executor.submit(process_single_folder, folder, existing_paths): folder for folder in chunk_dirs}
            
            for future in concurrent.futures.as_completed(future_to_folder):
                completed_folders += 1
                if completed_folders % 10 == 0 or completed_folders == total_folders:
                    logger.info(f"  [INGEST PROGRESS] Processed {completed_folders}/{total_folders} folders...")
                folder = future_to_folder[future]
                try:
                    tracks_list, cover_path = future.result()
                    if cover_path:
                        chunk_covers.add(cover_path)
                    if tracks_list:
                        chunk_new_tracks.extend(tracks_list)
                except Exception as e:
                    logger.error(f"Error processing folder '{folder}' during parallel ingestion: {e}")
                    
        if chunk_new_tracks:
            logger.info(f"Found {len(chunk_new_tracks)} new tracks to ingest. Batch writing to PostgreSQL in chunks of 100...")
            CHUNK_SIZE = 100
            batch_idx = 1
            total_batches = (len(chunk_new_tracks) - 1) // CHUNK_SIZE + 1
            for j in range(0, len(chunk_new_tracks), CHUNK_SIZE):
                sub_chunk = chunk_new_tracks[j : j + CHUNK_SIZE]
                logger.info(f"Ingesting batch of {len(sub_chunk)} tracks into PostgreSQL (Batch {batch_idx}/{total_batches})...")
                ingest_music_library.batch_ingest_tracks("postgresql", get_pg_conn_params(), sub_chunk)
                batch_idx += 1
        else:
            logger.info("All tracks in this batch are already in the database.")
            
        # B. Run VLM cover scanning for covers in this batch
        unprocessed_covers = filter_unprocessed_covers(conn, chunk_covers)
        if unprocessed_covers:
            logger.info(f"Found {len(unprocessed_covers)} unprocessed cover images in this batch. Calling VLM describer...")
            
            vlm_cmd = [
                sys.executable,
                "describe_photos.py",
                "--max-photos", str(len(unprocessed_covers)),
                "--batch-size", str(args.batch_size),
                "--max-workers", str(args.max_workers),
                "--no-json"
            ]
            
            unprocessed_folders = sorted(list(set(os.path.dirname(c) for c in unprocessed_covers)))
            estimated_len = len(" ".join(vlm_cmd)) + sum(len(f) + 8 for f in unprocessed_folders)
            
            if estimated_len < 6000:
                for folder in unprocessed_folders:
                    vlm_cmd.extend(["--dir", folder])
            else:
                vlm_cmd.extend(["--dir", args.dir])
                
            logger.info(f"Executing VLM describer for batch: {' '.join(vlm_cmd)}")
            subprocess.run(vlm_cmd, input=b"n\n", check=True)
        else:
            logger.info("All cover art images in this batch are already described.")
            
        # C. Run LLM Curation for this batch's directories
        logger.info(f"Starting database metadata curation for batch {p_idx + 1}...")
        run_llm_curation(conn, chunk_dirs, args.dry_run, None, args.limit_batches)

    # Step 4: Run final global sweep to curate any leftover tracks
    run_llm_curation(conn, None, args.dry_run, None, args.limit_batches)

    # Step 5: Run final database genre normalization sweep
    if not args.dry_run:
        normalize_database_genres(conn)

    conn.close()
    logger.info("Unified Cataloging Pipeline completed successfully.")


if __name__ == "__main__":
    main()
