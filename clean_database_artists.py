"""PostgreSQL Music Tracks Artist Metadata Cleaner.

Purpose:
    This script queries the active PostgreSQL catalog for music track records
    with incomplete metadata (where Artist, Album, or Genre is 'Unknown' or NULL),
    groups them by parent folder, synthesizes their true identity (Artist, Album,
    Genre, and Clean Titles) using a single LLM call per album, and commits
    the corrected values to the database.

Architecture and Mechanics:
    1. Database Retrieval: Pulls tracks with gaps in Artist, Album, or Genre.
    2. Directory Grouping: Groups tracks by their parent folder.
    3. Cover Art Association: Fetches associated cover artwork paths (including PDF-extracted).
    4. Group VLM / LLM Synthesis (Primary):
       - Queries the local Gemma model (/analyze) with the consolidated directory details,
         original JRiver metadata, cover visual descriptions, and complete tracklist.
       - The LLM reasons over these fields as a whole to output the true Artist, Album name,
         Genre, and cleaned track titles as a JSON block.
    5. Offline Fallback Heuristics: Uses deterministic filename split if LLM fails.
    6. Database Update: Bulk-commits updates to 'music_tracks' (artist, album, genre, title).

Execution Modes:
    - Command Line:
      python clean_database_artists.py [--dry-run]
"""

import os
import sys
import re
import json
import logging
import argparse
import requests
import psycopg2
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional, Any

# Ensure standard UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

PROJECT_DIR: str = os.path.dirname(os.path.abspath(__file__))

# Configure logging to show all details with timestamps
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

LOG_FILE: str = os.path.join(PROJECT_DIR, "gemma_cataloger.log")
file_log = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_log.setFormatter(formatter)
logger.addHandler(file_log)


def get_pg_connection() -> Any:
    """Creates a connection to the PostgreSQL database using env configurations."""
    load_dotenv(os.path.join(PROJECT_DIR, "auth", ".env"))
    db_name = os.getenv("DB_NAME", "photo_catalog")
    db_host = os.getenv("DB_HOST", "localhost")
    logger.info(f"[DB CONNECT] Connecting to PostgreSQL database '{db_name}' on '{db_host}'...")
    return psycopg2.connect(
        dbname=db_name,
        user=os.getenv("DB_USER", "postgres"),
        host=db_host,
        port=int(os.getenv("DB_PORT", "5432")),
        password=os.getenv("DB_PASSWORD", "")
    )


def is_valid_artist_name(name: str) -> bool:
    """Checks if a resolved artist name is clean and valid."""
    if not name:
        logger.debug("[VALIDATION] Artist candidate rejected: Empty or None")
        return False
    if name == "Unknown Artist":
        logger.debug("[VALIDATION] Artist candidate rejected: matches 'Unknown Artist'")
        return False
    if len(name) > 50:
        logger.info(f"[VALIDATION] Rejected candidate name '{name}' (Too long: {len(name)} chars, max 50)")
        return False
        
    lower_name = name.lower()
    junk_patterns = [
        r"\.mkv", r"\.mp4", r"\.avi", r"\.flv", r"\.flac", r"\.mp3", r"\.mov", r"\.mpg", r"\.iso",
        r"1080p", r"720p", r"bdrip", r"webrip", r"x264", r"h264", r"dvd", r"cdrip",
        r"video_ts", r"cbfm", r"sam tbs", r"nl subs", r"season", r"episode",
        r"part\d+", r"part \d+", r"vol\d+", r"vol \d+", r"track\d+", r"track \d+",
        r"disc\d+", r"disc \d+", r"\d+gb", r"\d+mb"
    ]
    for pattern in junk_patterns:
        if re.search(pattern, lower_name):
            logger.info(f"[VALIDATION] Rejected candidate name '{name}' (Matches junk filter pattern: '{pattern}')")
            return False
            
    if name.strip().isdigit() and len(name.strip()) <= 2:
        logger.info(f"[VALIDATION] Rejected candidate name '{name}' (Identified as simple track/disc number digits)")
        return False
        
    logger.info(f"[VALIDATION] Approved artist candidate name: '{name}'")
    return True


ACTIVE_LLM_URL: Optional[str] = None

class CurationServer:
    """Manages connections and status checks for curation LLM/VLM servers."""
    def __init__(self, name: str, url: str, mode: str, model_name: str = "gemma4-it-q4:latest") -> None:
        self.name = name
        self.url = url
        self.mode = mode
        self.model_name = model_name
        
    def is_alive(self) -> bool:
        """Tests server responsiveness with a lightweight status probe."""
        try:
            if self.mode == "ollama":
                res = requests.get(f"{self.url.replace('/api/generate', '/api/version')}", timeout=1.5)
                return res.status_code == 200
            else:
                res = requests.get(f"{self.url.replace('/analyze', '/docs')}", timeout=1.5)
                return res.status_code == 200
        except Exception:
            return False

# Load environment variables from the auth/.env file
load_dotenv(os.path.join(os.path.dirname(__file__), "auth", ".env"))

# Retrieve server URLs dynamically from environment variables
env_ollama_host = os.getenv("OLLAMA_HOST", "http://192.168.8.156:11434/api/generate")
env_fallback_host = os.getenv("FALLBACK_SERVER_URL", "http://192.168.8.193:11434/api/generate")
env_local_ollama = os.getenv("LOCAL_OLLAMA_HOST", "http://127.0.0.1:11434/api/generate")

# Normalize endpoints (ensure Ollama endpoints use /api/generate)
if "/api/chat" in env_ollama_host:
    env_ollama_host = env_ollama_host.replace("/api/chat", "/api/generate")
if "/api/chat" in env_fallback_host:
    env_fallback_host = env_fallback_host.replace("/api/chat", "/api/generate")
if "/api/chat" in env_local_ollama:
    env_local_ollama = env_local_ollama.replace("/api/chat", "/api/generate")

# Ensure generate suffix is present on Ollama URLs
for url_var in ("env_ollama_host", "env_fallback_host", "env_local_ollama"):
    val = locals()[url_var]
    if val and not val.endswith("/api/generate") and not val.endswith("/api/generate/"):
        locals()[url_var] = val.rstrip("/") + "/api/generate"

# Re-assign normalized local variables
env_ollama_host = locals()["env_ollama_host"]
env_fallback_host = locals()["env_fallback_host"]
env_local_ollama = locals()["env_local_ollama"]

ALL_CURATION_SERVERS: List[CurationServer] = [
    CurationServer("Remote Ollama (Giga)", env_fallback_host, "ollama", "gemma4-it-q4:latest"),
    CurationServer("Remote Ollama (Lenovo)", env_ollama_host, "ollama", "gemma2-2b-custom:latest"),
    CurationServer("Local Ollama", env_local_ollama, "ollama", "gemma4-it-q4:latest"),
    CurationServer("Remote VLM (Dell)", "http://192.168.8.113:8000/analyze", "vlm"),
    CurationServer("Local VLM (Workstation)", "http://127.0.0.1:8000/analyze", "vlm")
]

def get_active_curation_servers() -> List[CurationServer]:
    """Probes and returns all currently active curation servers in the network.
    Prioritizes Ollama servers to maximize performance and enable real-time token streaming.
    Bypasses VLM servers (including local VLM) for text curation if any Ollama server is responsive.
    """
    active_ollama: List[CurationServer] = []
    active_vlm: List[CurationServer] = []
    for srv in ALL_CURATION_SERVERS:
        logger.info(f"[PROBE] Checking curation server availability: {srv.name}...")
        if srv.is_alive():
            logger.info(f"[PROBE SUCCESS] Server {srv.name} is active.")
            if srv.mode == "ollama":
                active_ollama.append(srv)
            else:
                active_vlm.append(srv)
        else:
            logger.info(f"[PROBE FAILED] Server {srv.name} is offline.")
            
    active_servers = active_ollama.copy()
    
    # Always include the local VLM server if it is active, since the local RTX 5080 is the fastest GPU
    for srv in active_vlm:
        if "local" in srv.name.lower() or "workstation" in srv.name.lower():
            active_servers.append(srv)
            
    if active_servers:
        logger.info(f"[ROUTING] Using active curation server(s): {[s.name for s in active_servers]}")
        return active_servers
    else:
        logger.info(f"[ROUTING] Falling back to slow remote VLM server(s) for curation: {[s.name for s in active_vlm]}")
        return active_vlm

ACTIVE_LLM_URL: Optional[str] = None

def get_active_llm_url() -> str:
    """Probes available hosts and returns the active server endpoint URL."""
    global ACTIVE_LLM_URL
    if ACTIVE_LLM_URL is None:
        active = get_active_curation_servers()
        if active:
            ACTIVE_LLM_URL = active[0].url
        else:
            ACTIVE_LLM_URL = "http://127.0.0.1:8000/analyze"
    return ACTIVE_LLM_URL


def extract_group_metadata_via_llm(
    dir_path: str,
    album_name: str,
    genre_name: str,
    tracks: List[Tuple[str, str]],
    visual_desc: str, 
    visual_tags: List[str],
    server: Optional[Any] = None,
    active_server_count: int = 1
) -> Tuple[Optional[str], Optional[str], Optional[str], Dict[str, str], Dict[str, Dict[str, Any]]]:
    """Queries the local Gemma server to synthesize artist, album, genre, and titles."""
    dir_base = os.path.basename(dir_path)
    logger.info(f"[LLM SYNTHESIS START] Analyzing directory '{album_name} ({dir_base})' with {len(tracks)} tracks...")
    
    # Load prompt dynamically from music_curation_prompt.txt
    prompt_file = os.path.join(os.path.dirname(__file__), "music_curation_prompt.txt")
    if os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            template = f.read()
    else:
        logger.warning(f"Curation prompt file not found at {prompt_file}. Falling back to default prompt template.")
        template = (
            "Analyze the files and directory path to determine if this folder contains a musical release.\n"
            "Parent Directory Path: {dir_path}\n"
            "JRiver Album Name: {album_name}\n"
            "JRiver Genre Name: {genre_name}\n"
            "Cover Art Visual Description: {visual_desc}\n"
            "Cover Art Suggested Tags: {visual_tags}\n\n"
            "Complete Track List in this Directory:\n{track_list}\n"
            "Respond strictly with a JSON block in this exact structure:\n"
            "{\n"
            "  \"is_mixed_or_various\": true,\n"
            "  \"default_artist\": \"...\",\n"
            "  \"default_album\": \"...\",\n"
            "  \"default_genre\": \"...\",\n"
            "  \"tracks\": [\n"
            "    {\"file\": \"filename.ext\", \"is_music\": true, \"title\": \"...\", \"artist\": \"...\", \"album\": \"...\", \"genre\": \"...\"}\n"
            "  ]\n"
            "}"
        )

    track_list_str = ""
    for fpath, title in tracks:
        track_list_str += f"  - File: '{os.path.basename(fpath)}' | Title: '{title}'\n"
        
    prompt = template
    prompt = prompt.replace("{dir_path}", dir_path)
    prompt = prompt.replace("{album_name}", album_name)
    prompt = prompt.replace("{genre_name}", genre_name)
    prompt = prompt.replace("{visual_desc}", visual_desc)
    prompt = prompt.replace("{visual_tags}", ", ".join(visual_tags))
    prompt = prompt.replace("{track_list}", track_list_str)

    if server is None:
        url = get_active_llm_url()
        mode = "ollama" if "/api/generate" in url else "vlm"
        matching_srv = next((s for s in ALL_CURATION_SERVERS if s.url == url), None)
        model_name = matching_srv.model_name if matching_srv else "gemma4-it-q4:latest"
        class SimpleServer:
            def __init__(self, name: str, url: str, mode: str, model_name: str) -> None:
                self.name = name
                self.url = url
                self.mode = mode
                self.model_name = model_name
        server = SimpleServer("Default Server", url, mode, model_name)

    llm_url = server.url
    is_ollama = server.mode == "ollama"
    should_stream = is_ollama and (active_server_count == 1)
    
    if is_ollama:
        payload = {
            "model": server.model_name,
            "prompt": prompt,
            "stream": should_stream,
            "options": {
                "temperature": 0.7,
                "num_predict": 2048
            }
        }
    else:
        payload = {
            "prompt_text": prompt,
            "temperature": 0.7,
            "max_new_tokens": 2048
        }
    
    logger.info(f"\n--- [LLM PROMPT SENT TO {server.name}] ---\n{prompt}\n-----------------------------------------")
    logger.info(f"[LLM REQUEST] Submitting prompt payload for '{album_name} ({dir_base})' ({len(tracks)} tracks) to server at {llm_url}...")
    
    # Spawn background heartbeat logger to show progress during blocking inference wait (VLM only)
    import threading
    import time
    
    stop_event = threading.Event()
    def heartbeat_worker() -> None:
        start_time = time.time()
        while not stop_event.is_set():
            time.sleep(5.0)
            if stop_event.is_set():
                break
            elapsed = int(time.time() - start_time)
            logger.info(f"  [{server.name}] Curation in progress for '{album_name} ({dir_base})' | Elapsed: {elapsed}s")
            
    heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
    if not should_stream:
        heartbeat_thread.start()
    
    try:
        response_container = []
        def make_request() -> None:
            try:
                if is_ollama:
                    if should_stream:
                        res = requests.post(llm_url, json=payload, stream=True, timeout=300.0)
                        full_response = ""
                        sys.stdout.write("\n--- [OLLAMA REAL-TIME STREAMING RESPONSE] ---\n")
                        sys.stdout.flush()
                        for line in res.iter_lines():
                            if stop_event.is_set():
                                break
                            if line:
                                chunk_data = json.loads(line.decode('utf-8'))
                                text_chunk = chunk_data.get("response", "")
                                full_response += text_chunk
                                sys.stdout.write(text_chunk)
                                sys.stdout.flush()
                        sys.stdout.write("\n--- [OLLAMA STREAMING END] ---\n")
                        sys.stdout.flush()
                    else:
                        non_stream_payload = payload.copy()
                        non_stream_payload["stream"] = False
                        res = requests.post(llm_url, json=non_stream_payload, timeout=300.0)
                        full_response = res.json().get("response", "")
                    
                    class MockResponse:
                        def __init__(self, text: str, status_code: int) -> None:
                            self.text = text
                            self.status_code = status_code
                        def json(self) -> dict:
                            return {"response": self.text}
                            
                    response_container.append(MockResponse(full_response, 200))
                else:
                    res = requests.post(llm_url, json=payload, timeout=300.0)
                    response_container.append(res)
            except Exception as err:
                response_container.append(err)
                
        req_thread = threading.Thread(target=make_request, daemon=True)
        req_thread.start()
        
        try:
            while req_thread.is_alive():
                time.sleep(0.1) # Yield execution slices to ensure instant Ctrl+C interruption on Windows
        finally:
            stop_event.set()
            if not should_stream:
                heartbeat_thread.join(timeout=1.0)
            
        if not response_container:
            raise RuntimeError("Request thread exited without a response or error.")
            
        response = response_container[0]
        if isinstance(response, Exception):
            raise response
            
        if response.status_code == 200:
            raw_res: str = response.json().get("response", "").strip()
            logger.info(f"\n--- [LLM RAW RESPONSE FROM {server.name}] ---\n{raw_res}\n-----------------------------------------")
            
            if raw_res.startswith("```json"):
                raw_res = raw_res[7:]
            if raw_res.endswith("```"):
                raw_res = raw_res[:-3]
            raw_res = raw_res.strip()
            
            try:
                res_data = json.loads(raw_res)
            except json.JSONDecodeError as e:
                import ast
                try:
                    res_data = ast.literal_eval(raw_res)
                except Exception:
                    raise e
            
            # Extract defaults
            default_artist = res_data.get("default_artist", "").strip()
            default_album = res_data.get("default_album", "").strip()
            default_genre = res_data.get("default_genre", "").strip()
            
            if default_artist.lower() in ("unknown artist", "unknown", "unresolved", "n/a", ""):
                default_artist = "Unresolved Artist"
            if default_album.lower() in ("unknown album", "unknown", "unresolved", "n/a", ""):
                default_album = "Unresolved Album"
            if default_genre.lower() in ("unknown genre", "unknown", "unresolved", "n/a", ""):
                default_genre = "Unresolved Genre"
                
            track_updates: Dict[str, Dict[str, Any]] = {}
            cleaned_map: Dict[str, str] = {}
            
            # Try to parse track list if present
            if "tracks" in res_data and isinstance(res_data["tracks"], list):
                for item in res_data["tracks"]:
                    fname = item.get("file")
                    if not fname:
                        continue
                    is_music_track = item.get("is_music", True)
                    t_title = item.get("title", "")
                    t_artist = item.get("artist", "").strip()
                    t_album = item.get("album", "").strip()
                    t_genre = item.get("genre", "").strip()
                    
                    if not t_artist:
                        t_artist = default_artist
                    if not t_album:
                        t_album = default_album
                    if not t_genre:
                        t_genre = default_genre
                        
                    if not is_music_track:
                        t_artist = "Non-Music"
                        t_album = "Non-Music"
                        t_genre = "Non-Music"
                        
                    if t_artist.lower() in ("unknown artist", "unknown", "unresolved", "n/a", ""):
                        t_artist = "Unresolved Artist"
                    if t_album.lower() in ("unknown album", "unknown", "unresolved", "n/a", ""):
                        t_album = "Unresolved Album"
                    if t_genre.lower() in ("unknown genre", "unknown", "unresolved", "n/a", ""):
                        t_genre = "Unresolved Genre"
                        
                    track_updates[fname] = {
                        "artist": t_artist,
                        "album": t_album,
                        "genre": t_genre,
                        "title": t_title
                    }
                    cleaned_map[fname] = t_title
            
            # Fallback for old schema format output compatibility
            if not track_updates and ("cleaned_tracks" in res_data or "is_music" in res_data):
                is_music = res_data.get("is_music", True)
                old_artist = res_data.get("artist", default_artist).strip()
                old_album = res_data.get("album", default_album).strip()
                old_genre = res_data.get("genre", default_genre).strip()
                
                if not is_music:
                    old_artist = "Non-Music"
                    old_album = "Non-Music"
                    old_genre = "Non-Music"

                if old_artist.lower() in ("unknown artist", "unknown", "unresolved", "n/a", ""):
                    old_artist = "Unresolved Artist"
                if old_album.lower() in ("unknown album", "unknown", "unresolved", "n/a", ""):
                    old_album = "Unresolved Album"
                if old_genre.lower() in ("unknown genre", "unknown", "unresolved", "n/a", ""):
                    old_genre = "Unresolved Genre"
                    
                for item in res_data.get("cleaned_tracks", []):
                    fname = item.get("file")
                    t_title = item.get("title", "")
                    if fname:
                        track_updates[fname] = {
                            "artist": old_artist,
                            "album": old_album,
                            "genre": old_genre,
                            "title": t_title
                        }
                        cleaned_map[fname] = t_title
                        
            # Populate any missing tracks from the input tracks list using defaults
            for fpath, orig_title in tracks:
                fname = os.path.basename(fpath)
                if fname not in track_updates:
                    track_updates[fname] = {
                        "artist": default_artist,
                        "album": default_album,
                        "genre": default_genre,
                        "title": cleaned_map.get(fname, orig_title)
                    }
                    
            # Create a clean text-wrapped display card of the LLM returns
            width = 80
            border = "-" * width
            card_lines = [
                border,
                f" LLM CURATION RESULTS FOR FOLDER: '{album_name} ({dir_base})' ".center(width, "="),
                border,
                f"  Default Artist: {default_artist}",
                f"  Default Album:  {default_album}",
                f"  Default Genre:  {default_genre}",
                border,
                "  Track-by-Track Details:",
            ]
            for fname, details in track_updates.items():
                t_title = details.get("title", "")
                t_art = details.get("artist", default_artist)
                t_alb = details.get("album", default_album)
                t_gen = details.get("genre", default_genre)
                
                card_lines.append(f"    * File: '{fname}'")
                card_lines.append(f"      -> Title:  '{t_title}'")
                card_lines.append(f"         Artist: '{t_art}' | Album: '{t_alb}' | Genre: '{t_gen}'")
            card_lines.append(border)
            
            logger.info("\n" + "\n".join(card_lines))
            
            return default_artist, default_album, default_genre, cleaned_map, track_updates
        else:
            logger.error(f"[LLM HTTP ERROR] Local VLM returned status code {response.status_code}")
    except Exception as e:
        logger.warning(f"[LLM FAILURE] Failed to query local LLM server for directory {dir_base}: {e}")
        
    return None, None, None, {}, {}


def resolve_group_metadata_offline(
    dir_path: str, 
    tracks: List[Tuple[str, str]]
) -> Tuple[str, Optional[str], Optional[str], Dict[str, str], str]:
    """Applies offline heuristics to determine metadata for the group."""
    dir_base = os.path.basename(dir_path)
    logger.info(f"[HEURISTIC RUN] Running offline checks for directory: '{dir_base}'...")
    
    artist: str = "Unknown Artist"
    album: Optional[str] = None
    genre: Optional[str] = None
    cleaned_map: Dict[str, str] = {}
    method: str = "Unresolved"

    # Heuristic Fallback: Check track title separator dashes (Safe and deterministic)
    for fpath, title in tracks:
        if " - " in title:
            parts = [p.strip() for p in title.split(" - ", 1)]
            if len(parts) == 2 and is_valid_artist_name(parts[0]):
                artist = parts[0]
                method = "Track Title Separator Split"
                logger.info(f"[HEURISTIC SUCCESS] Found artist '{artist}' by splitting track title '{title}'")
                break

    # If an artist was resolved, clean titles by removing prefix
    if artist != "Unknown Artist":
        for fpath, title in tracks:
            fname = os.path.basename(fpath)
            clean_title = title
            prefix = f"{artist} - "
            if title.startswith(prefix):
                clean_title = title[len(prefix):]
            cleaned_map[fname] = clean_title
            logger.info(f"  - Cleaned title for '{fname}': '{title}' -> '{clean_title}'")

    if artist == "Unknown Artist" or artist.lower() in ("unknown artist", "unknown", "unresolved", "n/a", ""):
        artist = "Unresolved Artist"
    if album is None or album.lower() in ("unknown album", "unknown", "unresolved", "n/a", ""):
        album = "Unresolved Album"
    if genre is None or genre.lower() in ("unknown genre", "unknown", "unresolved", "n/a", ""):
        genre = "Unresolved Genre"

    return artist, album, genre, cleaned_map, method


def main() -> None:
    """Orchestrates database queries, groups tracks, updates records, and prints summaries."""
    parser = argparse.ArgumentParser(description="Clean gaps in PostgreSQL music catalog metadata.")
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Display changes without updating the database."
    )
    args = parser.parse_args()

    try:
        conn = get_pg_connection()
        cursor = conn.cursor()
    except Exception as e:
        logger.critical(f"Failed to connect to PostgreSQL: {e}")
        return

    # Select all tracks with any incomplete metadata fields
    logger.info("[DB FETCH] Querying PostgreSQL for tracks with missing Artist, Album, or Genre fields...")
    query = """
        SELECT file_path, title, artist, album, genre, album_art_path 
        FROM music_tracks 
        WHERE artist = 'Unknown Artist' OR artist IS NULL
           OR album = 'Unknown Album' OR album IS NULL
           OR genre = 'Unknown Genre' OR genre IS NULL;
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    if not rows:
        logger.info("[SUCCESS] No records with incomplete metadata found in the database. Everything is clean!")
        conn.close()
        return

    logger.info(f"[DB RETRIEVED] Loaded {len(rows)} database records with metadata gaps. Grouping by directory...")

    # Group tracks by parent directory path
    groups: Dict[str, Dict[str, Any]] = {}
    for file_path, title, artist, album, genre, album_art_path in rows:
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

    logger.info(f"[GROUPING COMPLETE] Grouped into {len(groups)} distinct directories. Commencing batch VLM/LLM synthesis...")

    updates: List[Tuple[str, str, str, str, str]] = []

    for dir_path, info in groups.items():
        album_art_path = info["album_art"]
        album_name = info["album"]
        genre_name = info["genre"]
        artist_name = info["artist_orig"]
        tracks = info["tracks"]
        dir_base = os.path.basename(dir_path)
        
        logger.info(f"\n[DIR PROCESSING] Processing folder: '{dir_base}' (Original tags: Artist='{artist_name}', Album='{album_name}', Genre='{genre_name}')")
        
        resolved_artist = None
        resolved_album = None
        resolved_genre = None
        cleaned_map: Dict[str, str] = {}
        method = "Unresolved"
        
        # Step 1: Query VLM Cover Art Details and run LLM Synthesis (Primary)
        if album_art_path:
            logger.info(f"  - Cover art path registered: '{album_art_path}'")
            logger.info(f"  - Querying photos table for cover metadata...")
            cursor.execute(
                "SELECT primary_subject, suggested_tags FROM photos WHERE full_path = %s",
                (album_art_path,)
            )
            row = cursor.fetchone()
            if row:
                visual_desc: str = row[0] or ""
                visual_tags: List[str] = []
                try:
                    visual_tags = json.loads(row[1]) if row[1] else []
                except Exception:
                    pass
                logger.info(f"  - Found cover art in photos: Desc Length={len(visual_desc)} chars, Tags={len(visual_tags)}")
                
                llm_artist, llm_album, llm_genre, llm_cleaned = extract_group_metadata_via_llm(
                    dir_path, album_name, genre_name, tracks, visual_desc, visual_tags
                )
                if llm_artist and is_valid_artist_name(llm_artist):
                    resolved_artist = llm_artist
                    resolved_album = llm_album
                    resolved_genre = llm_genre
                    cleaned_map = llm_cleaned
                    method = "LLM Directory Group VLM Synthesis"
            else:
                logger.info("  - Cover art path not found in photos table. Visual description is unavailable.")
        else:
            logger.info("  - No cover art registered for this group. Visual description is unavailable.")

        # Step 2: Fallback to offline heuristics if LLM failed
        if not resolved_artist:
            logger.info("  - LLM synthesis was unable to resolve artist. Running offline heuristic fallbacks...")
            h_artist, h_album, h_genre, h_cleaned, h_method = resolve_group_metadata_offline(dir_path, tracks)
            if h_artist != "Unknown Artist":
                resolved_artist = h_artist
                resolved_album = h_album
                resolved_genre = h_genre
                cleaned_map = h_cleaned
                method = h_method

        # Establish final values to write
        final_artist = resolved_artist if resolved_artist else artist_name
        final_album = resolved_album if resolved_album else album_name
        final_genre = resolved_genre if resolved_genre else genre_name

        if final_artist != "Unknown Artist" or final_album != "Unknown Album" or final_genre != "Unknown Genre":
            logger.info(f"[DECISION - CURATED] Method: {method} resolved metadata for '{dir_base}'")
            logger.info(f"  -> Artist: '{artist_name}' -> '{final_artist}'")
            logger.info(f"  -> Album:  '{album_name}' -> '{final_album}'")
            logger.info(f"  -> Genre:  '{genre_name}' -> '{final_genre}'")
            
            for fpath, orig_title in tracks:
                fname = os.path.basename(fpath)
                clean_title = cleaned_map.get(fname, orig_title)
                logger.info(f"  -> Preparing DB update tuple: artist='{final_artist}', album='{final_album}', genre='{final_genre}', title='{clean_title}' for path: {fpath}")
                updates.append((final_artist, final_album, final_genre, clean_title, fpath))
        else:
            logger.info(f"[DECISION - UNRESOLVED] Could not resolve any metadata gaps for folder: '{dir_base}'")
            for fpath, orig_title in tracks:
                logger.info(f"  -> Record unchanged: '{orig_title}'")

    if updates:
        if args.dry_run:
            logger.info(f"\n[DRY RUN] Would commit {len(updates)} metadata updates to PostgreSQL database. Transaction skipped.")
        else:
            logger.info(f"\n[DB COMMIT] Committing {len(updates)} metadata updates to PostgreSQL...")
            update_query = """
                UPDATE music_tracks 
                SET artist = %s, album = %s, genre = %s, title = %s 
                WHERE file_path = %s;
            """
            for final_artist, final_album, final_genre, clean_title, file_path in updates:
                logger.info(f"[DB WRITE] Executing UPDATE: artist='{final_artist}', album='{final_album}', genre='{final_genre}', title='{clean_title}' WHERE file_path='{file_path}'")
                cursor.execute(update_query, (final_artist, final_album, final_genre, clean_title, file_path))
            conn.commit()
            logger.info(f"[SUCCESS] Database successfully committed. {len(updates)} records curated.")
    else:
        logger.info("\n[FINISH] No records could be corrected. Visual metadata and filename parsing did not yield corrections.")

    conn.close()


if __name__ == "__main__":
    main()
