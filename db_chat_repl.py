"""Interactive CLI REPL client for the photo catalog database.

Purpose:
    This script provides an interactive Read-Eval-Print Loop (REPL) CLI chat client
    connected to the local photo catalog SQLite database (photo_catalog.db) and
    the offline Gemma 4 VLM. It enables users to ask natural language questions about
    the photo catalog, which the VLM answers by generating and executing SQL queries.

Architecture and Mechanics:
    1. WSL2 Server Control: Integrates with wsl_client to start and keep the vision
       server alive during the session.
    2. Dynamic Prompting: Reloads 'db_prompt.txt' dynamically on each query.
    3. Tool Call Parser: Parses '<tool_call>{"tool": "query_db", "sql": "..."}</tool_call>'
       blocks using regular expressions.
    4. SQLite Executor: Connects to 'photo_catalog.db' in read-only mode to retrieve
       records, formatting the results as clean markdown tables.
    5. Agent Loop: Runs a multi-step completion loop (up to 5 turns) to let the model
       reason over query results before delivering the final response.

Execution Modes:
    - Interactive CLI Shell: Run from a console terminal to start the chat loop.
      Command:
        python db_chat_repl.py
"""

import os
import sys
import textwrap
import re
import json
import sqlite3
import signal
import datetime
import requests
import psycopg2
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple, Any

# Import local wsl client module to manage model lifecycle
import wsl_client

# Load workspace environment variables
if os.path.exists("auth/.env"):
    load_dotenv("auth/.env")
else:
    load_dotenv()

# Reconfigure console streams for UTF-8 on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

# Try to import readline for history and editing capabilities
try:
    import readline
except ImportError:
    readline = None

SERVER_URL: str = "http://127.0.0.1:8000/analyze"
PROJECT_DIR: str = os.path.dirname(os.path.abspath(__file__))
DB_PATH: str = os.path.join(PROJECT_DIR, "photo_catalog.db")
PROMPT_FILE: str = "db_prompt.txt"

# Track absolute paths from the most recent SQL query to support /open index command
last_query_paths: List[str] = []


def sigint_handler(signum: int, frame: Any) -> None:
    """Handles SIGINT (Ctrl-C) to exit the client gracefully.

    Args:
        signum: The signal number (typically SIGINT).
        frame: The current execution frame object.

    Returns:
        None
    """
    print("\nExiting...")
    sys.exit(0)


def load_system_prompt(file_name: str = PROMPT_FILE) -> str:
    """Loads the system prompt template from an external file on disk.

    Args:
        file_name: The filename of the prompt template.

    Returns:
        The raw string content of the system prompt template.
    """
    file_path: str = os.path.join(PROJECT_DIR, file_name)
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"[Warning] Failed to read prompt template: {e}")

    # Fallback prompt in case the file cannot be accessed
    return (
        "You are a helpful assistant for the photo catalog database.\n"
        "=== SYSTEM CONTEXT ===\n"
        "Current local date/time: {current_time}\n"
        "Total photo records currently cataloged: {total_photos}\n"
    )


def get_total_photos_count() -> int:
    """Queries the database to return the total number of cataloged photos.

    Args:
        None

    Returns:
        The total number of records in the photos table.
    """
    is_testing = "pytest" in sys.argv[0] or "unittest" in sys.argv[0] or "PYTEST_CURRENT_TEST" in os.environ
    db_backend = "sqlite" if is_testing else os.getenv("DB_BACKEND", "postgresql").lower()
    
    if db_backend == "postgresql":
        conn = None
        try:
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
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM photos")
            row = cursor.fetchone()
            return row[0] if row else 0
        except Exception as e:
            print(f"[Warning] Failed to fetch total photos count from PostgreSQL: {e}")
            return 0
        finally:
            if conn:
                conn.close()
    else:
        if not os.path.exists(DB_PATH):
            return 0
        conn = None
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM photos")
            row = cursor.fetchone()
            return row[0] if row else 0
        except Exception as e:
            print(f"[Warning] Failed to fetch total photos count: {e}")
            return 0
        finally:
            if conn:
                conn.close()


def execute_sql(sql: str) -> Tuple[str, str, List[str]]:
    """Executes a database query against the photo catalog database.

    Args:
        sql: The SQL query string.

    Returns:
        A tuple of (raw_json_for_llm, terminal_display_for_user, paths_list).
    """
    is_testing = "pytest" in sys.argv[0] or "unittest" in sys.argv[0] or "PYTEST_CURRENT_TEST" in os.environ
    db_backend = "sqlite" if is_testing else os.getenv("DB_BACKEND", "postgresql").lower()

    conn = None
    try:
        if db_backend == "postgresql":
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
            
            # Connect in read-only mode to prevent mutation from generated SQL queries
            conn = psycopg2.connect(**db_conn_params)
            conn.set_session(readonly=True, autocommit=True)
            cursor = conn.cursor()
        else:
            if not os.path.exists(DB_PATH):
                err: str = f"Error: Database file not found at {DB_PATH}"
                return err, err, []
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            cursor = conn.cursor()

        # Rewrite parameter markers if query contains SQLite '?' but we are on PostgreSQL
        if db_backend == "postgresql":
            sql = sql.replace("?", "%s")
            
        cursor.execute(sql)

        if cursor.description:
            cols: List[str] = [desc[0] for desc in cursor.description]
            rows: List[Tuple[Any, ...]] = cursor.fetchall()
            if not rows:
                msg: str = "Query executed successfully. No rows returned."
                return msg, msg, []

            # Truncate the total return (list of rows) to 300 rows for CLI display
            if len(rows) > 300:
                rows = rows[:300]

            # --- Fetch full paths if we only have rel_path for RAG/VLM context enrichment ---
            rel_to_full: Dict[str, str] = {}
            if "rel_path" in cols:
                rel_paths_in_rows = [row[cols.index("rel_path")] for row in rows if row[cols.index("rel_path")] is not None]
                if rel_paths_in_rows:
                    try:
                        if db_backend == "postgresql":
                            conn2 = psycopg2.connect(**db_conn_params)
                            conn2.set_session(readonly=True, autocommit=True)
                            cursor2 = conn2.cursor()
                            placeholders = ",".join(["%s"] * len(rel_paths_in_rows))
                            cursor2.execute(f"SELECT rel_path, full_path FROM photos WHERE rel_path IN ({placeholders})", rel_paths_in_rows)
                        else:
                            conn2 = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
                            cursor2 = conn2.cursor()
                            placeholders = ",".join(["?"] * len(rel_paths_in_rows))
                            cursor2.execute(f"SELECT rel_path, full_path FROM photos WHERE rel_path IN ({placeholders})", rel_paths_in_rows)
                        
                        for r_path, f_path in cursor2.fetchall():
                            rel_to_full[r_path.lower()] = f_path
                        conn2.close()
                    except Exception as e:
                        print(f"[Warning] Failed to fetch full_path mappings for VLM: {e}")

            # --- 1. Construct raw markdown for VLM (Always ensuring full_path is present in context) ---
            # Slices cell strings to 2000 characters directly in Python to protect VRAM dynamically.
            # If the query returns more than 10 rows, we send only the first 5 rows as a sample
            # and a note about the total count to prevent swamping VRAM during VLM reasoning.
            vlm_cols = list(cols)
            has_appended_full = "full_path" not in cols and bool(rel_to_full)
            if has_appended_full:
                vlm_cols.append("full_path")

            raw_headers: str = f"| {' | '.join(vlm_cols)} |"
            raw_separator: str = f"| {' | '.join(['---'] * len(vlm_cols))} |"
            raw_lines: List[str] = [raw_headers, raw_separator]
            
            rel_idx = cols.index("rel_path") if "rel_path" in cols else -1

            total_count = len(rows)
            if total_count > 10:
                vlm_rows: List[Tuple[Any, ...]] = rows[:5]
                prefix_note = (
                    f"Query executed successfully. Returned {total_count} rows. "
                    f"Only the first 5 rows are shown below as a sample to save context window. "
                    f"All {total_count} rows have already been printed directly to the user's terminal. "
                    f"Refer the user to the printed list above and do NOT list individual paths or repeat descriptions in your response.\n\n"
                )
            else:
                vlm_rows = rows
                prefix_note = ""

            for row in vlm_rows:
                row_str: List[str] = []
                for idx_c, val in enumerate(row):
                    if val is None:
                        row_str.append("NULL")
                    elif isinstance(val, float):
                        row_str.append(f"{val:.3f}")
                    else:
                        val_str: str = str(val).replace("\n", " ")
                        if len(val_str) > 2000:
                            row_str.append(val_str[:1997] + "...")
                        else:
                            row_str.append(val_str)
                
                # Append full_path column value for RAG enrichment if missing
                if has_appended_full and rel_idx != -1:
                    rel_val = row[rel_idx]
                    f_path = rel_to_full.get(str(rel_val).lower(), "NULL") if rel_val is not None else "NULL"
                    row_str.append(f_path)
                
                raw_lines.append(f"| {' | '.join(row_str)} |")
            raw_markdown: str = prefix_note + "\n".join(raw_lines)

            # --- 2. Construct terminal display for User ---
            # If it's a single value (e.g. COUNT)
            if len(rows) == 1 and len(cols) == 1 and cols[0] not in ("full_path", "rel_path"):
                val_single = rows[0][0]
                term_display: str = str(val_single) if val_single is not None else "NULL"
                return raw_markdown, term_display, []

            # Check if we can format as indexed bullets with file paths/names
            path_col_idx: int = -1
            for name in ["full_path", "rel_path", "file_path"]:
                if name in cols:
                    path_col_idx = cols.index(name)
                    break
 
            if path_col_idx != -1:
                bullets: List[str] = []
                paths_list: List[str] = []
                 
                # Determine all metadata column indexes (excluding path columns)
                meta_cols: List[Tuple[int, str]] = [
                    (i, col_name) for i, col_name in enumerate(cols)
                    if col_name not in ("full_path", "rel_path", "file_path")
                ]

                # Detect console width dynamically (fallback to 80)
                try:
                    term_width = os.get_terminal_size().columns
                except Exception:
                    term_width = 80
                wrap_width = max(term_width - 6, 40)  # Account for indentation spacing

                for idx, row in enumerate(rows):
                    val_path = row[path_col_idx]
                    if val_path is not None:
                        val_str: str = str(val_path)
                        resolved_path = val_str
                        if cols[path_col_idx] == "rel_path" and val_str.lower() in rel_to_full:
                            resolved_path = rel_to_full[val_str.lower()]

                        win_path: str = os.path.normpath(resolved_path)
                        paths_list.append(win_path)
                        
                        # Build formatted lines for each bullet item
                        bullet_lines: List[str] = [f"[{idx + 1}] {win_path}"]
                        
                        for col_idx, col_name in meta_cols:
                            val_meta = row[col_idx]
                            if val_meta is None:
                                val_meta_clean = "NULL"
                            else:
                                val_meta_clean = str(val_meta).replace("\n", " ").strip()
                            
                            # Skip printing empty lists or empty values to keep output concise
                            if val_meta_clean in ("[]", "{}", ""):
                                continue
                                
                            meta_line = f"{col_name}: {val_meta_clean}"
                            # Wrap metadata cleanly with indented wrap boundaries
                            wrapped_meta = textwrap.wrap(meta_line, width=wrap_width, subsequent_indent="        ")
                            bullet_lines.extend(["    " + line for line in wrapped_meta])
                            
                        bullets.append("\n".join(bullet_lines))
                return raw_markdown, "\n".join(bullets), paths_list

            # Fallback: Truncated Markdown Table for clean terminal output
            term_headers: str = f"| {' | '.join(cols)} |"
            term_separator: str = f"| {' | '.join(['---'] * len(cols))} |"
            term_lines: List[str] = [term_headers, term_separator]
            for row in rows:
                row_str: List[str] = []
                for val in row:
                    if val is None:
                        row_str.append("NULL")
                    elif isinstance(val, float):
                        row_str.append(f"{val:.3f}")
                    else:
                        val_str: str = str(val).replace("\n", " ")
                        if len(val_str) > 120:
                            row_str.append(val_str[:117] + "...")
                        else:
                            row_str.append(val_str)
                term_lines.append(f"| {' | '.join(row_str)} |")
            return raw_markdown, "\n".join(term_lines), []
        else:
            if db_backend == "sqlite":
                conn.commit()
            msg: str = f"Query executed successfully. Rows affected: {cursor.rowcount}"
            return msg, msg, []
    except Exception as e:
        err: str = f"Error executing SQL: {e}"
        return err, err, []
    finally:
        if conn:
            conn.close()


def run_repl(remote: bool = False, model_name: str = "gemma4-it-q4:latest", host: str = "192.168.8.193", port: int = 11434) -> None:
    """Runs the interactive Read-Eval-Print Loop (REPL) CLI chat client.

    Loops prompting user input, formatting the payload, querying the
    FastAPI or remote Ollama endpoint, executing tool calls, and updating conversation history.

    Args:
        remote: If True, connects to the remote Ollama server instead of local WSL2 container.
        model_name: The name of the remote model to use.
        host: Host IP or hostname of the remote Ollama server.
        port: Connection port of the remote Ollama server.

    Returns:
        None

    Raises:
        SystemExit: If the model server fails to start or respond.
    """
    global last_query_paths
    import threading
    
    server_thread: Optional[threading.Thread] = None
    # Ensure WSL2 server is running if not in remote mode
    if not remote:
        print("[WSL2 Server] Starting local model server in a background thread...")
        
        def boot_server() -> None:
            if not wsl_client.start_wsl_server():
                print("\n[Error] Failed to start local WSL2 model server. VLM queries will fail.")
                
        server_thread = threading.Thread(target=boot_server, daemon=True, name="WSLServerBoot")
        server_thread.start()
    else:
        print(f"[Remote Mode] Connecting to model server at http://{host}:{port} using model '{model_name}'...")

    print("==================================================")
    print("  Gemma 4 Photo Catalog - Database Chat Client")
    print("==================================================")
    print("Instructions:")
    print("  * Type your question and press Enter.")
    print("  * To paste multiline text, type '/paste' and press Enter.")
    print("  * Type 'open <index>' or '/open <index>' to view a photo locally.")
    print("  * Type '/clear' or '/reset' to clear chat history.")
    print("  * Type 'exit' or 'quit' to close the client.")
    print("==================================================")
    print()

    # Register OS-level signal handler for SIGINT (Ctrl-C)
    signal.signal(signal.SIGINT, sigint_handler)

    session: requests.Session = requests.Session()
    chat_history: List[Dict[str, str]] = []

    while True:
        try:
            user_input: str = input("Prompt > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Exiting...")
            break

        if user_input.lower() in ("/clear", "/reset"):
            chat_history = []
            print("Conversation history cleared.")
            continue

        # Slash Command: /playlist [name]
        if user_input.lower().startswith("/playlist") or user_input.lower().startswith("playlist"):
            prefix_len = 9 if user_input.lower().startswith("/playlist") else 8
            playlist_name = user_input[prefix_len:].strip().strip("'\"")
            if not playlist_name:
                playlist_name = "ai_generated"
            if not playlist_name.lower().endswith(".m3u"):
                playlist_name += ".m3u"
                
            if not last_query_paths:
                print("[Playlist Error] ❌ No files found in the last query to generate a playlist.")
                continue
                
            # Filter only audio files
            audio_paths = [p for p in last_query_paths if p.lower().endswith((".flac", ".mp3", ".wav", ".m4a", ".ogg", ".wma", ".aac"))]
            if not audio_paths:
                print("[Playlist Error] ❌ None of the files in the last query are audio files.")
                continue
                
            playlist_dir = r"D:\Users\steven\Music\Playlists"
            try:
                os.makedirs(playlist_dir, exist_ok=True)
            except Exception:
                playlist_dir = os.path.join(PROJECT_DIR, "Playlists")
                os.makedirs(playlist_dir, exist_ok=True)
                
            playlist_path = os.path.join(playlist_dir, playlist_name)
            try:
                with open(playlist_path, "w", encoding="utf-8") as f:
                    for path in audio_paths:
                        f.write(path + "\n")
                print(f"[Playlist Success] ✅ Created M3U playlist with {len(audio_paths)} tracks:")
                print(f"   -> {playlist_path}")
            except Exception as ex:
                print(f"[Playlist Error] ❌ Failed to write playlist file: {ex}")
            continue

        # Slash Command: /play or /queue or /add
        cmd_normalized = user_input.lower()
        if any(cmd_normalized.startswith(x) for x in ("/play", "play", "/queue", "queue", "/add", "add")):
            prefix_len = 0
            is_queue = False
            for p in ("/play", "play", "/queue", "queue", "/add", "add"):
                if cmd_normalized.startswith(p):
                    prefix_len = len(p)
                    if "queue" in p or "add" in p:
                        is_queue = True
                    break
            
            target = user_input[prefix_len:].strip().strip("'\"")
            cmd_name = "Queue" if is_queue else "Play"
            
            tracks_to_play = []
            if not target:
                # Play/Queue all audio tracks from last query
                if not last_query_paths:
                    print(f"[{cmd_name} Error] ❌ No files found in the last query to {cmd_name.lower()}.")
                    continue
                tracks_to_play = [p for p in last_query_paths if p.lower().endswith((".flac", ".mp3", ".wav", ".m4a", ".ogg", ".wma", ".aac"))]
                if not tracks_to_play:
                    print(f"[{cmd_name} Error] ❌ None of the files in the last query are audio files.")
                    continue
            elif target.isdigit():
                # Play/Queue specific track by index
                idx = int(target)
                if 1 <= idx <= len(last_query_paths):
                    track_path = last_query_paths[idx - 1]
                    if track_path.lower().endswith((".flac", ".mp3", ".wav", ".m4a", ".ogg", ".wma", ".aac")):
                        tracks_to_play = [track_path]
                    else:
                        print(f"[{cmd_name} Error] ❌ File at index {idx} is not an audio file: {os.path.basename(track_path)}")
                        continue
                else:
                    print(f"[{cmd_name} Error] ❌ Index {idx} is out of range. Valid indexes: 1 to {len(last_query_paths)}.")
                    continue
            else:
                print(f"[{cmd_name} Error] ❌ Invalid parameter. Use '/{cmd_name.lower()}' to {cmd_name.lower()} all, or '/{cmd_name.lower()} <index>'.")
                continue
                
            action_desc = "queueing" if is_queue else "playing"
            print(f"[JRiver {cmd_name}] Preparing to queue {len(tracks_to_play)} track(s) on JRiver Media Center...")
            try:
                import urllib.parse
                # 1. Clear current queue only if not queueing/adding
                if not is_queue:
                    requests.get("http://127.0.0.1:52198/MCWS/v1/Playback/ClearPlaylist?Zone=0&ZoneType=ID", timeout=5)
                
                # 2. Add each track sequentially
                queued_count = 0
                for path in tracks_to_play:
                    encoded_path = urllib.parse.quote(path)
                    add_url = f"http://127.0.0.1:52198/MCWS/v1/Playback/PlayByFilename?Filenames={encoded_path}&Location=End&Zone=0&ZoneType=ID"
                    add_r = requests.get(add_url, timeout=5)
                    if add_r.status_code == 200:
                        queued_count += 1
                        
                # 3. Ensure play command is sent only if clearing & starting fresh
                if not is_queue:
                    requests.get("http://127.0.0.1:52198/MCWS/v1/Playback/Play?Zone=0&ZoneType=ID", timeout=5)
                
                print(f"[JRiver {cmd_name}] ✅ Successfully queued {queued_count} track(s) in JRiver!")
            except Exception as ex:
                print(f"[JRiver {cmd_name} Error] ❌ Failed to send command to JRiver: {ex}")
                print("   Ensure JRiver Media Center is running and Media Network (MCWS) is enabled on port 52198.")
            continue

        if (user_input.lower().startswith("/catalog") or 
            user_input.lower().startswith("catalog") or 
            user_input.lower().startswith("/run_cataloger") or 
            user_input.lower().startswith("run_cataloger")):
            
            prefix_len = 0
            for p in ("/cataloger", "cataloger", "/catalog", "catalog", "/run_cataloger", "run_cataloger"):
                if user_input.lower().startswith(p):
                    prefix_len = len(p)
                    break
            
            catalog_args_str = user_input[prefix_len:].strip()
            import shlex
            try:
                parsed_args = shlex.split(catalog_args_str)
            except Exception:
                parsed_args = catalog_args_str.split()

            print(f"[Cataloger Agent] Launching describe_photos.py with arguments: {parsed_args} in a background thread...")
            
            def run_cataloger_in_background(args_list: List[str]) -> None:
                import subprocess
                py_exe = os.path.join(os.path.dirname(PROJECT_DIR), "ltx2_env", "Scripts", "python.exe")
                if not os.path.exists(py_exe):
                    py_exe = "python"
                
                full_args = [py_exe, "describe_photos.py"]
                
                has_db = False
                has_dir = False
                for a in args_list:
                    if a.startswith("--db"):
                        has_db = True
                    if a.startswith("--dir"):
                        has_dir = True
                
                if not has_db:
                    full_args.extend(["--db", DB_PATH])
                if not has_dir:
                    full_args.extend(["--dir", "D:\\Users\\steven\\Pictures"])
                
                if "--no-json" not in args_list:
                    full_args.append("--no-json")
                if "--embed-exif" not in args_list:
                    full_args.append("--embed-exif")
                
                full_args.extend(args_list)
                
                try:
                    res = subprocess.run(
                        full_args,
                        cwd=PROJECT_DIR,
                        capture_output=True,
                        text=True,
                        timeout=7200
                    )
                    if res.returncode == 0:
                        print("\n[Cataloger Success] ✅ Cataloging run completed successfully!")
                        for line in (res.stderr.splitlines() + res.stdout.splitlines()):
                            if "Found" in line or "Saved" in line or "processed" in line or "Active VLM" in line or "images to process" in line:
                                print(f"[Cataloger Summary] {line.strip()}")
                        print("Prompt > ", end="", flush=True)
                    else:
                        print(f"\n[Cataloger Error] ❌ Cataloging failed (Exit Code {res.returncode}):\n{res.stderr or res.stdout}")
                        print("Prompt > ", end="", flush=True)
                except Exception as ex:
                    print(f"\n[Cataloger Error] ❌ Failed to launch cataloger: {ex}")
                    print("Prompt > ", end="", flush=True)

            threading.Thread(target=run_cataloger_in_background, args=(parsed_args,), daemon=True, name="CatalogerRun").start()
            continue

        if (user_input.lower().startswith("/merge") or 
            user_input.lower().startswith("merge") or 
            "merge agent" in user_input.lower() or 
            "lauch merge" in user_input.lower() or 
            "launch merge" in user_input.lower()):
            
            # Detect if user requested to overwrite/force update existing records
            overwrite = False
            user_args = user_input.lower()
            if any(k in user_args for k in ("overwrite", "-o", "all", "same", "force")):
                overwrite = True

            print(f"[Merge Agent] Launching the description merger script (overwrite={overwrite}) in a background thread...")
            def run_merge_in_background(force_overwrite: bool) -> None:
                import subprocess
                py_exe = os.path.join(os.path.dirname(PROJECT_DIR), "ltx2_env", "Scripts", "python.exe")
                if not os.path.exists(py_exe):
                    py_exe = "python"
                try:
                    merge_args = [py_exe, "merge_new_to_enriched.py"]
                    if force_overwrite:
                        merge_args.append("--overwrite")
                    res = subprocess.run(
                        merge_args,
                        cwd=PROJECT_DIR,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    if res.returncode == 0:
                        print("\n[Merge Agent Success] ✅ Master catalog description merge completed successfully!")
                        for line in (res.stderr.splitlines() + res.stdout.splitlines()):
                            if "Merge summary" in line or "Successfully wrote updated master" in line:
                                print(f"[Merge Agent Summary] {line.strip()}")
                        
                        # Trigger SQLite synchronization from updated photo_descriptions_enriched.json
                        print("[Merge Agent] Synchronizing updated master catalog to SQLite database...")
                        enriched_json = os.path.join(os.path.dirname(PROJECT_DIR), "photo_descriptions_enriched.json")
                        sync_res = subprocess.run(
                            [py_exe, "import_json_to_sqlite.py", "--source", enriched_json],
                            cwd=PROJECT_DIR,
                            capture_output=True,
                            text=True,
                            timeout=300
                        )
                        if sync_res.returncode == 0:
                            print("[Merge Agent Sync Success] ✅ SQLite database successfully synchronized!")
                            for line in (sync_res.stderr.splitlines() + sync_res.stdout.splitlines()):
                                if "Sync completed" in line or "DB is already up-to-date" in line:
                                    print(f"[Merge Agent Sync Summary] {line.strip()}")
                        else:
                            print(f"[Merge Agent Sync Error] ❌ SQLite synchronization failed (Exit Code {sync_res.returncode}):\n{sync_res.stderr or sync_res.stdout}")
                        print("Prompt > ", end="", flush=True)
                    else:
                        print(f"\n[Merge Agent Error] ❌ Merger failed (Exit Code {res.returncode}):\n{res.stderr or res.stdout}")
                        print("Prompt > ", end="", flush=True)
                except Exception as ex:
                    print(f"\n[Merge Agent Error] ❌ Failed to launch merger: {ex}")
                    print("Prompt > ", end="", flush=True)

            threading.Thread(target=run_merge_in_background, args=(overwrite,), daemon=True, name="MergeAgentRun").start()
            continue

        if user_input.lower().startswith("/open ") or user_input.lower().startswith("open "):
            prefix_len = 6 if user_input.lower().startswith("/open ") else 5
            target = user_input[prefix_len:].strip().strip("'\"")
            file_to_open = ""
            
            # Allow opening by numeric index corresponding to printed bullet items
            if target.isdigit():
                idx = int(target)
                if 1 <= idx <= len(last_query_paths):
                    file_to_open = last_query_paths[idx - 1]
                else:
                    print(f"[Error] Index {idx} is out of range. Valid indexes: 1 to {len(last_query_paths)}.")
                    continue
            else:
                file_to_open = target
                # Handle raw file:/// URL stripping if copy-pasted
                if file_to_open.startswith("file:///"):
                    file_to_open = file_to_open[8:]
                file_to_open = os.path.normpath(file_to_open)

            if file_to_open and os.path.exists(file_to_open):
                print(f"[Opening]: {file_to_open}...")
                try:
                    os.startfile(file_to_open)
                except Exception as e:
                    print(f"[Error] Failed to open file: {e}")
            elif file_to_open:
                print(f"[Error] File not found: {file_to_open}")
            continue

        # Check if the user input is a direct SQL query
        if user_input.lower().startswith("select ") or user_input.lower().startswith("with "):
            print(f"[Executing Direct SQL]: {user_input}")
            try:
                raw_markdown, term_display, paths_list = execute_sql(user_input)
                print(f"[Results]:\n{term_display}\n")
                if paths_list:
                    last_query_paths = paths_list
            except Exception as e:
                print(f"[SQL Error]: {e}")
            continue

        prompt_text: str = user_input

        # Handle multiline paste command
        if user_input.lower() == "/paste":
            print("[Multiline Mode] Paste text. Type '/end' on a separate line to finish and send.")
            multiline_lines: List[str] = []
            
            # Temporarily restore default SIGINT handler for KeyboardInterrupt support
            old_handler = signal.signal(signal.SIGINT, signal.SIG_DFL)
            try:
                while True:
                    try:
                        line: str = input("... ")
                    except EOFError:
                        break
                    if line.strip() == "/end":
                        break
                    multiline_lines.append(line)
            except KeyboardInterrupt:
                print("\n[Cancelled multiline input]")
                multiline_lines = []
            finally:
                # Restore the custom exit signal handler
                signal.signal(signal.SIGINT, sigint_handler)

            prompt_text = "\n".join(multiline_lines).strip()
            if not prompt_text:
                continue

        # Get total photos count and format system context
        total_photos: int = get_total_photos_count()
        current_time_str: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt_template: str = load_system_prompt()
        
        try:
            system_context: str = prompt_template.format(
                current_time=current_time_str,
                total_photos=total_photos
            )
        except KeyError as ke:
            print(f"[Warning] Formatting placeholder missing in prompt template: {ke}")
            system_context = prompt_template

        # Record the initial history length to allow restoring on failure
        initial_history_len: int = len(chat_history)
        chat_history.append({"role": "user", "content": prompt_text})

        # Format previous conversation history (excluding the current user prompt at the end)
        history_str: str = ""
        for msg in chat_history[:-1]:
            role: str = msg["role"].capitalize()
            content: str = msg["content"]
            history_str += f"{role}: {content}\n\n"

        # Assemble the active prompt string
        active_prompt: str = f"{system_context}\n\n"
        if history_str:
            active_prompt += f"=== CONVERSATION HISTORY ===\n{history_str}"
        active_prompt += f"=== CURRENT TURN ===\nUser: {prompt_text}\n"

        # Agent iteration loop to allow multiple database queries
        tool_executed: bool = False
        final_response_text: str = ""

        # Limit to 5 iterations to prevent infinite agent run-away loops
        for iteration in range(5):
            if remote:
                target_url: str = f"http://{host}:{port}/api/chat"
                messages: List[Dict[str, str]] = []
                messages.append({"role": "system", "content": system_context})
                for msg in chat_history:
                    messages.append({"role": msg["role"], "content": msg["content"]})

                payload: Dict[str, Any] = {
                    "model": model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_ctx": 65536,
                        "num_predict": 4096
                    }
                }
            else:
                target_url = SERVER_URL
                payload = {
                    "prompt_text": active_prompt,
                    "temperature": 0.2,
                    "max_new_tokens": 4096
                }

            if not remote and server_thread and server_thread.is_alive():
                print("[WSL2 Server] Local model server is still booting up (VRAM weight loading in progress). Waiting for boot to complete...")
                server_thread.join()

            if not tool_executed:
                print("Waiting for response...")
            else:
                print("Processing SQL query results...")

            try:
                response = session.post(target_url, json=payload, timeout=180.0)
                if response.status_code != 200:
                    print(f"\n[Error] Server returned status code {response.status_code}: {response.text}\n")
                    break

                result: Dict[str, Any] = response.json()
                if remote:
                    response_text: str = result.get("message", {}).get("content", "").strip()
                else:
                    response_text = result.get("response", "").strip()

                # Look for tool call tags in response
                tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', response_text, re.DOTALL)
                if tool_call_match:
                    tool_json_str: str = tool_call_match.group(1).strip()
                    print(f"\n[Executing Tool Call]: {tool_json_str}")
                    sql_query: str = ""

                    try:
                        tool_data: Dict[str, Any] = json.loads(tool_json_str)
                        sql_query = tool_data.get("sql", "")
                    except Exception:
                        # Fallback: Extract SELECT statement directly
                        sql_match = re.search(r'(SELECT\s+.*)', tool_json_str, re.IGNORECASE | re.DOTALL)
                        if sql_match:
                            sql_query = sql_match.group(1).strip()

                    if sql_query:
                        print(f"[Executing SQL]: {sql_query}")
                        raw_markdown, term_display, paths_list = execute_sql(sql_query)
                        print(f"[Results]:\n{term_display}\n")
                        
                        # Store paths list for /open index command
                        if paths_list:
                            last_query_paths = paths_list

                        # Record the intermediate turns directly in chat_history
                        chat_history.append({"role": "assistant", "content": response_text})
                        chat_history.append({"role": "user", "content": f"TOOL RESULT:\n{raw_markdown}"})

                        # Update active_prompt for local mode
                        active_prompt += f"Assistant: {response_text}\n\nUser: TOOL RESULT:\n{raw_markdown}\n\n"
                        tool_executed = True
                        continue
                    else:
                        print("[Error] Failed to parse SQL statement from tool call.")

                # If no tool call was matched, we've received the final answer
                final_response_text = response_text
                chat_history.append({"role": "assistant", "content": final_response_text})
                break

            except requests.RequestException as e:
                print(f"\n[Error] Failed to connect to server: {e}\n")
                break
            except Exception as e:
                print(f"\n[Error] An unexpected error occurred: {e}\n")
                break

        # Display response if successful, otherwise restore history
        if final_response_text:
            print("\nResponse:")
            print(final_response_text)
            print()

            # Record clean conversational turns in history, keeping the last 20 messages
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]
        else:
            # Restore history to before this turn if it failed
            chat_history = chat_history[:initial_history_len]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Interactive CLI REPL client for the photo catalog database.")
    parser.add_argument("--remote", action="store_true", help="Use remote Ollama server on ubunto-giga instead of local WSL2 container.")
    parser.add_argument("--model", type=str, default="gemma4-it-q4:latest", help="Model name to request from remote Ollama server.")
    parser.add_argument("--host", type=str, default="192.168.8.193", help="Remote host IP or hostname.")
    parser.add_argument("--port", type=int, default=11434, help="Remote host port.")
    args = parser.parse_args()

    run_repl(
        remote=args.remote,
        model_name=args.model,
        host=args.host,
        port=args.port
    )
