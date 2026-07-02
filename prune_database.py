"""Standalone Database Pruning Utility.

Purpose:
    This utility scans the SQLite database (and optionally the legacy JSON file)
    and prunes records for image files that no longer exist on disk.

Architecture and Mechanics:
    1. Safety Guards:
       - Checks Windows drive letters (e.g., 'D:\\'). If the drive is not mounted,
         skips pruning to prevent data loss.
       - Legacy path protection: Paths starting with '/Volumes/' (historical Mac paths)
         are skipped by default unless --prune-legacy is explicitly passed.
       - Dry-run by default: Prints a summary of target deletions without modifying the DB
         unless --commit is specified.
    2. Batch Execution: Runs SQLite DELETE commands within a transaction to maintain integrity.
    3. JSON Sync: Syncs the deletions to photo_descriptions.json if it exists.

Execution Modes:
    - Dry-run (Safe Preview):
      python prune_database.py
    - Commit (Execute Deletion):
      python prune_database.py --commit
    - Prune Legacy Mac paths:
      python prune_database.py --commit --prune-legacy
"""

import os
import sys
import json
import shutil
import sqlite3
import psycopg2
import argparse
from dotenv import load_dotenv
from typing import List, Dict, Set, Tuple, Any

# Ensure the cataloger's directory takes priority in path lookup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from describe_photos import PICTURE_DIRS, compute_rel_path

# Reconfigure console streams for UTF-8 on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

# Load workspace environment variables
PROJECT_DIR: str = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(PROJECT_DIR, "auth", ".env")):
    load_dotenv(os.path.join(PROJECT_DIR, "auth", ".env"))
elif os.path.exists("auth/.env"):
    load_dotenv("auth/.env")
else:
    load_dotenv()

DB_PATH: str = os.path.join(PROJECT_DIR, "photo_catalog.db")
JSON_PATH: str = os.path.join(os.path.dirname(PROJECT_DIR), "photo_descriptions.json")

def get_db_conn(db_path: str) -> Tuple[Any, str]:
    """Returns a tuple of (connection, backend_type) depending on DB_BACKEND configuration.
    
    If DB_BACKEND is 'postgresql', connects to PostgreSQL and returns (conn, 'postgresql').
    Otherwise, connects to SQLite and returns (conn, 'sqlite').
    """
    is_testing = "pytest" in sys.argv[0] or "unittest" in sys.argv[0] or "PYTEST_CURRENT_TEST" in os.environ
    db_backend = "sqlite" if is_testing else os.getenv("DB_BACKEND", "postgresql").lower()
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
        conn = psycopg2.connect(**db_conn_params)
        return conn, "postgresql"
    else:
        conn = sqlite3.connect(db_path)
        return conn, "sqlite"


def prune_database(db_path: str, json_path: str, dry_run: bool, prune_legacy: bool) -> None:
    """Scans the database and prunes missing files.

    Args:
        db_path: Absolute path to the SQLite database.
        json_path: Absolute path to the legacy JSON database.
        dry_run: If True, only log proposed changes.
        prune_legacy: If True, prune historical non-Windows paths (e.g., /Volumes/).

    Returns:
        None

    Raises:
        sqlite3.Error: For SQLite transaction issues.
    """
    print("==================================================")
    print("          Database Pruning Check Started")
    print("==================================================")
    print(f"Target DB:  {db_path}")
    print(f"Dry Run:    {dry_run}")
    print(f"Prune Legacy Mac Paths: {prune_legacy}")
    print("--------------------------------------------------")

    is_testing = "pytest" in sys.argv[0] or "unittest" in sys.argv[0] or "PYTEST_CURRENT_TEST" in os.environ
    db_backend = "sqlite" if is_testing else os.getenv("DB_BACKEND", "postgresql").lower()
    if db_backend != "postgresql" and not os.path.exists(db_path):
        print(f"[ERROR] Database file not found at: {db_path}")
        sys.exit(1)

    conn, backend = get_db_conn(db_path)
    cursor = conn.cursor()

    try:
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
            print(f"[WARNING] 'photos' table does not exist in the {backend} database. Nothing to prune.")
            return

        cursor.execute("SELECT id, full_path FROM photos")
        rows: List[Tuple[int, str]] = cursor.fetchall()
        print(f"Found {len(rows)} total records to verify.")

        to_delete_ids: List[int] = []
        to_delete_paths: List[str] = []
        to_update_records: List[Tuple[str, int, str]] = []  # (new_path, db_id, old_path)
        skipped_drives: Set[str] = set()
        skipped_legacy_count: int = 0

        for db_id, full_path in rows:
            # 1. If it exists at its original location, keep it!
            if os.path.isfile(full_path):
                continue

            # 2. Try to resolve the path on Windows active directories (Translation)
            rel_path = compute_rel_path(full_path)
            resolved = False
            for pic_dir in PICTURE_DIRS:
                candidate = os.path.normpath(os.path.join(pic_dir, rel_path))
                if os.path.isfile(candidate):
                    to_update_records.append((candidate, db_id, full_path))
                    resolved = True
                    break

            if resolved:
                continue

            # 3. Check for legacy Mac volume paths (that couldn't be resolved)
            if full_path.startswith("/Volumes/"):
                if not prune_legacy:
                    skipped_legacy_count += 1
                    continue

            # 4. Parse drive prefix on Windows (for unresolved paths)
            drive, _ = os.path.splitdrive(full_path)
            if drive:
                drive_root = drive + "\\"
                if not os.path.exists(drive_root):
                    if drive not in skipped_drives:
                        print(f"[SAFETY] Drive {drive} is not mounted. Skipping files on this drive.")
                        skipped_drives.add(drive)
                    continue

            # 5. Verify file presence on disk (it was already checked, so it is missing)
            to_delete_ids.append(db_id)
            to_delete_paths.append(full_path)

        print(f"Verification complete:")
        if to_update_records:
            print(f" -> Identified {len(to_update_records)} records to be migrated to active Windows paths.")
        if skipped_legacy_count > 0:
            print(f" -> Skipped {skipped_legacy_count} legacy Mac volume paths (use --prune-legacy to force clean).")
        if skipped_drives:
            print(f" -> Skipped files on unmounted drives: {', '.join(skipped_drives)}")
        print(f" -> Identified {len(to_delete_paths)} records pointing to missing files.")

        if not to_delete_paths and not to_update_records:
            print("\n[SUCCESS] No changes needed. Database is up to date.")
            return

        # Preview targeted translations
        if to_update_records:
            print("\nPreview of records targeted for translation to MSDOS/Windows:")
            preview_limit = 10
            for new_p, _, old_p in to_update_records[:preview_limit]:
                print(f" - [TRANSLATE] {old_p} -> {new_p}")
            if len(to_update_records) > preview_limit:
                print(f" ... and {len(to_update_records) - preview_limit} more files.")

        # Preview targeted deletions
        if to_delete_paths:
            print("\nPreview of files targeted for removal:")
            preview_limit = 10
            for p in to_delete_paths[:preview_limit]:
                print(f" - [PRUNE] {p}")
            if len(to_delete_paths) > preview_limit:
                print(f" ... and {len(to_delete_paths) - preview_limit} more files.")

        if dry_run:
            print(f"\n[DRY RUN] Would have updated {len(to_update_records)} records and deleted {len(to_delete_paths)} records. No changes made.")
            return

        # 1. Back up databases first
        if backend == "sqlite":
            print("\nBacking up database before pruning...")
            try:
                db_backup = db_path + ".bak"
                shutil.copy2(db_path, db_backup)
                print(f"[SUCCESS] SQLite database backed up to: {db_backup}")
            except Exception as bu_err:
                print(f"[ERROR] Database backup failed: {bu_err}. Aborting pruning for safety.")
                return
        else:
            print("\n[INFO] Skipping SQLite backup file creation for PostgreSQL backend.")

        if os.path.exists(json_path):
            try:
                json_backup = json_path + ".bak"
                shutil.copy2(json_path, json_backup)
                print(f"[SUCCESS] JSON database backed up to: {json_backup}")
            except Exception as bu_err:
                print(f"[ERROR] JSON backup failed: {bu_err}. Aborting pruning for safety.")
                return

        # 2. Execute Updates (Translations)
        if to_update_records:
            print(f"\nMigrating {len(to_update_records)} records to active Windows paths in {backend}...")
            query = "UPDATE photos SET full_path = %s WHERE id = %s" if backend == "postgresql" else "UPDATE photos SET full_path = ? WHERE id = ?"
            for new_p, db_id, _ in to_update_records:
                cursor.execute(query, (new_p, db_id))
            conn.commit()
            print(f"[SUCCESS] {backend} paths migrated.")

        # 3. Execute Deletions
        if to_delete_paths:
            print(f"\nDeleting {len(to_delete_paths)} records from {backend} database...")
            # Split into chunks to avoid SQLite parameter limit issues
            chunk_size = 999
            for i in range(0, len(to_delete_ids), chunk_size):
                chunk = to_delete_ids[i:i + chunk_size]
                if backend == "postgresql":
                    placeholders = ",".join("%s" for _ in chunk)
                else:
                    placeholders = ",".join("?" for _ in chunk)
                cursor.execute(f"DELETE FROM photos WHERE id IN ({placeholders})", chunk)
            conn.commit()
            print(f"[SUCCESS] {backend} database pruned.")

        # 4. Vacuum SQLite database to reclaim disk space
        if backend == "sqlite":
            print("Vacuuming SQLite database to reclaim disk space...")
            old_isolation = conn.isolation_level
            conn.isolation_level = None
            try:
                cursor.execute("VACUUM")
                print("[SUCCESS] SQLite database vacuumed.")
            except Exception as ve:
                print(f"[WARNING] SQLite database vacuum failed: {ve}")
            finally:
                conn.isolation_level = old_isolation

        # Sync changes to JSON if it exists
        if os.path.exists(json_path):
            print(f"Syncing changes to legacy JSON database: {json_path}")
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    deleted_paths_set = {p.lower() for p in to_delete_paths}
                    # Map old lowercase path -> new path
                    translation_map = {old_p.lower(): new_p for new_p, _, old_p in to_update_records}
                    
                    new_data = []
                    removed_json_count = 0
                    updated_json_count = 0
                    
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        orig_path = item.get("full_path", "")
                        orig_path_lower = orig_path.lower()
                        
                        if orig_path_lower in deleted_paths_set:
                            removed_json_count += 1
                            continue
                        
                        if orig_path_lower in translation_map:
                            item["full_path"] = translation_map[orig_path_lower]
                            updated_json_count += 1
                        
                        new_data.append(item)
                    
                    # Write updated JSON atomically
                    temp_json = json_path + ".tmp"
                    with open(temp_json, "w", encoding="utf-8") as f:
                        json.dump(new_data, f, indent=4)
                    os.replace(temp_json, json_path)
                    print(f"[SUCCESS] Updated {updated_json_count} paths and removed {removed_json_count} records in legacy JSON database.")
            except Exception as jsone:
                print(f"[ERROR] Failed to update JSON database: {jsone}")

    except sqlite3.Error as e:
        print(f"[ERROR] SQLite error occurred: {e}")
        conn.rollback()
    finally:
        conn.close()


def main() -> None:
    """CLI Entrypoint for the pruner tool."""
    parser = argparse.ArgumentParser(description="Prune database records for missing files.")
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Execute the deletions (without this, script runs in safe preview dry-run mode)."
    )
    parser.add_argument(
        "--prune-legacy",
        action="store_true",
        help="Prune legacy Mac Volume paths (/Volumes/) if the files are not accessible."
    )
    args = parser.parse_args()

    prune_database(DB_PATH, JSON_PATH, dry_run=not args.commit, prune_legacy=args.prune_legacy)


if __name__ == "__main__":
    main()
