"""SQL Query Loader Utility Module.

Purpose:
    Provides a centralized utility to dynamically load and cache external SQLite (.sql) files.
    This eliminates inline hardcoded SQL queries from Python files, complying with repository-wide
    architectural standards.

Architecture and Mechanics:
    - Path Resolution: Resolves target SQL file paths relative to a canonical 'sql/' project root folder.
    - I/O Caching: Implements an in-memory dictionary-based lookup cache (_SQL_CACHE). Once a SQL template
      is read from the filesystem, subsequent requests fetch it directly from memory to avoid disk I/O bottlenecks.
    - Character Encoding: Enforces explicit UTF-8 character encoding on all filesystem reads.

Execution Modes:
    - Library Import Mode: Imported as a utility module (`from sql_loader import get_sql`) by codebase scripts.
"""

import os
from typing import Dict
from dotenv import load_dotenv

# Load workspace environment variables
if os.path.exists("auth/.env"):
    load_dotenv("auth/.env")
else:
    load_dotenv()

# Global directory constant
SQL_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sql")

# In-memory dictionary query cache, keyed by (relative_path, db_backend)
_SQL_CACHE: Dict[tuple, str] = {}


def get_sql(relative_path: str, db_backend: str = "") -> str:
    """Loads a SQL query from the given relative path within the sql/ folder, caching the result.

    If db_backend is 'postgresql' (or if empty and DB_BACKEND is set to 'postgresql' in .env),
    SQLite '?' placeholders are dynamically replaced with PostgreSQL '%s' placeholders.

    Args:
        relative_path: Path to the SQL file relative to the sql/ directory
            (e.g. 'queries/update_photo.sql').
        db_backend: Optional database backend override ('sqlite' or 'postgresql').

    Returns:
        The content of the SQL file as a string.

    Raises:
        FileNotFoundError: If the SQL file does not exist at the specified path.
    """
    if not db_backend:
        import sys
        is_testing = "pytest" in sys.argv[0] or "unittest" in sys.argv[0] or "PYTEST_CURRENT_TEST" in os.environ
        db_backend = "sqlite" if is_testing else os.getenv("DB_BACKEND", "sqlite")
        
    db_backend = db_backend.lower()
    cache_key = (relative_path, db_backend)
    
    if cache_key not in _SQL_CACHE:
        full_path: str = os.path.normpath(os.path.join(SQL_DIR, relative_path))
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"SQL file not found at: {full_path}")
        
        with open(full_path, "r", encoding="utf-8") as f:
            query_content = f.read().strip()
            
        # Dynamically rewrite placeholders if using PostgreSQL
        if db_backend == "postgresql":
            query_content = query_content.replace("?", "%s")
            
        _SQL_CACHE[cache_key] = query_content
            
    return _SQL_CACHE[cache_key]
