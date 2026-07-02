"""Functional test to verify connection to the local PostgreSQL database on Windows.

Loads credentials dynamically from the .env configuration.
"""

import os
import sys
from typing import Any, Dict
import psycopg2
from dotenv import load_dotenv

# Load workspace environment variables
if os.path.exists("auth/.env"):
    load_dotenv("auth/.env")
else:
    load_dotenv()

def run_functional_test(conn_params: Dict[str, Any], mode_label: str) -> bool:
    """Runs connection and query assertions using the provided parameters.

    Args:
        conn_params: Database connection dictionary arguments.
        mode_label: Human-readable label describing the connection mode.

    Returns:
        True if the test succeeded, False otherwise.
    """
    print(f"\n[{mode_label}] Connecting using: {conn_params}")
    try:
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        print(f"[{mode_label}] Connection established successfully.")
        
        # Test query
        cursor.execute("SELECT COUNT(*) FROM photos")
        count = cursor.fetchone()[0]
        print(f"[{mode_label}] Query success! Total rows in 'photos' table: {count}")
        
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"[{mode_label}] Connection failed: {e}", file=sys.stderr)
        return False

def main() -> None:
    """Main execution entry point."""
    # Read settings directly from the workspace .env config
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = int(os.getenv("DB_PORT", "5432"))
    db_user = os.getenv("DB_USER", "postgres")
    
    # Load database password from auth/db_password.txt
    db_pass = ""
    pwd_path = os.path.join("auth", "db_password.txt")
    if os.path.exists(pwd_path):
        with open(pwd_path, "r", encoding="utf-8") as f:
            db_pass = f.read().strip()
            
    db_name = os.getenv("DB_NAME", "photo_catalog")
    
    local_params = {
        "dbname": db_name,
        "user": db_user,
        "password": db_pass,
        "host": db_host,
        "port": db_port
    }
    
    success = run_functional_test(local_params, "Local Windows PostgreSQL")
    
    if success:
        print("\nFunctional check PASSED.")
        sys.exit(0)
    else:
        print("\nAll functional connection attempts failed. Please verify PostgreSQL service status.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
