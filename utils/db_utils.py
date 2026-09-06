
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, List, Optional

DB_PATH = Path("outputs/stegoshield_audit.db")
_LOCK = threading.Lock()


def get_db_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Initialize scan audit table if not exists."""
    with _LOCK:
        conn = get_db_connection(db_path)
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS scan_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        sha256 TEXT,
                        file_size_bytes INTEGER,
                        dimensions TEXT,
                        probability REAL,
                        confidence REAL,
                        classification TEXT,
                        threshold REAL,
                        entropy REAL,
                        runtime_ms REAL,
                        status TEXT DEFAULT 'SUCCESS',
                        error_message TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_scan_timestamp ON scan_records(timestamp);
                    """
                )
        finally:
            conn.close()


def log_scan_record(
    timestamp: str,
    filename: str,
    sha256: Optional[str] = None,
    file_size_bytes: int = 0,
    dimensions: Optional[str] = None,
    probability: float = 0.0,
    confidence: float = 0.0,
    classification: str = "CLEAN",
    threshold: float = 0.73,
    entropy: float = 0.0,
    runtime_ms: float = 0.0,
    status: str = "SUCCESS",
    error_message: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> int:
    """Log a scan record to the audit database."""
    init_db(db_path)
    with _LOCK:
        conn = get_db_connection(db_path)
        try:
            with conn:
                cur = conn.execute(
                    """
                    INSERT INTO scan_records (
                        timestamp, filename, sha256, file_size_bytes, dimensions,
                        probability, confidence, classification, threshold,
                        entropy, runtime_ms, status, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        filename,
                        sha256,
                        file_size_bytes,
                        dimensions,
                        probability,
                        confidence,
                        classification,
                        threshold,
                        entropy,
                        runtime_ms,
                        status,
                        error_message,
                    ),
                )
                return cur.lastrowid
        finally:
            conn.close()


def get_scan_records(limit: int = 100, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve historical scan records from database."""
    init_db(db_path)
    with _LOCK:
        conn = get_db_connection(db_path)
        try:
            cur = conn.execute(
                """
                SELECT id, timestamp, filename, sha256, file_size_bytes, dimensions,
                       probability, confidence, classification, threshold,
                       entropy, runtime_ms, status, error_message
                FROM scan_records
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()


def clear_scan_records(db_path: Path = DB_PATH) -> None:
    """Clear all records from audit table."""
    init_db(db_path)
    with _LOCK:
        conn = get_db_connection(db_path)
        try:
            with conn:
                conn.execute("DELETE FROM scan_records")
        finally:
            conn.close()


def count_scan_records(db_path: Path = DB_PATH) -> int:
    """Count total records in audit table."""
    init_db(db_path)
    with _LOCK:
        conn = get_db_connection(db_path)
        try:
            cur = conn.execute("SELECT COUNT(*) FROM scan_records")
            return cur.fetchone()[0]
        finally:
            conn.close()
