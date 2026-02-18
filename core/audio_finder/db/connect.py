from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Generator
from pathlib import Path


def _ensure_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def _init_connection(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")

    schema_path = Path(__file__).parent / "schema.sql"
    conn.executescript(schema_path.read_text())


@contextlib.contextmanager
def get_conn(db_path: str) -> Generator[sqlite3.Connection]:
    _ensure_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        _init_connection(conn)
        yield conn
    finally:
        conn.close()


def init_schema(db_path: str) -> None:
    with get_conn(db_path):
        pass
