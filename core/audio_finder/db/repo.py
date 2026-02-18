from __future__ import annotations

import json
import sqlite3


def _lastrowid(cur: sqlite3.Cursor) -> int:
    """Return lastrowid, asserting it's not None (always true after INSERT)."""
    rowid = cur.lastrowid
    assert rowid is not None
    return rowid


# --- Tags ---


def get_or_create_tag(conn: sqlite3.Connection, name: str) -> int:
    row = conn.execute(
        "SELECT id FROM tags WHERE name = ?", (name,)
    ).fetchone()
    if row:
        return row[0]
    cur = conn.execute(
        "INSERT INTO tags (name) VALUES (?)", (name,)
    )
    return _lastrowid(cur)


# --- Media Files ---


def file_exists(conn: sqlite3.Connection, tag_id: int, sha256: str) -> int | None:
    row = conn.execute(
        "SELECT id FROM media_files WHERE tag_id = ? AND sha256 = ?",
        (tag_id, sha256),
    ).fetchone()
    return row[0] if row else None


def insert_media_file(
    conn: sqlite3.Connection,
    tag_id: int,
    path: str,
    sha256: str,
    duration_sec: float,
) -> int:
    cur = conn.execute(
        "INSERT INTO media_files (tag_id, path, sha256, duration_sec) "
        "VALUES (?, ?, ?, ?)",
        (tag_id, path, sha256, duration_sec),
    )
    return _lastrowid(cur)


def get_files_for_tag(conn: sqlite3.Connection, tag_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT id, path, sha256, duration_sec FROM media_files WHERE tag_id = ? ORDER BY id",
        (tag_id,),
    ).fetchall()
    return [
        {"id": r[0], "path": r[1], "sha256": r[2], "duration_sec": r[3]}
        for r in rows
    ]


# --- Fingerprint Profiles ---


def get_or_create_profile(conn: sqlite3.Connection, params: dict) -> int:
    params_json = json.dumps(params, sort_keys=True)
    row = conn.execute(
        "SELECT id FROM fp_profiles WHERE params_json = ?",
        (params_json,),
    ).fetchone()
    if row:
        return row[0]
    cur = conn.execute(
        "INSERT INTO fp_profiles (params_json) VALUES (?)",
        (params_json,),
    )
    return _lastrowid(cur)


# --- Fingerprints ---


def bulk_insert_fingerprints(
    conn: sqlite3.Connection,
    tag_id: int,
    profile_id: int,
    file_id: int,
    hashes: list[tuple[int, int]],
) -> int:
    """Insert fingerprints using executemany. hashes is list of (hash_value, t_frame)."""
    if not hashes:
        return 0

    conn.executemany(
        "INSERT INTO fingerprints (tag_id, profile_id, hash, file_id, t_frame) "
        "VALUES (?, ?, ?, ?, ?)",
        [(tag_id, profile_id, h, file_id, t) for h, t in hashes],
    )
    return len(hashes)


def load_fingerprints_for_file(
    conn: sqlite3.Connection,
    tag_id: int,
    profile_id: int,
    file_id: int,
) -> list[tuple[int, int]]:
    """Return list of (hash, t_frame) for a file."""
    rows = conn.execute(
        "SELECT hash, t_frame FROM fingerprints "
        "WHERE tag_id = ? AND profile_id = ? AND file_id = ?",
        (tag_id, profile_id, file_id),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def find_collisions_batch(
    conn: sqlite3.Connection,
    tag_id: int,
    profile_id: int,
    hashes: list[int],
    exclude_file_id: int | None = None,
) -> list[tuple[int, int, int]]:
    """Find all (hash, file_id, t_frame) that match the given hashes.

    Uses a temp table JOIN for large hash sets.
    """
    if not hashes:
        return []

    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _lookup_hashes (h INTEGER)")
    conn.execute("DELETE FROM _lookup_hashes")

    conn.executemany(
        "INSERT INTO _lookup_hashes (h) VALUES (?)",
        [(h,) for h in hashes],
    )

    exclude = ""
    params: list = [tag_id, profile_id]
    if exclude_file_id is not None:
        exclude = " AND f.file_id != ?"
        params.append(exclude_file_id)

    rows = conn.execute(
        "SELECT f.hash, f.file_id, f.t_frame "
        "FROM fingerprints f "
        "JOIN _lookup_hashes l ON f.hash = l.h "
        f"WHERE f.tag_id = ? AND f.profile_id = ?{exclude}",
        params,
    ).fetchall()

    conn.execute("DELETE FROM _lookup_hashes")

    return [(r[0], r[1], r[2]) for r in rows]


# --- Hash Stats (stop-word filtering) ---


def compute_hash_stats(
    conn: sqlite3.Connection, tag_id: int, profile_id: int
) -> None:
    conn.execute(
        "DELETE FROM tag_hash_stats WHERE tag_id = ? AND profile_id = ?",
        (tag_id, profile_id),
    )
    conn.execute(
        "INSERT INTO tag_hash_stats (tag_id, profile_id, hash, file_count) "
        "SELECT tag_id, profile_id, hash, COUNT(DISTINCT file_id) "
        "FROM fingerprints "
        "WHERE tag_id = ? AND profile_id = ? "
        "GROUP BY tag_id, profile_id, hash",
        (tag_id, profile_id),
    )


def load_stop_hashes(
    conn: sqlite3.Connection,
    tag_id: int,
    profile_id: int,
    max_file_ratio: float,
) -> set[int]:
    """Return hashes appearing in more than max_file_ratio fraction of files."""
    total = conn.execute(
        "SELECT COUNT(*) FROM media_files WHERE tag_id = ?", (tag_id,)
    ).fetchone()[0]
    if total == 0:
        return set()
    threshold = int(total * max_file_ratio)
    rows = conn.execute(
        "SELECT hash FROM tag_hash_stats "
        "WHERE tag_id = ? AND profile_id = ? AND file_count > ?",
        (tag_id, profile_id, threshold),
    ).fetchall()
    return {r[0] for r in rows}


# --- Vacuum ---


def vacuum_analyze_fingerprints(conn: sqlite3.Connection) -> None:
    conn.execute("ANALYZE fingerprints")


# --- Segments ---


def insert_segment(
    conn: sqlite3.Connection,
    tag_id: int,
    profile_id: int,
    canonical_file_id: int,
    start_frame: int,
    end_frame: int,
    duration_sec: float,
) -> int:
    cur = conn.execute(
        "INSERT INTO segments (tag_id, profile_id, canonical_file_id, "
        "start_frame, end_frame, duration_sec) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (tag_id, profile_id, canonical_file_id, start_frame, end_frame, duration_sec),
    )
    return _lastrowid(cur)


def insert_segment_match(
    conn: sqlite3.Connection,
    segment_id: int,
    file_id: int,
    start_frame: int,
    end_frame: int,
    start_sec: float,
    end_sec: float,
    score: float,
) -> int:
    cur = conn.execute(
        "INSERT INTO segment_matches "
        "(segment_id, file_id, start_frame, end_frame, start_sec, end_sec, score) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (segment_id, file_id, start_frame, end_frame, start_sec, end_sec, score),
    )
    return _lastrowid(cur)


def bulk_insert_segment_fingerprints(
    conn: sqlite3.Connection,
    segment_id: int,
    hashes: list[tuple[int, int]],
) -> int:
    """Insert canonical segment hashes. hashes is list of (hash_value, t_offset_frame)."""
    if not hashes:
        return 0
    conn.executemany(
        "INSERT INTO segment_fingerprints (segment_id, hash, t_offset_frame) "
        "VALUES (?, ?, ?)",
        [(segment_id, h, t) for h, t in hashes],
    )
    return len(hashes)


def load_segment_fingerprints(
    conn: sqlite3.Connection, segment_id: int
) -> list[tuple[int, int]]:
    """Return list of (hash, t_offset_frame) for a segment."""
    rows = conn.execute(
        "SELECT hash, t_offset_frame FROM segment_fingerprints WHERE segment_id = ?",
        (segment_id,),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def delete_analysis_for_tag(
    conn: sqlite3.Connection, tag_id: int, profile_id: int
) -> None:
    """Delete all previous analysis results for a tag/profile."""
    conn.execute(
        "DELETE FROM segment_fingerprints WHERE segment_id IN "
        "(SELECT id FROM segments WHERE tag_id = ? AND profile_id = ?)",
        (tag_id, profile_id),
    )
    conn.execute(
        "DELETE FROM segment_matches WHERE segment_id IN "
        "(SELECT id FROM segments WHERE tag_id = ? AND profile_id = ?)",
        (tag_id, profile_id),
    )
    conn.execute(
        "DELETE FROM segments WHERE tag_id = ? AND profile_id = ?",
        (tag_id, profile_id),
    )


def get_segments_for_tag(
    conn: sqlite3.Connection, tag_id: int, profile_id: int
) -> list[dict]:
    """Return segments with their matches, joined with file paths."""
    segments = conn.execute(
        "SELECT id, duration_sec FROM segments "
        "WHERE tag_id = ? AND profile_id = ? ORDER BY id",
        (tag_id, profile_id),
    ).fetchall()

    result = []
    for seg_id, duration in segments:
        matches = conn.execute(
            "SELECT m.file_id, f.path, m.start_sec, m.end_sec, "
            "m.end_sec - m.start_sec AS duration_sec, m.score "
            "FROM segment_matches m "
            "JOIN media_files f ON m.file_id = f.id "
            "WHERE m.segment_id = ? ORDER BY m.score DESC",
            (seg_id,),
        ).fetchall()
        result.append({
            "segment_id": seg_id,
            "duration_sec": duration,
            "matches": [
                {
                    "file_id": r[0],
                    "file_path": r[1],
                    "start_sec": r[2],
                    "end_sec": r[3],
                    "duration_sec": r[4],
                    "score": r[5],
                }
                for r in matches
            ],
        })
    return result
