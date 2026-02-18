CREATE TABLE IF NOT EXISTS tags (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS media_files (
    id          INTEGER PRIMARY KEY,
    tag_id      INTEGER NOT NULL REFERENCES tags(id),
    path        TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    duration_sec REAL,
    UNIQUE (tag_id, sha256)
);

CREATE TABLE IF NOT EXISTS fp_profiles (
    id          INTEGER PRIMARY KEY,
    params_json TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS fingerprints (
    tag_id      INTEGER NOT NULL,
    profile_id  INTEGER NOT NULL,
    hash        INTEGER NOT NULL,
    file_id     INTEGER NOT NULL,
    t_frame     INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fp_lookup
    ON fingerprints (tag_id, profile_id, hash, file_id, t_frame);

CREATE TABLE IF NOT EXISTS tag_hash_stats (
    tag_id      INTEGER NOT NULL,
    profile_id  INTEGER NOT NULL,
    hash        INTEGER NOT NULL,
    file_count  INTEGER NOT NULL,
    PRIMARY KEY (tag_id, profile_id, hash)
);

CREATE TABLE IF NOT EXISTS segments (
    id              INTEGER PRIMARY KEY,
    tag_id          INTEGER NOT NULL REFERENCES tags(id),
    profile_id      INTEGER NOT NULL REFERENCES fp_profiles(id),
    canonical_file_id INTEGER NOT NULL REFERENCES media_files(id),
    start_frame     INTEGER NOT NULL,
    end_frame       INTEGER NOT NULL,
    duration_sec    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS segment_matches (
    id          INTEGER PRIMARY KEY,
    segment_id  INTEGER NOT NULL REFERENCES segments(id),
    file_id     INTEGER NOT NULL REFERENCES media_files(id),
    start_frame INTEGER NOT NULL,
    end_frame   INTEGER NOT NULL,
    start_sec   REAL NOT NULL,
    end_sec     REAL NOT NULL,
    score       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS segment_fingerprints (
    segment_id      INTEGER NOT NULL REFERENCES segments(id),
    hash            INTEGER NOT NULL,
    t_offset_frame  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_segfp_segment
    ON segment_fingerprints (segment_id);
