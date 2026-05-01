CREATE TABLE IF NOT EXISTS notes (
    run_id TEXT PRIMARY KEY,
    body TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tags (
    run_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY (run_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);

CREATE TABLE IF NOT EXISTS benchmark_cache (
    run_id TEXT NOT NULL,
    checkpoint TEXT NOT NULL,
    device TEXT NOT NULL,
    amp INTEGER NOT NULL,
    avg_ms REAL NOT NULL,
    fps REAL NOT NULL,
    params INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, checkpoint, device, amp)
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    config_yaml TEXT,
    exp_dir TEXT,
    stdout_path TEXT,
    pid INTEGER,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
