PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS zones (
    zone_id TEXT PRIMARY KEY,
    embedding_dim INTEGER NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    doc_id TEXT NOT NULL,
    zone_id TEXT NOT NULL,
    record_type TEXT NOT NULL DEFAULT 'knowledge',
    text TEXT NOT NULL,
    filename TEXT,
    source_path TEXT,
    extension TEXT,
    chunk_index INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    char_len INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    mtime INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    file_type TEXT NOT NULL,
    language TEXT,
    is_deleted INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    FOREIGN KEY(zone_id) REFERENCES zones(zone_id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_zone_doc ON chunks(zone_id, doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_zone_record ON chunks(zone_id, record_type);
CREATE INDEX IF NOT EXISTS idx_chunks_zone_deleted ON chunks(zone_id, is_deleted);

CREATE TABLE IF NOT EXISTS chunk_vectors (
    chunk_id TEXT PRIMARY KEY,
    zone_id TEXT NOT NULL,
    vector BLOB NOT NULL,
    dim INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_chunk_vectors_zone ON chunk_vectors(zone_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    text,
    filename,
    content='chunks',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunk_fts(rowid, text, filename)
    VALUES (new.id, new.text, COALESCE(new.filename, ''));
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, text, filename)
    VALUES('delete', old.id, old.text, COALESCE(old.filename, ''));
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, text, filename)
    VALUES('delete', old.id, old.text, COALESCE(old.filename, ''));
    INSERT INTO chunk_fts(rowid, text, filename)
    VALUES (new.id, new.text, COALESCE(new.filename, ''));
END;
