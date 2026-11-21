from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path
from typing import Any

from .config import DATABASE_PATH


def _pack_vector(vector: list[float] | None) -> bytes | None:
    if not vector:
        return None
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack_vector(data: bytes | None) -> list[float] | None:
    if not data:
        return None
    dimensions = len(data) // 4
    return list(struct.unpack(f"<{dimensions}f", data))


class IndexStore:
    """SQLite persistence for repositories, file fingerprints, code chunks, and vectors."""

    def __init__(self, database_path: Path = DATABASE_PATH):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS repositories (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL UNIQUE,
                    embedding_model TEXT,
                    indexed_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS files (
                    repo_id TEXT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
                    path TEXT NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    size INTEGER NOT NULL,
                    digest TEXT NOT NULL,
                    line_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (repo_id, path)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    repo_id TEXT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
                    chunk_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    line_numbers TEXT NOT NULL,
                    symbol TEXT,
                    text TEXT NOT NULL,
                    embedding BLOB,
                    embedding_model TEXT,
                    PRIMARY KEY (repo_id, chunk_id),
                    FOREIGN KEY (repo_id, path) REFERENCES files(repo_id, path) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS chunks_by_repo_path ON chunks(repo_id, path, ordinal);
            """)
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(files)").fetchall()}
            if "line_count" not in columns:
                connection.execute("ALTER TABLE files ADD COLUMN line_count INTEGER NOT NULL DEFAULT 0")

    def ensure_repository(self, repo_id: str, name: str, path: str, model_name: str | None) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO repositories(id,name,path,embedding_model) VALUES(?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET name=excluded.name,path=excluded.path",
                (repo_id, name, path, model_name),
            )

    def get_files(self, repo_id: str) -> dict[str, dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT path,mtime_ns,size,digest FROM files WHERE repo_id=?", (repo_id,)).fetchall()
        return {row["path"]: dict(row) for row in rows}

    def get_file_chunks(self, repo_id: str, path: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM chunks WHERE repo_id=? AND path=? ORDER BY ordinal", (repo_id, path)
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["chunk_id"], "path": row["path"], "ordinal": row["ordinal"],
            "start": row["start_line"], "end": row["end_line"],
            "line_numbers": json.loads(row["line_numbers"]), "symbol": row["symbol"],
            "text": row["text"], "embedding": _unpack_vector(row["embedding"]),
            "embedding_model": row["embedding_model"],
        }

    def replace_file(self, repo_id: str, path: str, mtime_ns: int, size: int, digest: str, line_count: int,
                     chunks: list[dict[str, Any]], vectors: list[list[float]] | None, model_name: str | None) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM chunks WHERE repo_id=? AND path=?", (repo_id, path))
            connection.execute(
                "INSERT INTO files(repo_id,path,mtime_ns,size,digest,line_count) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(repo_id,path) DO UPDATE SET mtime_ns=excluded.mtime_ns,size=excluded.size,digest=excluded.digest,line_count=excluded.line_count",
                (repo_id, path, mtime_ns, size, digest, line_count),
            )
            for ordinal, chunk in enumerate(chunks):
                chunk_id = chunk["id"]
                vector = vectors[ordinal] if vectors and ordinal < len(vectors) else None
                connection.execute(
                    "INSERT INTO chunks(repo_id,chunk_id,path,ordinal,start_line,end_line,line_numbers,symbol,text,embedding,embedding_model) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (repo_id, chunk_id, path, ordinal, chunk["start"], chunk["end"],
                     json.dumps(chunk["line_numbers"]), chunk.get("symbol"), chunk["text"],
                     _pack_vector(vector), model_name if vector else None),
                )

    def set_embeddings(self, repo_id: str, path: str, vectors: list[list[float]], model_name: str) -> None:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT chunk_id FROM chunks WHERE repo_id=? AND path=? ORDER BY ordinal", (repo_id, path)
            ).fetchall()
            for row, vector in zip(rows, vectors):
                connection.execute(
                    "UPDATE chunks SET embedding=?,embedding_model=? WHERE repo_id=? AND chunk_id=?",
                    (_pack_vector(vector), model_name, repo_id, row["chunk_id"]),
                )

    def remove_files_except(self, repo_id: str, paths: set[str]) -> int:
        with self.connect() as connection:
            rows = connection.execute("SELECT path FROM files WHERE repo_id=?", (repo_id,)).fetchall()
            removed = [row["path"] for row in rows if row["path"] not in paths]
            connection.executemany("DELETE FROM files WHERE repo_id=? AND path=?", [(repo_id, path) for path in removed])
        return len(removed)

    def load_repository(self, repo_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with self.connect() as connection:
            file_rows = connection.execute(
                "SELECT path,size,line_count FROM files WHERE repo_id=? ORDER BY path COLLATE NOCASE", (repo_id,)
            ).fetchall()
            chunk_rows = connection.execute(
                "SELECT * FROM chunks WHERE repo_id=? ORDER BY path COLLATE NOCASE,ordinal", (repo_id,)
            ).fetchall()
        files = [{"path": row["path"], "bytes": row["size"], "lines": row["line_count"]} for row in file_rows]
        return files, [self._row_to_chunk(row) for row in chunk_rows]

    def embeddings_ready(self, repo_id: str, path: str, model_name: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total, SUM(CASE WHEN embedding IS NOT NULL AND embedding_model=? THEN 1 ELSE 0 END) AS ready "
                "FROM chunks WHERE repo_id=? AND path=?", (model_name, repo_id, path)
            ).fetchone()
        return row["total"] == row["ready"]

    def model_for_repository(self, repo_id: str) -> str | None:
        with self.connect() as connection:
            row = connection.execute("SELECT embedding_model FROM repositories WHERE id=?", (repo_id,)).fetchone()
        return row["embedding_model"] if row else None

    def repository_has_embeddings(self, repo_id: str, model_name: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total, SUM(CASE WHEN embedding IS NOT NULL AND embedding_model=? THEN 1 ELSE 0 END) AS ready "
                "FROM chunks WHERE repo_id=?", (model_name, repo_id)
            ).fetchone()
        return bool(row["total"] and row["total"] == row["ready"])

    def update_model(self, repo_id: str, model_name: str | None) -> None:
        with self.connect() as connection:
            connection.execute("UPDATE repositories SET embedding_model=?,indexed_at=CURRENT_TIMESTAMP WHERE id=?", (model_name, repo_id))
