import hashlib
import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

FTS_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def legacy_document_id(filename: str) -> str:
    digest = hashlib.sha256(f"legacy:{filename}".encode()).hexdigest()[:20]
    return f"legacy-{digest}"


class ChunkStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_hash TEXT NOT NULL UNIQUE,
                storage_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                page_count INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                chunk_size INTEGER NOT NULL,
                chunk_overlap INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                document TEXT NOT NULL,
                page INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_document_page ON chunks(document_id, page);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                id UNINDEXED, text, content='chunks', content_rowid='rowid'
            );
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
              INSERT INTO chunks_fts(rowid, id, text) VALUES (new.rowid, new.id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
              INSERT INTO chunks_fts(chunks_fts, rowid, id, text)
              VALUES('delete', old.rowid, old.id, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
              INSERT INTO chunks_fts(chunks_fts, rowid, id, text)
              VALUES('delete', old.rowid, old.id, old.text);
              INSERT INTO chunks_fts(rowid, id, text) VALUES (new.rowid, new.id, new.text);
            END;
            """
        )
        self._migrate_legacy_chunks()

    def _migrate_legacy_chunks(self) -> None:
        columns = self._table_columns("chunks")
        if not columns:
            return
        if "document_id" not in columns:
            self.connection.execute("ALTER TABLE chunks ADD COLUMN document_id TEXT")
        if "chunk_index" not in columns:
            self.connection.execute("ALTER TABLE chunks ADD COLUMN chunk_index INTEGER DEFAULT 0")

        legacy_rows = self.connection.execute(
            """
            SELECT document, COUNT(*) AS chunk_count, MAX(page) AS page_count
            FROM chunks
            WHERE document_id IS NULL OR document_id = ''
            GROUP BY document
            """
        ).fetchall()
        for row in legacy_rows:
            document_id = legacy_document_id(row["document"])
            timestamp = utc_now()
            self.connection.execute(
                """
                INSERT OR IGNORE INTO documents(
                    document_id, filename, file_hash, storage_path, size_bytes, page_count,
                    chunk_count, status, chunk_size, chunk_overlap, created_at, updated_at
                )
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    document_id,
                    row["document"],
                    document_id,
                    "",
                    0,
                    row["page_count"] or 0,
                    row["chunk_count"],
                    "legacy",
                    0,
                    0,
                    timestamp,
                    timestamp,
                ),
            )
            self.connection.execute(
                "UPDATE chunks SET document_id = ? WHERE document = ?",
                (document_id, row["document"]),
            )
        self.connection.commit()

    def _table_columns(self, table_name: str) -> set[str]:
        rows = self.connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {row["name"] for row in rows}

    def health_check(self) -> bool:
        self.connection.execute("SELECT 1").fetchone()
        return True

    def save_document(self, document: dict, rows: list[dict]) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO documents(
                    document_id, filename, file_hash, storage_path, size_bytes, page_count,
                    chunk_count, status, chunk_size, chunk_overlap, created_at, updated_at
                )
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(document_id) DO UPDATE SET
                    filename = excluded.filename,
                    file_hash = excluded.file_hash,
                    storage_path = excluded.storage_path,
                    size_bytes = excluded.size_bytes,
                    page_count = excluded.page_count,
                    chunk_count = excluded.chunk_count,
                    status = excluded.status,
                    chunk_size = excluded.chunk_size,
                    chunk_overlap = excluded.chunk_overlap,
                    updated_at = excluded.updated_at
                """,
                (
                    document["document_id"],
                    document["filename"],
                    document["file_hash"],
                    document["storage_path"],
                    document["size_bytes"],
                    document["page_count"],
                    document["chunk_count"],
                    document["status"],
                    document["chunk_size"],
                    document["chunk_overlap"],
                    document["created_at"],
                    document["updated_at"],
                ),
            )
            self.connection.execute(
                "DELETE FROM chunks WHERE document_id = ?", (document["document_id"],)
            )
            self._upsert_chunks(rows)

    def add(self, rows: list[dict]) -> None:
        with self.connection:
            self._upsert_chunks(rows)

    def _upsert_chunks(self, rows: list[dict]) -> None:
        if not rows:
            return
        self.connection.executemany(
            """
            INSERT INTO chunks(id, document_id, document, page, chunk_index, text, embedding)
            VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
              document_id = excluded.document_id,
              document = excluded.document,
              page = excluded.page,
              chunk_index = excluded.chunk_index,
              text = excluded.text,
              embedding = excluded.embedding
            """,
            [
                (
                    r["id"],
                    r.get("document_id", "test-document"),
                    r["document"],
                    r["page"],
                    r.get("chunk_index", 0),
                    r["text"],
                    json.dumps(r["embedding"]),
                )
                for r in rows
            ],
        )

    def get_document_by_hash(self, file_hash: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM documents WHERE file_hash = ?", (file_hash,)
        ).fetchone()

    def get_document(self, document_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()

    def list_documents(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            "SELECT * FROM documents ORDER BY created_at DESC, filename ASC"
        ).fetchall()

    def delete_document(self, document_id: str) -> bool:
        with self.connection:
            existing = self.get_document(document_id)
            if existing is None:
                return False
            self.connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            self.connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            return True

    def chunks_for_document(self, document_id: str) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT id, document_id, document, page, chunk_index, text
            FROM chunks
            WHERE document_id = ?
            ORDER BY page ASC, chunk_index ASC
            """,
            (document_id,),
        ).fetchall()

    def keyword_search(
        self, query: str, limit: int, document_ids: list[str] | None = None
    ) -> list[str]:
        tokens = [token.lower() for token in FTS_TOKEN_PATTERN.findall(query) if len(token) > 2]
        terms = " OR ".join(f'"{token}"' for token in tokens)
        if not terms:
            return []

        params: list[object] = [terms]
        document_filter = ""
        if document_ids:
            marks = ",".join("?" for _ in document_ids)
            document_filter = f" AND chunks.document_id IN ({marks})"
            params.extend(document_ids)
        params.append(limit)

        rows = self.connection.execute(
            f"""
            SELECT chunks.id
            FROM chunks_fts
            JOIN chunks ON chunks_fts.rowid = chunks.rowid
            WHERE chunks_fts MATCH ?{document_filter}
            ORDER BY bm25(chunks_fts)
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [row["id"] for row in rows]

    def vector_search(
        self, query_embedding: list[float], limit: int, document_ids: list[str] | None = None
    ) -> list[str]:
        params: list[object] = []
        document_filter = ""
        if document_ids:
            marks = ",".join("?" for _ in document_ids)
            document_filter = f" WHERE document_id IN ({marks})"
            params.extend(document_ids)
        rows = self.connection.execute(
            f"SELECT id, embedding FROM chunks{document_filter}", params
        ).fetchall()
        if not rows:
            return []
        query = np.asarray(query_embedding, dtype=np.float32)
        scored = []
        for row in rows:
            vector = np.asarray(json.loads(row["embedding"]), dtype=np.float32)
            score = float(
                np.dot(query, vector) / (np.linalg.norm(query) * np.linalg.norm(vector) + 1e-9)
            )
            scored.append((row["id"], score))
        return [item[0] for item in sorted(scored, key=lambda item: item[1], reverse=True)[:limit]]

    def get(self, ids: list[str]) -> dict[str, sqlite3.Row]:
        if not ids:
            return {}
        marks = ",".join("?" for _ in ids)
        rows = self.connection.execute(
            f"SELECT * FROM chunks WHERE id IN ({marks})", ids
        ).fetchall()
        return {row["id"]: row for row in rows}
