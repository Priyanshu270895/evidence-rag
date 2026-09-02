import json
import re
import sqlite3
from pathlib import Path

import numpy as np

FTS_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


class ChunkStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY, document TEXT NOT NULL, page INTEGER NOT NULL,
                text TEXT NOT NULL, embedding TEXT NOT NULL
            );
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

    def add(self, rows: list[dict]) -> None:
        self.connection.executemany(
            """
            INSERT INTO chunks(id, document, page, text, embedding)
            VALUES(?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
              document = excluded.document,
              page = excluded.page,
              text = excluded.text,
              embedding = excluded.embedding
            """,
            [
                (r["id"], r["document"], r["page"], r["text"], json.dumps(r["embedding"]))
                for r in rows
            ],
        )
        self.connection.commit()

    def keyword_search(self, query: str, limit: int) -> list[str]:
        tokens = [token.lower() for token in FTS_TOKEN_PATTERN.findall(query) if len(token) > 2]
        terms = " OR ".join(f'"{token}"' for token in tokens)
        if not terms:
            return []
        rows = self.connection.execute(
            "SELECT id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (terms, limit),
        ).fetchall()
        return [row["id"] for row in rows]

    def vector_search(self, query_embedding: list[float], limit: int) -> list[str]:
        rows = self.connection.execute("SELECT id, embedding FROM chunks").fetchall()
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
