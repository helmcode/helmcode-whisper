"""The searchable history: SQLite, full-text, and 4096-dimension vectors.

No vector extension. sqlite-vec would mean a native build on every platform, and
at this scale it buys nothing: 500 meetings is roughly 30k passages, and a brute
force dot product over 30k rows finishes faster than the network round trip that
produced the query vector. Vectors are L2-normalized on the way in, so cosine
similarity is a plain matrix multiply.

FTS5 ships with SQLite, so keyword search works whether or not the embedding
endpoint was reachable when a meeting was processed.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..api import HelmcodeClient
from .model import Segment

EMBED_BATCH = 32
# Passage size is a compromise: long enough that an embedding has something to
# grip, short enough that a hit points at a moment rather than a chapter. The
# first real transcript argued for the shorter end — a single passage covering a
# minute of a meeting mixes three topics, and one vector cannot represent three
# topics well enough for any of them to be findable.
MAX_PASSAGE_SECONDS = 45.0
MAX_PASSAGE_CHARS = 450

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meetings (
    id      TEXT PRIMARY KEY,
    title   TEXT NOT NULL,
    date    TEXT NOT NULL,
    path    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS passages (
    id          INTEGER PRIMARY KEY,
    meeting_id  TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    start       REAL NOT NULL,
    end         REAL NOT NULL,
    speaker     TEXT NOT NULL,
    text        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS passages_meeting ON passages(meeting_id);

CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(
    text,
    content='passages',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS vectors (
    passage_id  INTEGER PRIMARY KEY REFERENCES passages(id) ON DELETE CASCADE,
    dim         INTEGER NOT NULL,
    vec         BLOB NOT NULL
);
"""


@dataclass(frozen=True)
class Passage:
    start: float
    end: float
    speaker: str
    text: str


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(_SCHEMA)
    return connection


def build_passages(segments: list[Segment]) -> list[Passage]:
    """Group consecutive same-speaker segments into search-sized passages."""
    passages: list[Passage] = []
    buffer: list[Segment] = []

    def flush() -> None:
        if not buffer:
            return
        text = " ".join(segment.text.strip() for segment in buffer).strip()
        if text:
            passages.append(Passage(buffer[0].start, buffer[-1].end, buffer[0].speaker, text))
        buffer.clear()

    for segment in segments:
        if segment.dropped or not segment.text.strip():
            continue
        if buffer:
            same_speaker = segment.speaker == buffer[0].speaker
            too_long = segment.end - buffer[0].start > MAX_PASSAGE_SECONDS
            too_wide = sum(len(item.text) for item in buffer) > MAX_PASSAGE_CHARS
            if not same_speaker or too_long or too_wide:
                flush()
        buffer.append(segment)
    flush()
    return passages


def index_meeting(
    connection: sqlite3.Connection,
    client: HelmcodeClient | None,
    *,
    meeting_id: str,
    title: str,
    date: str,
    path: Path,
    passages: list[Passage],
    embed_model: str,
) -> dict[str, int]:
    """Replace this meeting's rows. Returns counts for the run summary."""
    connection.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
    connection.execute(
        "DELETE FROM passages_fts WHERE rowid IN (SELECT id FROM passages WHERE meeting_id = ?)",
        (meeting_id,),
    )
    connection.execute("DELETE FROM passages WHERE meeting_id = ?", (meeting_id,))
    connection.execute(
        "INSERT INTO meetings (id, title, date, path) VALUES (?, ?, ?, ?)",
        (meeting_id, title, date, str(path)),
    )

    passage_ids: list[int] = []
    for passage in passages:
        cursor = connection.execute(
            "INSERT INTO passages (meeting_id, start, end, speaker, text) VALUES (?, ?, ?, ?, ?)",
            (meeting_id, passage.start, passage.end, passage.speaker, passage.text),
        )
        passage_id = int(cursor.lastrowid)
        passage_ids.append(passage_id)
        connection.execute(
            "INSERT INTO passages_fts (rowid, text) VALUES (?, ?)", (passage_id, passage.text)
        )

    embedded = 0
    if client is not None and passages:
        for offset in range(0, len(passages), EMBED_BATCH):
            batch = passages[offset : offset + EMBED_BATCH]
            vectors = client.embed([item.text for item in batch], model=embed_model)
            for passage_id, vector in zip(passage_ids[offset:], vectors, strict=False):
                array = _normalize(np.asarray(vector, dtype=np.float32))
                connection.execute(
                    "INSERT OR REPLACE INTO vectors (passage_id, dim, vec) VALUES (?, ?, ?)",
                    (passage_id, array.size, array.tobytes()),
                )
                embedded += 1

    connection.commit()
    return {"passages": len(passages), "embedded": embedded}


def load_vectors(connection: sqlite3.Connection) -> tuple[list[int], np.ndarray]:
    rows = connection.execute("SELECT passage_id, dim, vec FROM vectors").fetchall()
    if not rows:
        return [], np.empty((0, 0), dtype=np.float32)
    dim = rows[0]["dim"]
    ids = [row["passage_id"] for row in rows]
    matrix = np.frombuffer(b"".join(row["vec"] for row in rows), dtype=np.float32)
    return ids, matrix.reshape(len(rows), dim)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector
