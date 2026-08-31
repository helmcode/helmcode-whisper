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
    vec         BLOB NOT NULL,
    model       TEXT
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
    _migrate(connection)
    return connection


def _migrate(connection: sqlite3.Connection) -> None:
    """Bring an index written by an older version up to the current schema.

    Only additive changes belong here. `vectors.model` is left NULL on the rows
    that predate it rather than guessed at: `load_vectors` treats an unlabelled
    vector as belonging to whatever model is configured now, which is what
    produced it in every case an 0.1 can have created, and the dimension check
    catches the one case where that assumption is wrong.
    """
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(vectors)")}
    if "model" not in columns:
        connection.execute("ALTER TABLE vectors ADD COLUMN model TEXT")
        connection.commit()


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
    delete_meeting(connection, meeting_id, commit=False)
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
            # strict: a batch that comes back short would otherwise pair the
            # wrong vector with the rest of the passages from here on, and every
            # search after it would quietly return the wrong moment.
            batch_ids = passage_ids[offset : offset + len(batch)]
            for passage_id, vector in zip(batch_ids, vectors, strict=True):
                array = _normalize(np.asarray(vector, dtype=np.float32))
                connection.execute(
                    "INSERT OR REPLACE INTO vectors (passage_id, dim, vec, model) "
                    "VALUES (?, ?, ?, ?)",
                    (passage_id, array.size, array.tobytes(), embed_model),
                )
                embedded += 1

    connection.commit()
    return {"passages": len(passages), "embedded": embedded}


def delete_meeting(
    connection: sqlite3.Connection, meeting_id: str, *, commit: bool = True
) -> int:
    """Remove one meeting from the index. Returns how many passages went.

    Deleting the folder is not enough on its own: the passages outlive it and
    search keeps offering sentences from a recording that is no longer there.

    `passages_fts` is an external-content table, so the cascade from `meetings`
    does not reach it and its rows have to go first, while the passage ids they
    are keyed on still exist.
    """
    removed = connection.execute(
        "SELECT COUNT(*) FROM passages WHERE meeting_id = ?", (meeting_id,)
    ).fetchone()[0]
    connection.execute(
        "DELETE FROM passages_fts WHERE rowid IN (SELECT id FROM passages WHERE meeting_id = ?)",
        (meeting_id,),
    )
    connection.execute("DELETE FROM passages WHERE meeting_id = ?", (meeting_id,))
    connection.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
    if commit:
        connection.commit()
    return int(removed)


def load_vectors(
    connection: sqlite3.Connection, *, model: str
) -> tuple[list[int], np.ndarray, int]:
    """Every vector `model` produced, plus how many were left out.

    Two embedding models in one index is not a matter of taste. Their vectors do
    not live in the same space, so a cosine similarity across them is a number
    with no meaning; and when the dimensions differ too — 4096 for
    qwen3-embedding, something else for anything that replaces it — the reshape
    below used to fail with `cannot reshape array of size 9216 into shape
    (3,4096)`, which tells the person who changed `HCW_EMBED_MODEL` nothing
    about what they changed.

    So the current model's vectors are the ones searched, and the rest are
    counted and reported rather than dropped in silence: they are not gone, they
    are waiting for `hcw process` to run again on the meetings they came from.

    Unlabelled rows predate the `model` column and are taken at face value. The
    dimension filter is what stops that assumption doing damage if it is wrong.
    """
    rows = connection.execute(
        "SELECT passage_id, dim, vec, model FROM vectors ORDER BY passage_id"
    ).fetchall()
    if not rows:
        return [], np.empty((0, 0), dtype=np.float32), 0

    usable = [row for row in rows if row["model"] is None or row["model"] == model]
    if usable:
        # The newest vector decides the dimension: after a model change it is
        # the one the query vector will match.
        dim = usable[-1]["dim"]
        usable = [row for row in usable if row["dim"] == dim]
    stale = len(rows) - len(usable)
    if not usable:
        return [], np.empty((0, 0), dtype=np.float32), stale

    ids = [row["passage_id"] for row in usable]
    matrix = np.frombuffer(b"".join(row["vec"] for row in usable), dtype=np.float32)
    return ids, matrix.reshape(len(usable), dim), stale


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector
