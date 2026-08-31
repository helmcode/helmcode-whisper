"""Passage building and the keyword fallback that works without an API key."""

from __future__ import annotations

import sqlite3

import numpy as np

from helmcode_whisper.pipeline import index
from helmcode_whisper.pipeline.model import ME, Segment


def test_passages_group_by_speaker() -> None:
    segments = [
        Segment(0.0, 2.0, "Hola a todos.", "mic", ME),
        Segment(2.0, 4.0, "Empezamos con el precio.", "mic", ME),
        Segment(4.0, 6.0, "Me parece bien.", "system", "SPEAKER_00"),
    ]
    passages = index.build_passages(segments)

    assert len(passages) == 2
    assert passages[0].speaker == ME
    assert passages[0].text == "Hola a todos. Empezamos con el precio."
    assert passages[1].speaker == "SPEAKER_00"


def test_dropped_segments_are_not_indexed() -> None:
    segments = [
        Segment(0.0, 2.0, "Eco del altavoz.", "mic", ME, dropped="echo"),
        Segment(0.0, 2.0, "Eco del altavoz.", "system", "SPEAKER_00"),
    ]
    passages = index.build_passages(segments)

    assert len(passages) == 1
    assert passages[0].speaker == "SPEAKER_00"


def test_a_long_monologue_is_split() -> None:
    segments = [
        Segment(start, start + 5.0, "palabra " * 30, "system", "SPEAKER_00")
        for start in range(0, 180, 5)
    ]
    passages = index.build_passages(segments)

    assert len(passages) > 1
    for passage in passages:
        assert passage.end - passage.start <= index.MAX_PASSAGE_SECONDS + 5.0


def test_keyword_search_works_without_embeddings(tmp_path) -> None:
    connection = index.connect(tmp_path / "index.sqlite3")
    stats = index.index_meeting(
        connection,
        None,  # no API client: this is the degraded path
        meeting_id="2026-08-11-pricing",
        title="Pricing",
        date="2026-08-11",
        path=tmp_path,
        passages=[index.Passage(0.0, 5.0, ME, "Subimos el precio por usuario en septiembre")],
        embed_model="qwen3-embedding",
    )

    assert stats == {"passages": 1, "embedded": 0}
    rows = connection.execute(
        "SELECT rowid FROM passages_fts WHERE passages_fts MATCH 'precio'"
    ).fetchall()
    assert len(rows) == 1


def test_reindexing_replaces_rather_than_duplicates(tmp_path) -> None:
    db = tmp_path / "index.sqlite3"
    connection = index.connect(db)
    passage = index.Passage(0.0, 5.0, ME, "Una sola vez")
    for _ in range(3):
        index.index_meeting(
            connection,
            None,
            meeting_id="same-meeting",
            title="Same",
            date="2026-08-11",
            path=tmp_path,
            passages=[passage],
            embed_model="qwen3-embedding",
        )

    count = connection.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
    assert count == 1


def test_deleting_a_meeting_takes_its_passages_and_vectors(tmp_path) -> None:
    """A folder deleted from disk must not keep answering searches."""
    import numpy as np

    connection = index.connect(tmp_path / "index.sqlite3")
    for meeting_id in ("a", "b"):
        index.index_meeting(
            connection,
            None,
            meeting_id=meeting_id,
            title=meeting_id,
            date="2026-08-12",
            path=tmp_path / meeting_id,
            passages=[index.Passage(0.0, 2.0, ME, f"hola desde {meeting_id}")],
            embed_model="embed",
        )
    connection.execute(
        "INSERT INTO vectors (passage_id, dim, vec) "
        "SELECT id, 2, ? FROM passages WHERE meeting_id = 'a'",
        (np.zeros(2, dtype=np.float32).tobytes(),),
    )
    connection.commit()

    removed = index.delete_meeting(connection, "a")

    assert removed == 1
    assert connection.execute("SELECT COUNT(*) FROM meetings").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM passages").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] == 0
    # The full-text table is external-content, so the cascade never reaches it.
    hits = connection.execute(
        "SELECT rowid FROM passages_fts WHERE passages_fts MATCH 'hola'"
    ).fetchall()
    assert len(hits) == 1
    connection.close()


def test_two_embedding_models_in_one_index_do_not_crash_the_search(tmp_path) -> None:
    """The bug this replaced: a raw numpy reshape error out of `hcw search`.

    Changing HCW_EMBED_MODEL left the archive holding vectors of two widths, and
    `load_vectors` reshaped the lot to the first row's dimension —
    "cannot reshape array of size 9216 into shape (3,4096)", from a command the
    user ran to look something up.
    """
    connection = index.connect(tmp_path / "index.sqlite3")
    connection.execute(
        "INSERT INTO meetings (id, title, date, path) VALUES ('m', 't', '2026-01-01', '/x')"
    )
    for model, dim in (("old-embed", 8), ("old-embed", 8), ("new-embed", 4)):
        cursor = connection.execute(
            "INSERT INTO passages (meeting_id, start, end, speaker, text) "
            "VALUES ('m', 0, 1, 'Me', 'hola')"
        )
        vector = np.ones(dim, dtype=np.float32)
        connection.execute(
            "INSERT INTO vectors (passage_id, dim, vec, model) VALUES (?, ?, ?, ?)",
            (cursor.lastrowid, dim, vector.tobytes(), model),
        )
    connection.commit()

    ids, matrix, stale = index.load_vectors(connection, model="new-embed")

    assert matrix.shape == (1, 4)
    assert len(ids) == 1
    # The two the current model cannot compare against are reported, not hidden.
    assert stale == 2
    connection.close()


def test_vectors_from_an_older_index_are_still_searchable(tmp_path) -> None:
    """Upgrading must not blank the archive, so an unlabelled vector counts."""
    db = tmp_path / "index.sqlite3"
    connection = index.connect(db)
    connection.execute(
        "INSERT INTO meetings (id, title, date, path) VALUES ('m', 't', '2026-01-01', '/x')"
    )
    cursor = connection.execute(
        "INSERT INTO passages (meeting_id, start, end, speaker, text) "
        "VALUES ('m', 0, 1, 'Me', 'hola')"
    )
    # Written before the `model` column existed.
    connection.execute(
        "INSERT INTO vectors (passage_id, dim, vec) VALUES (?, ?, ?)",
        (cursor.lastrowid, 4, np.ones(4, dtype=np.float32).tobytes()),
    )
    connection.commit()
    connection.close()

    connection = index.connect(db)
    ids, matrix, stale = index.load_vectors(connection, model="qwen3-embedding")

    assert len(ids) == 1
    assert matrix.shape == (1, 4)
    assert stale == 0
    connection.close()


def test_the_model_column_is_added_to_an_index_that_predates_it(tmp_path) -> None:
    db = tmp_path / "index.sqlite3"
    connection = sqlite3.connect(db)
    connection.executescript(
        """
        CREATE TABLE vectors (
            passage_id  INTEGER PRIMARY KEY,
            dim         INTEGER NOT NULL,
            vec         BLOB NOT NULL
        );
        """
    )
    connection.commit()
    connection.close()

    connection = index.connect(db)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(vectors)")}

    assert "model" in columns
    connection.close()
