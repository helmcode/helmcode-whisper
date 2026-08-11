"""Passage building and the keyword fallback that works without an API key."""

from __future__ import annotations

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
