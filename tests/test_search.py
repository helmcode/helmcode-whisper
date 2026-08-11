"""Rank fusion, and the self-contained HTML export."""

from __future__ import annotations

from helmcode_whisper.pipeline import search
from helmcode_whisper.pipeline.model import ME, Segment, Transcript
from helmcode_whisper.ui.html import render_html


def hit(passage_id: int) -> search.Hit:
    return search.Hit(
        passage_id=passage_id,
        meeting_title="m",
        meeting_date="2026-08-11",
        start=0.0,
        speaker=ME,
        text=f"passage {passage_id}",
        score=0.0,
        source="vector",
    )


def test_agreeing_signals_win() -> None:
    candidates = [hit(1), hit(2), hit(3)]
    # Vector and keyword both prefer 2; the reranker prefers 3.
    ordered = search._fuse(candidates, [[2, 1, 3], [2, 3, 1], [3, 2, 1]])

    assert [item.passage_id for item in ordered] == [2, 3, 1]


def test_one_dissenting_signal_cannot_overturn_the_others() -> None:
    """The bug this replaced: a near-tied rerank flipping a decisive vector win."""
    candidates = [hit(1), hit(2)]
    vector = [1, 2]
    keyword = [1]
    rerank = [2, 1]

    ordered = search._fuse(candidates, [vector, keyword, rerank])

    assert ordered[0].passage_id == 1


def test_a_passage_missing_from_a_ranking_still_places() -> None:
    candidates = [hit(1), hit(2)]
    # Only the vector stage saw passage 2 at all.
    ordered = search._fuse(candidates, [[1, 2], [1]])

    assert [item.passage_id for item in ordered] == [1, 2]
    assert ordered[1].score > 0


def test_exported_html_makes_no_network_requests() -> None:
    """A shared notes.html that phones home would make the README a lie."""
    transcript = Transcript(
        segments=[Segment(0.0, 2.0, "Hola & <adiós>", "mic", ME)],
        language="es",
        speakers=[ME],
    )
    html = render_html(
        {
            "summary": "Short one.",
            "decisions": [],
            "action_items": [],
            "open_questions": [],
            "quotes": [{"speaker": ME, "text": "Hola"}],
        },
        {"title": "Standup", "started_at": "2026-08-11T10:00:00", "duration_seconds": 120},
        transcript,
    )

    assert "http://" not in html
    assert "https://" not in html
    assert "<script" not in html.lower()
    # Content is escaped, not injected.
    assert "&amp; &lt;adiós&gt;" in html
