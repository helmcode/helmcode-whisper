"""Meeting folders, and the cache that makes `process` resumable."""

from __future__ import annotations

from datetime import datetime

from helmcode_whisper.store import Meeting, slugify


def test_slugify_handles_accents_and_punctuation() -> None:
    assert slugify("Revisión de precios — Q3 (2026)") == "revision-de-precios-q3-2026"
    assert slugify("¿¡!?") == "meeting"


def test_two_meetings_the_same_day_do_not_collide(tmp_path) -> None:
    when = datetime(2026, 8, 11, 10, 0, 0)
    first = Meeting.create(tmp_path, "standup", when)
    second = Meeting.create(tmp_path, "standup", when)

    assert first.path != second.path
    assert first.path.name == "2026-08-11-standup"
    assert second.path.name == "2026-08-11-standup-2"


def test_latest_returns_the_most_recent(tmp_path) -> None:
    Meeting.create(tmp_path, "older", datetime(2026, 8, 10, 9, 0, 0))
    newest = Meeting.create(tmp_path, "newer", datetime(2026, 8, 11, 9, 0, 0))

    assert Meeting.latest(tmp_path).path == newest.path


def test_meta_updates_merge(tmp_path) -> None:
    meeting = Meeting.create(tmp_path, "standup", datetime(2026, 8, 11, 10, 0, 0))
    meeting.save_meta({"duration_seconds": 120})
    meeting.save_meta({"language": "es"})

    meta = meeting.load_meta()
    assert meta["title"] == "standup"
    assert meta["duration_seconds"] == 120
    assert meta["language"] == "es"


def test_a_truncated_cache_entry_is_ignored_not_fatal(tmp_path) -> None:
    meeting = Meeting.create(tmp_path, "standup", datetime(2026, 8, 11, 10, 0, 0))
    meeting.write_cached_json({"ok": True}, "stt", "chunk.json")
    assert meeting.read_cached_json("stt", "chunk.json") == {"ok": True}

    meeting.cache_path("stt", "chunk.json").write_text('{"ok": tr', encoding="utf-8")
    assert meeting.read_cached_json("stt", "chunk.json") is None
