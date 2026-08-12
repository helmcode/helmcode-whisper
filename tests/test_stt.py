"""Transcription: one pool across both tracks, and a cache keyed to the audio."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from helmcode_whisper.pipeline import stt
from helmcode_whisper.pipeline.audio import Chunk
from helmcode_whisper.pipeline.model import ME, OTHERS
from helmcode_whisper.store import Meeting


class FakeClient:
    """Records what it was asked to transcribe, and how much of it at once."""

    def __init__(self, payload_for=None, rendezvous: int = 0) -> None:
        self.calls: list[Path] = []
        self._payload_for = payload_for or (lambda path: {"text": path.stem, "language": "es"})
        self._lock = threading.Lock()
        self._in_flight = 0
        self.peak_in_flight = 0
        # When set, every call blocks until this many are inside at once. A
        # fake that returns immediately never overlaps with anything, so
        # counting concurrency without it only ever measures 1.
        self._barrier = threading.Barrier(rendezvous) if rendezvous else None

    def transcribe(self, path: Path, *, language=None, model=None) -> dict:
        with self._lock:
            self.calls.append(path)
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            if self._barrier is not None:
                self._barrier.wait(timeout=10)
            return self._payload_for(path)
        finally:
            with self._lock:
                self._in_flight -= 1


@pytest.fixture
def meeting(tmp_path: Path) -> Meeting:
    (tmp_path / "m").mkdir()
    return Meeting(tmp_path / "m")


@pytest.fixture
def no_encoding(monkeypatch):
    """Skip ffmpeg; record the chunk paths it would have written."""
    written: list[Path] = []

    def fake_encode(source: Path, chunk: Chunk, destination: Path) -> Path:
        written.append(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"opus")
        return destination

    monkeypatch.setattr(stt, "encode_chunk", fake_encode)
    return written


def plans(*, mic_chunks: int = 2, system_chunks: int = 2, fingerprint: str = "aaaa1111"):
    def chunks(count: int) -> list[Chunk]:
        return [Chunk(index, index * 100.0, index * 100.0 + 90.0) for index in range(count)]

    return [
        stt.TrackPlan("mic", Path("mic-16k.wav"), fingerprint, chunks(mic_chunks)),
        stt.TrackPlan("system", Path("system-16k.wav"), fingerprint, chunks(system_chunks)),
    ]


def test_both_tracks_go_through_one_pool(meeting, no_encoding) -> None:
    # Six chunks, three per track, and every one of them has to be inside the
    # client at the same time before any is allowed to return. A pool per track
    # tops out at the size of one track, so this cannot pass without the single
    # pool spanning both.
    client = FakeClient(rendezvous=6)

    result = stt.transcribe(
        client,
        meeting,
        plans(mic_chunks=3, system_chunks=3),
        language=None,
        model="whisper",
        concurrency=6,
    )

    assert len(client.calls) == 6
    assert result.total_chunks == 6
    assert client.peak_in_flight == 6


def test_segments_land_on_the_track_they_came_from(meeting, no_encoding) -> None:
    client = FakeClient()

    result = stt.transcribe(
        client, meeting, plans(mic_chunks=1, system_chunks=1), language=None, model="whisper"
    )

    assert [segment.speaker for segment in result.segments["mic"]] == [ME]
    assert [segment.speaker for segment in result.segments["system"]] == [OTHERS]


def test_a_replayed_run_is_reported_as_fully_cached(meeting, no_encoding) -> None:
    client = FakeClient()
    work = plans(mic_chunks=2, system_chunks=2)

    first = stt.transcribe(client, meeting, work, language=None, model="whisper")
    second = stt.transcribe(client, meeting, work, language=None, model="whisper")

    assert first.cache_hits == 0
    assert first.fully_cached is False
    assert second.cache_hits == 4
    assert second.fully_cached is True
    # The second run sent nothing.
    assert len(client.calls) == 4


def test_force_re_sends_every_chunk(meeting, no_encoding) -> None:
    client = FakeClient()
    work = plans(mic_chunks=2, system_chunks=0)

    stt.transcribe(client, meeting, work, language=None, model="whisper")
    replayed = stt.transcribe(client, meeting, work, language=None, model="whisper", force=True)

    assert replayed.cache_hits == 0
    assert len(client.calls) == 4


def test_chunk_audio_is_keyed_to_the_track_it_came_from(meeting, no_encoding) -> None:
    """The bug this guards against.

    `--force` re-derives the chunk plan from a freshly prepared track. Naming
    the encoded chunks by index alone means chunk 3 of the new plan is handed
    the file chunk 3 of the old plan left behind, so a stretch of the meeting
    gets transcribed as a different stretch of the meeting.
    """
    client = FakeClient()

    stt.transcribe(
        client, meeting, plans(mic_chunks=1, system_chunks=0, fingerprint="old00000"),
        language=None, model="whisper",
    )
    stt.transcribe(
        client, meeting, plans(mic_chunks=1, system_chunks=0, fingerprint="new11111"),
        language=None, model="whisper",
    )

    assert "old00000" in no_encoding[0].name
    assert "new11111" in no_encoding[1].name
    assert no_encoding[0] != no_encoding[1]


def test_a_different_language_does_not_reuse_the_transcription(meeting, no_encoding) -> None:
    client = FakeClient()
    work = plans(mic_chunks=1, system_chunks=0)

    stt.transcribe(client, meeting, work, language="es", model="whisper")
    switched = stt.transcribe(client, meeting, work, language="en", model="whisper")

    assert switched.cache_hits == 0
    assert len(client.calls) == 2
    # ...but the audio itself is identical, so it is encoded only once.
    assert len(set(no_encoding)) == 1


def test_words_are_sliced_into_the_segment_that_contains_them(meeting, no_encoding) -> None:
    payload = {
        "language": "es",
        "segments": [
            {"id": 0, "start": 0.0, "end": 2.0, "text": "hola que tal"},
            {"id": 1, "start": 2.0, "end": 4.0, "text": "muy bien"},
        ],
        "words": [
            {"word": " hola", "start": 0.0, "end": 0.5},
            {"word": " que", "start": 0.6, "end": 1.0},
            {"word": " tal", "start": 1.1, "end": 1.9},
            {"word": " muy", "start": 2.1, "end": 2.6},
            {"word": " bien", "start": 2.7, "end": 3.5},
        ],
    }
    client = FakeClient(payload_for=lambda path: payload)

    result = stt.transcribe(
        client, meeting, plans(mic_chunks=1, system_chunks=0), language=None, model="whisper"
    )

    first, second = result.segments["mic"]
    assert [word.text.strip() for word in first.words] == ["hola", "que", "tal"]
    assert [word.text.strip() for word in second.words] == ["muy", "bien"]


def test_word_timestamps_are_offset_into_the_recording(meeting, no_encoding) -> None:
    payload = {
        "segments": [{"id": 0, "start": 1.0, "end": 2.0, "text": "hola"}],
        "words": [{"word": " hola", "start": 1.0, "end": 2.0}],
    }
    client = FakeClient(payload_for=lambda path: payload)
    work = [stt.TrackPlan("mic", Path("mic.wav"), "ffff0000", [Chunk(0, 300.0, 390.0)])]

    result = stt.transcribe(client, meeting, work, language=None, model="whisper")

    segment = result.segments["mic"][0]
    assert segment.start == pytest.approx(301.0)
    assert segment.words[0].start == pytest.approx(301.0)


def test_an_explicit_language_wins_over_the_detected_one(meeting, no_encoding) -> None:
    client = FakeClient(payload_for=lambda path: {"text": "hola", "language": "es"})

    result = stt.transcribe(
        client, meeting, plans(mic_chunks=1, system_chunks=0), language="en", model="whisper"
    )

    assert result.language == "en"


def test_nothing_to_do_is_not_a_cached_run(meeting, no_encoding) -> None:
    client = FakeClient()

    result = stt.transcribe(
        client, meeting, plans(mic_chunks=0, system_chunks=0), language=None, model="whisper"
    )

    assert result.total_chunks == 0
    assert result.fully_cached is False
    assert result.segments == {"mic": [], "system": []}
