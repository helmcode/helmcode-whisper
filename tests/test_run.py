"""Orchestration: when diarization runs, when it is skipped, and how it is abandoned."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from helmcode_whisper.config import Config
from helmcode_whisper.pipeline import run
from helmcode_whisper.pipeline.audio import Chunk
from helmcode_whisper.store import Meeting


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        api_key="test-key",
        base_url="https://api.example.invalid/v1",
        hf_token="hf-token",
        home=tmp_path,
        stt_model="whisper",
        notes_model="notes",
        embed_model="embed",
        rerank_model="rerank",
        stt_concurrency=6,
    )


@pytest.fixture
def meeting(tmp_path: Path) -> Meeting:
    (tmp_path / "m").mkdir()
    return Meeting(tmp_path / "m")


def track(name: str = "system", *, chunks: int = 1) -> run._Track:
    return run._Track(
        name=name,
        path=Path(f"{name}-16k.wav"),
        fingerprint="abcd1234",
        chunks=[Chunk(index, 0.0, 90.0) for index in range(chunks)],
        duration_seconds=95.0,
        speech_seconds=54.0,
        from_cache=False,
    )


def test_no_diarize_skips_without_starting_a_thread(config, meeting) -> None:
    diarization = run._Diarization(
        meeting, config, track(), enabled=False, force=False
    )
    diarization.start()

    diarized, device, cached = diarization.apply([])

    assert (diarized, device, cached) == (False, None, False)
    assert diarization._thread is None


def test_a_missing_system_track_skips(config, meeting) -> None:
    diarization = run._Diarization(meeting, config, None, enabled=True, force=False)
    diarization.start()

    assert diarization.apply([])[0] is False
    assert diarization._thread is None


def test_a_silent_system_track_skips(config, meeting) -> None:
    """No speech means no chunks, which is known before transcription runs."""
    diarization = run._Diarization(
        meeting, config, track(chunks=0), enabled=True, force=False
    )
    diarization.start()

    assert diarization.apply([])[0] is False
    assert diarization._thread is None


def test_the_worker_is_a_daemon_so_a_failed_run_does_not_hang(
    config, meeting, monkeypatch
) -> None:
    """What stops a 30-minute hang after an ordinary API error.

    If transcription raises, nothing ever collects this result. A non-daemon
    worker keeps the interpreter alive until diarization finishes, so the user
    reads their error message and then waits out the whole diarization for
    output nobody will look at.
    """
    release = threading.Event()

    def blocking_work(self) -> run._DiarizationOutcome:
        release.wait(timeout=10)
        return run._DiarizationOutcome()

    monkeypatch.setattr(run._Diarization, "_work", blocking_work)

    diarization = run._Diarization(meeting, config, track(), enabled=True, force=False)
    diarization.start()
    try:
        assert diarization._thread is not None
        assert diarization._thread.daemon is True
    finally:
        release.set()
        diarization._thread.join(timeout=10)


def test_a_worker_that_explodes_is_reported_not_raised(config, meeting, monkeypatch) -> None:
    def boom(self) -> None:
        raise RuntimeError("torch fell over")

    monkeypatch.setattr(run._Diarization, "_work", boom)

    diarization = run._Diarization(meeting, config, track(), enabled=True, force=False)
    diarization.start()

    diarized, device, cached = diarization.apply([])

    assert diarized is False
    assert device is None
    assert "torch fell over" in diarization._outcome.problem


def test_turns_are_reused_from_the_cache_without_loading_pyannote(
    config, meeting, monkeypatch
) -> None:
    meeting.write_cached_json(
        {"turns": [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}], "device": "cpu"},
        "diarization-abcd1234.json",
    )

    def fail(*args, **kwargs):
        raise AssertionError("availability() must not be reached on a cache hit")

    monkeypatch.setattr(run.diarize, "availability", fail)

    diarization = run._Diarization(meeting, config, track(), enabled=True, force=False)
    diarization.start()

    from helmcode_whisper.pipeline.model import OTHERS, Segment

    segments = [Segment(1.0, 3.0, "hola", "system", OTHERS)]
    diarized, device, cached = diarization.apply(segments)

    assert (diarized, device, cached) == (True, "cpu", True)
    assert segments[0].speaker == "SPEAKER_00"
