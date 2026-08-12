"""The machine-readable step stream, which a front end is entitled to trust."""

from __future__ import annotations

import io
import json
import threading

from helmcode_whisper.pipeline import progress


def read(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def test_the_default_progress_does_nothing_and_costs_nothing() -> None:
    """Callers should never have to check whether they were given one."""
    quiet = progress.Progress()

    quiet.event(event="start")
    quiet.step("prepare", "running")
    quiet.close()


def test_each_event_is_one_json_object_on_its_own_line() -> None:
    stream = io.StringIO()
    events = progress.JsonProgress(stream)

    events.event(event="start", meeting="a")
    events.step("prepare", "done", seconds=1.5)

    assert read(stream) == [
        {"event": "start", "meeting": "a"},
        {"event": "step", "step": "prepare", "state": "done", "seconds": 1.5},
    ]


def test_the_stream_is_ascii_whatever_the_pipe_encoding_is() -> None:
    """The bug this guards against was mojibake in an interchange format.

    On Windows a redirected stdout carries the locale's encoding, so a meeting
    called "Revisión" was written as cp1252 bytes into a stream any reader would
    decode as UTF-8. Escaped ASCII means the same thing down every pipe.
    """
    stream = io.StringIO()

    progress.JsonProgress(stream).event(event="start", title="Revisión de precios")

    raw = stream.getvalue()
    assert raw.isascii()
    assert json.loads(raw)["title"] == "Revisión de precios"


def test_events_from_many_threads_stay_whole() -> None:
    """Chunk completions arrive from the transcription pool.

    Two half-written lines are not JSON, and a reader parsing line by line has
    no way to recover from one.
    """
    stream = io.StringIO()
    events = progress.JsonProgress(stream)

    def emit(index: int) -> None:
        for step in range(20):
            events.event(event="chunks", done=step, total=20, worker=index)

    threads = [threading.Thread(target=emit, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    parsed = read(stream)
    assert len(parsed) == 120
    assert all(item["event"] == "chunks" for item in parsed)


def test_a_closed_reader_does_not_take_the_run_down() -> None:
    """The front end going away is its business, not the pipeline's."""
    stream = io.StringIO()
    events = progress.JsonProgress(stream)
    stream.close()

    events.event(event="start")  # must not raise


def test_the_diarization_estimate_comes_from_the_measured_factor() -> None:
    assert progress.estimate_diarization_seconds(3600) == int(
        3600 * progress.CPU_DIARIZATION_FACTOR
    )
    assert progress.estimate_diarization_seconds(600) == int(
        600 * progress.CPU_DIARIZATION_FACTOR
    )


def test_no_estimate_is_offered_for_hardware_nobody_measured() -> None:
    """A number that turns out five times wrong is worse than no number.

    Only CPU has been measured, and this is the step where the wait is long
    enough that people watch the estimate rather than glance at it.
    """
    assert progress.estimate_diarization_seconds(3600, device="cuda") is None
    assert progress.estimate_diarization_seconds(0) is None
    assert progress.estimate_diarization_seconds(-5) is None


def test_the_step_names_cover_the_pipeline() -> None:
    """A front end lays out its rows from this, so it has to be the real list."""
    assert progress.STEPS == ("prepare", "transcribe", "diarize", "merge", "notes", "index")
