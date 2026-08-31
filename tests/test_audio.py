"""Chunk planning: where to cut, and what never gets uploaded."""

from __future__ import annotations

from helmcode_whisper.pipeline import audio


def test_chunks_stay_under_the_api_limit() -> None:
    # Ten minutes of speech in twenty half-minute bursts.
    regions = [(index * 30.0, index * 30.0 + 25.0) for index in range(20)]
    chunks = audio.plan_chunks(regions)

    assert chunks
    for chunk in chunks:
        assert chunk.duration <= audio.MAX_CHUNK_SECONDS + 1e-6


def test_silence_between_regions_is_never_uploaded() -> None:
    # Two seconds of speech, an hour of nothing, two more seconds.
    regions = [(0.0, 2.0), (3600.0, 3602.0)]
    chunks = audio.plan_chunks(regions)

    assert len(chunks) == 2
    assert sum(chunk.duration for chunk in chunks) < 10.0


def test_a_long_unbroken_region_is_cut_with_overlap() -> None:
    # Comfortably past MAX_CHUNK_SECONDS, so the cut is forced whatever the
    # ceiling is set to. At 300 s this test stopped proving anything the day the
    # ceiling went from 110 s to 480.
    regions = [(0.0, audio.MAX_CHUNK_SECONDS * 3)]
    chunks = audio.plan_chunks(regions)

    assert len(chunks) > 2
    # Every chunk after the first starts inside its predecessor, and says so.
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.start < previous.end
        assert current.overlap_until == previous.end


def test_timestamps_are_absolute() -> None:
    """Chunk starts are offsets into the original recording, not into a splice."""
    regions = [(0.0, 10.0), (600.0, 610.0)]
    chunks = audio.plan_chunks(regions)

    assert chunks[-1].start >= 600.0


def test_tiny_regions_do_not_become_requests() -> None:
    regions = [(0.0, 0.2)]
    assert audio.plan_chunks(regions) == []
