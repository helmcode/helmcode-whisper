"""Speech to text against the Helmcode Whisper endpoint.

A 60-minute meeting is roughly 35 chunks per track, so 70 requests. Sending them
one after another would make `process` take longer than the meeting did; sending
all of them at once earns rate limits. A small fixed pool is the whole trick.

Every chunk result is cached under the meeting's `.cache/`, keyed by the audio's
content hash and the request parameters. Re-running `process` after a failure
downstream costs nothing and repeats no inference.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..api import HelmcodeClient
from ..store import Meeting, file_fingerprint
from .audio import Chunk, encode_chunk
from .model import ME, OTHERS, Segment, Word

# Concurrent uploads in flight. Four keeps a 60-minute meeting moving without
# turning a shared endpoint into a queue of our own making.
CONCURRENCY = 4


def transcribe_track(
    client: HelmcodeClient,
    meeting: Meeting,
    *,
    track: str,
    prepared_wav: Path,
    chunks: list[Chunk],
    language: str | None,
    model: str,
    force: bool = False,
    on_chunk_done: Callable[[], None] | None = None,
) -> tuple[list[Segment], str | None, int]:
    """Transcribe every chunk of one track.

    Returns the segments in wall-clock order, the detected language, and how
    many chunks came from cache — the last so the caller can tell a measured
    run from a replayed one when it writes down timings.
    """
    if not chunks:
        return [], None, 0

    fingerprint = file_fingerprint(prepared_wav, extra=f"{model}|{language or 'auto'}")
    speaker = ME if track == "mic" else OTHERS
    cache_hits = 0

    def transcribe_one(chunk: Chunk) -> dict[str, Any]:
        nonlocal cache_hits
        cache_key = f"{track}-{fingerprint}-{chunk.index:04d}.json"
        if not force:
            cached = meeting.read_cached_json("stt", cache_key)
            if cached is not None:
                cache_hits += 1
                return cached

        chunk_path = meeting.cache_path("chunks", f"{track}-{chunk.index:04d}.ogg")
        encode_chunk(prepared_wav, chunk, chunk_path)
        payload = client.transcribe(chunk_path, language=language, model=model)
        meeting.write_cached_json(payload, "stt", cache_key)
        return payload

    def worker(chunk: Chunk) -> tuple[Chunk, dict[str, Any]]:
        try:
            return chunk, transcribe_one(chunk)
        finally:
            if on_chunk_done:
                on_chunk_done()

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        results = list(pool.map(worker, chunks))

    segments: list[Segment] = []
    detected_language: str | None = None
    for chunk, payload in results:
        detected_language = detected_language or payload.get("language")
        segments.extend(_segments_from(payload, chunk, track, speaker))

    segments.sort(key=lambda segment: segment.start)
    return segments, detected_language, cache_hits


def _segments_from(
    payload: dict[str, Any], chunk: Chunk, track: str, speaker: str
) -> list[Segment]:
    raw = payload.get("segments")
    if not raw:
        # `verbose_json` without segments still carries the full text; better a
        # chunk-sized block than a hole in the transcript.
        text = (payload.get("text") or "").strip()
        return [Segment(chunk.start, chunk.end, text, track, speaker)] if text else []

    words = [
        Word(
            start=chunk.start + float(word.get("start", 0.0)),
            end=chunk.start + float(word.get("end", 0.0)),
            text=str(word.get("word") or word.get("text") or ""),
        )
        for word in (payload.get("words") or [])
    ]

    segments: list[Segment] = []
    for item in raw:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        start = chunk.start + float(item.get("start", 0.0))
        end = chunk.start + float(item.get("end", 0.0))
        # Drop what the previous chunk already covered, so a forced overlap does
        # not stutter in the final transcript.
        if chunk.overlap_until and start < chunk.overlap_until - 0.05:
            continue
        segments.append(
            Segment(
                start=start,
                end=max(end, start),
                text=text,
                track=track,
                speaker=speaker,
                confidence=item.get("avg_logprob"),
                # Words are returned for the chunk as a whole, so hand each
                # segment the ones that fall inside it.
                words=[word for word in words if start - 0.01 <= word.start < end + 0.01] or None,
            )
        )
    return segments
