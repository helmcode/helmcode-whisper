"""Speech to text against the Helmcode Whisper endpoint.

A 60-minute meeting is roughly 8 chunks per track, so 16 requests. Sending them
one after another wastes most of the wall clock waiting on HTTP; sending all of
them at once earns rate limits. A small fixed pool is the whole trick.

It used to be ~70 requests, back when the endpoint refused a chunk longer than
two minutes. The pool mattered more then and it still pays for itself, because
one request per track would leave the microphone waiting on the system audio.

One pool, spanning both tracks. The obvious structure — transcribe the
microphone, then transcribe the system audio, four at a time each — never puts
more than four requests on the wire and makes the second track wait for the
first to drain. Pooling the combined work list instead keeps every slot busy
until there is no work left, which on a two-track meeting is close to half the
wall clock for the same number of requests.

Every chunk result is cached under the meeting's `.cache/`, keyed by the content
hash of the audio it came from and by the request parameters. Re-running
`process` after a failure downstream costs nothing and repeats no inference.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..api import HelmcodeClient
from ..store import Meeting
from .audio import Chunk, encode_chunk
from .intervals import slice_between
from .model import ME, OTHERS, Segment, Word

# Concurrent uploads in flight, across every track. The ceiling is the API's,
# not ours: it allows five parallel requests per key and 429s the sixth. See
# `config.DEFAULT_STT_CONCURRENCY`, which is what the CLI actually passes.
DEFAULT_CONCURRENCY = 4


@dataclass(frozen=True)
class TrackPlan:
    """One prepared track and the chunks to transcribe from it."""

    track: str
    prepared_wav: Path
    # Content hash of `prepared_wav`, computed once by the caller. It keys both
    # the encoded chunks and their transcriptions, so audio that did not change
    # is neither re-encoded nor re-sent.
    fingerprint: str
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def speaker(self) -> str:
        return ME if self.track == "mic" else OTHERS


@dataclass(frozen=True)
class Transcription:
    segments: dict[str, list[Segment]]
    language: str | None
    cache_hits: int
    total_chunks: int

    @property
    def fully_cached(self) -> bool:
        """Whether this run did no inference at all.

        A replayed run finishes in a second, and its timings mean nothing. This
        is what stops them being written to meta.json as if they were measured.
        """
        return self.total_chunks > 0 and self.cache_hits == self.total_chunks


def transcribe(
    client: HelmcodeClient,
    meeting: Meeting,
    plans: list[TrackPlan],
    *,
    language: str | None,
    model: str,
    force: bool = False,
    concurrency: int = DEFAULT_CONCURRENCY,
    on_chunk_done: Callable[[], None] | None = None,
) -> Transcription:
    """Transcribe every chunk of every track through one pool."""
    empty = {plan.track: [] for plan in plans}
    work = [(plan, chunk) for plan in plans for chunk in plan.chunks]
    if not work:
        return Transcription(empty, language, 0, 0)

    parameters = _parameter_fingerprint(model, language)

    def run_one(item: tuple[TrackPlan, Chunk]) -> tuple[TrackPlan, Chunk, dict[str, Any], bool]:
        plan, chunk = item
        try:
            key = f"{plan.track}-{plan.fingerprint}-{parameters}-{chunk.index:04d}.json"
            if not force:
                cached = meeting.read_cached_json("stt", key)
                if cached is not None:
                    return plan, chunk, cached, True

            # The chunk audio is keyed by the track's content hash, not by index
            # alone. Indexes are reused between runs, and `--force` re-derives
            # the chunk plan from a freshly prepared track, so an index-only name
            # will happily hand the encoder's leftovers to a chunk that now
            # covers a different stretch of the meeting.
            chunk_path = meeting.cache_path(
                "chunks", f"{plan.track}-{plan.fingerprint}-{chunk.index:04d}.ogg"
            )
            encode_chunk(plan.prepared_wav, chunk, chunk_path)
            payload = client.transcribe(chunk_path, language=language, model=model)
            meeting.write_cached_json(payload, "stt", key)
            return plan, chunk, payload, False
        finally:
            if on_chunk_done:
                on_chunk_done()

    with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(work)))) as pool:
        results = list(pool.map(run_one, work))

    # Counted here rather than incremented inside the workers: `hits += 1` from
    # a pool is a race, and this number decides whether a run's timings are a
    # measurement or a replay.
    segments: dict[str, list[Segment]] = {track: [] for track in empty}
    detected: str | None = None
    cache_hits = 0
    for plan, chunk, payload, from_cache in results:
        cache_hits += int(from_cache)
        detected = detected or payload.get("language")
        segments[plan.track].extend(_segments_from(payload, chunk, plan.track, plan.speaker))

    for items in segments.values():
        items.sort(key=lambda segment: segment.start)
    return Transcription(segments, language or detected, cache_hits, len(work))


def _parameter_fingerprint(model: str, language: str | None) -> str:
    """Model and language, folded into something safe to put in a filename."""
    raw = f"{model}|{language or 'auto'}".encode()
    return hashlib.sha256(raw).hexdigest()[:8]


def _segments_from(
    payload: dict[str, Any], chunk: Chunk, track: str, speaker: str
) -> list[Segment]:
    raw = payload.get("segments")
    if not raw:
        # `verbose_json` without segments still carries the full text; better a
        # chunk-sized block than a hole in the transcript.
        text = (payload.get("text") or "").strip()
        return [Segment(chunk.start, chunk.end, text, track, speaker)] if text else []

    words = sorted(
        (
            Word(
                start=chunk.start + float(word.get("start", 0.0)),
                end=chunk.start + float(word.get("end", 0.0)),
                text=str(word.get("word") or word.get("text") or ""),
            )
            for word in (payload.get("words") or [])
        ),
        key=lambda word: word.start,
    )
    word_starts = [word.start for word in words]

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
        # Words come back for the chunk as a whole. Both lists are in time
        # order, so each segment's share is a slice rather than a filter over
        # every word in the chunk.
        first, last = slice_between(word_starts, start - 0.01, end + 0.01)
        segments.append(
            Segment(
                start=start,
                end=max(end, start),
                text=text,
                track=track,
                speaker=speaker,
                confidence=item.get("avg_logprob"),
                words=words[first:last] or None,
            )
        )
    return segments
