"""Fuse the two tracks into one transcript, and deal with the echo.

The problem nobody warns you about: unless the user wears headphones, their
microphone picks up the remote audio coming out of the speakers. Both tracks
then contain the same words, and a naive merge produces a transcript where every
remote sentence is said twice — once by the actual speaker and once, slightly
garbled, by "Me".

The fix uses the two facts that make an echo an echo: it lands at the same time
as the original, and it says the same thing. A microphone segment is dropped
when it overlaps a system segment in time *and* their texts are similar. Both
conditions are needed — people talk over each other constantly, and agreeing
with someone is not the same as echoing them.

Dropped segments stay in `transcript.json` with a reason attached, so the call
is auditable instead of a silent deletion.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from .model import Segment

# Fraction of the mic segment that has to sit inside a system segment.
_MIN_TIME_OVERLAP = 0.5
# Text similarity above which two overlapping segments are the same words.
_MIN_TEXT_SIMILARITY = 0.62
# Below this many characters, similarity is noise: "sí", "ya", "ok" match
# everything. Short mic segments are always kept.
_MIN_ECHO_CHARS = 12

_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")


def merge(
    mic_segments: list[Segment],
    system_segments: list[Segment],
    *,
    suppress_echo: bool = True,
) -> list[Segment]:
    if suppress_echo and mic_segments and system_segments:
        _mark_echoes(mic_segments, system_segments)

    merged = [*mic_segments, *system_segments]
    # Ties go to the remote track: when both start at the same instant it is
    # nearly always the remote speaker with the mic trailing them.
    merged.sort(key=lambda segment: (segment.start, 0 if segment.track == "system" else 1))
    return merged


def _mark_echoes(mic_segments: list[Segment], system_segments: list[Segment]) -> None:
    normalized_system = [(segment, _normalize(segment.text)) for segment in system_segments]

    for segment in mic_segments:
        text = _normalize(segment.text)
        if len(text) < _MIN_ECHO_CHARS:
            continue
        duration = max(segment.end - segment.start, 1e-6)

        for other, other_text in normalized_system:
            if other.end < segment.start:
                continue
            if other.start > segment.end:
                break
            overlap = min(segment.end, other.end) - max(segment.start, other.start)
            if overlap / duration < _MIN_TIME_OVERLAP:
                continue
            if _similarity(text, other_text) >= _MIN_TEXT_SIMILARITY:
                segment.dropped = "echo"
                break


def _normalize(text: str) -> str:
    return _SPACES.sub(" ", _PUNCTUATION.sub(" ", text.lower())).strip()


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    # Length alone rules most pairs out, and SequenceMatcher is the expensive
    # part of a step that runs over every overlapping pair in the meeting.
    ratio = len(left) / len(right) if len(left) < len(right) else len(right) / len(left)
    if ratio < 0.5:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def rename_speakers(segments: list[Segment], names: dict[str, str]) -> None:
    """Apply human names to SPEAKER_xx labels, e.g. from meta.json."""
    for segment in segments:
        if segment.speaker in names:
            segment.speaker = names[segment.speaker]


def speaker_list(segments: list[Segment]) -> list[str]:
    seen: dict[str, None] = {}
    for segment in segments:
        if not segment.dropped:
            seen.setdefault(segment.speaker, None)
    return list(seen)
