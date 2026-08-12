"""Fuse the two tracks into one transcript, and deal with the echo.

The problem nobody warns you about: unless the user wears headphones, their
microphone picks up the remote audio coming out of the speakers. Both tracks
then contain the same words, and a naive merge produces a transcript where every
remote sentence is said twice — once by the actual speaker and once, slightly
garbled, by "Me".

The two facts that make an echo an echo are that it lands at the same time as
the original and that it says the same thing. Both are needed — people talk
over each other constantly, and agreeing with someone is not the same as
echoing them.

Getting the *same thing* part right took a real recording to work out. The
first version compared each microphone segment against each system segment: it
required half the mic segment to sit inside one system segment and their texts
to be similar. Measured against an hour of audio played through speakers, that
caught 41% of the echo, and the reason was not the thresholds. The two tracks
are transcribed independently and Whisper draws segment boundaries wherever it
likes on each, so the echoed copy is cut differently from the original. A mic
segment routinely straddles two system segments and matches neither well
enough; one segment in that recording was 96% echo by content and intersected
no system segment at all.

So the comparison is against the *window* rather than a segment: everything the
remote side said while this mic segment was open, pooled, and asked what
fraction of the mic segment's words appear in it. Function words are excluded
by length, because "el", "de" and "que" are in every window ever recorded and
inflate the score of things nobody echoed. On the two recordings this was tuned
against, echo scores 0.87-1.00 and genuine speech 0.00-0.31 — the threshold
sits in the gap rather than against either edge, because the expensive mistake
here is deleting something the user actually said.

Dropped segments stay in `transcript.json` with a reason attached, so the call
is auditable instead of a silent deletion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .intervals import IntervalIndex
from .model import Segment

# How far either side of a mic segment to look for what was playing. Small:
# the point is still that an echo is simultaneous, and a wide window would
# start matching things said a sentence ago.
_ECHO_WINDOW_SECONDS = 1.0

# Fraction of the mic segment's content words that must appear in that window.
# Measured on two real recordings: echo lands at 0.87-1.00 and genuine speech
# at 0.00-0.31, so this sits in the gap. Nearer the floor than the ceiling on
# purpose — leaving an echo in is visible and recoverable, deleting what
# somebody said is neither.
_MIN_ECHO_CONTAINMENT = 0.65

# Words shorter than this are dropped before comparing. Spanish function words
# are three letters or fewer almost without exception, they appear in every
# window, and counting them pushes unrelated speech up toward the threshold.
_MIN_WORD_LENGTH = 4

# Below this many content words there is not enough left to judge: "sí, claro"
# reduces to one word, which either matches or does not on a coin flip.
_MIN_ECHO_WORDS = 4

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


@dataclass(frozen=True)
class _Utterance:
    """A system segment reduced to what echo detection compares."""

    start: float
    end: float
    words: frozenset[str]


def _content_words(text: str) -> frozenset[str]:
    return frozenset(
        word for word in _normalize(text).split() if len(word) >= _MIN_WORD_LENGTH
    )


def _mark_echoes(mic_segments: list[Segment], system_segments: list[Segment]) -> None:
    # Split once and index once. Rescanning every system segment for every
    # microphone segment is a thousand times a thousand on an hour-long
    # meeting.
    index = IntervalIndex(
        [_Utterance(item.start, item.end, _content_words(item.text)) for item in system_segments]
    )

    for segment in mic_segments:
        words = _content_words(segment.text)
        if len(words) < _MIN_ECHO_WORDS:
            continue

        window: set[str] = set()
        for other in index.overlapping(
            segment.start - _ECHO_WINDOW_SECONDS, segment.end + _ECHO_WINDOW_SECONDS
        ):
            window |= other.words
        if not window:
            continue

        if len(words & window) / len(words) >= _MIN_ECHO_CONTAINMENT:
            segment.dropped = "echo"


def _normalize(text: str) -> str:
    return _SPACES.sub(" ", _PUNCTUATION.sub(" ", text.lower())).strip()


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
