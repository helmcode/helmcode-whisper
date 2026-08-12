"""Step events for readers that are not a terminal.

`process` already reports itself beautifully to a person: hairlines, a spinner,
a summary. None of that is parseable. A front end that wants to show "separating
voices, about eight minutes left" has two options — scrape output written for
humans, which breaks the first time a label is reworded, or be told.

So it can be told. With `--progress-json` the pipeline emits one JSON object per
line to stdout and the human-readable output moves to stderr, which keeps the
two audiences on separate streams instead of interleaved on one.

The events are deliberately flat and additive: a reader should be able to ignore
a field it does not know and a new field should never break one that already
works.

    {"event": "step", "step": "transcribe", "state": "running"}
    {"event": "chunks", "done": 34, "total": 69}
    {"event": "step", "step": "transcribe", "state": "done", "seconds": 100.5}
    {"event": "step", "step": "diarize", "state": "running", "estimate_seconds": 1549}
"""

from __future__ import annotations

import json
import sys
import threading
from typing import Any, TextIO

# Diarization on CPU, as a fraction of the recording's length. Measured across
# 5, 10, 20 and 60-minute runs on one machine: 0.42, 0.43, 0.55 and 0.43. The
# spread is real, so this is a guide for a progress bar and nothing more — it is
# never presented as a duration the pipeline promises to hit.
CPU_DIARIZATION_FACTOR = 0.45

STEPS = ("prepare", "transcribe", "diarize", "merge", "notes", "index")


class Progress:
    """Does nothing. The default, so callers never have to check for None."""

    def event(self, **fields: Any) -> None:
        return

    def step(self, name: str, state: str, **fields: Any) -> None:
        self.event(event="step", step=name, state=state, **fields)

    def close(self) -> None:
        return


class JsonProgress(Progress):
    """One JSON object per line, flushed immediately.

    Flushed because the reader is watching a pipe: a buffered progress event is
    the same as no progress event. Locked because chunk completions arrive from
    the transcription pool and two half-written lines are not JSON.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout
        self._lock = threading.Lock()
        # Belt to the ASCII braces below. On Windows a redirected stdout arrives
        # with the locale's encoding, and the first accented character in a
        # meeting title came out as cp1252 inside a stream a reader is entitled
        # to treat as UTF-8.
        reconfigure = getattr(self._stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass

    def event(self, **fields: Any) -> None:
        # `ensure_ascii=True`, deliberately. This is an interchange stream whose
        # encoding is decided by whatever pipe it was handed, so the safe move is
        # to emit bytes that mean the same thing in all of them and let the
        # reader's JSON parser turn ó back into ó.
        line = json.dumps(fields, ensure_ascii=True)
        with self._lock:
            try:
                self._stream.write(line + "\n")
                self._stream.flush()
            except (BrokenPipeError, ValueError):
                # The reader went away. That is its business, not a reason to
                # take down a run that is otherwise working.
                pass


def estimate_diarization_seconds(audio_seconds: float, device: str = "cpu") -> int | None:
    """A rough guide for the progress bar, or None when we would be guessing.

    Only CPU has been measured. Returning None for a GPU is the honest answer:
    a made-up number that turns out to be five times wrong is worse for trust
    than an unknown duration, and this is the step where the wait is long
    enough that people watch the estimate.
    """
    if device != "cpu" or audio_seconds <= 0:
        return None
    return int(audio_seconds * CPU_DIARIZATION_FACTOR)
