"""Audio preparation: normalize, find speech, and cut chunks the API will accept.

The Helmcode transcription endpoint caps a request at 25 MB and around two
minutes of audio; anything longer comes back as a 524. So the real work here is
deciding *where* to cut a 60-minute recording into ~35 pieces without slicing
through the middle of a sentence.

The answer is voice activity detection. webrtcvad marks 30 ms frames as speech
or not; adjacent speech is merged, short gaps are bridged, and chunks are packed
up to the limit on those boundaries. Long silences fall outside every chunk and
are never uploaded — which is also how "trim the silence" is implemented, and it
keeps timestamps honest because each chunk carries its offset into the original.

Only when a single stretch of speech runs past the limit do we cut mid-sentence,
and then the pieces overlap so the model has context on both sides of the seam.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

# Whisper wants 16 kHz mono; webrtcvad only accepts 8/16/32/48 kHz. One
# resample at the top of the pipeline serves both.
SAMPLE_RATE = 16_000

# The endpoint's limit is ~120 s. Stopping at 110 leaves room for the encoder's
# rounding and for a chunk that ends on a long word.
MAX_CHUNK_SECONDS = 110.0
# Below this, a chunk costs a full request for almost no speech.
MIN_CHUNK_SECONDS = 1.0
# Speech separated by less than this is one utterance, not two.
BRIDGE_GAP_SECONDS = 0.45
# Keep a breath either side of detected speech so words are not clipped.
PAD_SECONDS = 0.30
# Context repeated across a forced mid-sentence cut.
OVERLAP_SECONDS = 2.0

_VAD_FRAME_MS = 30
_VAD_AGGRESSIVENESS = 2


class AudioError(RuntimeError):
    pass


@dataclass(frozen=True)
class Chunk:
    """A slice of a track, in seconds relative to the start of the recording."""

    index: int
    start: float
    end: float
    # Where the previous chunk stopped being authoritative. Segments before this
    # point are duplicates of the previous chunk's tail and get dropped.
    overlap_until: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise AudioError(
            "ffmpeg is not on PATH. It is required to resample and encode audio. "
            "Install it with `winget install Gyan.FFmpeg`, `apt install ffmpeg` or "
            "`brew install ffmpeg`."
        )
    return path


def _run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(
        [require_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AudioError(f"ffmpeg failed: {result.stderr.strip()[:400]}")


def prepare_track(source: Path, destination: Path) -> Path:
    """Resample a recorded track to 16 kHz mono and normalize its loudness.

    Normalization matters more than it looks: a laptop microphone and a
    conference stream routinely differ by 20 dB, and the quiet one transcribes
    noticeably worse. `loudnorm` is single-pass here — a second pass would be
    more accurate and is not worth the extra minute on a 60-minute file.
    """
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-i", str(source),
            "-af", "loudnorm=I=-18:TP=-2:LRA=11",
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            str(destination),
        ]
    )
    return destination


def encode_chunk(source16k: Path, chunk: Chunk, destination: Path) -> Path:
    """Cut one chunk out of the prepared track as Opus.

    Opus at 24 kbps mono is about 360 KB for two minutes — two orders of
    magnitude under the 25 MB request limit, and small enough that upload time
    stops mattering next to inference time.
    """
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-ss", f"{chunk.start:.3f}",
            "-t", f"{chunk.duration:.3f}",
            "-i", str(source16k),
            "-c:a", "libopus",
            "-b:a", "24k",
            "-application", "voip",
            str(destination),
        ]
    )
    return destination


def track_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate if info.samplerate else 0.0


def speech_regions(path16k: Path) -> list[tuple[float, float]]:
    """Contiguous stretches of speech, in seconds."""
    import webrtcvad

    audio, samplerate = sf.read(str(path16k), dtype="int16", always_2d=False)
    if samplerate != SAMPLE_RATE:
        raise AudioError(f"expected {SAMPLE_RATE} Hz, got {samplerate}")
    if audio.ndim > 1:
        audio = audio[:, 0]

    vad = webrtcvad.Vad(_VAD_AGGRESSIVENESS)
    frame_length = int(SAMPLE_RATE * _VAD_FRAME_MS / 1000)
    frame_count = len(audio) // frame_length

    # Convert the whole track to bytes once and hand webrtcvad slices of it.
    # An hour of audio is 120,000 frames, and calling `.tobytes()` on each row
    # of a reshaped array allocates 120,000 buffers to say what slicing an
    # existing `bytes` says for free.
    raw = audio[: frame_count * frame_length].tobytes()
    stride = frame_length * audio.itemsize

    flags = np.fromiter(
        (
            vad.is_speech(raw[offset : offset + stride], SAMPLE_RATE)
            for offset in range(0, frame_count * stride, stride)
        ),
        dtype=bool,
        count=frame_count,
    )
    return _flags_to_regions(flags, _VAD_FRAME_MS / 1000, len(audio) / SAMPLE_RATE)


def _flags_to_regions(
    flags: np.ndarray, frame_seconds: float, total_seconds: float
) -> list[tuple[float, float]]:
    regions: list[tuple[float, float]] = []
    start: int | None = None
    for index, is_speech in enumerate(flags):
        if is_speech and start is None:
            start = index
        elif not is_speech and start is not None:
            regions.append((start * frame_seconds, index * frame_seconds))
            start = None
    if start is not None:
        regions.append((start * frame_seconds, len(flags) * frame_seconds))

    padded: list[tuple[float, float]] = []
    for begin, end in regions:
        begin = max(0.0, begin - PAD_SECONDS)
        end = min(total_seconds, end + PAD_SECONDS)
        if padded and begin - padded[-1][1] <= BRIDGE_GAP_SECONDS:
            padded[-1] = (padded[-1][0], end)
        else:
            padded.append((begin, end))
    return padded


def plan_chunks(regions: list[tuple[float, float]]) -> list[Chunk]:
    """Pack speech regions into chunks at or under the API's duration limit."""
    chunks: list[Chunk] = []
    current_start: float | None = None
    current_end: float | None = None

    def flush() -> None:
        nonlocal current_start, current_end
        if current_start is not None and current_end is not None:
            if current_end - current_start >= MIN_CHUNK_SECONDS:
                chunks.append(Chunk(len(chunks), current_start, current_end))
        current_start = current_end = None

    for begin, end in regions:
        if current_start is None:
            current_start, current_end = begin, end
        elif end - current_start <= MAX_CHUNK_SECONDS:
            current_end = end
        else:
            flush()
            current_start, current_end = begin, end

        # One unbroken region longer than the limit: cut it, with overlap.
        while current_end is not None and current_end - current_start > MAX_CHUNK_SECONDS:
            cut = current_start + MAX_CHUNK_SECONDS
            chunks.append(Chunk(len(chunks), current_start, cut))
            current_start = cut - OVERLAP_SECONDS

    flush()

    # Mark which chunks start inside their predecessor so the transcript
    # assembler can drop the repeated words instead of stuttering.
    marked: list[Chunk] = []
    previous_end = 0.0
    for chunk in chunks:
        overlap_until = previous_end if chunk.start < previous_end else 0.0
        marked.append(Chunk(chunk.index, chunk.start, chunk.end, overlap_until))
        previous_end = chunk.end
    return marked
