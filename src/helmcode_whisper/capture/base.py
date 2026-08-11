"""Track recording: one audio stream in, one WAV file out.

Two tracks are recorded and never mixed: the microphone is "me", the system
loopback is "everyone else". That split hands us half the diarization for free
and leaves pyannote with only the remote track to untangle.

Audio callbacks must not block, so they only copy their buffer into a queue; a
writer thread does the file I/O. Everything is captured at the device's native
sample rate and downmixed to mono — resampling is `process`'s job, and doing it
here would put an avoidable computation in the real-time path.
"""

from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

# A stalled stream shorter than this is ordinary scheduling jitter; anything
# longer is a real gap and gets filled with silence. See _write_loop.
_GAP_THRESHOLD_SECONDS = 0.25


class CaptureError(RuntimeError):
    """The requested audio device could not be opened."""


@dataclass(frozen=True)
class DeviceInfo:
    """A device we could record from, in terms the CLI can print."""

    index: int
    name: str
    channels: int
    samplerate: int
    backend: str
    kind: str  # "mic" or "system"


class TrackRecorder(ABC):
    """Writes one input stream to a mono WAV file until stopped."""

    def __init__(self, device: DeviceInfo, path: Path) -> None:
        self.device = device
        self.path = path
        self._queue: queue.Queue[tuple[np.ndarray, float] | None] = queue.Queue(maxsize=256)
        self._writer: threading.Thread | None = None
        self._frames = 0
        self._peak = 0.0
        self._max_peak = 0.0
        self._dropped = 0
        self._padded_frames = 0
        self._started_at = 0.0
        self._error: BaseException | None = None

    # ── public surface ───────────────────────────────────────────

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def duration(self) -> float:
        return self._frames / self.device.samplerate if self.device.samplerate else 0.0

    @property
    def level(self) -> float:
        """Peak amplitude since the last read, for the recording meter."""
        peak, self._peak = self._peak, 0.0
        return peak

    @property
    def max_peak(self) -> float:
        """Loudest sample of the whole take.

        A muted or blocked input still opens, still delivers blocks, and still
        writes a perfectly valid WAV — of silence. Without this, the first sign
        of a dead microphone is an empty transcript twenty minutes later.
        """
        return self._max_peak

    @property
    def dropped_blocks(self) -> int:
        """Blocks lost because the writer could not keep up. Should stay at 0."""
        return self._dropped

    @property
    def padded_seconds(self) -> float:
        """Silence inserted to keep this track on the wall clock. See _write_loop."""
        return self._padded_frames / self.device.samplerate if self.device.samplerate else 0.0

    def start(self) -> None:
        self._writer = threading.Thread(target=self._write_loop, name=f"write-{self.path.stem}")
        self._writer.start()
        try:
            self._started_at = time.monotonic()
            self._open_stream()
        except BaseException:
            self._queue.put(None)
            self._writer.join(timeout=5)
            raise

    def stop(self) -> None:
        try:
            self._close_stream()
        finally:
            # An empty block stamped now, so a track that went quiet before the
            # end is padded out to the same length as the others.
            self._submit(np.zeros(0, dtype=np.float32))
            self._queue.put(None)
            if self._writer:
                self._writer.join(timeout=10)
        if self._error:
            raise CaptureError(f"recording {self.path.name} failed: {self._error}") from self._error

    # ── for subclasses ───────────────────────────────────────────

    @abstractmethod
    def _open_stream(self) -> None: ...

    @abstractmethod
    def _close_stream(self) -> None: ...

    def _submit(self, block: np.ndarray) -> None:
        """Called from the audio thread. Never blocks, never raises."""
        mono = block if block.ndim == 1 else block.mean(axis=1)
        try:
            self._queue.put_nowait((mono.astype(np.float32, copy=True), time.monotonic()))
        except queue.Full:
            # Dropping a block beats stalling the audio device; the count is
            # reported at the end of `record` so a struggling machine is visible
            # rather than silently producing a transcript with holes in it.
            self._dropped += 1

    # ── writer thread ────────────────────────────────────────────

    def _write_loop(self) -> None:
        """Write blocks, filling wall-clock gaps with silence.

        A WASAPI loopback device delivers nothing at all while the machine is
        silent — it does not hand over blocks of zeros. Left alone, a 90-second
        recording with two minutes of quiet in it produces a file containing
        only the noisy parts, spliced together. The mic track has no such gaps,
        so the two tracks drift apart and every timestamp in the merged
        transcript is wrong from the first silence onwards.

        The fix is to trust the clock rather than the block count: when a block
        arrives later than the frames written so far can account for, the
        difference was silence, and silence is written.
        """
        samplerate = self.device.samplerate
        try:
            with sf.SoundFile(
                self.path,
                mode="w",
                samplerate=samplerate,
                channels=1,
                subtype="PCM_16",
            ) as handle:
                while True:
                    item = self._queue.get()
                    if item is None:
                        return
                    block, arrived_at = item

                    # The callback fires once the block has been captured, so
                    # its arrival marks the block's end on the wall clock.
                    expected_end = (arrived_at - self._started_at) * samplerate
                    gap = expected_end - block.shape[0] - self._frames
                    if gap > _GAP_THRESHOLD_SECONDS * samplerate:
                        silence = np.zeros(int(gap), dtype=np.float32)
                        handle.write(silence)
                        self._frames += silence.shape[0]
                        self._padded_frames += silence.shape[0]

                    peak = float(np.abs(block).max()) if block.size else 0.0
                    self._peak = max(self._peak, peak)
                    self._max_peak = max(self._max_peak, peak)
                    self._frames += block.shape[0]
                    handle.write(block)
        except BaseException as exc:  # surfaced by stop()
            self._error = exc
            # Drain so a full queue cannot wedge the audio callback.
            while True:
                if self._queue.get() is None:
                    return


class SoundDeviceRecorder(TrackRecorder):
    """PortAudio capture. The microphone everywhere; system audio on Linux/macOS."""

    def __init__(self, device: DeviceInfo, path: Path) -> None:
        super().__init__(device, path)
        self._stream = None

    def _open_stream(self) -> None:
        import sounddevice as sd

        def callback(indata, _frames, _time, status) -> None:  # noqa: ANN001
            del status  # over/underruns are visible in dropped_blocks
            self._submit(indata)

        try:
            self._stream = sd.InputStream(
                device=self.device.index,
                channels=self.device.channels,
                samplerate=self.device.samplerate,
                dtype="float32",
                blocksize=0,
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:  # sounddevice raises PortAudioError subclasses
            raise CaptureError(f"could not open '{self.device.name}': {exc}") from exc

    def _close_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
