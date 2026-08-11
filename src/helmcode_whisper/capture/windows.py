"""System audio on Windows, via WASAPI loopback.

Windows exposes every output device as a hidden loopback *input*, which is the
cleanest capture path of the three platforms: no virtual cable, no extra
install. PortAudio's Python binding does not expose the flag reliably, so this
uses PyAudioWPatch — a maintained fork of PyAudio that does.
"""

from __future__ import annotations

import numpy as np

from .base import CaptureError, DeviceInfo, TrackRecorder

BACKEND = "wasapi-loopback"


def find_system_device() -> DeviceInfo | None:
    """The loopback of the current default output device, or None."""
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        return None

    audio = pyaudio.PyAudio()
    try:
        try:
            wasapi = audio.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            return None

        default_output = audio.get_device_info_by_index(wasapi["defaultOutputDevice"])
        device = default_output
        if not default_output.get("isLoopbackDevice"):
            # The loopback shares the speaker's name with a suffix appended.
            device = next(
                (
                    candidate
                    for candidate in audio.get_loopback_device_info_generator()
                    if default_output["name"] in candidate["name"]
                ),
                None,
            )
        if device is None:
            return None

        return DeviceInfo(
            index=int(device["index"]),
            name=str(device["name"]),
            channels=int(device["maxInputChannels"]),
            samplerate=int(device["defaultSampleRate"]),
            backend=BACKEND,
            kind="system",
        )
    finally:
        audio.terminate()


class WasapiLoopbackRecorder(TrackRecorder):
    def __init__(self, device: DeviceInfo, path) -> None:  # noqa: ANN001
        super().__init__(device, path)
        self._audio = None
        self._stream = None

    def _open_stream(self) -> None:
        import pyaudiowpatch as pyaudio

        channels = self.device.channels

        def callback(in_data, _frame_count, _time_info, _status):  # noqa: ANN001
            block = np.frombuffer(in_data, dtype=np.float32)
            if channels > 1:
                block = block.reshape(-1, channels)
            self._submit(block)
            return (None, pyaudio.paContinue)

        self._audio = pyaudio.PyAudio()
        try:
            self._stream = self._audio.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=self.device.samplerate,
                input=True,
                input_device_index=self.device.index,
                frames_per_buffer=2048,
                stream_callback=callback,
            )
        except Exception as exc:
            self._audio.terminate()
            self._audio = None
            raise CaptureError(f"could not open loopback '{self.device.name}': {exc}") from exc

    def _close_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._audio is not None:
            self._audio.terminate()
            self._audio = None
