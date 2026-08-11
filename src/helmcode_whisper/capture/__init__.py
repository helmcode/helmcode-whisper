"""Device discovery and recorder construction, per platform."""

from __future__ import annotations

import sys
from pathlib import Path

from .base import CaptureError, DeviceInfo, SoundDeviceRecorder, TrackRecorder

__all__ = [
    "CaptureError",
    "DeviceInfo",
    "TrackRecorder",
    "find_mic_device",
    "find_system_device",
    "list_input_devices",
    "make_mic_recorder",
    "make_system_recorder",
    "system_capture_hint",
]


def find_mic_device() -> DeviceInfo | None:
    """The default input device. Present on every platform we support."""
    try:
        import sounddevice as sd
    except ImportError:
        return None

    try:
        device = sd.query_devices(kind="input")
    except Exception:
        return None
    if not device or device.get("max_input_channels", 0) <= 0:
        return None

    index = device.get("index")
    if index is None:
        default = sd.default.device
        index = default[0] if isinstance(default, (list, tuple)) else default

    return DeviceInfo(
        index=int(index),
        name=str(device["name"]),
        channels=min(int(device["max_input_channels"]), 2),
        samplerate=int(device["default_samplerate"]),
        backend="portaudio",
        kind="mic",
    )


def find_system_device() -> DeviceInfo | None:
    """The system-audio device for this platform, or None if unavailable."""
    if sys.platform == "win32":
        from . import windows

        return windows.find_system_device()
    if sys.platform == "darwin":
        from . import macos

        return macos.find_system_device()
    from . import linux

    return linux.find_system_device()


def system_capture_hint() -> str:
    """What to tell the user when system capture is missing on this platform."""
    if sys.platform == "win32":
        return (
            "No WASAPI loopback device found. Check that an output device is "
            "active, then run `hcw devices`."
        )
    if sys.platform == "darwin":
        return (
            "No BlackHole device found. macOS cannot capture system audio without "
            "a virtual device — see the macOS setup section of the README."
        )
    return (
        "No PulseAudio/PipeWire monitor source found. Check `pactl get-default-sink` "
        "and that PortAudio can see its monitor, then run `hcw devices`."
    )


def make_mic_recorder(device: DeviceInfo, path: Path) -> TrackRecorder:
    return SoundDeviceRecorder(device, path)


def make_system_recorder(device: DeviceInfo, path: Path) -> TrackRecorder:
    if device.backend == "wasapi-loopback":
        from .windows import WasapiLoopbackRecorder

        return WasapiLoopbackRecorder(device, path)
    return SoundDeviceRecorder(device, path)


def list_input_devices() -> list[DeviceInfo]:
    """Every recordable input, for `hcw devices`."""
    devices: list[DeviceInfo] = []
    try:
        import sounddevice as sd

        for index, device in enumerate(sd.query_devices()):
            if device.get("max_input_channels", 0) <= 0:
                continue
            devices.append(
                DeviceInfo(
                    index=index,
                    name=str(device["name"]),
                    channels=int(device["max_input_channels"]),
                    samplerate=int(device["default_samplerate"]),
                    backend="portaudio",
                    kind="mic",
                )
            )
    except Exception:
        pass

    if sys.platform == "win32":
        try:
            import pyaudiowpatch as pyaudio

            audio = pyaudio.PyAudio()
            try:
                for device in audio.get_loopback_device_info_generator():
                    devices.append(
                        DeviceInfo(
                            index=int(device["index"]),
                            name=str(device["name"]),
                            channels=int(device["maxInputChannels"]),
                            samplerate=int(device["defaultSampleRate"]),
                            backend="wasapi-loopback",
                            kind="system",
                        )
                    )
            finally:
                audio.terminate()
        except Exception:
            pass

    return devices
