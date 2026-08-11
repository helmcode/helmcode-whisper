"""System audio on Linux, via the PipeWire/PulseAudio monitor source.

Every sink has a `.monitor` source carrying exactly what it plays. Asking
`pactl` for the default sink and recording its monitor is the precise answer;
scanning PortAudio's device list for a monitor is the fallback for setups
without `pactl` on PATH.

NOT TESTED: written against the PulseAudio/PipeWire contract but not yet run on
a Linux machine. See the platform support table in the README.
"""

from __future__ import annotations

import subprocess

from .base import DeviceInfo

BACKEND = "pulse-monitor"


def _default_sink_monitor() -> str | None:
    try:
        result = subprocess.run(
            ["pactl", "get-default-sink"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sink = result.stdout.strip()
    return f"{sink}.monitor" if sink else None


def find_system_device() -> DeviceInfo | None:
    """A monitor source exposed through PortAudio, or None."""
    try:
        import sounddevice as sd
    except ImportError:
        return None

    try:
        devices = sd.query_devices()
    except Exception:
        return None

    inputs = [
        (index, device)
        for index, device in enumerate(devices)
        if device.get("max_input_channels", 0) > 0
    ]

    wanted = _default_sink_monitor()
    ordered = sorted(
        inputs,
        key=lambda pair: (
            # Exact default-sink monitor first, then anything monitor-ish.
            0 if wanted and wanted in pair[1]["name"] else 1,
            0 if "monitor" in pair[1]["name"].lower() else 1,
        ),
    )
    for index, device in ordered:
        if wanted and wanted in device["name"]:
            return _to_info(index, device)
        if "monitor" in device["name"].lower():
            return _to_info(index, device)
    return None


def _to_info(index: int, device: dict) -> DeviceInfo:
    return DeviceInfo(
        index=index,
        name=str(device["name"]),
        channels=int(device["max_input_channels"]),
        samplerate=int(device["default_samplerate"]),
        backend=BACKEND,
        kind="system",
    )
