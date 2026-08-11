"""System audio on macOS, via a BlackHole virtual device.

macOS has no software route to the system mix: an app can only record what the
OS hands it, and the OS hands it nothing. The workaround is a virtual audio
device (BlackHole) plus a Multi-Output Device so sound reaches both the speakers
and the virtual device — a manual setup step the README documents rather than
hides. A native ScreenCaptureKit helper would remove it, and is on the roadmap.

NOT TESTED: no macOS machine in the loop yet. See the platform table in the
README.
"""

from __future__ import annotations

from .base import DeviceInfo

BACKEND = "blackhole"

# BlackHole ships in 2ch, 16ch and 64ch variants; the 2ch one is what the README
# tells people to install, but accept any of them.
_MARKERS = ("blackhole", "loopback audio", "soundflower")


def find_system_device() -> DeviceInfo | None:
    try:
        import sounddevice as sd
    except ImportError:
        return None

    try:
        devices = sd.query_devices()
    except Exception:
        return None

    for index, device in enumerate(devices):
        if device.get("max_input_channels", 0) <= 0:
            continue
        name = str(device["name"]).lower()
        if any(marker in name for marker in _MARKERS):
            return DeviceInfo(
                index=index,
                name=str(device["name"]),
                # A 16ch BlackHole carrying a stereo program would waste 14
                # silent channels through the whole pipeline; two is enough for
                # anything we downmix to mono anyway.
                channels=min(int(device["max_input_channels"]), 2),
                samplerate=int(device["default_samplerate"]),
                backend=BACKEND,
                kind="system",
            )
    return None
