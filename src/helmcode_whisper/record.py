"""`hcw record` — capture a meeting to two WAV files.

Deliberately offline. Nothing in this module imports the API client: a recording
must never fail, stall, or wait on a network that happens to be down.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from rich.live import Live
from rich.table import Table
from rich.text import Text

from .capture import (
    TrackRecorder,
    find_mic_device,
    find_system_device,
    make_mic_recorder,
    make_system_recorder,
    system_capture_hint,
)
from .config import Config
from .store import Meeting
from .ui import console, eyebrow, hairline, status_line

RECORDING_NOTICE = (
    "Recording a conversation without telling the other people in it is illegal in "
    "many places, and a bad idea everywhere. Tell them, and get their agreement, "
    "before you start."
)


def record(
    config: Config, title: str, *, mic_only: bool = False, seconds: float | None = None
) -> Meeting:
    config.home.mkdir(parents=True, exist_ok=True)

    mic_device = find_mic_device()
    if mic_device is None:
        raise RuntimeError("No microphone found. Run `hcw devices` to see what is available.")

    system_device = None if mic_only else find_system_device()

    console().print()
    console().print(eyebrow("before you record"))
    console().print(Text(RECORDING_NOTICE, style="secondary"))
    console().print()

    hairline("devices")
    status_line("ok", f"mic     {mic_device.name}", f"{mic_device.samplerate} Hz")
    if system_device:
        status_line("ok", f"system  {system_device.name}", f"{system_device.samplerate} Hz")
    else:
        status_line(
            "warn",
            "system  not captured — recording the microphone only",
            "" if mic_only else system_capture_hint(),
        )

    meeting = Meeting.create(config.home, title, datetime.now())
    recorders: list[tuple[str, TrackRecorder]] = [
        ("me", make_mic_recorder(mic_device, meeting.mic_wav))
    ]
    if system_device:
        recorders.append(("others", make_system_recorder(system_device, meeting.system_wav)))

    started = time.monotonic()
    for _, recorder in recorders:
        recorder.start()

    console().print()
    stop_hint = f"stopping after {seconds:.0f}s" if seconds else "press Ctrl+C to stop"
    console().print(Text(f"  recording — {stop_hint}", style="accent"))
    console().print()

    try:
        with Live(_meters(recorders, 0.0), console=console(), refresh_per_second=8) as live:
            while True:
                time.sleep(0.12)
                elapsed = time.monotonic() - started
                live.update(_meters(recorders, elapsed))
                if seconds is not None and elapsed >= seconds:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        for _, recorder in recorders:
            recorder.stop()

    duration = max((recorder.duration for _, recorder in recorders), default=0.0)
    dropped = sum(recorder.dropped_blocks for _, recorder in recorders)
    meeting.save_meta(
        {
            "duration_seconds": round(duration, 2),
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "tracks": {
                label: {
                    "file": recorder.path.name,
                    "device": recorder.device.name,
                    "backend": recorder.device.backend,
                    "samplerate": recorder.device.samplerate,
                    "seconds": round(recorder.duration, 2),
                    "silence_padded_seconds": round(recorder.padded_seconds, 2),
                }
                for label, recorder in recorders
            },
            "dropped_blocks": dropped,
        }
    )

    console().print()
    hairline("recorded")
    for label, recorder in recorders:
        # -60 dBFS. Anything quieter over a whole meeting is not a quiet room,
        # it is a device that never delivered audio.
        silent = recorder.max_peak < 0.001
        status_line(
            "warn" if silent else "ok",
            f"{label:<7} {recorder.path.name}",
            _hms(recorder.duration) if not silent else "SILENT — recorded nothing but zeros",
        )
        if silent:
            status_line(
                "warn",
                f"the {label} track has no audio in it",
                _silence_hint(recorder.device.kind),
            )
    if dropped:
        status_line(
            "warn",
            f"{dropped} audio blocks dropped — the transcript may have gaps",
            "the machine could not keep up with the writer thread",
        )
    console().print()
    console().print(
        Text.assemble(("  next  ", "tertiary"), (f"hcw process {meeting.path}", "accent"))
    )
    console().print()
    return meeting


def _silence_hint(kind: str) -> str:
    if kind == "mic":
        return (
            "check that the microphone is not muted and that Windows privacy settings, "
            "or your OS equivalent, let this terminal use it"
        )
    return "check that sound was actually playing through the captured output device"


def _meters(recorders: list[tuple[str, TrackRecorder]], elapsed: float) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(width=2)
    table.add_column(style="tertiary", width=7)
    table.add_column()
    table.add_column(style="mono")
    for label, recorder in recorders:
        table.add_row("", label, _bar(recorder.level), _hms(recorder.duration))
    table.add_row("", "", "", "")
    table.add_row("", "elapsed", Text(_hms(elapsed), style="accent"), "")
    return table


def _bar(level: float, width: int = 28) -> Text:
    """A peak meter drawn as a hairline that fills with the accent colour."""
    filled = min(width, int(level * width * 1.6))
    bar = Text()
    bar.append("─" * filled, style="accent.solid")
    bar.append("─" * (width - filled), style="hairline")
    return bar


def _hms(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def resolve_meeting(config: Config, target: str | None) -> Meeting:
    """A path, a folder name under HCW_HOME, or the most recent meeting."""
    if target:
        path = Path(target).expanduser()
        if not path.is_dir():
            path = config.home / target
        if not path.is_dir():
            raise RuntimeError(f"No meeting at {target}")
        return Meeting(path)

    meeting = Meeting.latest(config.home)
    if meeting is None:
        raise RuntimeError(f"No meetings in {config.home}. Record one with `hcw record`.")
    return meeting
