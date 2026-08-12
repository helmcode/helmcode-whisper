"""The `hcw` command line."""

from __future__ import annotations

import shutil
import sys
from typing import Annotated

import typer

from . import __version__
from .config import ConfigError, load_config
from .ui import console, err_console, hairline, status_line

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Self-hosted meeting notes. Records mic + system audio, transcribes and "
    "summarizes it with open models on Helmcode, and keeps every byte on your machine.",
)


@app.command()
def record(
    title: Annotated[
        str, typer.Option("--title", "-t", help="Meeting title; becomes the folder name.")
    ] = "meeting",
    mic_only: Annotated[
        bool, typer.Option("--mic-only", help="Skip system audio even if it is available.")
    ] = False,
    seconds: Annotated[
        float | None,
        typer.Option("--seconds", "-s", help="Stop automatically after N seconds."),
    ] = None,
) -> None:
    """Record a meeting. Works entirely offline."""
    from .record import record as run_record

    _run(lambda: run_record(load_config(), title, mic_only=mic_only, seconds=seconds))


@app.command()
def devices() -> None:
    """List the audio inputs this machine can record from."""
    from rich.table import Table

    from .capture import (
        find_mic_device,
        find_system_device,
        list_input_devices,
        system_capture_hint,
    )

    mic = find_mic_device()
    system = find_system_device()

    console().print()
    hairline("selected")
    status_line("ok", f"mic     {mic.name}" if mic else "mic     none found")
    if system:
        status_line("ok", f"system  {system.name}", system.backend)
    else:
        status_line("warn", "system  none found", system_capture_hint())

    table = Table(box=None, pad_edge=False, padding=(0, 2))
    table.add_column("#", style="tertiary", justify="right")
    table.add_column("device", style="secondary")
    table.add_column("ch", style="tertiary", justify="right")
    table.add_column("rate", style="tertiary", justify="right")
    table.add_column("backend", style="tertiary")
    for device in list_input_devices():
        table.add_row(
            str(device.index),
            device.name,
            str(device.channels),
            f"{device.samplerate}",
            device.backend,
        )

    console().print()
    hairline("all inputs")
    console().print(table)
    console().print()


@app.command()
def doctor() -> None:
    """Check that everything `record` and `process` need is in place."""
    config = load_config()

    console().print()
    hairline("environment")
    supported = sys.version_info >= (3, 11)
    status_line(
        "ok" if supported else "err",
        f"python  {sys.version.split()[0]}",
        "" if supported else "3.11+ required",
    )
    ffmpeg = shutil.which("ffmpeg")
    status_line(
        "ok" if ffmpeg else "err",
        "ffmpeg  " + (ffmpeg or "not found on PATH"),
        "" if ffmpeg else "required to split and encode audio",
    )
    try:
        config.home.mkdir(parents=True, exist_ok=True)
        status_line("ok", f"storage {config.home}")
    except OSError as exc:
        status_line("err", f"storage {config.home}", str(exc))

    hairline("audio")
    from .capture import find_mic_device, find_system_device, system_capture_hint

    mic = find_mic_device()
    status_line("ok" if mic else "err", f"mic     {mic.name if mic else 'none found'}")
    system = find_system_device()
    if system:
        status_line("ok", f"system  {system.name}", system.backend)
    else:
        status_line("warn", "system  none found", system_capture_hint())
    try:
        import webrtcvad  # noqa: F401

        status_line("ok", "vad     webrtcvad")
    except ImportError:
        status_line("err", "vad     webrtcvad not installed", "reinstall the package")

    hairline("helmcode api")
    status_line(
        "ok" if config.api_key else "warn",
        f"key     {'set' if config.api_key else 'not set'}",
        "" if config.api_key else "record works without it; process and search do not",
    )
    status_line("ok", f"url     {config.base_url}")
    if config.api_key:
        from .api import ApiError, HelmcodeClient

        try:
            with HelmcodeClient(config) as client:
                available = client.models()
        except ApiError as exc:
            status_line("err", "models  unreachable", str(exc))
        else:
            for label, wanted in (
                ("stt", config.stt_model),
                ("notes", config.notes_model),
                ("embed", config.embed_model),
                ("rerank", config.rerank_model),
            ):
                present = wanted in available
                status_line(
                    "ok" if present else "warn",
                    f"{label:<7} {wanted}",
                    "" if present else "not in /models — set the matching HCW_*_MODEL",
                )

    hairline("diarization")
    from .pipeline.diarize import availability

    usable, reason = availability()
    status_line(
        "ok" if usable else "skip",
        "pyannote ready" if usable else "pyannote unavailable",
        reason or "",
    )
    status_line(
        "ok" if config.hf_token else "skip",
        f"hf token {'set' if config.hf_token else 'not set'}",
        "" if config.hf_token else "needed only for pyannote weights",
    )
    console().print()


@app.command()
def process(
    meeting: Annotated[
        str | None, typer.Argument(help="Meeting folder. Defaults to the most recent one.")
    ] = None,
    language: Annotated[
        str | None,
        typer.Option("--language", "-l", help="ISO-639-1 hint, e.g. es. Auto if omitted."),
    ] = None,
    no_diarize: Annotated[
        bool, typer.Option("--no-diarize", help="Skip pyannote; keep the me/others split only.")
    ] = False,
    no_index: Annotated[
        bool, typer.Option("--no-index", help="Skip embeddings; the meeting stays out of search.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Ignore cached step results and redo everything.")
    ] = False,
    progress_json: Annotated[
        bool,
        typer.Option(
            "--progress-json",
            help="Emit one JSON step event per line on stdout, and move the human-readable "
            "output to stderr. For front ends, so they do not have to scrape it.",
        ),
    ] = False,
) -> None:
    """Transcribe, diarize, summarize and index a recorded meeting."""
    from .pipeline.progress import JsonProgress, Progress
    from .pipeline.run import run_process
    from .record import resolve_meeting

    events: Progress = Progress()
    if progress_json:
        # Before the first console() call, or the terminal interface has already
        # claimed stdout and the two audiences end up interleaved on one pipe.
        from .ui.theme import send_human_output_to_stderr

        send_human_output_to_stderr()
        events = JsonProgress()

    def go() -> None:
        config = load_config()
        try:
            run_process(
                config,
                resolve_meeting(config, meeting),
                language=language,
                diarize_enabled=not no_diarize,
                index_enabled=not no_index,
                force=force,
                events=events,
            )
        except Exception as exc:
            # A reader watching the event stream should be told the run died,
            # rather than having the pipe close on it and left guessing.
            events.event(event="error", message=f"{type(exc).__name__}: {exc}")
            raise

    _run(go)


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What you are looking for, in your own words.")],
    limit: Annotated[int, typer.Option("--limit", "-n", help="How many passages to show.")] = 5,
) -> None:
    """Search everything you have ever recorded, by meaning."""
    from .pipeline.search import run_search

    _run(lambda: run_search(load_config(), query, limit=limit))


@app.command(name="list")
def list_meetings() -> None:
    """List recorded meetings."""
    from rich.table import Table

    from .store import Meeting

    config = load_config()
    meetings = Meeting.all(config.home)
    if not meetings:
        console().print(f"\n  No meetings in {config.home}\n", style="tertiary")
        return

    table = Table(box=None, pad_edge=False, padding=(0, 2))
    table.add_column("meeting", style="secondary")
    table.add_column("length", style="tertiary", justify="right")
    table.add_column("notes", style="tertiary")
    for item in meetings:
        meta = item.load_meta()
        seconds = int(meta.get("duration_seconds") or 0)
        table.add_row(
            item.path.name,
            f"{seconds // 60}m{seconds % 60:02d}s" if seconds else "-",
            "yes" if item.notes_md.is_file() else "-",
        )
    console().print()
    hairline("meetings")
    console().print(table)
    console().print()


@app.callback(invoke_without_command=True)
def _root(
    version: Annotated[bool, typer.Option("--version", help="Print the version and exit.")] = False,
) -> None:
    if version:
        console().print(f"helmcode-whisper {__version__}")
        raise typer.Exit()


def _run(action) -> None:  # noqa: ANN001
    """Turn expected failures into a clean message instead of a traceback."""
    try:
        action()
    except ConfigError as exc:
        err_console().print(f"\n  {exc}\n", style="error")
        raise typer.Exit(code=2) from None
    except RuntimeError as exc:
        err_console().print(f"\n  {exc}\n", style="error")
        raise typer.Exit(code=1) from None


def main() -> None:
    app()


if __name__ == "__main__":
    main()
