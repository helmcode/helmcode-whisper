"""`hcw process` — the whole pipeline, resumable at every step.

Each step writes its result to the meeting's `.cache/` before the next one
starts. A crash in the merge does not re-run an hour of transcription, and a
second `process` after a fixed template only pays for the notes. `--force`
throws the cache away.

The steps are ordered by what they need rather than by the order they appear in
the output. Both tracks are prepared at once, and diarization — local, CPU-bound
and by far the most expensive step — starts as soon as the system track is ready
and runs underneath the transcription requests, which spend their time waiting
on the network. Nothing downstream of the transcript can start early, so that
overlap is where the wall clock is won.

Step timings land in `meta.json`, which is where the numbers in the article come
from — measured on the machine that ran it, not estimated.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.text import Text

from ..api import ApiError, HelmcodeClient
from ..config import Config
from ..store import Meeting, file_fingerprint
from ..ui import console, eyebrow, hairline, progress, status_line
from ..ui.html import render_html
from . import audio, diarize, index, merge, notes, stt
from .model import Segment, Transcript


def run_process(
    config: Config,
    meeting: Meeting,
    *,
    language: str | None = None,
    diarize_enabled: bool = True,
    index_enabled: bool = True,
    force: bool = False,
) -> None:
    audio.require_ffmpeg()
    config.require_api_key()

    meta = meeting.load_meta()
    timings: dict[str, float] = {}
    # Which steps were replayed from cache. A cached run finishes in a second
    # and its timings mean nothing; without this flag they would silently
    # overwrite the real measurement in meta.json.
    cached_steps: dict[str, bool] = {}

    console().print()
    console().print(eyebrow("processing"))
    console().print(Text(f"  {meeting.path}", style="tertiary"))
    console().print()

    sources = _present_tracks(meeting)
    if not sources:
        raise RuntimeError(f"No audio files in {meeting.path}")

    with HelmcodeClient(config) as client:
        # ── 1-2. prepare and plan, both tracks at once ───────────
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(sources)) as pool:
            tracks = list(
                pool.map(
                    lambda item: _prepare_track(meeting, item[0], item[1], force=force),
                    sources.items(),
                )
            )
        # Printed here rather than inside the workers: two threads writing to
        # the same console interleave their lines.
        for track in tracks:
            status_line(
                "ok" if track.chunks else "warn",
                f"{track.name:<7} {track.duration_seconds / 60:.1f} min",
                f"{track.speech_seconds / 60:.1f} min of speech in {len(track.chunks)} chunks"
                + ("" if track.chunks else " — this track will not be transcribed"),
            )
        timings["prepare"] = time.monotonic() - started
        cached_steps["prepare"] = all(track.from_cache for track in tracks)

        total_chunks = sum(len(track.chunks) for track in tracks)
        if not total_chunks:
            raise RuntimeError("No speech detected in the recording.")

        # ── 3. diarize and transcribe, together ──────────────────
        diarization = _Diarization(
            meeting,
            config,
            next((track for track in tracks if track.name == "system"), None),
            enabled=diarize_enabled,
            force=force,
        )
        diarization.start()

        started = time.monotonic()
        with progress() as bar:
            task = bar.add_task("transcribing", total=total_chunks)
            transcription = stt.transcribe(
                client,
                meeting,
                [track.plan for track in tracks],
                language=language,
                model=config.stt_model,
                force=force,
                concurrency=config.stt_concurrency,
                on_chunk_done=lambda: bar.advance(task),
            )
        timings["transcribe"] = time.monotonic() - started
        cached_steps["transcribe"] = transcription.fully_cached

        segments = transcription.segments
        detected_language = transcription.language
        status_line(
            "ok",
            f"transcribed {sum(len(items) for items in segments.values())} segments",
            f"language {detected_language or 'unknown'}",
        )

        # ── 4. collect the diarization that has been running ─────
        # This measures what diarization still costs the pipeline, which is what
        # it had left to do when the last chunk came back — not what it cost in
        # total. `_Diarization.seconds` records that separately, so meta.json
        # keeps both and neither can be mistaken for the other.
        started = time.monotonic()
        diarized, device, cached_steps["diarize"] = diarization.apply(segments.get("system") or [])
        timings["diarize"] = time.monotonic() - started

        # ── 5. merge ─────────────────────────────────────────────
        merged = merge.merge(segments.get("mic", []), segments.get("system", []))
        echoes = sum(1 for segment in merged if segment.dropped == "echo")
        if echoes:
            status_line(
                "warn",
                f"{echoes} microphone segments dropped as echo",
                "the remote audio was picked up by the mic — use headphones",
            )

        transcript = Transcript(
            segments=merged,
            language=detected_language,
            speakers=merge.speaker_list(merged),
            diarized=diarized,
        )
        meeting.transcript_json.write_text(
            json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        status_line(
            "ok", f"transcript  {len(transcript.kept)} turns", ", ".join(transcript.speakers)
        )

        # ── 6. notes ─────────────────────────────────────────────
        started = time.monotonic()
        duration_minutes = (meta.get("duration_seconds") or _span(merged)) / 60
        cached_notes = None if force else meeting.read_cached_json("notes.json")
        cached_steps["notes"] = bool(cached_notes)
        if cached_notes:
            note_data, note_stats = cached_notes["notes"], cached_notes["stats"]
            status_line("skip", "notes from cache", "--force to regenerate")
        else:
            note_data, note_stats = notes.generate_notes(
                client,
                transcript,
                model=config.notes_model,
                title=str(meta.get("title", meeting.path.name)),
                date=str(meta.get("started_at", ""))[:10],
                duration_minutes=duration_minutes,
            )
            meeting.write_cached_json({"notes": note_data, "stats": note_stats}, "notes.json")
            status_line(
                "ok",
                f"notes generated via {note_stats.get('mode')}",
                f"{note_stats.get('prompt_tokens', 0)} in / "
                f"{note_stats.get('completion_tokens', 0)} out tokens",
            )
        timings["notes"] = time.monotonic() - started

        meta = meeting.save_meta(
            {
                "language": detected_language,
                "speakers": transcript.speakers,
                "diarized": diarized,
                "diarization_device": device,
                "duration_seconds": meta.get("duration_seconds") or round(_span(merged), 2),
            }
        )
        meeting.notes_md.write_text(notes.render_markdown(note_data, meta), encoding="utf-8")
        meeting.notes_html.write_text(
            render_html(note_data, meta, transcript), encoding="utf-8"
        )

        # ── 7. index ─────────────────────────────────────────────
        started = time.monotonic()
        index_stats = {"passages": 0, "embedded": 0}
        if index_enabled:
            index_stats = _index_meeting(config, client, meeting, meta, transcript)
        else:
            status_line("skip", "indexing disabled", "--no-index")
        timings["index"] = time.monotonic() - started
        cached_steps["index"] = False

    run_record = {
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "timings_seconds": {key: round(value, 2) for key, value in timings.items()},
        # What diarization cost on its own, as opposed to what it cost the
        # pipeline after overlapping with transcription. The gap between this
        # and timings_seconds["diarize"] is what the overlap saved.
        "diarization_seconds": round(diarization.seconds, 2),
        "from_cache": cached_steps,
        "chunks": {track.name: len(track.chunks) for track in tracks},
        "stt_concurrency": config.stt_concurrency,
        "notes_tokens": {
            "prompt": note_stats.get("prompt_tokens", 0),
            "completion": note_stats.get("completion_tokens", 0),
            "mode": note_stats.get("mode"),
        },
        "index": index_stats,
    }
    # Every run is appended rather than overwritten. A replayed run finishes in
    # a second, and letting it overwrite the one that actually did the work
    # would quietly destroy the only measurement of this meeting.
    history = [*meta.get("runs", []), run_record]
    meeting.save_meta(
        {
            **run_record,
            "runs": history,
            "transcript_chars": len(transcript.as_text()),
            "echo_segments_dropped": echoes,
        }
    )

    _print_summary(meeting, note_data, timings)


# ── preparation ──────────────────────────────────────────────────


@dataclass(frozen=True)
class _Track:
    """A prepared track and everything later steps need to know about it."""

    name: str
    path: Path
    fingerprint: str
    chunks: list[audio.Chunk]
    duration_seconds: float
    speech_seconds: float
    from_cache: bool

    @property
    def plan(self) -> stt.TrackPlan:
        return stt.TrackPlan(self.name, self.path, self.fingerprint, self.chunks)


def _present_tracks(meeting: Meeting) -> dict[str, Path]:
    tracks: dict[str, Path] = {}
    if meeting.mic_wav.is_file() and meeting.mic_wav.stat().st_size > 44:
        tracks["mic"] = meeting.mic_wav
    if meeting.system_wav.is_file() and meeting.system_wav.stat().st_size > 44:
        tracks["system"] = meeting.system_wav
    return tracks


def _prepare_track(meeting: Meeting, name: str, source: Path, *, force: bool) -> _Track:
    """Resample one track and work out where to cut it. Runs off the main thread.

    The chunk plan is cached alongside the prepared audio. It is a deterministic
    function of that audio, and recomputing it means reading the whole track
    back and running voice activity detection over every 30 ms of it — a few
    seconds per hour of recording, paid on every run including the ones that
    have nothing else left to do.
    """
    target = meeting.cache_path("prepared", f"{name}-16k.wav")
    if force and target.is_file():
        target.unlink()
    was_prepared = target.is_file()
    audio.prepare_track(source, target)

    # One hash of the prepared track, shared by the chunk encoder, the
    # transcription cache and the diarization cache. It used to be computed
    # separately by each of them, which is three passes over ~115 MB per track.
    fingerprint = file_fingerprint(target)

    cache_key = f"plan-{name}-{fingerprint}.json"
    cached = None if force else meeting.read_cached_json(cache_key)
    if cached:
        return _Track(
            name=name,
            path=target,
            fingerprint=fingerprint,
            chunks=[audio.Chunk(**chunk) for chunk in cached["chunks"]],
            duration_seconds=cached["duration_seconds"],
            speech_seconds=cached["speech_seconds"],
            from_cache=was_prepared,
        )

    regions = audio.speech_regions(target)
    track = _Track(
        name=name,
        path=target,
        fingerprint=fingerprint,
        chunks=audio.plan_chunks(regions),
        duration_seconds=audio.track_duration(target),
        speech_seconds=sum(end - begin for begin, end in regions),
        from_cache=was_prepared,
    )
    meeting.write_cached_json(
        {
            "chunks": [chunk.__dict__ for chunk in track.chunks],
            "duration_seconds": track.duration_seconds,
            "speech_seconds": track.speech_seconds,
        },
        cache_key,
    )
    return track


# ── diarization, in the background ───────────────────────────────


@dataclass
class _DiarizationOutcome:
    turns: list[diarize.Turn] | None = None
    device: str | None = None
    from_cache: bool = False
    # Why there are no turns, and how loudly to say so. A missing pyannote is
    # an expected configuration; pyannote blowing up halfway through is not.
    problem: str | None = None
    level: str = "skip"
    seconds: float = 0.0
    details: list[str] = field(default_factory=list)


class _Diarization:
    """Speaker turns for the system track, computed while transcription runs.

    Diarization needs the prepared system audio and nothing else. It does not
    need the transcript, which is the only reason it can start this early — and
    it should, because it is local CPU work competing with a step that is almost
    entirely idle waiting on HTTP. On the first real recording it was 46 s of a
    70 s run.

    The worker never prints and never touches shared state: everything it learns
    comes back through `_DiarizationOutcome`, and `apply` reports it from the
    main thread once the progress bar is gone.

    It is a daemon thread rather than a pool because of what happens when
    something upstream fails. If transcription raises — an expired key, a
    network that went away — nothing ever collects this result, and a
    non-daemon worker holds the interpreter open until it finishes: the user
    reads their error message and then watches the command hang for as long as
    diarization takes, which on a 60-minute meeting is half an hour. A daemon
    thread is abandoned at exit, which is the right answer for work whose
    output nobody is going to read.
    """

    def __init__(
        self,
        meeting: Meeting,
        config: Config,
        track: _Track | None,
        *,
        enabled: bool,
        force: bool,
    ) -> None:
        self._meeting = meeting
        self._config = config
        self._track = track
        self._force = force
        self._thread: threading.Thread | None = None
        self._outcome = _DiarizationOutcome()
        self.seconds = 0.0

        self._skip: str | None = None
        if not enabled:
            self._skip = "--no-diarize"
        elif track is None:
            self._skip = "no system track to separate"
        elif not track.chunks:
            self._skip = "no speech on the system track"

    def start(self) -> None:
        if self._skip is not None:
            return
        self._thread = threading.Thread(target=self._run, name="diarize", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        # Timed out here, around everything the thread does, so that it is
        # directly comparable with how long the main thread waited. Timing only
        # the pyannote call would leave the import — torch is seconds on its
        # own — outside the number, and `diarization_seconds` minus the wait
        # would come out negative on a short meeting.
        began = time.monotonic()
        try:
            outcome = self._work()
        except BaseException as exc:  # noqa: BLE001 — reported, never raised here
            outcome = _DiarizationOutcome(problem=f"{type(exc).__name__}: {exc}", level="warn")
        outcome.seconds = time.monotonic() - began
        self._outcome = outcome

    def _work(self) -> _DiarizationOutcome:
        assert self._track is not None
        cache_key = f"diarization-{self._track.fingerprint}.json"
        cached = None if self._force else self._meeting.read_cached_json(cache_key)
        if cached:
            return _DiarizationOutcome(
                turns=[diarize.Turn(**turn) for turn in cached["turns"]],
                device=cached.get("device"),
                from_cache=True,
            )

        usable, reason = diarize.availability()
        if not usable:
            return _DiarizationOutcome(problem=reason or "pyannote is unavailable")

        try:
            turns, device = diarize.diarize(self._track.path, self._config.hf_token)
        except diarize.DiarizationUnavailable as exc:
            return _DiarizationOutcome(problem=str(exc))
        except Exception as exc:
            # Diarization is the optional step. Whatever pyannote, torch or a
            # model download does in there, the meeting still has a transcript
            # and notes waiting on the other side of it.
            return _DiarizationOutcome(problem=f"{type(exc).__name__}: {exc}", level="warn")

        self._meeting.write_cached_json(
            {"turns": [turn.__dict__ for turn in turns], "device": device}, cache_key
        )
        return _DiarizationOutcome(turns=turns, device=device)

    def apply(self, segments: list[Segment]) -> tuple[bool, str | None, bool]:
        """Wait for the turns and label the segments. (diarized, device, cached)."""
        if self._skip is not None:
            status_line("skip", "diarization skipped", self._skip)
            return False, None, False

        assert self._thread is not None
        self._thread.join()
        outcome = self._outcome
        self.seconds = outcome.seconds

        if outcome.turns is None:
            if outcome.level == "warn":
                status_line("warn", f"diarization failed: {outcome.problem}")
            status_line("skip", "diarization unavailable — keeping the me/others split",
                        outcome.problem or "")
            return False, None, False

        if not segments:
            status_line("skip", "diarization skipped", "the system track transcribed to nothing")
            return False, None, outcome.from_cache

        if outcome.from_cache:
            status_line("skip", f"diarization from cache ({len(outcome.turns)} turns)")
        else:
            status_line(
                "ok",
                f"diarization  {len(outcome.turns)} turns on {outcome.device}",
                f"{outcome.seconds:.0f}s, alongside transcription",
            )

        # Split first, then label: cutting on word timestamps is what lets a
        # segment containing two people become two segments instead of one
        # wrong label.
        refined = diarize.split_by_speaker(segments, outcome.turns)
        for segment, label in zip(
            refined, diarize.label_segments(refined, outcome.turns), strict=True
        ):
            segment.speaker = label

        segments[:] = refined
        return True, outcome.device, outcome.from_cache


# ── indexing and output ──────────────────────────────────────────


def _index_meeting(
    config: Config,
    client: HelmcodeClient,
    meeting: Meeting,
    meta: dict[str, Any],
    transcript: Transcript,
) -> dict[str, int]:
    passages = index.build_passages(transcript.segments)
    connection = index.connect(config.db_path)
    try:
        try:
            stats = index.index_meeting(
                connection,
                client,
                meeting_id=meeting.path.name,
                title=str(meta.get("title", meeting.path.name)),
                date=str(meta.get("started_at", ""))[:10],
                path=meeting.path,
                passages=passages,
                embed_model=config.embed_model,
            )
            status_line(
                "ok", f"indexed  {stats['passages']} passages", f"{stats['embedded']} embedded"
            )
        except ApiError as exc:
            stats = index.index_meeting(
                connection,
                None,
                meeting_id=meeting.path.name,
                title=str(meta.get("title", meeting.path.name)),
                date=str(meta.get("started_at", ""))[:10],
                path=meeting.path,
                passages=passages,
                embed_model=config.embed_model,
            )
            status_line(
                "warn",
                f"indexed  {stats['passages']} passages, keyword search only",
                f"embeddings unavailable: {exc}",
            )
    finally:
        connection.close()
    return stats


def _span(segments: list[Segment]) -> float:
    return max((segment.end for segment in segments), default=0.0)


def _print_summary(meeting: Meeting, note_data: dict[str, Any], timings: dict[str, float]) -> None:
    console().print()
    hairline("summary")
    console().print(Text(f"  {note_data.get('summary', '')}", style="secondary"))

    if note_data.get("decisions"):
        console().print()
        console().print(eyebrow("decisions"))
        for item in note_data["decisions"]:
            console().print(Text(f"  · {item}", style="secondary"))

    if note_data.get("action_items"):
        console().print()
        console().print(eyebrow("action items"))
        for item in note_data["action_items"]:
            who = " · ".join(filter(None, [item.get("owner"), item.get("due")]))
            line = Text.assemble(("  □ ", "tertiary"), (item["task"], "secondary"))
            if who:
                line.append(f"  {who}", style="accent")
            console().print(line)

    console().print()
    hairline("files")
    for path in (meeting.notes_md, meeting.notes_html, meeting.transcript_json):
        status_line("ok", path.name, str(path.parent))
    total = sum(timings.values())
    console().print()
    console().print(
        Text(
            "  " + "  ".join(f"{key} {value:.0f}s" for key, value in timings.items())
            + f"   total {total:.0f}s",
            style="tertiary",
        )
    )
    console().print()
