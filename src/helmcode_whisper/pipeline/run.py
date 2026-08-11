"""`hcw process` — the whole pipeline, resumable at every step.

Each step writes its result to the meeting's `.cache/` before the next one
starts. A crash in the merge does not re-run an hour of transcription, and a
second `process` after a fixed template only pays for the notes. `--force`
throws the cache away.

Step timings land in `meta.json`, which is where the numbers in the article come
from — measured on the machine that ran it, not estimated.
"""

from __future__ import annotations

import json
import time
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

    tracks = _present_tracks(meeting)
    if not tracks:
        raise RuntimeError(f"No audio files in {meeting.path}")

    with HelmcodeClient(config) as client:
        # ── 1-2. prepare and plan ────────────────────────────────
        started = time.monotonic()
        prepared: dict[str, Path] = {}
        plans: dict[str, list[audio.Chunk]] = {}
        prepare_hits = 0
        for track, source in tracks.items():
            target = meeting.cache_path("prepared", f"{track}-16k.wav")
            if force and target.is_file():
                target.unlink()
            prepare_hits += 1 if target.is_file() else 0
            prepared[track] = audio.prepare_track(source, target)
            regions = audio.speech_regions(prepared[track])
            plans[track] = audio.plan_chunks(regions)
            speech = sum(end - begin for begin, end in regions)
            status_line(
                "ok" if plans[track] else "warn",
                f"{track:<7} {audio.track_duration(prepared[track]) / 60:.1f} min",
                f"{speech / 60:.1f} min of speech in {len(plans[track])} chunks"
                + ("" if plans[track] else " — this track will not be transcribed"),
            )
        timings["prepare"] = time.monotonic() - started
        cached_steps["prepare"] = prepare_hits == len(tracks)

        total_chunks = sum(len(chunks) for chunks in plans.values())
        if not total_chunks:
            raise RuntimeError("No speech detected in the recording.")

        # ── 3. transcribe ────────────────────────────────────────
        started = time.monotonic()
        segments: dict[str, list[Segment]] = {}
        detected_language = language
        stt_cache_hits = 0
        with progress() as bar:
            task = bar.add_task("transcribing", total=total_chunks)
            for track, chunks in plans.items():
                track_segments, found, hits = stt.transcribe_track(
                    client,
                    meeting,
                    track=track,
                    prepared_wav=prepared[track],
                    chunks=chunks,
                    language=language,
                    model=config.stt_model,
                    force=force,
                    on_chunk_done=lambda: bar.advance(task),
                )
                segments[track] = track_segments
                detected_language = detected_language or found
                stt_cache_hits += hits
        timings["transcribe"] = time.monotonic() - started
        cached_steps["transcribe"] = stt_cache_hits == total_chunks
        status_line(
            "ok",
            f"transcribed {sum(len(items) for items in segments.values())} segments",
            f"language {detected_language or 'unknown'}",
        )

        # ── 4. diarize, locally ──────────────────────────────────
        started = time.monotonic()
        diarized = False
        device = None
        cached_steps["diarize"] = False
        if diarize_enabled and "system" in segments and segments["system"]:
            diarized, device, cached_steps["diarize"] = _diarize_system_track(
                meeting, config, prepared["system"], segments["system"], force=force
            )
        elif diarize_enabled:
            status_line("skip", "diarization skipped", "no system track to separate")
        else:
            status_line("skip", "diarization disabled", "--no-diarize")
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
        "from_cache": cached_steps,
        "chunks": {track: len(chunks) for track, chunks in plans.items()},
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


def _present_tracks(meeting: Meeting) -> dict[str, Path]:
    tracks: dict[str, Path] = {}
    if meeting.mic_wav.is_file() and meeting.mic_wav.stat().st_size > 44:
        tracks["mic"] = meeting.mic_wav
    if meeting.system_wav.is_file() and meeting.system_wav.stat().st_size > 44:
        tracks["system"] = meeting.system_wav
    return tracks


def _diarize_system_track(
    meeting: Meeting,
    config: Config,
    prepared: Path,
    segments: list[Segment],
    *,
    force: bool,
) -> tuple[bool, str | None, bool]:
    """Returns (diarized, device, came_from_cache)."""
    fingerprint = file_fingerprint(prepared)
    cache_key = f"diarization-{fingerprint}.json"
    cached = None if force else meeting.read_cached_json(cache_key)

    if cached:
        turns = [diarize.Turn(**turn) for turn in cached["turns"]]
        device = cached.get("device")
        status_line("skip", f"diarization from cache ({len(turns)} turns)")
    else:
        usable, reason = diarize.availability()
        if not usable:
            status_line(
                "skip",
                "diarization unavailable — keeping the me/others split",
                reason or "",
            )
            return False, None, False
        console().print(
            Text("  diarizing locally — this runs on CPU and takes a while", style="tertiary")
        )
        try:
            turns, device = diarize.diarize(prepared, config.hf_token)
        except diarize.DiarizationUnavailable as exc:
            status_line("skip", "diarization skipped", str(exc))
            return False, None, False
        except Exception as exc:
            # Diarization is the optional step. Whatever pyannote, torch or a
            # model download does in there, the meeting still has a transcript
            # and notes waiting on the other side of it.
            status_line("warn", f"diarization failed: {type(exc).__name__}: {exc}")
            status_line("skip", "keeping the me/others split")
            return False, None, False
        meeting.write_cached_json(
            {"turns": [turn.__dict__ for turn in turns], "device": device}, cache_key
        )
        status_line("ok", f"diarization  {len(turns)} turns on {device}")

    # Split first, then label: cutting on word timestamps is what lets a segment
    # containing two people become two segments instead of one wrong label.
    refined = diarize.split_by_speaker(segments, turns)
    for segment, label in zip(refined, diarize.label_segments(refined, turns), strict=True):
        segment.speaker = label

    segments[:] = refined
    return True, device, bool(cached)


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
