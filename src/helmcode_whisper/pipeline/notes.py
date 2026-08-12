"""Structured notes from a transcript, via DeepSeek V4 Flash.

The prompt lives in `templates/notes.md`, not in this file. Personalising what
the notes say is the whole reason to run your own tool instead of buying one,
and a prompt buried in Python is not personalisable.

What the template does *not* control is the shape of the output: that is fixed
by `NOTES_SCHEMA` and enforced by the API's structured-output mode. The two
serve different jobs — the schema keeps `notes.md`, `notes.html` and the search
index from having to guess, and the template decides tone, language and what
counts as a decision.
"""

from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from ..api import ApiError, HelmcodeClient
from .model import Transcript

# Structured output is documented as validated on qwen3.6 and gemma4, not on
# deepseek-v4-flash. Rather than pick a different model, ask for the strictest
# mode and walk down: json_schema, then json_object, then a plain request with
# tolerant parsing. Which rung it landed on is reported in meta.json.
_STRUCTURED_MODES = ("json_schema", "json_object", "plain")

# Past this many characters the transcript is summarized in blocks and the
# partial results merged. A 60-minute meeting is roughly 55k characters, so the
# single-pass path covers the common case.
MAP_REDUCE_THRESHOLD = 120_000
_BLOCK_CHARS = 60_000
# Block summaries in flight at once. Smaller than the transcription pool: these
# are long generations rather than short ones, and there are only ever a handful.
_BLOCK_CONCURRENCY = 3

NOTES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "decisions", "action_items", "open_questions", "quotes"],
    "properties": {
        "summary": {
            "type": "string",
            "description": "5 to 8 sentences covering what the meeting was about.",
        },
        "decisions": {"type": "array", "items": {"type": "string"}},
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["task", "owner", "due"],
                "properties": {
                    "task": {"type": "string"},
                    "owner": {"type": "string", "description": "Empty when not stated."},
                    "due": {"type": "string", "description": "Empty when not stated."},
                },
            },
        },
        "open_questions": {"type": "array", "items": {"type": "string"}},
        "quotes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["speaker", "text"],
                "properties": {"speaker": {"type": "string"}, "text": {"type": "string"}},
            },
        },
    },
}

_COMMENT_BLOCK = re.compile(r"<!--.*?-->", re.DOTALL)


class NotesError(RuntimeError):
    pass


# ── template ─────────────────────────────────────────────────────


def find_template() -> Path:
    """First match wins: env override, project-local, then the packaged copy."""
    candidates: list[Path] = []
    override = os.environ.get("HCW_NOTES_TEMPLATE")
    if override:
        candidates.append(Path(override).expanduser())
    candidates.append(Path.cwd() / "templates" / "notes.md")
    candidates.append(Path(__file__).resolve().parent.parent / "templates" / "notes.md")
    # Running from a checkout: src/helmcode_whisper/pipeline -> repo root.
    candidates.append(Path(__file__).resolve().parents[3] / "templates" / "notes.md")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise NotesError(
        "No notes template found. Expected templates/notes.md, or set HCW_NOTES_TEMPLATE."
    )


def render_prompt(template: Path, transcript_text: str, context: dict[str, str]) -> str:
    body = _COMMENT_BLOCK.sub("", template.read_text(encoding="utf-8")).strip()
    for key, value in context.items():
        body = body.replace("{{" + key + "}}", value)
    return body.replace("{{TRANSCRIPT}}", transcript_text)


# ── generation ───────────────────────────────────────────────────


def generate_notes(
    client: HelmcodeClient,
    transcript: Transcript,
    *,
    model: str,
    title: str,
    date: str,
    duration_minutes: float,
    template: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (notes, stats). Stats carry token usage for NUMBERS.md."""
    template = template or find_template()
    text = transcript.as_text()
    if not text.strip():
        raise NotesError("The transcript is empty; there is nothing to summarize.")

    context = {
        "TITLE": title,
        "DATE": date,
        "DURATION": f"{duration_minutes:.0f}",
        "SPEAKERS": ", ".join(transcript.speakers) or "unknown",
    }

    if len(text) <= MAP_REDUCE_THRESHOLD:
        prompt = render_prompt(template, text, context)
        notes, stats = _complete(client, prompt, model=model)
        return notes, stats

    blocks = _split_transcript(text, _BLOCK_CHARS)
    total_stats = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0, "mode": None}
    memo = _ModeMemo()

    def summarize_block(numbered: tuple[int, str]) -> tuple[dict[str, Any], dict[str, Any]]:
        position, block = numbered
        block_context = dict(context)
        block_context["TITLE"] = f"{title} (part {position} of {len(blocks)})"
        return _complete(
            client, render_prompt(template, block, block_context), model=model, memo=memo
        )

    # The blocks do not depend on each other — each one summarizes its own slice
    # and only the final pass sees them together — so they go out concurrently.
    # Sequentially this step is one full round trip per block, and it only runs
    # at all on transcripts long enough that there are several of them.
    with ThreadPoolExecutor(max_workers=min(_BLOCK_CONCURRENCY, len(blocks))) as pool:
        results = list(pool.map(summarize_block, enumerate(blocks, start=1)))

    partials = [notes for notes, _ in results]
    for _, stats in results:
        _accumulate(total_stats, stats)

    merged = _merge_partials(partials)
    # One more pass so the summary reads as one meeting rather than N recaps.
    summary_prompt = render_prompt(
        template,
        "\n\n".join(f"Part {i}: {part.get('summary', '')}" for i, part in enumerate(partials, 1)),
        context,
    )
    final, stats = _complete(client, summary_prompt, model=model, memo=memo)
    _accumulate(total_stats, stats)
    merged["summary"] = final.get("summary") or merged["summary"]
    return merged, total_stats


class _ModeMemo:
    """The structured-output mode that has already worked, within one run.

    The ladder exists because the API documents `json_schema` as validated on
    other models than this one. When it works — which is what every measured
    run has shown — there is nothing to remember. When it does not, this is
    what stops each of a long transcript's blocks paying separately for the
    same two rejected requests.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mode: str | None = None

    def ladder(self) -> tuple[str, ...]:
        with self._lock:
            known = self._mode
        if known is None:
            return _STRUCTURED_MODES
        return (known, *(mode for mode in _STRUCTURED_MODES if mode != known))

    def worked(self, mode: str) -> None:
        with self._lock:
            self._mode = mode


def _complete(
    client: HelmcodeClient, prompt: str, *, model: str, memo: _ModeMemo | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    memo = memo or _ModeMemo()
    last_error: Exception | None = None
    for mode in memo.ladder():
        response_format: dict[str, Any] | None = None
        if mode == "json_schema":
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "meeting_notes", "strict": True, "schema": NOTES_SCHEMA},
            }
        elif mode == "json_object":
            response_format = {"type": "json_object"}

        instruction = prompt
        if mode != "json_schema":
            instruction = (
                f"{prompt}\n\nAnswer with a single JSON object and nothing else, "
                f"matching this schema:\n{json.dumps(NOTES_SCHEMA, ensure_ascii=False)}"
            )

        try:
            payload = client.chat(
                [{"role": "user", "content": instruction}],
                model=model,
                response_format=response_format,
                temperature=0.2,
            )
            content = payload["choices"][0]["message"]["content"]
            notes = _coerce(_parse_json(content))
        except (ApiError, KeyError, ValueError) as exc:
            last_error = exc
            continue

        memo.worked(mode)
        usage = payload.get("usage") or {}
        return notes, {
            "mode": mode,
            "requests": 1,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }

    raise NotesError(f"The notes model returned nothing usable: {last_error}")


def _parse_json(content: str) -> dict[str, Any]:
    content = content.strip()
    try:
        return json.loads(content)
    except ValueError:
        pass
    # A fenced block, or prose wrapped around the object.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    braces = re.search(r"\{.*\}", content, re.DOTALL)
    if braces:
        return json.loads(braces.group(0))
    raise ValueError("no JSON object in the response")


def _coerce(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize a loosely-shaped response into the schema's shape."""
    summary = data.get("summary") or ""
    if isinstance(summary, list):
        summary = " ".join(str(item) for item in summary)

    def string_list(key: str) -> list[str]:
        value = data.get(key) or []
        if isinstance(value, str):
            value = [value]
        return [str(item).strip() for item in value if str(item).strip()]

    action_items = []
    for item in data.get("action_items") or []:
        if isinstance(item, str):
            action_items.append({"task": item, "owner": "", "due": ""})
        elif isinstance(item, dict) and (item.get("task") or "").strip():
            action_items.append(
                {
                    "task": str(item["task"]).strip(),
                    "owner": str(item.get("owner") or "").strip(),
                    "due": str(item.get("due") or "").strip(),
                }
            )

    quotes = []
    for item in data.get("quotes") or []:
        if isinstance(item, dict) and (item.get("text") or "").strip():
            quotes.append(
                {
                    "speaker": str(item.get("speaker") or "").strip(),
                    "text": str(item["text"]).strip(),
                }
            )

    return {
        "summary": str(summary).strip(),
        "decisions": string_list("decisions"),
        "action_items": action_items,
        "open_questions": string_list("open_questions"),
        "quotes": quotes,
    }


def _split_transcript(text: str, block_chars: int) -> list[str]:
    lines = text.split("\n")
    blocks: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if size + len(line) > block_chars and current:
            blocks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        blocks.append("\n".join(current))
    return blocks


def _merge_partials(partials: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "summary": " ".join(part.get("summary", "") for part in partials).strip(),
        "decisions": [],
        "action_items": [],
        "open_questions": [],
        "quotes": [],
    }
    seen: set[str] = set()
    for part in partials:
        for key in ("decisions", "open_questions"):
            for item in part.get(key, []):
                if item.lower() not in seen:
                    seen.add(item.lower())
                    merged[key].append(item)
        merged["action_items"].extend(part.get("action_items", []))
        merged["quotes"].extend(part.get("quotes", []))
    return merged


def _accumulate(total: dict[str, Any], stats: dict[str, Any]) -> None:
    total["requests"] += stats.get("requests", 0)
    total["prompt_tokens"] += stats.get("prompt_tokens", 0)
    total["completion_tokens"] += stats.get("completion_tokens", 0)
    total["mode"] = total["mode"] or stats.get("mode")


# ── rendering ────────────────────────────────────────────────────


def render_markdown(notes: dict[str, Any], meta: dict[str, Any]) -> str:
    started = meta.get("started_at", "")
    try:
        when = datetime.fromisoformat(started).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        when = started

    minutes = (meta.get("duration_seconds") or 0) / 60
    speakers = meta.get("speakers") or []

    lines = [f"# {meta.get('title', 'Meeting')}", ""]
    detail = [when, f"{minutes:.0f} min"]
    if speakers:
        detail.append(", ".join(speakers))
    lines.append(" · ".join(part for part in detail if part))
    lines += ["", "## Summary", "", notes["summary"], ""]

    if notes["decisions"]:
        lines += ["## Decisions", ""]
        lines += [f"- {item}" for item in notes["decisions"]]
        lines.append("")

    if notes["action_items"]:
        lines += ["## Action items", ""]
        for item in notes["action_items"]:
            suffix = []
            if item.get("owner"):
                suffix.append(item["owner"])
            if item.get("due"):
                suffix.append(item["due"])
            tail = f" — {' · '.join(suffix)}" if suffix else ""
            lines.append(f"- [ ] {item['task']}{tail}")
        lines.append("")

    if notes["open_questions"]:
        lines += ["## Open questions", ""]
        lines += [f"- {item}" for item in notes["open_questions"]]
        lines.append("")

    if notes["quotes"]:
        lines += ["## Quotes", ""]
        for quote in notes["quotes"]:
            lines.append(f"> {quote['text']}")
            lines.append(">")
            lines.append(f"> — {quote['speaker'] or 'unknown'}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"
