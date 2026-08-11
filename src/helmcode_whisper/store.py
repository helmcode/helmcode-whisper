"""On-disk layout for a meeting, and the cache that makes `process` resumable.

    ~/helmcode-whisper/2026-08-11-pricing-review/
        audio-mic.wav        audio-system.wav
        transcript.json      notes.md      notes.html
        meta.json
        .cache/              chunk audio + per-step results

Nothing here is uploaded anywhere. The cache is what lets a failed merge skip
straight past a completed transcription instead of paying for it twice.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(title: str, *, max_length: int = 48) -> str:
    normalized = unicodedata.normalize("NFKD", title)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = _SLUG_STRIP.sub("-", ascii_only).strip("-")
    return slug[:max_length].strip("-") or "meeting"


@dataclass(frozen=True)
class Meeting:
    path: Path

    # ── files ────────────────────────────────────────────────────

    @property
    def mic_wav(self) -> Path:
        return self.path / "audio-mic.wav"

    @property
    def system_wav(self) -> Path:
        return self.path / "audio-system.wav"

    @property
    def transcript_json(self) -> Path:
        return self.path / "transcript.json"

    @property
    def notes_md(self) -> Path:
        return self.path / "notes.md"

    @property
    def notes_html(self) -> Path:
        return self.path / "notes.html"

    @property
    def meta_json(self) -> Path:
        return self.path / "meta.json"

    @property
    def cache_dir(self) -> Path:
        return self.path / ".cache"

    # ── lifecycle ────────────────────────────────────────────────

    @classmethod
    def create(cls, home: Path, title: str, started_at: datetime) -> Meeting:
        base = f"{started_at:%Y-%m-%d}-{slugify(title)}"
        path = home / base
        # Two meetings a day can share a title; never clobber the first one.
        suffix = 2
        while path.exists():
            path = home / f"{base}-{suffix}"
            suffix += 1
        path.mkdir(parents=True)
        meeting = cls(path)
        meeting.save_meta(
            {
                "title": title,
                "started_at": started_at.isoformat(timespec="seconds"),
                "tool_version": "0.1.0",
            }
        )
        return meeting

    @classmethod
    def latest(cls, home: Path) -> Meeting | None:
        """The meeting that started most recently.

        Not the most recently *modified* one: processing an old meeting would
        make it the newest by mtime, and then a bare `hcw process` would keep
        picking it instead of the recording that just finished. The start time
        in meta.json is the fact that actually answers the question; the folder
        name, which begins with the date, is the fallback when it is missing.
        """
        candidates = [cls(p) for p in home.glob("*/") if (p / "meta.json").is_file()]
        if not candidates:
            return None
        return max(candidates, key=lambda meeting: meeting._sort_key())

    def _sort_key(self) -> tuple[str, str]:
        started = str(self.load_meta().get("started_at") or "")
        return (started, self.path.name)

    @classmethod
    def all(cls, home: Path) -> list[Meeting]:
        return sorted(
            (cls(p) for p in home.glob("*/") if (p / "meta.json").is_file()),
            key=lambda m: m.path.name,
        )

    # ── metadata ─────────────────────────────────────────────────

    def load_meta(self) -> dict[str, Any]:
        if not self.meta_json.is_file():
            return {}
        return json.loads(self.meta_json.read_text(encoding="utf-8"))

    def save_meta(self, updates: dict[str, Any]) -> dict[str, Any]:
        meta = self.load_meta()
        meta.update(updates)
        self.meta_json.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return meta

    @property
    def title(self) -> str:
        return self.load_meta().get("title") or self.path.name

    # ── cache ────────────────────────────────────────────────────

    def cache_path(self, *parts: str) -> Path:
        path = self.cache_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def read_cached_json(self, *parts: str) -> Any | None:
        path = self.cache_dir.joinpath(*parts)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            # A cache entry truncated by a crash is not worth a hard failure.
            return None

    def write_cached_json(self, payload: Any, *parts: str) -> None:
        path = self.cache_path(*parts)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def file_fingerprint(path: Path, *, extra: str = "") -> str:
    """A short content hash, used to key cached API results to their input."""
    digest = hashlib.sha256()
    digest.update(extra.encode("utf-8"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]
