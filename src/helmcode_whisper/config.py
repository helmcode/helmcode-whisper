"""Configuration, read once from the environment and a local `.env`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://api.helmcode.com/v1"

# Model ids as served by the Helmcode API. Overridable so the tool survives the
# day these change without a release.
DEFAULT_STT_MODEL = "whisper"
DEFAULT_NOTES_MODEL = "deepseek-v4-flash"
DEFAULT_EMBED_MODEL = "qwen3-embedding"
DEFAULT_RERANK_MODEL = "rerank"

# Transcription requests in flight at once, across both tracks. At the current
# chunk size an hour-long meeting is about 16 of them, so this decides less than
# it used to, but it is still the difference between one wave and several.
#
# Four, because the API enforces `max_parallel_requests: 5` per key and answers
# a sixth with a 429. Measured, not guessed: six produced
# "Limit type: max_parallel_requests. Current limit: 5, Remaining: 0" against
# the real endpoint. Four leaves a slot free so a `doctor` or a `search` running
# alongside does not push `process` over the edge.
DEFAULT_STT_CONCURRENCY = 4


class ConfigError(RuntimeError):
    """Raised when a command needs configuration the user has not provided."""


@dataclass(frozen=True)
class Config:
    api_key: str | None
    base_url: str
    hf_token: str | None
    home: Path
    stt_model: str
    notes_model: str
    embed_model: str
    rerank_model: str
    stt_concurrency: int

    def require_api_key(self) -> str:
        if not self.api_key:
            raise ConfigError(
                "HELMCODE_API_KEY is not set. Copy .env.example to .env and fill it in. "
                "(`hcw record` works without it; `process` and `search` do not.)"
            )
        return self.api_key

    @property
    def db_path(self) -> Path:
        return self.home / "index.sqlite3"


def _find_dotenv() -> Path | None:
    """Look for `.env` in the working directory, its parents, then HCW_HOME."""
    cwd = Path.cwd()
    for directory in (cwd, *cwd.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    home_env = _home_from_env() / ".env"
    return home_env if home_env.is_file() else None


def _home_from_env() -> Path:
    raw = os.environ.get("HCW_HOME")
    return Path(raw).expanduser() if raw else Path.home() / "helmcode-whisper"


@lru_cache(maxsize=1)
def load_config() -> Config:
    dotenv_path = _find_dotenv()
    if dotenv_path:
        # Real environment variables win over the file, so `HCW_HOME=... hcw ...`
        # behaves the way anyone would expect.
        load_dotenv(dotenv_path, override=False)

    base_url = os.environ.get("HELMCODE_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    return Config(
        api_key=os.environ.get("HELMCODE_API_KEY") or None,
        base_url=base_url,
        hf_token=os.environ.get("HF_TOKEN") or None,
        home=_home_from_env(),
        stt_model=os.environ.get("HCW_STT_MODEL", DEFAULT_STT_MODEL),
        notes_model=os.environ.get("HCW_NOTES_MODEL", DEFAULT_NOTES_MODEL),
        embed_model=os.environ.get("HCW_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        rerank_model=os.environ.get("HCW_RERANK_MODEL", DEFAULT_RERANK_MODEL),
        stt_concurrency=_positive_int("HCW_STT_CONCURRENCY", DEFAULT_STT_CONCURRENCY),
    )


def _positive_int(name: str, default: int) -> int:
    """An environment override that refuses to be zero, negative or nonsense."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a whole number, not {raw!r}.") from exc
    if value < 1:
        raise ConfigError(f"{name} must be at least 1, not {value}.")
    return value
