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
    )
