"""The only code in this package that opens a network connection.

Every request goes through a single `httpx.Client` bound to `HELMCODE_BASE_URL`
with relative paths, so there is no way for a caller to reach another host by
accident. `tests/test_no_egress.py` enforces that no other URL is hardcoded
anywhere in the package.

The one other host involved in the project is huggingface.co, which pyannote
contacts once to download diarization weights — no meeting content is sent
there, and it can be avoided entirely by pre-downloading the models.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx

from .config import Config

# The API times out audio requests around the two-minute mark; give a chunked
# upload generous room before we call it a failure ourselves.
_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=10.0)

_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 524}
_MAX_ATTEMPTS = 4


class ApiError(RuntimeError):
    """A request failed in a way retrying will not fix."""


class HelmcodeClient:
    """A thin, OpenAI-compatible client for the endpoints this tool uses."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = httpx.Client(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.require_api_key()}",
                "User-Agent": "helmcode-whisper/0.1",
            },
            timeout=_TIMEOUT,
        )

    def __enter__(self) -> HelmcodeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # ── plumbing ─────────────────────────────────────────────────

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = self._client.request(method, path, **kwargs)
            except httpx.TransportError as exc:  # network hiccup, DNS, timeout
                last_error = exc
            else:
                if response.status_code < 400:
                    return response.json()
                if response.status_code not in _RETRY_STATUS:
                    raise ApiError(_describe(response))
                last_error = ApiError(_describe(response))

            if attempt < _MAX_ATTEMPTS - 1:
                # Plain exponential backoff: 1s, 2s, 4s. The failure mode we
                # actually hit is a burst of concurrent chunk uploads, and the
                # concurrency cap already keeps that burst small.
                time.sleep(2**attempt)

        raise ApiError(f"{path} failed after {_MAX_ATTEMPTS} attempts: {last_error}")

    # ── endpoints ────────────────────────────────────────────────

    def models(self) -> list[str]:
        payload = self._request("GET", "/models")
        return [entry["id"] for entry in payload.get("data", [])]

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """POST /audio/transcriptions with per-segment timestamps.

        The endpoint caps requests at 25 MB and roughly two minutes of audio, so
        callers must hand it pre-split chunks; see `pipeline.audio`.
        """
        # Both granularities in one request: `segments` gives sentence-shaped
        # text and `words` gives the boundaries needed to cut a segment where
        # the speaker changes. Asking for both is one request, not two.
        data: dict[str, Any] = {
            "model": model or self._config.stt_model,
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["segment", "word"],
        }
        if language:
            data["language"] = language

        with audio_path.open("rb") as handle:
            files = {"file": (audio_path.name, handle, "application/ogg")}
            return self._request("POST", "/audio/transcriptions", data=data, files=files)

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self._config.notes_model,
            "messages": messages,
        }
        if response_format:
            payload["response_format"] = response_format
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        return self._request("POST", "/chat/completions", json=payload)

    def embed(self, inputs: list[str], *, model: str | None = None) -> list[list[float]]:
        payload = self._request(
            "POST",
            "/embeddings",
            json={"model": model or self._config.embed_model, "input": inputs},
        )
        # The API returns entries with their original index, but does not promise
        # order; sort so a caller can zip the result against its input.
        entries = sorted(payload["data"], key=lambda item: item["index"])
        return [entry["embedding"] for entry in entries]

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": model or self._config.rerank_model,
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n
        result = self._request("POST", "/rerank", json=payload)
        return result.get("results", result.get("data", []))


def _describe(response: httpx.Response) -> str:
    try:
        body = response.json()
        message = body.get("error", {}).get("message") or body.get("message") or str(body)
    except ValueError:
        message = response.text[:400]
    return f"HTTP {response.status_code}: {message}"
