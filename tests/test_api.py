"""Retries, rate limits, and the multipart body that has to survive them."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from helmcode_whisper import api
from helmcode_whisper.api import ApiError, HelmcodeClient
from helmcode_whisper.config import Config


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        api_key="test-key",
        base_url="https://api.example.invalid/v1",
        hf_token=None,
        home=tmp_path,
        stt_model="whisper",
        notes_model="notes",
        embed_model="embed",
        rerank_model="rerank",
        stt_concurrency=4,
    )


@pytest.fixture
def no_sleep(monkeypatch):
    """Retries back off for seconds; the test does not need to live them."""
    slept: list[float] = []
    monkeypatch.setattr(api.time, "sleep", slept.append)
    return slept


def client_with(config: Config, handler) -> HelmcodeClient:
    client = HelmcodeClient(config)
    client._client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=config.base_url,
        headers={"Authorization": "Bearer test-key"},
    )
    return client


# ── the multipart body across a retry ────────────────────────────


def test_a_retried_upload_sends_the_whole_file_again(config, tmp_path, no_sleep) -> None:
    """The failure this would cause is silent.

    `transcribe` opens the chunk once and the retry loop reuses that handle. If
    httpx did not rewind it, attempt two would post an empty file part and the
    endpoint would answer with an empty transcription — a hole in the meeting
    that nothing downstream could detect.
    """
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sizes.append(len(request.content))
        if len(sizes) < 3:
            return httpx.Response(503, json={"error": {"message": "later"}})
        return httpx.Response(200, json={"text": "hola", "segments": []})

    chunk = tmp_path / "chunk.ogg"
    chunk.write_bytes(b"OggS" + b"\x00" * 4096)

    with client_with(config, handler) as client:
        assert client.transcribe(chunk)["text"] == "hola"

    assert len(sizes) == 3
    assert len(set(sizes)) == 1, f"body size changed between attempts: {sizes}"
    assert sizes[0] > 4096


# ── rate limiting ────────────────────────────────────────────────


PARALLEL_LIMIT = {
    "error": {
        "message": (
            "Rate limit exceeded for api_key: abc. "
            "Limit type: max_parallel_requests. Current limit: 5, Remaining: 0."
        )
    }
}


def test_the_parallel_limit_names_the_setting_that_fixes_it(config, no_sleep) -> None:
    """Measured against the real API: it allows 5 in flight and 429s the sixth."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json=PARALLEL_LIMIT)

    with client_with(config, handler) as client:  # noqa: SIM117
        with pytest.raises(ApiError, match="HCW_STT_CONCURRENCY"):
            client.chat([{"role": "user", "content": "hi"}])


def test_an_unrelated_429_is_not_given_the_concurrency_advice(config, no_sleep) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "monthly quota exhausted"}})

    with client_with(config, handler) as client:  # noqa: SIM117
        with pytest.raises(ApiError) as caught:
            client.chat([{"role": "user", "content": "hi"}])

    assert "HCW_STT_CONCURRENCY" not in str(caught.value)


def test_retry_after_overrides_the_backoff(config, no_sleep) -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"Retry-After": "7"}, json=PARALLEL_LIMIT)
        return httpx.Response(200, json={"choices": []})

    with client_with(config, handler) as client:
        client.chat([{"role": "user", "content": "hi"}])

    assert no_sleep == [7.0]


def test_a_nonsense_retry_after_falls_back_to_the_backoff(config, no_sleep) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503, headers={"Retry-After": "Wed, 12 Aug 2026 09:05:09 GMT"}, json={}
        )

    with client_with(config, handler) as client:  # noqa: SIM117
        with pytest.raises(ApiError):
            client.chat([{"role": "user", "content": "hi"}])

    assert no_sleep == [1, 2, 4]


def test_retry_after_is_clamped_so_a_bad_header_cannot_stall_a_run(config, no_sleep) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, headers={"Retry-After": "86400"}, json={})

    with client_with(config, handler) as client:  # noqa: SIM117
        with pytest.raises(ApiError):
            client.chat([{"role": "user", "content": "hi"}])

    assert no_sleep == [60.0, 60.0, 60.0]


def test_a_client_error_is_not_retried(config, no_sleep) -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    with client_with(config, handler) as client:  # noqa: SIM117
        with pytest.raises(ApiError, match="bad key"):
            client.chat([{"role": "user", "content": "hi"}])

    assert len(attempts) == 1
    assert no_sleep == []
