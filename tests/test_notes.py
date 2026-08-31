"""Template rendering and the tolerant parsing behind the structured-output fallback."""

from __future__ import annotations

import json
import threading

import pytest

from helmcode_whisper.api import ApiError
from helmcode_whisper.pipeline import notes
from helmcode_whisper.pipeline.model import ME, Segment, Transcript


def test_the_packaged_template_is_found() -> None:
    template = notes.find_template()
    assert template.is_file()
    assert "{{TRANSCRIPT}}" in template.read_text(encoding="utf-8")


def test_rendering_strips_the_editor_comments(tmp_path) -> None:
    template = tmp_path / "notes.md"
    template.write_text(
        "<!-- instructions for the human -->\nMeeting: {{TITLE}}\n\n{{TRANSCRIPT}}\n",
        encoding="utf-8",
    )
    rendered = notes.render_prompt(template, "00:00 Me: hola", {"TITLE": "Q3 review"})

    assert "instructions for the human" not in rendered
    assert "Meeting: Q3 review" in rendered
    assert "00:00 Me: hola" in rendered


def test_json_is_recovered_from_a_fenced_block() -> None:
    payload = {
        "summary": "s",
        "decisions": [],
        "action_items": [],
        "open_questions": [],
        "quotes": [],
    }
    content = f"Sure, here you go:\n```json\n{json.dumps(payload)}\n```\nHope that helps."

    assert notes._parse_json(content) == payload


def test_coerce_accepts_a_loosely_shaped_answer() -> None:
    result = notes._coerce(
        {
            "summary": ["first sentence.", "second sentence."],
            "decisions": "ship on Friday",
            "action_items": ["write the README", {"task": "measure", "owner": "Borja"}],
            "quotes": [{"text": "we ship Friday"}, {"speaker": "nobody"}],
        }
    )

    assert result["summary"] == "first sentence. second sentence."
    assert result["decisions"] == ["ship on Friday"]
    assert result["action_items"][0] == {"task": "write the README", "owner": "", "due": ""}
    assert result["action_items"][1]["owner"] == "Borja"
    # A quote with no text is not a quote.
    assert len(result["quotes"]) == 1
    assert result["open_questions"] == []


ANSWER = json.dumps(
    {
        "summary": "Resumen.",
        "decisions": ["enviar el jueves"],
        "action_items": [],
        "open_questions": [],
        "quotes": [],
    }
)


class ChatClient:
    """A chat endpoint that can reject the strict modes and count concurrency."""

    def __init__(self, *, rejects: tuple[str, ...] = (), rendezvous: int = 0) -> None:
        self.modes: list[str] = []
        self._rejects = rejects
        self._lock = threading.Lock()
        self._barrier = threading.Barrier(rendezvous) if rendezvous else None

    def chat(self, messages, *, model=None, response_format=None, temperature=None, **_):
        mode = (response_format or {}).get("type", "plain")
        with self._lock:
            self.modes.append(mode)
            position = len(self.modes)
        if mode in self._rejects:
            raise ApiError(f"HTTP 400: {mode} not supported")
        # Only the first wave rendezvouses. The joining pass that follows the
        # blocks is a single call by design and would wait for company forever.
        if self._barrier is not None and position <= self._barrier.parties:
            self._barrier.wait(timeout=10)
        return {
            "choices": [{"message": {"content": ANSWER}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }


def long_transcript() -> Transcript:
    """Long enough to take the map-reduce path."""
    line = "x" * 400
    segments = [
        Segment(float(index), float(index) + 1, line, "mic", ME)
        for index in range(notes.MAP_REDUCE_THRESHOLD // 400 + 20)
    ]
    return Transcript(segments=segments, speakers=[ME])


def test_long_transcripts_summarize_their_blocks_concurrently(tmp_path) -> None:
    template = tmp_path / "notes.md"
    template.write_text("{{TITLE}}\n{{TRANSCRIPT}}\n", encoding="utf-8")
    transcript = long_transcript()
    blocks = len(notes._split_transcript(transcript.as_text(), notes._BLOCK_CHARS))
    assert blocks > 1, "the fixture must be long enough to be split"

    # Every block has to be inside the client at once before any may return.
    # Sequential block summaries cannot satisfy that barrier.
    client = ChatClient(rendezvous=min(blocks, notes._BLOCK_CONCURRENCY))

    result, stats = notes.generate_notes(
        client,
        transcript,
        model="deepseek-v4-flash",
        title="Largo",
        date="2026-08-12",
        duration_minutes=90,
        template=template,
    )

    assert result["summary"]
    assert stats["requests"] == blocks + 1  # one per block, plus the joining pass


def test_a_rejected_schema_is_not_retried_for_every_block(tmp_path) -> None:
    """The point of remembering the mode.

    Without the memo each block rediscovers that json_schema is refused, so a
    transcript split into four blocks pays four rejected requests instead of one.
    """
    template = tmp_path / "notes.md"
    template.write_text("{{TITLE}}\n{{TRANSCRIPT}}\n", encoding="utf-8")
    client = ChatClient(rejects=("json_schema",))

    notes.generate_notes(
        client,
        long_transcript(),
        model="deepseek-v4-flash",
        title="Largo",
        date="2026-08-12",
        duration_minutes=90,
        template=template,
    )

    # The concurrent first wave may each hit the rejection before any of them
    # has learned; what must not happen is every later call repeating it.
    assert client.modes.count("json_schema") <= notes._BLOCK_CONCURRENCY
    assert "json_object" in client.modes


def test_an_empty_transcript_is_refused_rather_than_summarized() -> None:
    with pytest.raises(notes.NotesError, match="empty"):
        notes.generate_notes(
            ChatClient(),
            Transcript(segments=[]),
            model="deepseek-v4-flash",
            title="Vacía",
            date="2026-08-12",
            duration_minutes=0,
        )


def test_markdown_leaves_out_empty_sections() -> None:
    rendered = notes.render_markdown(
        {
            "summary": "It was short.",
            "decisions": [],
            "action_items": [{"task": "send the deck", "owner": "Ana", "due": "Friday"}],
            "open_questions": [],
            "quotes": [],
        },
        {"title": "Standup", "started_at": "2026-08-11T10:00:00", "duration_seconds": 600},
    )

    assert "## Decisions" not in rendered
    assert "- [ ] send the deck — Ana · Friday" in rendered
    assert "# Standup" in rendered


def test_a_sentence_can_be_both_a_decision_and_an_open_question() -> None:
    """One dedup set covered both sections, so the second appearance vanished."""
    partials = [
        {"summary": "a", "decisions": ["Ship on Friday"], "open_questions": [], "quotes": []},
        {"summary": "b", "decisions": [], "open_questions": ["Ship on Friday"], "quotes": []},
    ]

    merged = notes._merge_partials(partials)

    assert merged["decisions"] == ["Ship on Friday"]
    assert merged["open_questions"] == ["Ship on Friday"]


def test_a_section_still_dedupes_within_itself() -> None:
    partials = [
        {"summary": "a", "decisions": ["Ship on Friday"], "open_questions": [], "quotes": []},
        {"summary": "b", "decisions": ["ship on friday"], "open_questions": [], "quotes": []},
    ]

    merged = notes._merge_partials(partials)

    assert merged["decisions"] == ["Ship on Friday"]
