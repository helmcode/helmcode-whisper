"""Template rendering and the tolerant parsing behind the structured-output fallback."""

from __future__ import annotations

import json

from helmcode_whisper.pipeline import notes


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
