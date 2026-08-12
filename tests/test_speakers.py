"""Proposing speaker names, including the answers that must not be trusted."""

from __future__ import annotations

import json

import pytest

from helmcode_whisper.api import ApiError
from helmcode_whisper.pipeline import speakers
from helmcode_whisper.pipeline.model import ME, Segment, Transcript


class FakeClient:
    def __init__(self, answer, *, raw: str | None = None) -> None:
        self.answer = answer
        self.raw = raw
        self.prompts: list[str] = []

    def chat(self, messages, *, model=None, response_format=None, temperature=None, **_):
        self.prompts.append(messages[0]["content"])
        content = self.raw if self.raw is not None else json.dumps({"speakers": self.answer})
        return {"choices": [{"message": {"content": content}}]}


def transcript() -> Transcript:
    return Transcript(
        segments=[
            Segment(0.0, 3.0, "Buenos días, soy Ana.", "system", "SPEAKER_00"),
            Segment(3.0, 6.0, "Hola Ana, ¿empezamos?", "mic", ME),
            Segment(6.0, 9.0, "Yo lo veo bien.", "system", "SPEAKER_01"),
        ],
        speakers=["SPEAKER_00", ME, "SPEAKER_01"],
    )


def test_labels_come_back_in_speaking_order() -> None:
    assert speakers.labels_in(transcript()) == ["SPEAKER_00", ME, "SPEAKER_01"]


def test_a_named_speaker_is_proposed_with_its_evidence() -> None:
    client = FakeClient([
        {"label": "SPEAKER_00", "name": "Ana", "confidence": "high", "evidence": "soy Ana"},
        {"label": ME, "name": "", "confidence": "low", "evidence": ""},
        {"label": "SPEAKER_01", "name": "", "confidence": "low", "evidence": ""},
    ])

    proposals = speakers.propose_names(client, transcript(), model="m")

    assert [p.label for p in proposals] == ["SPEAKER_00", ME, "SPEAKER_01"]
    assert proposals[0].name == "Ana"
    assert proposals[0].usable is True
    assert proposals[2].usable is False


def test_every_label_gets_an_entry_even_when_the_model_skips_it() -> None:
    """A label the model ignored is unnamed, not missing."""
    client = FakeClient([
        {"label": "SPEAKER_00", "name": "Ana", "confidence": "high", "evidence": "soy Ana"},
    ])

    proposals = speakers.propose_names(client, transcript(), model="m")

    assert len(proposals) == 3
    assert proposals[1].name == ""
    assert proposals[1].confidence == "low"


def test_a_label_nobody_asked_about_is_discarded() -> None:
    client = FakeClient([
        {"label": "SPEAKER_99", "name": "Nadie", "confidence": "high", "evidence": "x"},
        {"label": "SPEAKER_00", "name": "Ana", "confidence": "high", "evidence": "soy Ana"},
    ])

    proposals = speakers.propose_names(client, transcript(), model="m")

    assert "SPEAKER_99" not in [p.label for p in proposals]
    assert len(proposals) == 3


def test_a_name_outside_the_invitee_list_is_thrown_away() -> None:
    """The candidate list is a constraint, not a hint.

    A model handed five invitees and answering with a sixth name has invented
    somebody, and the invention arrives looking exactly like a real answer.
    """
    client = FakeClient([
        {"label": "SPEAKER_00", "name": "Ana", "confidence": "high", "evidence": "soy Ana"},
        {"label": "SPEAKER_01", "name": "Roberto", "confidence": "high", "evidence": "inventado"},
        {"label": ME, "name": "", "confidence": "low", "evidence": ""},
    ])

    proposals = speakers.propose_names(
        client, transcript(), model="m", candidates=["Ana Pérez", "Ana", "Luis"]
    )

    named = {p.label: p.name for p in proposals}
    assert named["SPEAKER_00"] == "Ana"
    assert named["SPEAKER_01"] == ""  # Roberto was not invited


def test_the_invitee_list_reaches_the_prompt() -> None:
    client = FakeClient([])

    speakers.propose_names(client, transcript(), model="m", candidates=["Ana", "Luis"])

    assert "- Ana" in client.prompts[0]
    assert "- Luis" in client.prompts[0]


def test_an_unparseable_answer_is_an_api_error_not_a_crash() -> None:
    client = FakeClient([], raw="lo siento, no puedo")

    with pytest.raises(ApiError, match="speaker names"):
        speakers.propose_names(client, transcript(), model="m")


def test_a_bad_confidence_value_falls_back_to_low() -> None:
    client = FakeClient([
        {"label": "SPEAKER_00", "name": "Ana", "confidence": "absolutísima", "evidence": "x"},
    ])

    proposals = speakers.propose_names(client, transcript(), model="m")

    assert proposals[0].confidence == "low"


def test_an_empty_transcript_asks_nothing() -> None:
    client = FakeClient([])

    assert speakers.propose_names(client, Transcript(segments=[]), model="m") == []
    assert client.prompts == []


def test_a_long_transcript_is_truncated_before_it_is_sent() -> None:
    long = Transcript(
        segments=[
            Segment(float(i), float(i) + 1, "x" * 500, "system", "SPEAKER_00")
            for i in range(400)
        ],
        speakers=["SPEAKER_00"],
    )
    client = FakeClient([])

    speakers.propose_names(client, long, model="m")

    assert len(client.prompts[0]) < speakers.MAX_TRANSCRIPT_CHARS + 4000
