"""Who is SPEAKER_01?

Diarization separates voices and has no idea what they are called. The names are
usually already in the transcript: people introduce themselves at the start of a
meeting, and they address each other by name all the way through. So this reads
the conversation back and asks which label belongs to whom.

It *proposes*. It does not decide. Every answer carries the words that suggested
it and how sure the model is, and the caller is expected to put a person in
front of the result before anything gets renamed — attributing a decision to the
wrong colleague in a set of notes is worse than leaving SPEAKER_01 in.

When the meeting's invitees are known, they are passed in as the candidate list.
That turns an open question into a closed one: not "what is this person called"
but "which of these five people is this", which is a different and much easier
question.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..api import ApiError, HelmcodeClient
from .model import ME, Transcript

# Enough transcript for names to appear without paying for an hour of it twice.
# Introductions cluster at the start, and direct address is spread throughout,
# so the head of the conversation is where the evidence is densest.
MAX_TRANSCRIPT_CHARS = 60_000

NAMES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["speakers"],
    "properties": {
        "speakers": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["label", "name", "confidence", "evidence"],
                "properties": {
                    "label": {"type": "string", "description": "The label as given, verbatim."},
                    "name": {
                        "type": "string",
                        "description": "The person's name, or an empty string if it is not "
                        "stated in the transcript. Never guess.",
                    },
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "evidence": {
                        "type": "string",
                        "description": "The short quote from the transcript that shows it, "
                        "or an empty string when there is no name.",
                    },
                },
            },
        }
    },
}

_PROMPT = """\
Below is a meeting transcript. Each line is one turn, prefixed with a timestamp
and the label of whoever was speaking.

Work out which real person each label refers to.

Rules, in order of importance:
- A name must be stated in the transcript. Someone introducing themselves
  ("soy Ana", "Ana al habla") is the strongest evidence. Being addressed
  directly ("gracias, Ana", "¿Ana, lo tienes?") is the next strongest — the
  name in that case belongs to whoever speaks *around* it, not to the speaker
  of the line containing it, so read the turns either side before deciding.
- If a label's name is not stated, return an empty name with confidence "low".
  An empty answer is correct and useful. A guess is neither.
- Do not infer names from roles, topics, gender, or how likely a name is.
- `{me}` is the person who made the recording. Name them only if they are
  addressed or introduce themselves like anyone else.
- Return exactly one entry per label listed, using the label verbatim.

Labels in this transcript: {labels}
{candidates}
Transcript:
{transcript}
"""

_CANDIDATES = """\
These people were invited to the meeting. Every name you return should be one
of them, matched to the label that best fits the transcript. If a label matches
none of them, return an empty name rather than inventing one:
{names}
"""


@dataclass(frozen=True)
class Proposal:
    label: str
    name: str
    confidence: str
    evidence: str

    @property
    def usable(self) -> bool:
        return bool(self.name.strip())


def labels_in(transcript: Transcript) -> list[str]:
    """Speaker labels present, in the order they first speak."""
    seen: dict[str, None] = {}
    for segment in transcript.kept:
        seen.setdefault(segment.speaker, None)
    return list(seen)


def propose_names(
    client: HelmcodeClient,
    transcript: Transcript,
    *,
    model: str,
    candidates: list[str] | None = None,
) -> list[Proposal]:
    """Ask who each label is. Returns one proposal per label, names may be empty."""
    labels = labels_in(transcript)
    if not labels:
        return []

    text = transcript.as_text()
    if len(text) > MAX_TRANSCRIPT_CHARS:
        text = text[:MAX_TRANSCRIPT_CHARS]

    prompt = _PROMPT.format(
        me=ME,
        labels=", ".join(labels),
        candidates=_CANDIDATES.format(names="\n".join(f"- {name}" for name in candidates))
        if candidates
        else "",
        transcript=text,
    )

    payload = client.chat(
        [{"role": "user", "content": prompt}],
        model=model,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "speaker_names", "strict": True, "schema": NAMES_SCHEMA},
        },
        temperature=0.0,
    )
    try:
        data = json.loads(payload["choices"][0]["message"]["content"])
    except (KeyError, IndexError, ValueError) as exc:
        raise ApiError(f"the model returned nothing usable for speaker names: {exc}") from exc

    return _coerce(data, labels, candidates)


def _coerce(
    data: dict[str, Any], labels: list[str], candidates: list[str] | None
) -> list[Proposal]:
    """Keep only answers about labels we asked about, and names we allowed."""
    allowed = {name.strip().casefold(): name.strip() for name in (candidates or [])}
    by_label: dict[str, Proposal] = {}

    for item in data.get("speakers") or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if label not in labels or label in by_label:
            continue

        name = str(item.get("name") or "").strip()
        # A candidate list is a constraint, not a hint. The model was told to
        # pick from it; anything else is a hallucination wearing a real name.
        if name and allowed:
            name = allowed.get(name.casefold(), "")

        confidence = str(item.get("confidence") or "low")
        by_label[label] = Proposal(
            label=label,
            name=name,
            confidence=confidence if confidence in {"high", "medium", "low"} else "low",
            evidence=str(item.get("evidence") or "").strip(),
        )

    # A label the model skipped is an unnamed label, not a missing one.
    return [
        by_label.get(label, Proposal(label, "", "low", "")) for label in labels
    ]
