"""The shape of a transcript, shared by every step after transcription."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# The label given to the person who did the recording. The microphone track is
# them by definition, which is the half of diarization that costs nothing.
ME = "Me"

# Everyone on the remote track before pyannote has had a look. When diarization
# runs it replaces this with SPEAKER_00, SPEAKER_01 and so on; when it does not,
# this is what the transcript says, and it should read like a word rather than a
# placeholder nobody filled in.
OTHERS = "Others"


@dataclass
class Word:
    """One word with its own timestamps, used to cut segments precisely."""

    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    track: str  # "mic" or "system"
    speaker: str = ME
    # Set when the segment was dropped as an echo of the other track; kept in
    # the file so the decision is auditable rather than invisible.
    dropped: str | None = None
    confidence: float | None = None
    # Working data for the diarization split. Deliberately not serialized: it
    # would triple the size of transcript.json to record something no later step
    # reads.
    words: list[Word] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("words", None)
        return {key: value for key, value in data.items() if value is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Segment:
        return cls(
            start=float(data["start"]),
            end=float(data["end"]),
            text=str(data["text"]),
            track=str(data.get("track", "system")),
            speaker=str(data.get("speaker", ME)),
            dropped=data.get("dropped"),
            confidence=data.get("confidence"),
        )


@dataclass
class Transcript:
    segments: list[Segment] = field(default_factory=list)
    language: str | None = None
    speakers: list[str] = field(default_factory=list)
    diarized: bool = False

    @property
    def kept(self) -> list[Segment]:
        return [segment for segment in self.segments if not segment.dropped]

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "speakers": self.speakers,
            "diarized": self.diarized,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transcript:
        return cls(
            segments=[Segment.from_dict(item) for item in data.get("segments", [])],
            language=data.get("language"),
            speakers=list(data.get("speakers", [])),
            diarized=bool(data.get("diarized", False)),
        )

    def as_text(self) -> str:
        """The transcript as the notes model sees it: one line per turn."""
        lines = []
        for segment in self.kept:
            stamp = f"[{int(segment.start) // 60:02d}:{int(segment.start) % 60:02d}]"
            lines.append(f"{stamp} {segment.speaker}: {segment.text.strip()}")
        return "\n".join(lines)
