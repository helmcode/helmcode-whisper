"""Diarization is optional, and 'optional' has to survive ugly failures.

The acceptance criterion is that the pipeline runs to completion without
pyannote. Not installed is the easy case. The one that actually happened is
pyannote being installed and refusing to load: on Windows with Smart App Control
enabled, torch's unsigned `shm.dll` is blocked by code integrity policy and the
import raises OSError, which an `except ImportError` sails straight past.
"""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from helmcode_whisper.pipeline import diarize
from helmcode_whisper.pipeline.model import OTHERS, Segment, Word


def words(text: str, start: float, step: float = 0.5) -> list[Word]:
    """One word per `step` seconds, the way the API returns them."""
    return [
        Word(start + index * step, start + (index + 1) * step, f" {token}")
        for index, token in enumerate(text.split())
    ]


@pytest.fixture
def pyannote_raises(monkeypatch):
    """Make importing pyannote fail with a given exception."""

    def install(exception: Exception) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args, **kwargs):
            if name.startswith("pyannote"):
                raise exception
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

    return install


def test_missing_pyannote_is_reported_not_raised(pyannote_raises) -> None:
    pyannote_raises(ModuleNotFoundError("No module named 'pyannote'", name="pyannote"))

    usable, reason = diarize.availability()

    assert usable is False
    assert "not installed" in reason


def test_a_broken_install_is_not_reported_as_a_missing_one(pyannote_raises) -> None:
    """The failure this machine actually produces.

    torch upgraded in place leaves pyannote importing a symbol its installed
    torch no longer exports. That is an ImportError but not a missing module,
    and answering it with "pyannote is not installed" sends the user to
    reinstall something they already have.
    """
    pyannote_raises(
        ImportError("cannot import name 'NP_SUPPORTED_MODULES' from 'torch._dynamo.utils'")
    )

    usable, reason = diarize.availability()

    assert usable is False
    assert "not installed" not in reason
    assert "version mismatch" in reason


def test_a_missing_dependency_names_the_dependency(pyannote_raises) -> None:
    pyannote_raises(ModuleNotFoundError("No module named 'torch'", name="torch"))

    usable, reason = diarize.availability()

    assert usable is False
    assert "torch is missing" in reason


def test_a_blocked_native_library_is_also_survivable(pyannote_raises) -> None:
    pyannote_raises(OSError("[WinError 4551] blocked by application control"))

    usable, reason = diarize.availability()

    assert usable is False
    assert "will not load" in reason
    assert "WinError 4551" in reason


def test_diarize_raises_its_own_error_so_process_can_carry_on(pyannote_raises) -> None:
    pyannote_raises(OSError("[WinError 4551] blocked by application control"))

    with pytest.raises(diarize.DiarizationUnavailable):
        diarize.diarize(Path("nonexistent.wav"), "hf_token")


def test_a_missing_token_is_caught_before_any_download(monkeypatch) -> None:
    # Pin availability rather than relying on this machine having a working
    # pyannote: the assertion is about the token check, and letting the
    # environment decide whether the test runs at all makes it useless in CI,
    # where pyannote is deliberately not installed.
    monkeypatch.setattr(diarize, "availability", lambda: (True, None))

    with pytest.raises(diarize.DiarizationUnavailable, match="HF_TOKEN"):
        diarize.diarize(Path("nonexistent.wav"), None)


def test_segments_take_the_speaker_they_overlap_most() -> None:
    turns = [
        diarize.Turn(0.0, 10.0, "SPEAKER_00"),
        diarize.Turn(10.0, 20.0, "SPEAKER_01"),
    ]
    segments = [
        Segment(1.0, 3.0, "primero", "system", OTHERS),
        Segment(9.0, 15.0, "a caballo", "system", OTHERS),  # 1s vs 5s
        Segment(12.0, 14.0, "segundo", "system", OTHERS),
    ]

    assert diarize.label_segments(segments, turns) == [
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_01",
    ]


def test_a_segment_holding_two_people_is_cut_between_them() -> None:
    """Whisper segments follow punctuation, not speakers. 18 turns once
    collapsed into 3 labels because of exactly this."""
    spoken = words("uno dos tres cuatro cinco seis siete ocho", 0.0)
    segment = Segment(0.0, 4.0, "uno dos tres cuatro cinco seis siete ocho", "system", OTHERS)
    segment.words = spoken
    turns = [
        diarize.Turn(0.0, 2.0, "SPEAKER_00"),
        diarize.Turn(2.0, 4.0, "SPEAKER_01"),
    ]

    result = diarize.split_by_speaker([segment], turns)

    assert len(result) == 2
    assert result[0].speaker == "SPEAKER_00"
    assert result[0].text == "uno dos tres cuatro"
    assert result[1].speaker == "SPEAKER_01"
    assert result[1].text == "cinco seis siete ocho"
    assert result[1].start == pytest.approx(2.0)


def test_a_single_stray_word_does_not_shatter_a_sentence() -> None:
    spoken = words("uno dos tres cuatro cinco seis siete ocho", 0.0)
    segment = Segment(0.0, 4.0, "uno dos tres cuatro cinco seis siete ocho", "system", OTHERS)
    segment.words = spoken
    # One word's worth of someone else in the middle: an overlap, not a turn.
    turns = [
        diarize.Turn(0.0, 2.0, "SPEAKER_00"),
        diarize.Turn(2.0, 2.5, "SPEAKER_01"),
        diarize.Turn(2.5, 4.0, "SPEAKER_00"),
    ]

    result = diarize.split_by_speaker([segment], turns)

    assert len(result) == 1
    assert result[0].text == "uno dos tres cuatro cinco seis siete ocho"


def test_segments_without_word_timestamps_pass_through() -> None:
    segment = Segment(0.0, 4.0, "sin palabras", "system", OTHERS)
    turns = [diarize.Turn(0.0, 2.0, "SPEAKER_00"), diarize.Turn(2.0, 4.0, "SPEAKER_01")]

    assert diarize.split_by_speaker([segment], turns) == [segment]


def test_a_segment_overlapping_nothing_keeps_its_label() -> None:
    """Silence between turns is a bad place to guess."""
    turns = [diarize.Turn(0.0, 5.0, "SPEAKER_00")]
    segments = [Segment(100.0, 101.0, "suelto", "system", OTHERS)]

    assert diarize.label_segments(segments, turns) == [OTHERS]
