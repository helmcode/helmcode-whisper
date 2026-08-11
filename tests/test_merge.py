"""Echo suppression: the bug that ruins a demo if nobody wears headphones."""

from __future__ import annotations

from helmcode_whisper.pipeline import merge
from helmcode_whisper.pipeline.model import ME, Segment


def mic(start: float, end: float, text: str) -> Segment:
    return Segment(start, end, text, "mic", ME)


def system(start: float, end: float, text: str, speaker: str = "SPEAKER_00") -> Segment:
    return Segment(start, end, text, "system", speaker)


def test_the_mic_hearing_the_speakers_is_dropped() -> None:
    remote = system(10.0, 14.0, "El precio final sube a cuarenta y dos euros por usuario")
    echo = mic(10.1, 14.1, "el precio final sube a cuarenta y dos euros por usuario.")

    merged = merge.merge([echo], [remote])

    assert echo.dropped == "echo"
    assert [segment for segment in merged if not segment.dropped] == [remote]


def test_talking_over_someone_is_not_an_echo() -> None:
    """Same time, different words. People interrupt; that has to survive."""
    remote = system(10.0, 14.0, "El precio final sube a cuarenta y dos euros por usuario")
    interruption = mic(11.0, 12.0, "Espera, eso no es lo que acordamos en marzo")

    merge.merge([interruption], [remote])

    assert interruption.dropped is None


def test_saying_the_same_thing_later_is_not_an_echo() -> None:
    """Same words, different time. Repeating yourself is not feedback."""
    remote = system(10.0, 14.0, "El precio final sube a cuarenta y dos euros por usuario")
    later = mic(300.0, 304.0, "El precio final sube a cuarenta y dos euros por usuario")

    merge.merge([later], [remote])

    assert later.dropped is None


def test_short_agreements_are_always_kept() -> None:
    """"ya", "sí", "ok" match everything; length is the guard."""
    remote = system(10.0, 14.0, "Si, claro, lo vemos manana sin problema")
    agreement = mic(10.5, 11.0, "Si, claro")

    merge.merge([agreement], [remote])

    assert agreement.dropped is None


def test_merged_transcript_is_ordered_by_time() -> None:
    segments = merge.merge(
        [mic(5.0, 6.0, "una cosa"), mic(20.0, 21.0, "otra cosa")],
        [system(1.0, 2.0, "primero"), system(10.0, 11.0, "en medio")],
    )
    starts = [segment.start for segment in segments]
    assert starts == sorted(starts)


def test_speaker_list_ignores_dropped_segments() -> None:
    echo = mic(10.0, 14.0, "El precio final sube a cuarenta y dos euros por usuario")
    remote = system(10.0, 14.0, "El precio final sube a cuarenta y dos euros por usuario")
    segments = merge.merge([echo], [remote])

    assert merge.speaker_list(segments) == ["SPEAKER_00"]
