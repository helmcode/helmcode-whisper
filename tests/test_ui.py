"""The terminal output, and the encoding that used to be able to kill a run."""

from __future__ import annotations

import io

import pytest

from helmcode_whisper.ui import theme

# Everything the interface draws with that cp1252 cannot represent: the hairline
# rules, the recording meter, the checkbox beside each action item, and the
# separator in the summary line.
NON_LATIN1 = "─ □ · ┃"


def cp1252_stream() -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252")


def test_a_cp1252_stream_cannot_write_the_interface() -> None:
    """The premise. If this ever stops raising, the fix below is unnecessary."""
    with pytest.raises(UnicodeEncodeError):
        stream = cp1252_stream()
        stream.write(NON_LATIN1)
        stream.flush()


def test_the_console_makes_the_stream_speak_utf8_first() -> None:
    """`process` used to die here.

    On a redirected or legacy Windows console, stdout arrives as cp1252 and the
    summary is the first thing to print a character it cannot encode — after
    the transcript is written, the notes are written and the requests are paid
    for. The run ended in a traceback with all the work already done.
    """
    stream = cp1252_stream()

    theme._speak_utf8(stream)
    stream.write(NON_LATIN1)
    stream.flush()

    assert stream.encoding.lower().replace("-", "") == "utf8"


def test_a_stream_that_cannot_be_reconfigured_is_left_alone() -> None:
    class Fixed:
        def write(self, text: str) -> int:
            return len(text)

    theme._speak_utf8(Fixed())  # must not raise
