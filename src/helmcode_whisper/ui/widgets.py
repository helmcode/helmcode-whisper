"""Reusable pieces of terminal chrome.

The site draws structure with 0.5px hairlines and flat rectangles — no rounded
corners, no heavy boxes. The terminal equivalent is a thin horizontal rule and
tables with no borders, which is what everything here builds on.
"""

from __future__ import annotations

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from .theme import console


def eyebrow(text: str) -> Text:
    """A section label: uppercase mono with wide tracking, like `.eyebrow`."""
    return Text(" ".join(text.upper()), style="eyebrow")


def hairline(label: str | None = None) -> None:
    """A `.hairline` divider, optionally with a label sitting on it."""
    if label:
        console().rule(eyebrow(label), style="hairline", align="left")
    else:
        console().rule(style="hairline")


def kv_table(rows: list[tuple[str, str]], *, key_style: str = "tertiary") -> Table:
    """Label/value pairs, aligned, no borders. The site's field-label pattern."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style=key_style, justify="left", no_wrap=True)
    table.add_column(style="secondary")
    for key, value in rows:
        table.add_row(key, value)
    return table


def status_line(state: str, message: str, detail: str = "") -> None:
    """One line of `doctor`-style output: a marker, a message, a dim detail."""
    markers = {
        "ok": ("ok", "+"),
        "warn": ("warn", "!"),
        "err": ("err", "x"),
        "skip": ("tertiary", "-"),
    }
    style, glyph = markers.get(state, ("secondary", " "))
    line = Text.assemble((f" {glyph} ", style), (message, "secondary"))
    if detail:
        line.append(f"  {detail}", style="tertiary")
    console().print(line)


def progress() -> Progress:
    """A progress bar in the accent colour, for the long steps of `process`."""
    return Progress(
        SpinnerColumn(style="accent"),
        TextColumn("[secondary]{task.description}"),
        BarColumn(
            bar_width=32,
            complete_style="accent.solid",
            finished_style="ok",
            pulse_style="accent.solid",
        ),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console(),
        transient=False,
    )
