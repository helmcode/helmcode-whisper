"""The Helmcode design system, ported to the terminal.

Source of truth is the site's `src/styles/tokens.css`. That file states its
colours in oklch and layers text on the background with alpha; a terminal wants
opaque sRGB, so the values below are the resolved equivalents:

    --color-accent       oklch(51.1% .262 276.966)      -> #4f39f6
    --color-accent-text  oklch(70%   .18  276.966)      -> #848fff
    --color-text-secondary  rgba(255,255,255,.55) over #0a0a0a -> #919191

Keep them in sync by hand when the site's tokens move. Nothing else in this
package is allowed to name a colour.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.theme import Theme

# ─── COLOR: BASE ──────────────────────────────────────────────────
BG = "#0a0a0a"
SURFACE = "#111111"
SURFACE_RAISED = "#161616"
BORDER = "#202020"

# ─── COLOR: TEXTO ─────────────────────────────────────────────────
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#919191"
TEXT_TERTIARY = "#606060"
TEXT_MONO = "#b6b6b6"

# ─── COLOR: ACENTO ────────────────────────────────────────────────
ACCENT = "#4f39f6"
ACCENT_TEXT = "#848fff"

# ─── COLOR: ESTADO ────────────────────────────────────────────────
ERROR = "#dc5050"

# ─── COLOR: TERMINAL ──────────────────────────────────────────────
TERMINAL_RED = "#ff5f56"
TERMINAL_YELLOW = "#ffbd2e"
TERMINAL_GREEN = "#27c93f"

HELMCODE_THEME = Theme(
    {
        # Text roles, mirroring the site's three levels of emphasis.
        "primary": TEXT_PRIMARY,
        "secondary": TEXT_SECONDARY,
        "tertiary": TEXT_TERTIARY,
        "mono": TEXT_MONO,
        "accent": ACCENT_TEXT,
        "accent.solid": ACCENT,
        # The site writes labels in uppercase mono with wide tracking; the
        # tracking is faked at the callsite with spaces where it matters.
        "label": f"bold {TEXT_TERTIARY}",
        "eyebrow": f"bold {ACCENT_TEXT}",
        # Status. The site's traffic-light trio is the only saturated colour it
        # allows outside the accent, so status reuses it rather than inventing.
        "ok": TERMINAL_GREEN,
        "warn": TERMINAL_YELLOW,
        "err": TERMINAL_RED,
        "error": ERROR,
        # Structure.
        "hairline": BORDER,
        "rule.line": BORDER,
        "path": f"underline {TEXT_MONO}",
        # Transcript.
        "speaker.me": f"bold {ACCENT_TEXT}",
        "speaker.other": f"bold {TEXT_PRIMARY}",
        "timestamp": TEXT_TERTIARY,
        # Rich built-ins that would otherwise land on default ANSI colours.
        "progress.percentage": TEXT_SECONDARY,
        "progress.remaining": TEXT_TERTIARY,
        "progress.elapsed": TEXT_TERTIARY,
        "bar.complete": ACCENT,
        "bar.finished": TERMINAL_GREEN,
        "bar.pulse": ACCENT,
        "repr.number": TEXT_MONO,
        "repr.str": TEXT_SECONDARY,
    }
)

_console: Console | None = None
_err_console: Console | None = None
_human_output_on_stderr = False


def send_human_output_to_stderr() -> None:
    """Move the terminal interface off stdout, before anything is printed.

    For `--progress-json`, where stdout belongs to the machine-readable event
    stream. Two audiences on one pipe means a reader has to tell hairlines from
    JSON, so they get a stream each.

    Must be called before the first `console()`, which is why the CLI does it
    while parsing arguments rather than on the way into the pipeline.
    """
    global _human_output_on_stderr, _console
    _human_output_on_stderr = True
    _console = None


def _speak_utf8(stream: object) -> None:
    """Stop a character being able to kill the process.

    Windows still hands Python a cp1252 stdout in plenty of situations —
    a redirected pipe, a legacy console — and every part of this interface is
    drawn with characters cp1252 does not have: the hairline rules, the level
    meters, the ☐ beside each action item. Encoding one of those raises
    UnicodeEncodeError from inside the print, which on `process` means the
    transcript is written, the notes are written, the requests are paid for,
    and the run still ends in a traceback.

    Worst case the terminal cannot render what it is sent and shows the wrong
    glyph. That is a cosmetic problem. Losing the run is not.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:  # not a text stream, e.g. already wrapped in tests
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        pass


def console() -> Console:
    """The one stdout console. Themed, and reused so Rich shares its state."""
    global _console
    if _console is None:
        target = sys.stderr if _human_output_on_stderr else sys.stdout
        _speak_utf8(target)
        _console = Console(
            theme=HELMCODE_THEME, highlight=False, stderr=_human_output_on_stderr
        )
    return _console


def err_console() -> Console:
    """Errors and warnings go to stderr so `hcw search ... > file` stays clean."""
    global _err_console
    if _err_console is None:
        _speak_utf8(sys.stderr)
        _err_console = Console(theme=HELMCODE_THEME, highlight=False, stderr=True)
    return _err_console
