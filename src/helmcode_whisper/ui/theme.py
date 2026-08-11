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


def console() -> Console:
    """The one stdout console. Themed, and reused so Rich shares its state."""
    global _console
    if _console is None:
        _console = Console(theme=HELMCODE_THEME, highlight=False)
    return _console


def err_console() -> Console:
    """Errors and warnings go to stderr so `hcw search ... > file` stays clean."""
    global _err_console
    if _err_console is None:
        _err_console = Console(theme=HELMCODE_THEME, highlight=False, stderr=True)
    return _err_console
