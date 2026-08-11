"""`notes.html` — the meeting, rendered in the Helmcode design system.

Self-contained by design: no CDN, no webfont request, no analytics. Opening this
file must not produce a single network request, or the privacy claim on the tin
would be false for the one artefact people actually share.

The CSS is a subset of the site's `src/styles/tokens.css`, kept in the same
order and with the same names so the two can be diffed by eye.
"""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any

from ..pipeline.model import ME, Transcript

_CSS = """
:root {
  --color-bg: #0a0a0a;
  --color-surface: #111111;
  --color-border: rgba(255,255,255,0.09);
  --color-text-primary: #ffffff;
  --color-text-secondary: rgba(255,255,255,0.55);
  --color-text-tertiary: rgba(255,255,255,0.35);
  --color-accent: oklch(51.1% .262 276.966);
  --color-accent-text: oklch(70% .18 276.966);
  --color-accent-dim: oklch(51.1% .262 276.966 / 0.10);
  --font-sans: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-mono: 'Roboto Mono', ui-monospace, 'Cascadia Mono', 'Courier New', monospace;
  --tracking-mono-sm: 0.07em;
  --leading-body: 1.75;
  --space-2: 8px; --space-3: 12px; --space-4: 16px; --space-6: 24px;
  --space-8: 32px; --space-12: 48px; --space-16: 64px;
  --text-max-width: 720px;
}

*, *::before, *::after { box-sizing: border-box; }

html {
  background: var(--color-bg);
  color: var(--color-text-primary);
  font-family: var(--font-sans);
  font-size: 16px;
  color-scheme: dark;
  -webkit-font-smoothing: antialiased;
}

body {
  margin: 0 auto;
  padding: var(--space-16) var(--space-6) var(--space-16);
  max-width: var(--text-max-width);
  line-height: var(--leading-body);
}

.eyebrow {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: var(--tracking-mono-sm);
  text-transform: uppercase;
  color: var(--color-text-tertiary);
  margin: 0 0 var(--space-4);
}

h1 {
  font-size: clamp(28px, 4vw, 44px);
  font-weight: 500;
  letter-spacing: -0.02em;
  line-height: 1.12;
  margin: 0 0 var(--space-3);
}

h2 {
  font-size: 18px;
  font-weight: 500;
  margin: var(--space-12) 0 var(--space-4);
}

.meta {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--color-text-tertiary);
  margin: 0 0 var(--space-8);
}

.hairline { border: none; border-top: 0.5px solid var(--color-border); margin: 0; }

p, li { color: var(--color-text-secondary); }

ul { padding-left: 1.1em; margin: 0; }
li { margin-bottom: var(--space-2); }

.task { list-style: none; margin-left: -1.1em; }
.task::before { content: '□'; color: var(--color-text-tertiary); margin-right: var(--space-2); }
.task .who {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--color-accent-text);
  margin-left: var(--space-2);
}

blockquote {
  margin: 0 0 var(--space-6);
  padding: var(--space-3) var(--space-4);
  background: var(--color-accent-dim);
  border-left: 3px solid var(--color-accent);
  color: var(--color-text-primary);
}
blockquote .who {
  display: block;
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: var(--tracking-mono-sm);
  text-transform: uppercase;
  color: var(--color-text-tertiary);
  margin-top: var(--space-2);
}

.turn {
  display: grid;
  grid-template-columns: 64px 1fr;
  gap: var(--space-4);
  margin-bottom: var(--space-3);
}
.turn .t {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--color-text-tertiary);
  padding-top: 4px;
}
.turn .who {
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: var(--tracking-mono-sm);
  text-transform: uppercase;
  color: var(--color-text-tertiary);
}
.turn.me .who { color: var(--color-accent-text); }
.turn p { margin: 0; }

footer {
  margin-top: var(--space-16);
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--color-text-tertiary);
}
footer a { color: var(--color-text-secondary); }

@media (max-width: 640px) {
  body { padding: var(--space-8) var(--space-4); }
  .turn { grid-template-columns: 1fr; gap: var(--space-2); }
}
"""


def render_html(notes: dict[str, Any], meta: dict[str, Any], transcript: Transcript) -> str:
    title = html.escape(str(meta.get("title", "Meeting")))
    started = str(meta.get("started_at", ""))
    try:
        when = datetime.fromisoformat(started).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        when = started
    minutes = (meta.get("duration_seconds") or 0) / 60
    speakers = ", ".join(transcript.speakers) or "unknown"

    parts: list[str] = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{title}</title>",
        f"<style>{_CSS}</style>",
        "</head><body>",
        '<p class="eyebrow">meeting notes</p>',
        f"<h1>{title}</h1>",
        f'<p class="meta">{html.escape(when)} · {minutes:.0f} min · {html.escape(speakers)}</p>',
        '<hr class="hairline">',
        "<h2>Summary</h2>",
        f"<p>{html.escape(notes.get('summary', ''))}</p>",
    ]

    if notes.get("decisions"):
        parts.append("<h2>Decisions</h2><ul>")
        parts += [f"<li>{html.escape(item)}</li>" for item in notes["decisions"]]
        parts.append("</ul>")

    if notes.get("action_items"):
        parts.append("<h2>Action items</h2><ul>")
        for item in notes["action_items"]:
            who = " · ".join(filter(None, [item.get("owner"), item.get("due")]))
            suffix = f'<span class="who">{html.escape(who)}</span>' if who else ""
            parts.append(f'<li class="task">{html.escape(item["task"])}{suffix}</li>')
        parts.append("</ul>")

    if notes.get("open_questions"):
        parts.append("<h2>Open questions</h2><ul>")
        parts += [f"<li>{html.escape(item)}</li>" for item in notes["open_questions"]]
        parts.append("</ul>")

    if notes.get("quotes"):
        parts.append("<h2>Quotes</h2>")
        for quote in notes["quotes"]:
            speaker = html.escape(quote.get("speaker") or "unknown")
            parts.append(
                f'<blockquote>{html.escape(quote["text"])}'
                f'<span class="who">{speaker}</span></blockquote>'
            )

    parts.append("<h2>Transcript</h2>")
    for segment in transcript.kept:
        stamp = f"{int(segment.start) // 60:02d}:{int(segment.start) % 60:02d}"
        css_class = "turn me" if segment.speaker == ME else "turn"
        parts.append(
            f'<div class="{css_class}"><div class="t">{stamp}</div><div>'
            f'<div class="who">{html.escape(segment.speaker)}</div>'
            f"<p>{html.escape(segment.text.strip())}</p></div></div>"
        )

    parts.append(
        '<footer>Generated locally by helmcode-whisper. Audio and transcript never '
        "left this machine except for inference on the Helmcode API.</footer>"
    )
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"
