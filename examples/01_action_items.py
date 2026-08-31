"""Every open action item across every meeting you have recorded.

The point of this example is what it does *not* import. `notes.json` is plain
JSON in a folder on your disk, so the integration surface for most things you
would want to build is the standard library and a glob. No API key, no network,
and it keeps working if this tool is uninstalled tomorrow.

    python examples/01_action_items.py
    python examples/01_action_items.py --owner Ana
    python examples/01_action_items.py --since 2026-08-01

Shapes are in docs/DATA.md.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def home() -> Path:
    """Where meetings live. The same rule the tool itself follows."""
    raw = os.environ.get("HCW_HOME")
    return Path(raw).expanduser() if raw else Path.home() / "helmcode-whisper"


def meetings(root: Path, since: str | None = None):
    """Every processed meeting, oldest first.

    A folder without notes.json was recorded but never processed, which is a
    normal state rather than an error: `record` works offline and `process`
    happens later.
    """
    for folder in sorted(root.iterdir() if root.is_dir() else []):
        notes = folder / "notes.json"
        if not notes.is_file():
            continue
        meta_path = folder / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        date = str(meta.get("started_at", ""))[:10] or folder.name[:10]
        if since and date < since:
            continue
        yield {
            "id": folder.name,
            "date": date,
            "title": meta.get("title") or folder.name,
            "notes": json.loads(notes.read_text(encoding="utf-8")),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--owner", help="Only items assigned to this person.")
    parser.add_argument("--since", help="Only meetings on or after this date (YYYY-MM-DD).")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args()

    root = home()
    collected = []
    for meeting in meetings(root, since=args.since):
        for item in meeting["notes"].get("action_items", []):
            # `owner` is always present and is the empty string when nobody was
            # named. Matching case-insensitively on a substring, because the
            # transcript says "Ana" and the diarization says "SPEAKER_01".
            owner = item.get("owner", "")
            if args.owner and args.owner.lower() not in owner.lower():
                continue
            collected.append({**item, "meeting": meeting["id"], "date": meeting["date"],
                              "title": meeting["title"]})

    if args.json:
        print(json.dumps(collected, ensure_ascii=False, indent=2))
        return 0

    if not collected:
        if not root.is_dir():
            print(f"No meetings at {root}. Record one first, then `hcw process`.")
            return 1
        print("Nothing to do. Either you are on top of things or nobody committed to anything.")
        return 0

    width = max(len(item["title"]) for item in collected)
    for item in collected:
        tail = " · ".join(part for part in (item.get("owner"), item.get("due")) if part)
        print(f"  {item['date']}  {item['title']:<{width}}  {item['task']}"
              + (f"   [{tail}]" if tail else ""))
    plural = "" if len(collected) == 1 else "s"
    print(f"\n  {len(collected)} action item{plural}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
