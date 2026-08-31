"""Search your meetings from Python, without inheriting a print statement.

`search_hits` is deliberately separate from the command that prints: it returns
the hits, whether semantic recall was available, and how many indexed passages
the configured embedding model cannot compare against. That is enough to build a
UI, a bot, or an editor plugin on.

    python examples/03_search_from_python.py "what did we agree about pricing"
    python examples/03_search_from_python.py "enterprise tier" --limit 3

Needs HELMCODE_API_KEY for the meaning-based half. Without it the search
degrades to keyword matching and says so, which is the third return value
earning its keep.
"""

from __future__ import annotations

import argparse

from helmcode_whisper.config import ConfigError, load_config
from helmcode_whisper.pipeline.search import search_hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query", help="What you are looking for, in your own words.")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    config = load_config()
    try:
        hits, semantic, stale = search_hits(config, args.query, limit=args.limit)
    except (ConfigError, RuntimeError) as exc:
        # RuntimeError is "no index yet", which is a normal first-run state.
        print(exc)
        return 1

    if not semantic:
        print("  keyword only: no embeddings available, so meaning-based matches "
              "are missing\n")
    if stale:
        print(f"  {stale} passages were embedded with another model and are outside "
              "the semantic search\n")

    if not hits:
        print(f"  nothing found for {args.query!r}")
        return 0

    for hit in hits:
        stamp = f"{int(hit.start) // 60:02d}:{int(hit.start) % 60:02d}"
        # `meeting_id` is the folder name, which is what lets a UI open the
        # recording rather than only showing the sentence.
        print(f"  {hit.meeting_date}  {hit.meeting_title}  {stamp}  {hit.speaker}")
        print(f"    {hit.text}")
        print(f"    {config.home / hit.meeting_id}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
