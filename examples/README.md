# Examples

Three short programs, each showing one way to build on this. Run them from a
checkout: the first needs nothing installed at all, the other two need the
package importable.

The shapes they read are documented in [../docs/DATA.md](../docs/DATA.md).

## 01_action_items.py

Every open action item across every meeting, with owners and due dates.

```bash
python examples/01_action_items.py
python examples/01_action_items.py --owner Ana --since 2026-08-01
python examples/01_action_items.py --json | jq '.[] | select(.due != "")'
```

Look at what it imports: `argparse`, `json`, `os`, `pathlib`. Nothing from this
package, no API key, no network. `notes.json` is plain JSON in a folder, and for
most of what you would want to build that is the whole integration surface. It
keeps working if you uninstall the tool tomorrow.

## 02_drive_process.py

Run `process` from your own program and follow it step by step, which is what a
UI needs.

```bash
python examples/02_drive_process.py
python examples/02_drive_process.py 2026-08-31-sprint-review --force
```

`--progress-json` puts one JSON object per line on stdout and moves the human
output to stderr, so you never scrape a progress bar. Two details in here are
the ones that bite people:

- The display is ordered by the `steps` list from the `start` event, not by
  arrival. Diarization announces itself as running before transcription does,
  because it genuinely is running underneath, and a reader that trusts arrival
  order draws the pipeline wrong.
- Unknown event types and unknown fields are skipped rather than treated as
  errors. That is what keeps a reader working across versions.

## 03_search_from_python.py

Search from Python without inheriting a print statement.

```bash
python examples/03_search_from_python.py "what did we agree about pricing"
python examples/03_search_from_python.py "enterprise tier" --limit 3
```

`search_hits` returns three things: the hits, whether semantic recall was
available, and how many indexed passages the configured embedding model cannot
compare against. The second and third exist so a UI can say "these results are
worse than usual, and here is why" instead of quietly returning less.

Each hit carries `meeting_id`, which is the folder name, so a UI can open the
recording rather than only showing the sentence.

## Building something else

The rest of the surface, in rough order of how stable it is:

| you want | use |
|---|---|
| notes, decisions, action items | `notes.json` in the meeting folder |
| the transcript with speakers | `transcript.json`, skipping segments with `dropped` |
| timings, tokens, devices | `meta.json` |
| progress while it runs | `hcw process --progress-json` |
| search | `pipeline.search.search_hits(config, query)` |
| running the pipeline in-process | `pipeline.run.run_process(config, meeting, events=...)` |
| a meeting folder | `store.Meeting(path)` |

The files and the event stream are the parts meant to be depended on. The Python
functions are there and are used by the CLI, but this is a 0.1 and they can move.
