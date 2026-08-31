"""Run `process` from your own program and follow it step by step.

This is the shape a UI wants: `--progress-json` puts one JSON object per line on
stdout and moves everything human to stderr, so you read structured events and
never scrape a progress bar.

    python examples/02_drive_process.py
    python examples/02_drive_process.py 2026-08-31-sprint-review --force

What it demonstrates, beyond reading lines:

  * Ordering the display by the `steps` array from the `start` event rather than
    by arrival. Diarization announces itself before transcription does, because
    it really is running underneath, and a UI that trusts arrival order draws
    the pipeline wrong.
  * Ignoring event types and fields it does not know, which is what keeps a
    reader working across versions.

Events are documented in docs/DATA.md.
"""

from __future__ import annotations

import json
import subprocess
import sys


def main() -> int:
    command = [sys.executable, "-m", "helmcode_whisper.cli", "process",
               "--progress-json", *sys.argv[1:]]

    # stderr is left attached to the terminal: it carries the human-readable
    # output and any traceback, and swallowing it is how you end up debugging a
    # silent failure.
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True,
                               encoding="utf-8", bufsize=1)

    order: list[str] = []
    state: dict[str, dict] = {}
    assert process.stdout is not None

    for line in process.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            # Not our line. Never let a stray print take the reader down.
            continue

        kind = event.get("event")
        if kind == "start":
            order = list(event.get("steps") or [])
            total = event.get("duration_seconds") or 0
            print(f"{event.get('title')}  ({total / 60:.1f} min of audio)")
            print(f"  steps: {' -> '.join(order)}\n")

        elif kind == "step":
            step = event.get("step")
            state[step] = event
            mark = {"running": "..", "done": "ok", "failed": "!!", "skipped": "--"}.get(
                event.get("state"), "??"
            )
            detail = []
            if event.get("seconds") is not None:
                detail.append(f"{event['seconds']:.1f}s")
            if event.get("from_cache"):
                detail.append("from cache")
            if event.get("device"):
                detail.append(event["device"])
            if event.get("reason"):
                detail.append(str(event["reason"]))
            # Unknown fields are simply not printed. That is the whole contract.
            print(f"  [{mark}] {step:<10} {'  '.join(detail)}")

        elif kind == "chunks":
            done, total = event.get("done", 0), event.get("total", 0)
            print(f"       transcribing {done}/{total}", end="\r")

        elif kind == "done":
            print("\n" + "-" * 60)
            timings = event.get("timings_seconds") or {}
            # Ordered by the pipeline's own step list, not by dict order.
            for step in order:
                if step in timings:
                    cached = " (cached)" if state.get(step, {}).get("from_cache") else ""
                    print(f"  {step:<10} {timings[step]:>6.1f}s{cached}")
            print(f"  {'total':<10} {event.get('total_seconds', 0):>6.1f}s")
            summary = (event.get("summary") or "").strip()
            if summary:
                print(f"\n  {summary[:300]}{'...' if len(summary) > 300 else ''}")

    code = process.wait()
    if code != 0:
        print(f"\nprocess exited {code}; its error is on stderr above.", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
