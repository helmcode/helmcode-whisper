# Contributing

## Getting set up

```bash
git clone https://github.com/helmcode/helmcode-whisper
cd helmcode-whisper
uv venv && uv pip install -e ".[dev]"
cp .env.example .env          # then paste your HELMCODE_API_KEY
uv run pytest -q              # a couple of seconds, none of it networked
uv run ruff check src tests examples
```

Add diarization only if you are touching it, because it pulls torch:

```bash
uv pip install -e ".[diarize]" --torch-backend=auto
```

`--torch-backend=auto` matters. PyPI's Windows wheel for torch has no CUDA in
it, so without the flag diarization silently runs on the CPU and takes six or
seven times longer. `hcw doctor` will tell you which device it found.

`hcw doctor` is the first thing to run when anything is off, and the first
thing to paste into an issue.

## The shape of the thing

```
src/helmcode_whisper/
    cli.py            every command, and nothing else
    api.py            the only module that opens a network connection
    config.py         read once, from the environment and .env
    store.py          a meeting on disk
    capture/          per-platform audio input
    pipeline/         one module per step, in the order they run
    ui/               terminal output and the HTML export
templates/            the notes prompt, which is a file on purpose
docs/DATA.md          what a meeting leaves behind, and the event stream
examples/             three short programs that build on the output
```

`pipeline/` is worth reading in the order `run.py` calls it: `audio`, `stt`,
`diarize`, `merge`, `notes`, `index`. Every module has a docstring explaining
why it does the thing it does, and those are the real documentation.

## What the tests do and do not cover

No test touches the network or an audio device. The suite covers chunk planning,
echo suppression, response parsing, storage, the search index and the
progress-event stream, all against fixtures. That is what makes it fast enough
to run on every save and honest enough to gate a merge.

Anything that needs the real API is run by hand, and the number that comes back
goes in the commit message rather than into a test that would fail on somebody
else's key.

Two tests are load-bearing in a way that is easy to miss:

- **`test_no_egress.py`** reads the whole package looking for absolute URLs and
  fails on anything outside a short allowlist. The README's privacy claim is
  only true while that passes. If you add a host, the README's table has to
  change in the same commit, and you should expect to be asked why.
- **`test_api.py`** pins the client to relative paths, which is what makes
  `HELMCODE_BASE_URL` mean anything.

## Things that will trip you up

**The structured-output ladder in `notes.py` is not defensive coding.** The API
documents `json_schema` as validated on other models than the default, and a
reasoning model will sometimes answer with `content: null`. Every rung catches
`ValueError` and walks down. If you add a failure mode, raise something in that
family or the run dies at the most expensive step.

**Timings in `meta.json` are only measurements when `from_cache` is false.** A
replayed run finishes in a second. Do not compare numbers without checking.

**Diarization is optional and has to stay optional.** Every path through
`process` must finish with the me/others split when pyannote is missing, broken,
or blocked by a gated repo. `test_diarize.py` exists to keep that true.

**Two tracks share one clock.** Timestamps everywhere are absolute seconds from
the start of the recording, never offsets into a chunk or a splice. Chunk
planning is where that is easiest to break.

## Commits

One change per commit, and a message that says why rather than what. The diff
already says what. If a number convinced you, put the number in the message:
most of the interesting decisions in this repository are recorded that way and
nowhere else.

Run the linter and the tests before you push. CI runs Linux and Windows against
Python 3.11 and 3.12, and macOS is deliberately not claimed.

## Reporting something

Open an issue with the output of `hcw doctor`, your platform, and what you
expected instead. For anything about audio capture, the `devices` block from
`hcw devices` too. For anything about transcription, the `chunks` block from
`meta.json`.

macOS and Linux capture are implemented and untested, so reports there are
worth more than they look.
