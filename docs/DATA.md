# What a meeting leaves behind

Every shape on this page was copied out of a real run, not designed on paper.
If something here disagrees with what your own `~/helmcode-whisper` contains,
trust the folder and open an issue.

```
~/helmcode-whisper/
    index.sqlite3                       search across every meeting
    2026-08-31-sprint-review/
        audio-mic.wav                   your microphone
        audio-system.wav                everyone else
        transcript.json                 segments, timestamps, speakers
        notes.json                      the notes as structure
        notes.md                        the same notes, for reading
        notes.html                      the same notes, self-contained
        meta.json                        devices, timings, tokens, history
        .cache/                          per-step results, safe to delete
```

The folder name is the integration key: `<date>-<slug of the title>`, and it is
what `meta.json`, the search index and the event stream all call the meeting.

`notes.md` and `notes.html` are for humans. Build on `notes.json`, because an
owner and a due date cannot be recovered from a rendered bullet without
guessing.

---

## notes.json

Five keys, always present, and the shape is enforced by a JSON schema on the way
out of the model rather than hoped for.

```json
{
  "summary": "La reunión revisó el estado del proyecto de notas de reuniones...",
  "decisions": [
    "La diarización se deja activada por defecto ahora que corre en GPU."
  ],
  "action_items": [
    {
      "task": "Documentar cuánto tarda la diarización en cada dispositivo",
      "owner": "SPEAKER_01",
      "due": "viernes"
    }
  ],
  "open_questions": [
    "Quién puede probar la captura en macOS."
  ],
  "quotes": [
    { "speaker": "SPEAKER_01", "text": "En CPU no se sostiene una hora." }
  ]
}
```

| field | type | notes |
|---|---|---|
| `summary` | string | 5 to 8 sentences. Never empty on a successful run. |
| `decisions` | array of string | Can be empty. A meeting can decide nothing. |
| `action_items` | array of object | `task`, `owner`, `due`, all three always present |
| `action_items[].owner` | string | Empty string when nobody was named |
| `action_items[].due` | string | Empty string when no date was said. Free text, as spoken: "viernes", "next sprint". Not a date type, because the transcript rarely gives you one. |
| `open_questions` | array of string | Things raised and left open |
| `quotes` | array of object | `speaker` and `text` |

The section names are fixed by `NOTES_SCHEMA` in
`src/helmcode_whisper/pipeline/notes.py`. Change them there and the renderers,
the HTML export and this page all need to follow. What each section *means* is
decided by `templates/notes.md`, which is yours to edit.

`owner` and `quotes[].speaker` carry whatever the transcript called that person:
`Me` for the microphone track, and `SPEAKER_00` upward for the remote side
unless diarization was skipped, in which case the remote side is `Others`.

---

## transcript.json

```json
{
  "language": "es",
  "speakers": ["Me", "SPEAKER_00", "SPEAKER_01"],
  "diarized": true,
  "segments": [
    {
      "start": 0.0,
      "end": 25.36,
      "text": "Buenos días, repasamos el estado del proyecto.",
      "track": "mic",
      "speaker": "Me",
      "confidence": -0.0437
    }
  ]
}
```

| field | type | notes |
|---|---|---|
| `language` | string or null | ISO-639-1, detected or passed with `--language` |
| `speakers` | array of string | In first-appearance order |
| `diarized` | bool | False when pyannote was skipped or unavailable |
| `segments[].start`, `.end` | float | Seconds from the start of the recording, absolute. Both tracks share one clock. |
| `segments[].track` | string | `mic` or `system` |
| `segments[].speaker` | string | `Me`, `Others`, or `SPEAKER_nn` |
| `segments[].confidence` | float or absent | Whisper's `avg_logprob`. Negative, closer to zero is better. |
| `segments[].dropped` | string or absent | Present only on a segment the merge threw away, and it says why. `echo` is the one you will see. |

**Segments with a `dropped` key are still in the file on purpose.** Filter them
out unless you are auditing: `[s for s in segments if not s.get("dropped")]`.
Deleting somebody's words silently would be worse than keeping them with a
reason attached.

Word-level timestamps are asked for and used, to cut a segment where the speaker
changes mid-sentence, but they are not serialized. They would roughly triple the
size of this file to record something no later step reads.

---

## meta.json

The record of what happened, and the only place the numbers live.

```json
{
  "title": "sprint review",
  "started_at": "2026-08-31T11:20:00",
  "tool_version": "0.1.0",
  "duration_seconds": 66.43,
  "devices": { "mic": "...", "system": "..." },
  "language": "es",
  "speakers": ["Me", "SPEAKER_00"],
  "diarized": true,
  "diarization_device": "cuda",
  "processed_at": "2026-08-31T15:50:43",
  "timings_seconds": {
    "prepare": 1.47, "transcribe": 2.75,
    "diarize": 8.05, "notes": 41.89, "index": 1.98
  },
  "diarization_seconds": 10.8,
  "from_cache": { "prepare": false, "notes": false },
  "chunks": { "mic": 1, "system": 1 },
  "stt_concurrency": 4,
  "notes_tokens": { "prompt": 1142, "completion": 8724, "mode": "json_schema" },
  "index": { "passages": 11, "embedded": 11 },
  "transcript_chars": 1096,
  "echo_segments_dropped": 0,
  "runs": [ { "processed_at": "...", "timings_seconds": {} } ]
}
```

Three of these repay a closer look.

**`timings_seconds.diarize` and `diarization_seconds` are different numbers and
both are true.** The first is what diarization cost the pipeline, which is
whatever it had left to do when the transcription requests came back. The second
is what it cost on its own. The gap between them is what the overlap saved.

**`from_cache` tells you whether a timing is a measurement.** A replayed run
finishes in a second, and its timings mean nothing. If you are collecting
numbers, drop any step whose `from_cache` is true.

**`runs` is append-only.** Every `process` on this meeting adds an entry rather
than overwriting, so a cached re-run cannot destroy the one measurement that did
the work. The top-level keys mirror the most recent run.

`notes_tokens.mode` is which rung of the structured-output ladder worked:
`json_schema`, `json_object` or `plain`. It varies between runs on the same
recording, so treat it as a fact about that run and not about the model.

---

## The event stream

```bash
hcw process --progress-json
```

One JSON object per line on stdout, human output moved to stderr so the two
audiences do not interleave. Real output, elided:

```json
{"event": "start", "meeting": "2026-08-31-sprint-review", "title": "sprint review", "duration_seconds": 66.43, "steps": ["prepare", "transcribe", "diarize", "merge", "notes", "index"]}
{"event": "step", "step": "prepare", "state": "running"}
{"event": "step", "step": "prepare", "state": "done", "seconds": 1.47, "from_cache": false, "speech_seconds": 66.6, "chunks": 2}
{"event": "step", "step": "diarize", "state": "running", "estimate_seconds": 29, "reason": null}
{"event": "step", "step": "transcribe", "state": "running", "total": 2}
{"event": "chunks", "done": 1, "total": 2}
{"event": "step", "step": "transcribe", "state": "done", "seconds": 2.75, "from_cache": false, "segments": 5, "language": "es"}
{"event": "step", "step": "diarize", "state": "done", "seconds": 8.05, "waited_seconds": 8.05, "device": "cuda", "from_cache": false, "reason": null}
{"event": "step", "step": "merge", "state": "done", "turns": 11, "speakers": ["Me", "SPEAKER_00"], "echoes_dropped": 0}
{"event": "step", "step": "notes", "state": "done", "seconds": 41.89, "from_cache": false, "mode": "json_schema", "prompt_tokens": 1142, "completion_tokens": 8724}
{"event": "step", "step": "index", "state": "done", "seconds": 1.98, "passages": 11, "embedded": 11}
{"event": "done", "meeting": "...", "timings_seconds": {}, "total_seconds": 56.2, "summary": "...", "speakers": []}
```

Three event types, and the contract is narrow on purpose:

- `start` once, carrying the step names you are about to see. Read the list
  rather than hardcoding it.
- `step` for every transition. `state` is `running`, `done`, `failed` or
  `skipped`. A `skipped` or `failed` step carries `reason`.
- `chunks` while transcribing, `done` out of `total`.
- `done` once at the end.

**Steps do not arrive in the order `start` lists them.** Diarization announces
itself as running before transcription does, because it is running, underneath.
Order your UI by the `steps` array, not by arrival.

**Fields are additive.** Ignore a field you do not recognise and a new one will
not break a reader that already works. Nothing is removed without a version
bump.

A step that ends `failed` does not mean the run failed. Diarization is optional,
and `process` finishes with the me/others split when pyannote cannot run.

---

## The search index

`index.sqlite3`, one file for every meeting you have ever recorded. Tables:
`meetings`, `passages`, `passages_fts` (FTS5, external content) and `vectors`.

Query it directly if you want, but two things will bite:

- `passages_fts` is an external-content table, so a cascade from `meetings` does
  not reach it. `index.delete_meeting` exists for that reason. Deleting rows by
  hand leaves the keyword index offering sentences from a meeting that is gone.
- `vectors.model` records which embedding model produced each row. Vectors from
  two models are not comparable, and `index.load_vectors` filters on it. If you
  add rows, set it.

From Python, `pipeline.search.search_hits(config, query)` returns hits, whether
semantic recall was available, and how many passages the current embedding model
cannot see. See `examples/03_search_from_python.py`.
