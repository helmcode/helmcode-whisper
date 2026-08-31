"""Speaker diarization, on this machine and nowhere else.

This is the one step that is deliberately *not* an API call. Voice is biometric
data under the GDPR, and the point of the project is that the recording's
identity information never leaves the laptop. pyannote runs locally on CPU (or a
local CUDA GPU if there is one); the only network access is a one-time download
of the model weights from Hugging Face, which carries no meeting content.

Only the system track is diarized. The microphone track is one known person by
construction, so half the problem was already solved at capture time.

Everything here is optional: without pyannote installed, or without an HF token,
`process` runs to completion with the me/others split it already has.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .intervals import IntervalIndex


@contextmanager
def _quiet_import():
    """Silence pyannote's torchcodec warning.

    Importing pyannote prints forty lines of traceback when torchcodec cannot
    load its shared libraries — routine on Windows, where it needs an FFmpeg
    build most people do not have. It is noise here because this module decodes
    audio itself and never asks torchcodec for anything.
    """
    # Windows without Developer Mode cannot make symlinks, so huggingface_hub
    # warns — twice per model — that its cache will use more disk. True, and
    # nothing the user can act on from here.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    # torch 2.13 logs "triton not found; flop counting will not work" through
    # its own logger the first time this module is imported. triton has no
    # Windows build, flop counting is not something this tool asks for, and the
    # line lands in the middle of `hcw doctor` looking like a problem.
    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
    with warnings.catch_warnings():
        # `message` is matched from the start of the warning text, and this one
        # opens with a newline and runs for forty lines — hence the (?s) so `.`
        # crosses them.
        warnings.filterwarnings("ignore", message="(?s).*torchcodec.*")
        yield


# Every gated Hugging Face repo the diarization pipeline reaches for. The first
# two are the ones every tutorial names; the third is pulled by pyannote.audio 4
# while loading the 3.1 pipeline, and its 403 arrives after the other two have
# already downloaded — which reads as "I accepted the terms and it still fails".
GATED_REPOS = (
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
    "pyannote/speaker-diarization-community-1",
)


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


class DiarizationUnavailable(RuntimeError):
    """pyannote or its weights are not usable; the caller should carry on."""


def availability() -> tuple[bool, str | None]:
    """Whether pyannote can be imported here, and why not when it cannot.

    `except ImportError` is not enough. torch loads native libraries at import
    time, and those fail in ways that are not import errors at all: on Windows
    with Smart App Control enabled the unsigned `torch/lib/shm.dll` is blocked
    by code integrity policy and the import raises `OSError [WinError 4551]`.
    Anything that stops pyannote loading has to leave `process` running with the
    me/others split rather than take the whole pipeline down with it.
    """
    try:
        with _quiet_import():
            import pyannote.audio  # noqa: F401
    except ModuleNotFoundError as exc:
        # "No module named 'pyannote'" means it was never installed. The same
        # exception naming any other module means pyannote is here and one of
        # the things it imports is not — a different problem with a different
        # fix, and telling someone to install what they already have is the
        # least useful thing this function could do.
        if (exc.name or "").partition(".")[0] == "pyannote":
            return False, (
                "pyannote is not installed. Install it with "
                "`uv pip install 'helmcode-whisper[diarize]' --torch-backend=auto`, or keep "
                "the me/others split. The flag is what gets a torch that can see your GPU; "
                "without it PyPI hands Windows a CPU-only build."
            )
        return False, (
            f"pyannote is installed but {exc.name} is missing. Reinstall the extra with "
            "`uv pip install --reinstall 'helmcode-whisper[diarize]' --torch-backend=auto`."
        )
    except ImportError as exc:
        # An ImportError that is not a missing module is a version mismatch:
        # pyannote reaching for a symbol the installed torch does not export,
        # or torch failing to import itself after a partial in-place upgrade.
        return False, (
            f"pyannote is installed but will not import: {exc}. This is usually a torch "
            "version mismatch — reinstall the extra with "
            "`uv pip install --reinstall 'helmcode-whisper[diarize]' --torch-backend=auto`."
        )
    except Exception as exc:
        return False, f"pyannote is installed but will not load: {type(exc).__name__}: {exc}"
    return True, None


def available() -> bool:
    return availability()[0]


@dataclass(frozen=True)
class Device:
    """Where diarization will run, and what to do if that is the slow answer."""

    name: str  # "cuda" or "cpu"
    gpu: str | None = None
    # Set when the machine has a CUDA GPU that the installed torch cannot use.
    # This is the trap the whole extra walks into on Windows, so it gets said
    # out loud rather than left as a number nobody notices.
    advice: str | None = None


def device() -> Device | None:
    """The device pyannote would pick, resolved without loading any weights.

    Worth reporting on its own because the answer is wrong by default on
    Windows and silently so. `pip install torch` there resolves to the CPU-only
    wheel on PyPI, which is 122 MB against the 527 MB Linux build that carries
    CUDA, so `torch.cuda.is_available()` is False on a machine with a perfectly
    good GPU and the slowest step in the pipeline quietly takes ten times
    longer than it needs to.
    """
    try:
        with _quiet_import():
            import torch
    except Exception:
        return None

    if torch.cuda.is_available():
        try:
            return Device("cuda", gpu=torch.cuda.get_device_name(0))
        except Exception:
            return Device("cuda")

    advice = None
    if torch.version.cuda is None and _machine_has_an_nvidia_gpu():
        advice = (
            "this machine has an NVIDIA GPU and this torch was built without CUDA. "
            "Reinstall it from PyTorch's own index, which PyPI does not mirror: "
            "`uv pip install -e \".[diarize]\" --torch-backend=auto`"
        )
    return Device("cpu", advice=advice)


def _machine_has_an_nvidia_gpu() -> bool:
    """Ask the driver, not torch. Cheap, and the whole point is that torch is wrong."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        finished = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return finished.returncode == 0 and bool(finished.stdout.strip())


def diarize(wav_path: Path, hf_token: str | None) -> tuple[list[Turn], str]:
    """Return speaker turns for one track, plus the device it ran on."""
    usable, reason = availability()
    if not usable:
        raise DiarizationUnavailable(reason or "pyannote is unavailable")
    if not hf_token:
        raise DiarizationUnavailable(
            "HF_TOKEN is not set. pyannote needs it once to download its weights, and the "
            "terms of every gated repo it pulls have to be accepted on huggingface.co "
            f"first: {', '.join(GATED_REPOS)}."
        )

    import soundfile as sf
    import torch

    with _quiet_import():
        from pyannote.audio import Pipeline

    # pyannote renamed the argument: `use_auth_token` in 3.1, `token` since.
    # Supporting both keeps the tool working across the versions people actually
    # have installed instead of pinning one and calling it a requirement.
    pipeline = None
    errors: list[str] = []
    for keyword in ("token", "use_auth_token"):
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", **{keyword: hf_token}
            )
            break
        except TypeError as exc:
            errors.append(f"{keyword}: {exc}")
        except Exception as exc:
            raise DiarizationUnavailable(
                f"could not load pyannote/speaker-diarization-3.1: {exc}. Most often this "
                "means one of the gated repos has not been accepted yet — check all of "
                f"them: {', '.join(GATED_REPOS)}."
            ) from exc

    if pipeline is None:
        raise DiarizationUnavailable(
            "pyannote's from_pretrained rejected both token arguments, so either the token "
            "is not authorized for pyannote/speaker-diarization-3.1 or this version takes a "
            "third name for it: " + "; ".join(errors)
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))

    # pyannote is happy to use every core; on a laptop that makes the machine
    # unusable for the length of a 60-minute meeting. Leave one alone.
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

    # Decode the audio ourselves rather than handing pyannote a path. Its
    # built-in decoding goes through torchcodec, which needs a matching FFmpeg
    # shared build and simply does not load on a stock Windows install. We
    # already produced this file at 16 kHz mono, so reading it with soundfile is
    # both more reliable and one less thing that can differ between machines.
    samples, samplerate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples.T.copy())  # pyannote wants (channel, time)

    with warnings.catch_warnings():
        # pyannote's pooling layer takes the standard deviation of windows that
        # are sometimes a single frame long, and torch warns about the degrees
        # of freedom every time. It is normal, it is internal to the model, and
        # there is nothing the person running a meeting can do about it — but
        # it lands five lines of traceback in the middle of the output.
        warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")
        result = pipeline({"waveform": waveform, "sample_rate": samplerate})

    # pyannote 3 returns an Annotation. pyannote 4 wraps it in a DiarizeOutput
    # carrying two of them, and describes `exclusive_speaker_diarization` as the
    # one "adapted to downstream transcription" — it drops overlapping turns,
    # which is exactly right when the job is to hand each transcript segment a
    # single speaker.
    annotation = result
    for attribute in ("exclusive_speaker_diarization", "speaker_diarization"):
        candidate = getattr(result, attribute, None)
        if candidate is not None:
            annotation = candidate
            break

    if not hasattr(annotation, "itertracks"):
        raise DiarizationUnavailable(
            f"pyannote returned a {type(result).__name__} this version does not know how "
            "to read. Report it with your pyannote.audio version."
        )

    turns = [
        Turn(float(segment.start), float(segment.end), str(label))
        for segment, _, label in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda turn: turn.start)
    return turns, device


# A speaker change has to hold for at least this many words to be believed.
# Below it, a single word landing inside someone else's turn — a "sí", an
# overlap at a handover — would shatter a sentence into confetti.
MIN_WORDS_PER_TURN = 3


def split_by_speaker(segments: list, turns: list[Turn]) -> list:
    """Cut transcript segments where the speaker changes mid-sentence.

    Whisper decides segment boundaries from prosody and punctuation, and it is
    perfectly happy to put two people in one segment: on the first real
    recording, pyannote found 18 turns and Whisper returned 3 segments, so 18
    speaker changes collapsed into 3 labels.

    Word-level timestamps fix that. Each word gets the speaker whose turn covers
    it, consecutive words with the same speaker are regrouped, and a segment is
    re-cut on those groups. Segments transcribed without word timestamps pass
    through untouched.
    """
    from .model import Segment

    index = IntervalIndex(turns)
    refined: list = []
    for segment in segments:
        words = segment.words
        if not words or len(words) < MIN_WORDS_PER_TURN * 2:
            refined.append(segment)
            continue

        groups: list[tuple[str, list]] = []
        for word in words:
            turn = index.covering((word.start + word.end) / 2)
            speaker = turn.speaker if turn else segment.speaker
            if groups and groups[-1][0] == speaker:
                groups[-1][1].append(word)
            else:
                groups.append((speaker, [word]))

        groups = _absorb_short_groups(groups)
        if len(groups) < 2:
            refined.append(segment)
            continue

        for speaker, group in groups:
            text = "".join(word.text for word in group).strip()
            if not text:
                continue
            refined.append(
                Segment(
                    start=group[0].start,
                    end=group[-1].end,
                    text=text,
                    track=segment.track,
                    speaker=speaker,
                    confidence=segment.confidence,
                    words=group,
                )
            )

    refined.sort(key=lambda item: item.start)
    return refined


def _absorb_short_groups(groups: list[tuple[str, list]]) -> list[tuple[str, list]]:
    """Fold runs shorter than MIN_WORDS_PER_TURN into the neighbour beside them."""
    while len(groups) > 1:
        shortest = min(range(len(groups)), key=lambda index: len(groups[index][1]))
        if len(groups[shortest][1]) >= MIN_WORDS_PER_TURN:
            break
        # Prefer the longer neighbour; at the edges there is only one choice.
        left = groups[shortest - 1] if shortest > 0 else None
        right = groups[shortest + 1] if shortest < len(groups) - 1 else None
        if left is None:
            target = right
        elif right is None:
            target = left
        else:
            target = left if len(left[1]) >= len(right[1]) else right

        target[1].extend(groups[shortest][1])
        target[1].sort(key=lambda word: word.start)
        groups.pop(shortest)

        merged: list[tuple[str, list]] = []
        for speaker, words in groups:
            if merged and merged[-1][0] == speaker:
                merged[-1][1].extend(words)
            else:
                merged.append((speaker, words))
        groups = merged
    return groups


def label_segments(segments, turns: list[Turn]) -> list[str]:
    """Give each transcript segment the speaker it overlaps with the most.

    Whisper's segments and pyannote's turns are drawn on the same timeline but
    with different boundaries, so this is an overlap vote rather than a lookup.
    A segment that matches nothing keeps whatever it came in with — that happens
    on very short interjections, where guessing is worse than abstaining.
    """
    index = IntervalIndex(turns)
    labels: list[str] = []
    for segment in segments:
        best_speaker = segment.speaker
        best_overlap = 0.0
        for turn in index.overlapping(segment.start, segment.end):
            overlap = min(segment.end, turn.end) - max(segment.start, turn.start)
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, turn.speaker
        labels.append(best_speaker)
    return labels
