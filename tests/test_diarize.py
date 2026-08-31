"""Diarization is optional, and 'optional' has to survive ugly failures.

The acceptance criterion is that the pipeline runs to completion without
pyannote. Not installed is the easy case. The one that actually happened is
pyannote being installed and refusing to load: on Windows with Smart App Control
enabled, torch's unsigned `shm.dll` is blocked by code integrity policy and the
import raises OSError, which an `except ImportError` sails straight past.
"""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from helmcode_whisper.pipeline import diarize
from helmcode_whisper.pipeline.model import OTHERS, Segment, Word


def words(text: str, start: float, step: float = 0.5) -> list[Word]:
    """One word per `step` seconds, the way the API returns them."""
    return [
        Word(start + index * step, start + (index + 1) * step, f" {token}")
        for index, token in enumerate(text.split())
    ]


@pytest.fixture
def pyannote_raises(monkeypatch):
    """Make importing pyannote fail with a given exception."""

    def install(exception: Exception) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args, **kwargs):
            if name.startswith("pyannote"):
                raise exception
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

    return install


def test_missing_pyannote_is_reported_not_raised(pyannote_raises) -> None:
    pyannote_raises(ModuleNotFoundError("No module named 'pyannote'", name="pyannote"))

    usable, reason = diarize.availability()

    assert usable is False
    assert "not installed" in reason


def test_a_broken_install_is_not_reported_as_a_missing_one(pyannote_raises) -> None:
    """The failure this machine actually produces.

    torch upgraded in place leaves pyannote importing a symbol its installed
    torch no longer exports. That is an ImportError but not a missing module,
    and answering it with "pyannote is not installed" sends the user to
    reinstall something they already have.
    """
    pyannote_raises(
        ImportError("cannot import name 'NP_SUPPORTED_MODULES' from 'torch._dynamo.utils'")
    )

    usable, reason = diarize.availability()

    assert usable is False
    assert "not installed" not in reason
    assert "version mismatch" in reason


def test_a_missing_dependency_names_the_dependency(pyannote_raises) -> None:
    pyannote_raises(ModuleNotFoundError("No module named 'torch'", name="torch"))

    usable, reason = diarize.availability()

    assert usable is False
    assert "torch is missing" in reason


def test_a_blocked_native_library_is_also_survivable(pyannote_raises) -> None:
    pyannote_raises(OSError("[WinError 4551] blocked by application control"))

    usable, reason = diarize.availability()

    assert usable is False
    assert "will not load" in reason
    assert "WinError 4551" in reason


def test_diarize_raises_its_own_error_so_process_can_carry_on(pyannote_raises) -> None:
    pyannote_raises(OSError("[WinError 4551] blocked by application control"))

    with pytest.raises(diarize.DiarizationUnavailable):
        diarize.diarize(Path("nonexistent.wav"), "hf_token")


def test_a_missing_token_is_caught_before_any_download(monkeypatch) -> None:
    # Pin availability rather than relying on this machine having a working
    # pyannote: the assertion is about the token check, and letting the
    # environment decide whether the test runs at all makes it useless in CI,
    # where pyannote is deliberately not installed.
    monkeypatch.setattr(diarize, "availability", lambda: (True, None))

    with pytest.raises(diarize.DiarizationUnavailable, match="HF_TOKEN"):
        diarize.diarize(Path("nonexistent.wav"), None)


def test_segments_take_the_speaker_they_overlap_most() -> None:
    turns = [
        diarize.Turn(0.0, 10.0, "SPEAKER_00"),
        diarize.Turn(10.0, 20.0, "SPEAKER_01"),
    ]
    segments = [
        Segment(1.0, 3.0, "primero", "system", OTHERS),
        Segment(9.0, 15.0, "a caballo", "system", OTHERS),  # 1s vs 5s
        Segment(12.0, 14.0, "segundo", "system", OTHERS),
    ]

    assert diarize.label_segments(segments, turns) == [
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_01",
    ]


def test_a_segment_holding_two_people_is_cut_between_them() -> None:
    """Whisper segments follow punctuation, not speakers. 18 turns once
    collapsed into 3 labels because of exactly this."""
    spoken = words("uno dos tres cuatro cinco seis siete ocho", 0.0)
    segment = Segment(0.0, 4.0, "uno dos tres cuatro cinco seis siete ocho", "system", OTHERS)
    segment.words = spoken
    turns = [
        diarize.Turn(0.0, 2.0, "SPEAKER_00"),
        diarize.Turn(2.0, 4.0, "SPEAKER_01"),
    ]

    result = diarize.split_by_speaker([segment], turns)

    assert len(result) == 2
    assert result[0].speaker == "SPEAKER_00"
    assert result[0].text == "uno dos tres cuatro"
    assert result[1].speaker == "SPEAKER_01"
    assert result[1].text == "cinco seis siete ocho"
    assert result[1].start == pytest.approx(2.0)


def test_a_single_stray_word_does_not_shatter_a_sentence() -> None:
    spoken = words("uno dos tres cuatro cinco seis siete ocho", 0.0)
    segment = Segment(0.0, 4.0, "uno dos tres cuatro cinco seis siete ocho", "system", OTHERS)
    segment.words = spoken
    # One word's worth of someone else in the middle: an overlap, not a turn.
    turns = [
        diarize.Turn(0.0, 2.0, "SPEAKER_00"),
        diarize.Turn(2.0, 2.5, "SPEAKER_01"),
        diarize.Turn(2.5, 4.0, "SPEAKER_00"),
    ]

    result = diarize.split_by_speaker([segment], turns)

    assert len(result) == 1
    assert result[0].text == "uno dos tres cuatro cinco seis siete ocho"


def test_segments_without_word_timestamps_pass_through() -> None:
    segment = Segment(0.0, 4.0, "sin palabras", "system", OTHERS)
    turns = [diarize.Turn(0.0, 2.0, "SPEAKER_00"), diarize.Turn(2.0, 4.0, "SPEAKER_01")]

    assert diarize.split_by_speaker([segment], turns) == [segment]


def test_a_segment_overlapping_nothing_keeps_its_label() -> None:
    """Silence between turns is a bad place to guess."""
    turns = [diarize.Turn(0.0, 5.0, "SPEAKER_00")]
    segments = [Segment(100.0, 101.0, "suelto", "system", OTHERS)]

    assert diarize.label_segments(segments, turns) == [OTHERS]


# ── which device diarization will actually use ────────────────────
#
# This is the check that would have caught the sample run in the README saying
# "on cpu" on a machine with a CUDA GPU sitting idle.


class _FakeTorch:
    class cuda:  # noqa: N801 - mirroring torch's own lowercase namespace
        _available = False
        _name = "NVIDIA GeForce GTX 1660 Ti"

        @classmethod
        def is_available(cls) -> bool:
            return cls._available

        @classmethod
        def get_device_name(cls, index: int) -> str:
            return cls._name

    class version:  # noqa: N801
        cuda = None


def _install_fake_torch(monkeypatch, *, cuda_available: bool, built_with_cuda: str | None):
    import sys

    fake = _FakeTorch
    fake.cuda._available = cuda_available
    fake.version.cuda = built_with_cuda
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


def test_device_reports_cuda_when_torch_can_see_it(monkeypatch) -> None:
    _install_fake_torch(monkeypatch, cuda_available=True, built_with_cuda="13.0")

    where = diarize.device()

    assert where is not None
    assert where.name == "cuda"
    assert "1660" in (where.gpu or "")
    assert where.advice is None


def test_a_cpu_only_torch_next_to_an_nvidia_gpu_says_what_to_do(monkeypatch) -> None:
    """The Windows trap: PyPI's torch wheel has no CUDA, so the GPU goes unused."""
    _install_fake_torch(monkeypatch, cuda_available=False, built_with_cuda=None)
    monkeypatch.setattr(diarize, "_machine_has_an_nvidia_gpu", lambda: True)

    where = diarize.device()

    assert where is not None
    assert where.name == "cpu"
    assert where.advice is not None
    assert "--torch-backend" in where.advice


def test_cpu_without_a_gpu_is_not_a_problem_worth_mentioning(monkeypatch) -> None:
    _install_fake_torch(monkeypatch, cuda_available=False, built_with_cuda=None)
    monkeypatch.setattr(diarize, "_machine_has_an_nvidia_gpu", lambda: False)

    where = diarize.device()

    assert where is not None
    assert where.name == "cpu"
    assert where.advice is None


def test_a_cuda_build_that_cannot_see_the_gpu_gets_no_wheel_advice(monkeypatch) -> None:
    """torch has CUDA compiled in, so a missing GPU is a driver question."""
    _install_fake_torch(monkeypatch, cuda_available=False, built_with_cuda="13.0")
    monkeypatch.setattr(diarize, "_machine_has_an_nvidia_gpu", lambda: True)

    where = diarize.device()

    assert where is not None
    assert where.name == "cpu"
    assert where.advice is None


def test_device_is_unknown_rather_than_fatal_when_torch_will_not_import(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "torch":
            raise OSError("[WinError 4551] Smart App Control blocked torch/lib/shm.dll")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)

    assert diarize.device() is None


# ── the three gated repos, asked about together ────────────────────
#
# pyannote asks for the third only after the first two have downloaded, so left
# to its own order a missing acceptance reads as "I accepted the terms and it
# still fails" several hundred megabytes in.


class _GatedRepoError(Exception):
    pass


class _RepositoryNotFoundError(Exception):
    pass


class _HfHubHTTPError(Exception):
    pass


def _install_fake_hub(monkeypatch, blocked: dict[str, Exception]):
    import sys
    import types

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def model_info(self, repo):
            if repo in blocked:
                raise blocked[repo]
            return {"id": repo}

    hub = types.ModuleType("huggingface_hub")
    hub.HfApi = FakeApi
    utils = types.ModuleType("huggingface_hub.utils")
    utils.GatedRepoError = _GatedRepoError
    utils.RepositoryNotFoundError = _RepositoryNotFoundError
    utils.HfHubHTTPError = _HfHubHTTPError
    hub.utils = utils
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", utils)


def test_all_three_gated_repos_accepted(monkeypatch) -> None:
    _install_fake_hub(monkeypatch, blocked={})

    results = diarize.gated_access("hf_pretend")

    assert results is not None
    assert len(results) == len(diarize.GATED_REPOS)
    assert all(item.ok for item in results)


def test_the_third_repo_is_named_when_it_is_the_one_blocking(monkeypatch) -> None:
    """community-1 is the one every guide leaves out."""
    third = "pyannote/speaker-diarization-community-1"
    assert third in diarize.GATED_REPOS
    _install_fake_hub(monkeypatch, blocked={third: _GatedRepoError()})

    results = diarize.gated_access("hf_pretend")

    assert results is not None
    blocked = [item for item in results if not item.ok]
    assert [item.repo for item in blocked] == [third]
    assert blocked[0].url == f"https://huggingface.co/{third}"
    assert blocked[0].reason == "terms not accepted"


def test_a_404_is_reported_like_the_403_it_really_is(monkeypatch) -> None:
    """An unauthorized token cannot see a gated repo at all, so HF answers 404."""
    repo = diarize.GATED_REPOS[0]
    _install_fake_hub(monkeypatch, blocked={repo: _RepositoryNotFoundError()})

    results = diarize.gated_access("hf_pretend")

    assert results is not None
    blocked = [item for item in results if not item.ok]
    assert len(blocked) == 1
    assert blocked[0].reason == "not visible to this token"


def test_no_token_means_there_is_nothing_to_check() -> None:
    assert diarize.gated_access(None) is None
    assert diarize.gated_access("") is None


def test_without_huggingface_hub_the_check_is_skipped_not_failed(monkeypatch) -> None:
    """`doctor` reports what it can and never raises."""
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("huggingface_hub"):
            raise ImportError("no huggingface_hub")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)

    assert diarize.gated_access("hf_pretend") is None
