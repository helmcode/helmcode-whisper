"""The privacy claim, enforced.

The README says meeting content only ever reaches `HELMCODE_BASE_URL`. That is
only true if no module can reach anywhere else, so this test reads the whole
package looking for absolute URLs and fails on anything that is not the
configured default.

huggingface.co is the documented exception and the list below is deliberately
narrow about it: the repo names diarization needs, and nothing that could carry
a transcript. `hcw doctor` asks that host whether those three repos have been
accepted, which sends a token and three repo names; pyannote then downloads the
weights themselves. Both are disclosed in the README's own table.
"""

from __future__ import annotations

import re
from pathlib import Path

import helmcode_whisper
from helmcode_whisper.config import DEFAULT_BASE_URL

# How the f-string in `diarize.RepoAccess.url` reads in the source.
HF_REPO_PLACEHOLDER = "{self.repo}"

PACKAGE_ROOT = Path(helmcode_whisper.__file__).parent
URL = re.compile(r"https?://[^\s\"'<>)]+")

# The one host meeting content is allowed to reach, plus the project's own
# links, which are text rather than request targets.
ALLOWED = {
    DEFAULT_BASE_URL,
    "https://helmcode.com",
    "https://huggingface.co/pyannote/speaker-diarization-3.1",
    "https://huggingface.co/pyannote/segmentation-3.0",
    "https://huggingface.co/pyannote/speaker-diarization-community-1",
    # `RepoAccess.url`, which builds a link to one of the three repos above for
    # the person who has to go and accept its terms. Allowed as the template it
    # is written as, so a different path on that host still fails this test.
    f"https://huggingface.co/{HF_REPO_PLACEHOLDER}",
}


def test_no_unexpected_urls_in_package() -> None:
    offenders: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        for match in URL.findall(path.read_text(encoding="utf-8")):
            url = match.rstrip(".,;:")
            if url not in ALLOWED:
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}: {url}")
    assert not offenders, "unexpected hosts in the package: " + "; ".join(offenders)


def test_client_is_bound_to_the_configured_base_url() -> None:
    """Every endpoint method must use a relative path, or the binding is moot."""
    source = (PACKAGE_ROOT / "api.py").read_text(encoding="utf-8")
    for call in re.findall(r'self\._request\(\s*"[A-Z]+",\s*("[^"]+")', source):
        assert call.startswith('"/'), f"absolute request path in api.py: {call}"
