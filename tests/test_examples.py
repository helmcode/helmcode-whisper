"""The examples have to keep working, or they are worse than no examples.

Nothing here touches the network or a real meeting. It checks that each example
parses, that it runs end to end against a fixture meeting, and that the one
which reads notes.json does so without importing this package, which is the
whole point it is making.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SCRIPTS = sorted(EXAMPLES.glob("*.py"))


def test_there_are_examples() -> None:
    """A glob that matches nothing would make every test below vacuous."""
    assert SCRIPTS, f"no examples found in {EXAMPLES}"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_example_parses(script: Path) -> None:
    ast.parse(script.read_text(encoding="utf-8"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_example_documents_itself(script: Path) -> None:
    """Each one opens with a docstring saying what it is for and how to run it."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    doc = ast.get_docstring(tree)
    assert doc, f"{script.name} has no module docstring"
    assert "python examples/" in doc, f"{script.name} does not show how to run it"


def test_action_items_needs_nothing_from_this_package() -> None:
    """Its argument is that a folder of JSON is enough. Imports have to agree."""
    tree = ast.parse((EXAMPLES / "01_action_items.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "helmcode_whisper" not in imported


def _fixture_home(root: Path) -> Path:
    """A meeting folder with the two files the examples read, and nothing else."""
    home = root / "home"
    meeting = home / "2026-08-31-sprint-review"
    meeting.mkdir(parents=True)
    (meeting / "notes.json").write_text(
        json.dumps(
            {
                "summary": "Estado del proyecto y siguientes pasos.",
                "decisions": ["Dejar la diarización activada por defecto."],
                "action_items": [
                    {"task": "Medir la diarización en GPU", "owner": "Ana", "due": "viernes"},
                    {"task": "Probar la captura en macOS", "owner": "", "due": ""},
                ],
                "open_questions": [],
                "quotes": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (meeting / "meta.json").write_text(
        json.dumps({"title": "sprint review", "started_at": "2026-08-31T11:20:00"}),
        encoding="utf-8",
    )
    return home


def _run(script: str, home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLES / script), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        # The real environment with HCW_HOME redirected, rather than a minimal
        # one built by hand: Windows needs SYSTEMROOT to start a process at all,
        # and hardcoding it is how a test passes here and fails on the Linux
        # runner for a reason that has nothing to do with the example.
        env={**os.environ, "HCW_HOME": str(home), "PYTHONUTF8": "1",
             "PYTHONIOENCODING": "utf-8"},
        timeout=120,
    )


def test_action_items_reads_a_meeting(tmp_path: Path) -> None:
    finished = _run("01_action_items.py", _fixture_home(tmp_path))

    assert finished.returncode == 0, finished.stderr
    assert "Medir la diarización en GPU" in finished.stdout
    assert "Ana" in finished.stdout
    assert "2 action items" in finished.stdout


def test_action_items_filters_by_owner(tmp_path: Path) -> None:
    finished = _run("01_action_items.py", _fixture_home(tmp_path), "--owner", "ana")

    assert finished.returncode == 0, finished.stderr
    assert "Medir la diarización en GPU" in finished.stdout
    assert "Probar la captura en macOS" not in finished.stdout
    # Singular, because one is one.
    assert "1 action item\n" in finished.stdout


def test_action_items_emits_valid_json(tmp_path: Path) -> None:
    finished = _run("01_action_items.py", _fixture_home(tmp_path), "--json")

    assert finished.returncode == 0, finished.stderr
    items = json.loads(finished.stdout)
    assert [item["task"] for item in items] == [
        "Medir la diarización en GPU",
        "Probar la captura en macOS",
    ]
    assert items[0]["meeting"] == "2026-08-31-sprint-review"


def test_action_items_says_something_useful_with_no_meetings(tmp_path: Path) -> None:
    """The first-run state is not an error worth a traceback."""
    finished = _run("01_action_items.py", tmp_path / "nothing-here")

    assert finished.returncode == 1
    assert "Record one first" in finished.stdout
    assert "Traceback" not in finished.stderr


def test_action_items_ignores_a_recorded_but_unprocessed_meeting(tmp_path: Path) -> None:
    """`record` works offline, so a folder without notes.json is normal."""
    home = _fixture_home(tmp_path)
    (home / "2026-08-30-standup").mkdir()

    finished = _run("01_action_items.py", home)

    assert finished.returncode == 0, finished.stderr
    assert "2 action items" in finished.stdout
