"""The interval index, including the overlap cases a start-time search gets wrong."""

from __future__ import annotations

from dataclasses import dataclass

from helmcode_whisper.pipeline.intervals import IntervalIndex, slice_between


@dataclass(frozen=True)
class Span:
    start: float
    end: float
    label: str = ""


def test_covering_finds_the_interval_around_an_instant() -> None:
    index = IntervalIndex([Span(0.0, 5.0, "a"), Span(5.0, 10.0, "b")])

    assert index.covering(2.5).label == "a"
    assert index.covering(7.5).label == "b"


def test_covering_returns_none_in_a_gap() -> None:
    index = IntervalIndex([Span(0.0, 1.0, "a"), Span(9.0, 10.0, "b")])

    assert index.covering(5.0) is None


def test_covering_sees_past_a_later_short_interval() -> None:
    """The case a binary search on start times alone gets wrong.

    A long turn starting at 0 still covers second 8 even though a short turn
    started at 7 and finished at 7.5. Searching starts alone lands on the short
    one, finds it does not cover 8, and wrongly concludes nobody was talking.
    """
    index = IntervalIndex([Span(0.0, 20.0, "long"), Span(7.0, 7.5, "short")])

    assert index.covering(8.0).label == "long"
    assert index.covering(7.2).label == "short"  # the later start wins a tie


def test_overlapping_returns_every_intersecting_interval() -> None:
    index = IntervalIndex(
        [Span(0.0, 2.0, "a"), Span(1.5, 4.0, "b"), Span(3.0, 6.0, "c"), Span(20.0, 21.0, "far")]
    )

    assert [span.label for span in index.overlapping(1.0, 3.5)] == ["a", "b", "c"]
    assert [span.label for span in index.overlapping(10.0, 11.0)] == []


def test_overlapping_does_not_miss_an_interval_that_started_much_earlier() -> None:
    index = IntervalIndex([Span(0.0, 100.0, "long"), Span(50.0, 51.0, "short")])

    assert [span.label for span in index.overlapping(60.0, 61.0)] == ["long"]


def test_the_index_sorts_and_leaves_the_caller_s_list_alone() -> None:
    original = [Span(5.0, 6.0, "b"), Span(0.0, 1.0, "a")]

    index = IntervalIndex(original)

    assert [span.label for span in index.items] == ["a", "b"]
    assert [span.label for span in original] == ["b", "a"]


def test_an_empty_index_answers_without_complaining() -> None:
    index = IntervalIndex([])

    assert len(index) == 0
    assert index.covering(1.0) is None
    assert list(index.overlapping(0.0, 10.0)) == []


def test_slice_between_is_half_open() -> None:
    starts = [0.0, 1.0, 2.0, 3.0]

    assert slice_between(starts, 1.0, 3.0) == (1, 3)
    assert slice_between(starts, 1.5, 2.5) == (2, 3)
    assert slice_between(starts, 10.0, 20.0) == (4, 4)
