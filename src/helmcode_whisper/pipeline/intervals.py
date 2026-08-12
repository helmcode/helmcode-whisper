"""Interval lookups that do not get slower as the meeting gets longer.

Three steps of the pipeline ask the same question of a sorted list of timed
things: what covers this instant, and what overlaps this span. Diarization asks
it once per word and once per segment, echo suppression once per microphone
segment. Each one is a linear rescan if written the obvious way, and at
95 seconds of audio the obvious way is invisible — a few hundred iterations.

An hour of audio is where it stops being invisible. A thousand speaker turns
against ten thousand words is ten million comparisons in interpreted Python,
for a step whose actual work is a dictionary lookup. Hence a binary search.

The subtlety is that intervals overlap, so searching the start times alone is
wrong: the turn covering an instant may begin well before the last turn that
started before it. The prefix maximum of the end times fixes that. It is
non-decreasing by construction, so it can be searched too, and it answers
"could anything at or before this position still reach the time I am asking
about" — which is exactly the guard a backward scan needs to stop early.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterator
from typing import Generic, Protocol, TypeVar


class Interval(Protocol):
    """Anything with a start and an end, in seconds."""

    start: float
    end: float


ItemT = TypeVar("ItemT", bound=Interval)


class IntervalIndex(Generic[ItemT]):
    """A sorted, searchable view over intervals. Does not mutate its input."""

    def __init__(self, items: list[ItemT]) -> None:
        self.items: list[ItemT] = sorted(items, key=lambda item: item.start)
        self._starts = [item.start for item in self.items]
        self._max_end: list[float] = []
        highest = float("-inf")
        for item in self.items:
            highest = max(highest, item.end)
            self._max_end.append(highest)

    def __len__(self) -> int:
        return len(self.items)

    def covering(self, when: float) -> ItemT | None:
        """The interval containing `when`, or None.

        When several contain it — overlapping speech — the one that started
        latest wins, which is the one that best describes who is talking now.
        """
        for position in range(bisect_right(self._starts, when) - 1, -1, -1):
            if self._max_end[position] < when:
                break
            item = self.items[position]
            if item.start <= when <= item.end:
                return item
        return None

    def overlapping(self, start: float, end: float) -> Iterator[ItemT]:
        """Every interval intersecting [start, end], in start order.

        The binary search skips everything that ends before `start`; the
        per-item check is still needed because the prefix maximum only promises
        that *something* from that position onwards reaches `start`, not that
        every interval does.

        One interval spanning the whole recording pins the search at the
        beginning and makes this linear again. Speaker turns are short and, on
        the exclusive diarization this pipeline asks for, non-overlapping, so
        that degenerate case does not arise here — and an interval tree to
        guard against it would be more machinery than the problem deserves.
        """
        position = bisect_left(self._max_end, start)
        while position < len(self.items):
            item = self.items[position]
            if item.start >= end:
                return
            if item.end >= start:
                yield item
            position += 1


def slice_between(starts: list[float], start: float, end: float) -> tuple[int, int]:
    """The half-open range of positions in a sorted list falling in [start, end).

    For the one case that indexes points rather than intervals: handing each
    transcript segment the words that fall inside it.
    """
    return bisect_left(starts, start), bisect_left(starts, end)
