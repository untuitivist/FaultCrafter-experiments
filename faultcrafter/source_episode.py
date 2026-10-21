from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Iterable


@dataclass(frozen=True)
class SourceEpisodeCell:
    device: int
    record: int
    fault: int
    condition: tuple[float, ...]

    def __post_init__(self):
        if any(type(value) is not int or value < 0 for value in (self.device, self.record, self.fault)):
            raise ValueError("source episode identifiers must be nonnegative integers")
        if not isinstance(self.condition, tuple) or len(self.condition) != 8:
            raise ValueError("source episode requires four physical values and four masks")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in self.condition):
            raise ValueError("source episode coordinates must be finite numbers")
        if any(value not in (0, 1) for value in self.condition[4:]):
            raise ValueError("source episode masks must be independent binary presence flags")


@dataclass(frozen=True)
class SourceConditionEpisode:
    support_indices: tuple[int, ...]
    query_indices: tuple[int, ...]
    query_conditions: tuple[tuple[int, tuple[float, ...]], ...]
    sample_count: int


class SourceEpisodeRegistry:
    def __init__(self, entries: Iterable[SourceEpisodeCell]):
        entries = tuple(entries)
        if not entries or any(not isinstance(entry, SourceEpisodeCell) for entry in entries):
            raise ValueError("source episode registry requires typed source cells")
        records = {(entry.device, entry.record): entry for entry in entries}
        if len(records) != len(entries):
            raise ValueError("source episode registry contains duplicate cells")
        self.records = MappingProxyType(records)

    def split(self, cells: Iterable[tuple[int, int]], step: int) -> SourceConditionEpisode:
        cells = tuple(cells)
        if type(step) is not int or step < 1 or not cells:
            raise ValueError("source episode requires a nonempty batch and positive integer step")
        conditions = {}
        for cell in cells:
            if not isinstance(cell, tuple) or len(cell) != 2 or any(type(value) is not int or value < 0 for value in cell):
                raise ValueError("source episode batch requires integer device/record pairs")
            if cell not in self.records:
                raise ValueError("source episode cannot read an unregistered source cell")
            entry = self.records[cell]
            conditions.setdefault(entry.device, set()).add(entry.condition)
        query_conditions = tuple((device, ordered[(step - 1) % len(ordered)])
                                 for device, available in sorted(conditions.items())
                                 if len(ordered := sorted(available)) >= 2)
        selected = set(query_conditions)
        query = tuple(index for index, cell in enumerate(cells)
                      if (self.records[cell].device, self.records[cell].condition) in selected)
        query_set = set(query)
        support = tuple(index for index in range(len(cells)) if index not in query_set)
        return SourceConditionEpisode(support, query, query_conditions, len(cells))
