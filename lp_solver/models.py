from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class Rune:
    glyph: str
    index: int
    gp_value: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Page:
    page_id: str
    source_path: str
    lines: List[str]
    runes: List[Rune]

    @property
    def rune_count(self) -> int:
        return len(self.runes)
