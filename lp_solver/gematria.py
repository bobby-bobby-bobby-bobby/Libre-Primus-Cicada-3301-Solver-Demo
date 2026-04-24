from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

RUNES = [
    "ᚠ", "ᚢ", "ᚦ", "ᚩ", "ᚱ", "ᚳ", "ᚷ", "ᚹ", "ᚻ", "ᚾ", "ᛁ", "ᛄ", "ᛇ", "ᛈ", "ᛉ", "ᛋ",
    "ᛏ", "ᛒ", "ᛖ", "ᛗ", "ᛚ", "ᛝ", "ᛟ", "ᛞ", "ᚪ", "ᚫ", "ᚣ", "ᛡ", "ᛠ",
]

# Gematria Primus values (canonical ordering from LP community references).
GP_VALUES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109,
]

RUNE_TO_INDEX: Dict[str, int] = {r: i for i, r in enumerate(RUNES)}
RUNE_TO_GP: Dict[str, int] = {r: GP_VALUES[i] for i, r in enumerate(RUNES)}

LATIN_FALLBACK = "futhorcw hni jïp xst bemlŋod aæyia".replace(" ", "")
LATIN_TO_RUNE = {c: RUNES[i] for i, c in enumerate(LATIN_FALLBACK[: len(RUNES)])}


def rune_to_index(glyph: str) -> int:
    return RUNE_TO_INDEX[glyph]


def rune_to_gp(glyph: str) -> int:
    return RUNE_TO_GP[glyph]


def runes_to_indices(runes: Sequence[str]) -> List[int]:
    return [rune_to_index(r) for r in runes]


def runes_to_gp_values(runes: Sequence[str]) -> List[int]:
    return [rune_to_gp(r) for r in runes]


def rune_stream_from_text(text: str) -> List[str]:
    out: List[str] = []
    for ch in text:
        if ch in RUNE_TO_INDEX:
            out.append(ch)
        else:
            low = ch.lower()
            if low in LATIN_TO_RUNE:
                out.append(LATIN_TO_RUNE[low])
    return out


def all_runes() -> Iterable[str]:
    return RUNES
