from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Iterable, List

from .gematria import rune_stream_from_text, rune_to_gp
from .models import Page, Rune

RUNIC_RANGE = re.compile(r"[\u16A0-\u16FF]")


def ensure_repo_cloned(repo_url: str, target_dir: Path) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists() and any(target_dir.iterdir()):
        return
    subprocess.run(["git", "clone", "--quiet", repo_url, str(target_dir)], check=True)


def _candidate_text_files(root: Path) -> Iterable[Path]:
    exts = {".txt", ".md", ".rst", ".json"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and ".git" not in p.parts:
            yield p


def parse_pages(repo_root: Path) -> List[Page]:
    pages: List[Page] = []
    for path in _candidate_text_files(repo_root):
        text = path.read_text(encoding="utf-8", errors="ignore")
        runes = rune_stream_from_text(text)
        if not runes and not RUNIC_RANGE.search(text):
            continue
        rune_objs = [Rune(glyph=r, index=i, gp_value=rune_to_gp(r)) for i, r in enumerate(runes)]
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        pages.append(Page(page_id=path.stem, source_path=str(path), lines=lines, runes=rune_objs))
    return pages


def persist_pages(pages: List[Page], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for page in pages:
        payload = {
            "page_id": page.page_id,
            "source_path": page.source_path,
            "lines": page.lines,
            "runes": [{"glyph": r.glyph, "index": r.index, "gp_value": r.gp_value} for r in page.runes],
        }
        (out_dir / f"{page.page_id}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
