"""
Liber Primus data loader.

Workflow
--------
1. Auto-clone (or pull) the LP GitHub repository into *lp_data_dir*.
2. Scan for files that contain rune Unicode characters.
3. Parse each file into a :class:`~src.data.page.Page` object.
4. Fall back to built-in sample text when no external data is found.

The public repo ``rtkd/iddqd`` contains the LP transcription as plain
Unicode ``.txt`` files, one per page/section.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .page import Page
from .rune import RUNE_BY_UNICODE

log = logging.getLogger(__name__)

# Characters that unambiguously identify rune content
_RUNE_SET = frozenset(RUNE_BY_UNICODE.keys())

# ---------------------------------------------------------------------------
# Minimal built-in LP content (publicly known solved welcome segment)
# Used as an offline fallback so the engine can always run.
# ---------------------------------------------------------------------------
_FALLBACK_PAGES: Dict[str, str] = {
    "welcome": (
        "\u16A9\u16BE \u16A6\u16D6\u16D6\u16BE \u16A0\u16C1\u16B1\u16CB\u16CF "
        "\u16C1\u16CB \u16A6\u16D6\u16D6\u16BE \u16C1\u16BE\u16BE\u16A9\u16C3\u16BE\u16BE\u16CF\u16C1 "
        "\u16C1\u16CF \u16C1\u16CB \u16A6\u16D6\u16D6\u16BE \u16BE\u16A9\u16C3\u16D6\u16D6\u16CF\u16C1"
    ),
    "wisdom": (
        "\u16C1\u16BE \u16CF\u16BB\u16D6\u16CF \u16A0\u16C1\u16B1\u16CB\u16CF\u16BE "
        "\u16CF\u16BB\u16D6\u16CF \u16C1\u16CB\u16BE\u16CF\u16C1 \u16A6\u16D6\u16D6\u16BE"
    ),
}


def _has_runes(text: str) -> bool:
    """Return True if *text* contains at least one rune character."""
    return any(ch in _RUNE_SET for ch in text)


def _parse_rune_file(filepath: Path, page_id: str) -> Optional[Page]:
    """Parse a single rune-containing file into a :class:`Page`."""
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.warning("Cannot read %s: %s", filepath, exc)
        return None

    if not _has_runes(text):
        return None

    all_lines = text.splitlines()
    rune_lines = [ln for ln in all_lines if _has_runes(ln)]
    if not rune_lines:
        return None

    raw_runes = "\n".join(rune_lines)
    return Page(
        page_id=page_id,
        source_file=str(filepath),
        raw_runes=raw_runes,
        lines=rune_lines,
        metadata={
            "source": str(filepath),
            "total_lines": len(all_lines),
            "rune_lines": len(rune_lines),
        },
    )


def _create_fallback_pages() -> List[Page]:
    """Create minimal Page objects from the built-in fallback texts."""
    pages = []
    for pid, text in _FALLBACK_PAGES.items():
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            lines = [text]
        pages.append(
            Page(
                page_id=f"fallback_{pid}",
                source_file="builtin",
                raw_runes=text,
                lines=lines,
                metadata={"source": "builtin_fallback"},
            )
        )
    return pages


def clone_lp_repo(repo_url: str, target_dir: str) -> bool:
    """
    Clone or update the Liber Primus repository.

    Parameters
    ----------
    repo_url : str
        HTTPS URL of the upstream git repository.
    target_dir : str
        Local path where the repo should be checked out.

    Returns
    -------
    bool
        ``True`` if the directory is ready to scan, ``False`` on failure.
    """
    target = Path(target_dir)

    # Already cloned – attempt a fast-forward pull.
    if target.exists() and (target / ".git").exists():
        log.info("LP repo already present at %s – pulling updates.", target_dir)
        try:
            subprocess.run(
                ["git", "-C", str(target), "pull", "--ff-only"],
                check=True,
                capture_output=True,
                timeout=60,
            )
        except subprocess.CalledProcessError as exc:
            log.warning(
                "git pull failed (continuing with existing data): %s",
                exc.stderr.decode(errors="replace").strip(),
            )
        return True

    target.mkdir(parents=True, exist_ok=True)
    log.info("Cloning LP repo %s → %s", repo_url, target_dir)
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", repo_url, str(target)],
            check=True,
            capture_output=True,
            timeout=180,
        )
        log.info("Clone successful.")
        return True
    except subprocess.CalledProcessError as exc:
        log.error(
            "Clone failed: %s",
            exc.stderr.decode(errors="replace").strip(),
        )
        return False
    except FileNotFoundError:
        log.error("'git' executable not found; cannot clone LP repo.")
        return False


def find_rune_files(search_dir: str) -> List[Path]:
    """
    Recursively find files in *search_dir* that contain rune characters.

    Only common plain-text extensions are checked to avoid binary files.
    """
    search_path = Path(search_dir)
    if not search_path.exists():
        return []

    candidates: List[Path] = []
    for ext in ("*.txt", "*.md", "*.text", "*.utf8", "*.dat"):
        for fp in search_path.rglob(ext):
            try:
                # Peek at a small prefix to decide quickly.
                sample = fp.read_bytes()[:1024].decode("utf-8", errors="replace")
                if _has_runes(sample):
                    candidates.append(fp)
            except OSError:
                pass

    # De-duplicate (rglob can return duplicates on some platforms).
    seen = set()
    result = []
    for p in candidates:
        key = p.resolve()
        if key not in seen:
            seen.add(key)
            result.append(p)
    return sorted(result)


def load_pages(
    lp_data_dir: str,
    repo_url: str,
    max_pages: Optional[int] = None,
) -> List[Page]:
    """
    Load all Liber Primus pages, returning a list of :class:`Page` objects.

    Steps
    -----
    1. Attempt to clone / pull *repo_url* into *lp_data_dir*.
    2. Scan for rune-containing files.
    3. Parse each file into a Page.
    4. If nothing was found, fall back to built-in sample pages.

    Parameters
    ----------
    lp_data_dir : str
        Local directory for the cloned LP data.
    repo_url : str
        Remote git URL for the LP transcription repository.
    max_pages : int, optional
        Hard limit on the number of pages to load (useful for smoke tests).

    Returns
    -------
    List[Page]
        Pages sorted by page_id, with at least one entry guaranteed.
    """
    Path(lp_data_dir).mkdir(parents=True, exist_ok=True)

    # Best-effort clone / update.
    clone_ok = clone_lp_repo(repo_url, lp_data_dir)
    if not clone_ok:
        log.warning("Repo clone failed; scanning existing directory anyway.")

    rune_files = find_rune_files(lp_data_dir)
    log.info(
        "Found %d potential rune file(s) under %s.", len(rune_files), lp_data_dir
    )

    pages: List[Page] = []
    for fp in rune_files:
        page_id = fp.stem.lower().replace(" ", "_")
        page = _parse_rune_file(fp, page_id)
        if page is not None:
            pages.append(page)
        if max_pages is not None and len(pages) >= max_pages:
            break

    if not pages:
        log.warning(
            "No rune pages found in %s. Using built-in fallback pages.",
            lp_data_dir,
        )
        pages = _create_fallback_pages()

    pages.sort(key=lambda p: p.page_id)
    log.info(
        "Loaded %d page(s) with %d total runes.",
        len(pages),
        sum(p.num_runes for p in pages),
    )
    return pages
