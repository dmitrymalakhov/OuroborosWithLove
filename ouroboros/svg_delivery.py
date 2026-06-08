"""Helpers for turning final-response SVG code blocks into Telegram files."""

from __future__ import annotations

import datetime as _dt
import pathlib
import re
from dataclasses import dataclass
from typing import List, Tuple


SVG_MIME_TYPE = "image/svg+xml"
MAX_AUTO_SVG_FILES = 30

_FENCED_BLOCK_RE = re.compile(r"```(?P<lang>[^\n`]*)\n(?P<body>.*?)```", re.DOTALL)
_SVG_RE = re.compile(r"(?is)(<svg\b.*?</svg>)")
_SVG_NAME_RE = re.compile(r"(?iu)([\w .()_-]+?\.svg)")
_INVALID_FILENAME_CHARS_RE = re.compile(r'[\\/:\0<>|?*"]+')


@dataclass(frozen=True)
class SvgAttachment:
    """A generated SVG file ready to be sent as a Telegram document."""

    rel_path: str
    filename: str
    size_bytes: int


def extract_svg_attachments(
    text: str,
    drive_root: pathlib.Path,
    task_id: str = "",
) -> Tuple[str, List[SvgAttachment]]:
    """Write SVG fenced code blocks from a final answer to files.

    Returns the answer with those code blocks replaced by short attachment notes
    and metadata for send_document events.
    """
    if not isinstance(text, str) or "```" not in text or "<svg" not in text.lower():
        return text, []

    attachments: List[SvgAttachment] = []
    used_filenames: set[str] = set()
    parts: List[str] = []
    last_end = 0

    output_dir = drive_root / "exports" / "svg" / _safe_task_dir(task_id)

    for match in _FENCED_BLOCK_RE.finditer(text):
        parts.append(text[last_end:match.start()])
        last_end = match.end()

        svg_text = _extract_svg(match.group("body"))
        if not svg_text or len(attachments) >= MAX_AUTO_SVG_FILES:
            parts.append(match.group(0))
            continue

        index = len(attachments) + 1
        filename = _unique_filename(
            _filename_from_prefix(text[:match.start()], index),
            used_filenames,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = _dedupe_path(output_dir / filename)
        file_path.write_text(svg_text.rstrip() + "\n", encoding="utf-8")
        rel_path = file_path.relative_to(drive_root).as_posix()
        attachments.append(
            SvgAttachment(
                rel_path=rel_path,
                filename=file_path.name,
                size_bytes=file_path.stat().st_size,
            )
        )
        parts.append(f"Файл отправлен вложением: `{file_path.name}`")

    parts.append(text[last_end:])
    if not attachments:
        return text, []
    return _compact_attachment_text("".join(parts)), attachments


def _extract_svg(block_body: str) -> str:
    match = _SVG_RE.search((block_body or "").strip())
    if not match:
        return ""
    return match.group(1).strip()


def _filename_from_prefix(prefix: str, index: int) -> str:
    for line in reversed([line.strip() for line in prefix.splitlines() if line.strip()][-8:]):
        cleaned = line.strip("`*_#-: \t")
        match = _SVG_NAME_RE.search(cleaned)
        if match:
            return _sanitize_svg_filename(match.group(1), index)
    return f"generated-svg-{index}.svg"


def _sanitize_svg_filename(name: str, index: int) -> str:
    cleaned = pathlib.PurePosixPath(str(name or "").replace("\\", "/")).name
    cleaned = cleaned.strip().strip("`*_[]() ")
    cleaned = _INVALID_FILENAME_CHARS_RE.sub("_", cleaned).strip(" .")
    if not cleaned:
        cleaned = f"generated-svg-{index}.svg"
    if not cleaned.lower().endswith(".svg"):
        cleaned = f"{cleaned}.svg"
    if cleaned.lower() == ".svg":
        cleaned = f"generated-svg-{index}.svg"
    if len(cleaned) > 100:
        cleaned = f"{cleaned[:96].rstrip(' .')}.svg"
    return cleaned


def _unique_filename(filename: str, used: set[str]) -> str:
    base = pathlib.PurePosixPath(filename).stem
    suffix = pathlib.PurePosixPath(filename).suffix or ".svg"
    candidate = filename
    n = 2
    while candidate.lower() in used:
        candidate = f"{base}-{n}{suffix}"
        n += 1
    used.add(candidate.lower())
    return candidate


def _dedupe_path(path: pathlib.Path) -> pathlib.Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    n = 2
    while True:
        candidate = path.with_name(f"{stem}-{n}{suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def _safe_task_dir(task_id: str) -> str:
    raw = str(task_id or "").strip()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-._")
    if safe:
        return safe[:80]
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _compact_attachment_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    compacted: List[str] = []
    blank = False
    for line in lines:
        if line.strip():
            compacted.append(line)
            blank = False
        elif not blank:
            compacted.append("")
            blank = True
    return "\n".join(compacted).strip()
