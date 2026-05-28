"""Document analysis tools.

Extracts text and structure from user-provided documents so the agent can
summarize, critique, answer questions, or turn them into tasks.
"""

from __future__ import annotations

import re
import datetime
import ipaddress
import socket
import subprocess
import tempfile
import urllib.parse
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET

import requests

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.utils import clip_text, safe_relpath


DEFAULT_MAX_CHARS = 30_000
DEFAULT_MAX_PAGES = 25
DEFAULT_MAX_SLIDES = 80
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_XML_PART_BYTES = 10 * 1024 * 1024
MAX_ARCHIVE_FILES = 80
MAX_ARCHIVE_MEMBER_BYTES = 50 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 150 * 1024 * 1024

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl",
    ".xml", ".html", ".htm", ".log", ".py", ".js", ".ts", ".css",
}

ANALYSIS_TYPES = {"summary", "critique", "extract_tasks", "answer_question", "raw"}
ARCHIVE_EXTENSIONS = {".zip"}
DOCUMENT_EXTENSIONS = {".pdf", ".pptx", ".docx", ".ppt"} | TEXT_EXTENSIONS | ARCHIVE_EXTENSIONS


@dataclass
class ExtractedDocument:
    kind: str
    metadata: Dict[str, str] = field(default_factory=dict)
    sections: List[Tuple[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _clean_limit(value: int, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(value)
    except Exception:
        return default
    return max(minimum, min(value, maximum))


def _normalize_analysis_type(value: str) -> str:
    value = (value or "summary").strip().lower()
    if value not in ANALYSIS_TYPES:
        return "summary"
    return value


def _resolve_document_path(ctx: ToolContext, path: str, source: str) -> Path:
    if not path or not isinstance(path, str):
        raise ValueError("path must be a non-empty string")

    source = (source or "drive").strip().lower()
    if source not in {"drive", "repo"}:
        raise ValueError("source must be 'drive' or 'repo'")

    if source == "repo" and str(ctx.user_role or "user").lower() != "admin":
        raise PermissionError("source='repo' is admin-only in multi-user mode")

    root = ctx.repo_dir if source == "repo" else ctx.drive_root
    root = root.resolve()
    resolved = (root / safe_relpath(path)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError("Path traversal is not allowed")

    if not resolved.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    if not resolved.is_file():
        raise ValueError(f"Document path is not a file: {path}")
    if resolved.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"Document is too large: {resolved.stat().st_size} bytes")
    return resolved


def _safe_output_path(root: Path, rel: str) -> Path:
    rel = safe_relpath(rel)
    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise ValueError("Path traversal is not allowed")
    return path


def _safe_filename(name: str, fallback: str = "download") -> str:
    name = str(name or "").replace("\\", "_").replace("/", "_").replace(":", "_")
    name = "".join(ch for ch in name if ord(ch) >= 32)
    name = re.sub(r"\s+", " ", name).strip(" .")
    if not name:
        name = fallback
    if len(name) > 180:
        suffix = Path(name).suffix[:20]
        stem = Path(name).stem[:140].strip(" .") or fallback
        name = stem + suffix
    return name


def _zip_member_safe_name(name: str) -> str:
    parts = []
    for part in Path(str(name).replace("\\", "/")).parts:
        if part in ("", ".", "..") or part.startswith("/"):
            continue
        parts.append(_safe_filename(part, fallback="entry"))
    return "/".join(parts)


def _read_plain_text(path: Path) -> ExtractedDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    return ExtractedDocument(
        kind="text",
        metadata={"encoding": "utf-8"},
        sections=[("Text", text)],
    )


def _try_import_pdf_reader():
    try:
        from pypdf import PdfReader  # type: ignore

        return PdfReader, "pypdf"
    except Exception:
        pass
    try:
        from PyPDF2 import PdfReader  # type: ignore

        return PdfReader, "PyPDF2"
    except Exception:
        return None, ""


def _extract_pdf_with_python(path: Path, max_pages: int) -> ExtractedDocument | None:
    PdfReader, library = _try_import_pdf_reader()
    if PdfReader is None:
        return None

    warnings: List[str] = []
    reader = PdfReader(str(path))
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception:
            warnings.append("PDF is encrypted and could not be decrypted with an empty password.")

    total_pages = len(reader.pages)
    sections: List[Tuple[str, str]] = []
    for idx in range(1, min(total_pages, max_pages) + 1):
        page = reader.pages[idx - 1]
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            text = f"[Could not extract page text: {type(exc).__name__}: {exc}]"
        sections.append((f"Page {idx}", text.strip() or "[No extractable text]"))

    if total_pages > max_pages:
        warnings.append(f"Only the first {max_pages} of {total_pages} pages were extracted.")

    return ExtractedDocument(
        kind="pdf",
        metadata={"pages": str(total_pages), "extractor": library},
        sections=sections,
        warnings=warnings,
    )


def _extract_pdf_with_pdftotext(path: Path, max_pages: int) -> ExtractedDocument | None:
    try:
        proc = subprocess.run(
            ["pdftotext", "-layout", "-f", "1", "-l", str(max_pages), str(path), "-"],
            text=True,
            capture_output=True,
            timeout=45,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if proc.returncode != 0:
        return ExtractedDocument(
            kind="pdf",
            metadata={"extractor": "pdftotext"},
            warnings=[proc.stderr.strip() or "pdftotext failed"],
        )

    return ExtractedDocument(
        kind="pdf",
        metadata={"extractor": "pdftotext", "pages": f"first {max_pages}"},
        sections=[("PDF text", proc.stdout.strip() or "[No extractable text]")],
    )


def _extract_pdf(path: Path, max_pages: int) -> ExtractedDocument:
    extracted = _extract_pdf_with_python(path, max_pages)
    if extracted is not None:
        return extracted

    extracted = _extract_pdf_with_pdftotext(path, max_pages)
    if extracted is not None:
        return extracted

    return ExtractedDocument(
        kind="pdf",
        warnings=[
            "No PDF extractor is available. Install pypdf or pdftotext to analyze PDF files.",
        ],
    )


def _paragraphs_from_xml(xml_bytes: bytes) -> List[str]:
    root = ET.fromstring(xml_bytes)
    paragraphs: List[str] = []
    for para in root.iter():
        if not para.tag.endswith("}p"):
            continue
        parts = []
        for node in para.iter():
            if node.tag.endswith("}t") and node.text:
                parts.append(node.text)
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def _read_zip_xml(zf: zipfile.ZipFile, name: str) -> bytes:
    info = zf.getinfo(name)
    if info.file_size > MAX_XML_PART_BYTES:
        raise ValueError(f"XML part is too large: {name} ({info.file_size} bytes)")
    return zf.read(name)


def _number_from_name(name: str) -> int:
    match = re.search(r"(\d+)", name)
    if not match:
        return 0
    return int(match.group(1))


def _extract_pptx(path: Path, max_slides: int) -> ExtractedDocument:
    warnings: List[str] = []
    sections: List[Tuple[str, str]] = []
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        slides = sorted(
            [name for name in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)],
            key=_number_from_name,
        )
        notes = {
            _number_from_name(name): name
            for name in names
            if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", name)
        }

        for idx, slide_name in enumerate(slides[:max_slides], start=1):
            slide_num = _number_from_name(slide_name)
            try:
                slide_lines = _paragraphs_from_xml(_read_zip_xml(zf, slide_name))
            except Exception as exc:
                slide_lines = [f"[Could not extract slide text: {type(exc).__name__}: {exc}]"]

            parts = []
            if slide_lines:
                parts.append("\n".join(slide_lines))
            else:
                parts.append("[No extractable slide text]")

            notes_name = notes.get(slide_num)
            if notes_name:
                try:
                    note_lines = _paragraphs_from_xml(_read_zip_xml(zf, notes_name))
                except Exception as exc:
                    note_lines = [f"[Could not extract speaker notes: {type(exc).__name__}: {exc}]"]
                if note_lines:
                    parts.append("Speaker notes:\n" + "\n".join(note_lines))

            sections.append((f"Slide {idx}", "\n\n".join(parts)))

    if len(slides) > max_slides:
        warnings.append(f"Only the first {max_slides} of {len(slides)} slides were extracted.")

    return ExtractedDocument(
        kind="pptx",
        metadata={"slides": str(len(slides)), "extractor": "zip+xml"},
        sections=sections,
        warnings=warnings,
    )


def _extract_docx(path: Path) -> ExtractedDocument:
    with zipfile.ZipFile(path) as zf:
        try:
            paragraphs = _paragraphs_from_xml(_read_zip_xml(zf, "word/document.xml"))
        except KeyError:
            paragraphs = []
    return ExtractedDocument(
        kind="docx",
        metadata={"extractor": "zip+xml"},
        sections=[("Document text", "\n".join(paragraphs) or "[No extractable text]")],
    )


def _extract_zip(path: Path, max_pages: int, max_slides: int, max_archive_files: int) -> ExtractedDocument:
    warnings: List[str] = []
    sections: List[Tuple[str, str]] = []
    analyzed = 0
    skipped = 0
    total_uncompressed = 0
    listed: List[str] = []
    scan_infos: List[zipfile.ZipInfo] = []

    with zipfile.ZipFile(path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        for info in infos:
            safe_name = _zip_member_safe_name(info.filename)
            if not safe_name:
                skipped += 1
                warnings.append(f"Skipped unsafe archive entry: {info.filename}")
                continue
            total_uncompressed += int(info.file_size or 0)
            if total_uncompressed > MAX_ARCHIVE_TOTAL_BYTES:
                warnings.append(f"Stopped after archive total exceeded {MAX_ARCHIVE_TOTAL_BYTES} bytes.")
                break
            listed.append(f"{safe_name} ({info.file_size} bytes)")
            scan_infos.append(info)

        with tempfile.TemporaryDirectory(prefix="ouroboros_zip_") as tmp:
            tmp_root = Path(tmp).resolve()
            for info in scan_infos:
                if analyzed >= max_archive_files:
                    warnings.append(f"Only the first {max_archive_files} supported archive files were analyzed.")
                    break
                safe_name = _zip_member_safe_name(info.filename)
                if not safe_name:
                    continue
                suffix = Path(safe_name).suffix.lower()
                if suffix in ARCHIVE_EXTENSIONS:
                    skipped += 1
                    warnings.append(f"Nested archive skipped: {safe_name}")
                    continue
                if suffix not in DOCUMENT_EXTENSIONS:
                    skipped += 1
                    continue
                if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                    skipped += 1
                    warnings.append(f"Skipped oversized archive entry: {safe_name} ({info.file_size} bytes)")
                    continue

                member_path = (tmp_root / safe_name).resolve()
                try:
                    member_path.relative_to(tmp_root)
                except ValueError:
                    skipped += 1
                    warnings.append(f"Skipped unsafe archive entry: {info.filename}")
                    continue
                member_path.parent.mkdir(parents=True, exist_ok=True)
                member_path.write_bytes(zf.read(info))

                extracted = _extract_document(
                    member_path,
                    max_pages=max_pages,
                    max_slides=max_slides,
                    max_archive_files=max_archive_files,
                    allow_archives=False,
                )
                analyzed += 1
                for warning in extracted.warnings:
                    warnings.append(f"{safe_name}: {warning}")
                if extracted.sections:
                    for title, text in extracted.sections:
                        sections.append((f"{safe_name} / {title}", text))
                else:
                    sections.append((safe_name, "[No extractable content]"))

    if listed:
        sections.insert(0, ("Archive contents", "\n".join(listed[:200])))
    if skipped:
        warnings.append(f"Skipped {skipped} unsupported, nested, or unsafe archive entries.")

    return ExtractedDocument(
        kind="zip",
        metadata={
            "entries": str(len(listed)),
            "analyzed_files": str(analyzed),
            "extractor": "zip",
        },
        sections=sections,
        warnings=warnings,
    )


def _extract_document(
    path: Path,
    max_pages: int,
    max_slides: int,
    max_archive_files: int = MAX_ARCHIVE_FILES,
    allow_archives: bool = True,
) -> ExtractedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path, max_pages)
    if suffix == ".pptx":
        return _extract_pptx(path, max_slides)
    if suffix == ".docx":
        return _extract_docx(path)
    if suffix == ".zip" and allow_archives:
        return _extract_zip(path, max_pages=max_pages, max_slides=max_slides, max_archive_files=max_archive_files)
    if suffix in TEXT_EXTENSIONS:
        return _read_plain_text(path)
    if suffix == ".ppt":
        return ExtractedDocument(
            kind="ppt",
            warnings=["Legacy .ppt files are not supported. Convert the file to .pptx first."],
        )
    return ExtractedDocument(
        kind=suffix.lstrip(".") or "unknown",
        warnings=[
            "Unsupported document type. Supported: PDF, ZIP, PPTX, DOCX, TXT, MD, CSV, JSON, HTML, XML, code files.",
        ],
    )


def _analysis_instruction(analysis_type: str, question: str) -> str:
    if analysis_type == "summary":
        return "Summarize the document: main thesis, key points, important details, and gaps."
    if analysis_type == "critique":
        return "Critique the document: strengths, weak claims, missing evidence, contradictions, and concrete improvements."
    if analysis_type == "extract_tasks":
        return "Extract actionable tasks, owners if mentioned, deadlines if mentioned, risks, and dependencies."
    if analysis_type == "answer_question":
        if question:
            return f"Answer this question using only the extracted document content: {question}"
        return "Answer the user's question using only the extracted document content."
    return "Return the extracted document content without additional interpretation."


def _format_result(
    path: Path,
    source: str,
    result: ExtractedDocument,
    analysis_type: str,
    question: str,
    max_chars: int,
) -> str:
    lines = [
        "# Document Analysis",
        "",
        f"- source: {source}",
        f"- file: {path.name}",
        f"- type: {result.kind}",
        f"- size_bytes: {path.stat().st_size}",
    ]
    for key, value in sorted(result.metadata.items()):
        lines.append(f"- {key}: {value}")
    if result.warnings:
        lines.append("- warnings:")
        for warning in result.warnings:
            lines.append(f"  - {warning}")

    lines.extend([
        "",
        "## Requested analysis",
        _analysis_instruction(analysis_type, question),
        "",
        "## Extracted content",
    ])

    if result.sections:
        for title, text in result.sections:
            lines.extend(["", f"### {title}", text.strip() or "[Empty]"])
    else:
        lines.append("[No extractable content]")

    raw = "\n".join(lines).strip() + "\n"
    clipped = clip_text(raw, max_chars)
    if clipped != raw:
        clipped += f"\n\n[Output clipped to {max_chars} characters. Increase max_chars or narrow max_pages/max_slides for more content.]\n"
    return clipped


def _validate_download_url(url: str) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(str(url or "").strip())
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http and https URLs are supported")
    if not parsed.hostname:
        raise ValueError("URL must include a hostname")
    host = parsed.hostname
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve host: {host}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            raise ValueError(f"Refusing to download from private or local address: {ip}")
    return parsed


def _filename_from_download(url: str, content_disposition: str, content_type: str) -> str:
    cd = content_disposition or ""
    match = re.search(r"filename\*=([^']*)''([^;]+)", cd, flags=re.I)
    if match:
        return _safe_filename(urllib.parse.unquote(match.group(2)))
    match = re.search(r'filename="?([^";]+)"?', cd, flags=re.I)
    if match:
        return _safe_filename(match.group(1))

    parsed = urllib.parse.urlparse(url)
    name = Path(urllib.parse.unquote(parsed.path or "")).name
    if not name:
        name = "download"
    safe = _safe_filename(name)
    if "." not in safe:
        if "zip" in content_type:
            safe += ".zip"
        elif "pdf" in content_type:
            safe += ".pdf"
    return safe


def _dedupe_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for idx in range(2, 10_000):
        candidate = path.with_name(f"{stem}_{idx}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find free filename for {path.name}")


def _download_url_to_drive(
    ctx: ToolContext,
    url: str,
    output_path: str = "",
    max_bytes: int = MAX_FILE_BYTES,
) -> str:
    parsed = _validate_download_url(url)
    max_bytes = _clean_limit(max_bytes, MAX_FILE_BYTES, 1_000, MAX_FILE_BYTES)
    with requests.get(parsed.geturl(), stream=True, allow_redirects=True, timeout=(10, 60)) as resp:
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        filename = _filename_from_download(parsed.geturl(), resp.headers.get("content-disposition", ""), content_type)
        if output_path and output_path.strip():
            rel = safe_relpath(output_path)
            if rel.endswith("/"):
                rel = rel + filename
            elif not Path(rel).suffix:
                rel = str(Path(rel) / filename)
        else:
            day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
            rel = str(Path("downloads") / day / filename)

        out_path = _dedupe_path(_safe_output_path(ctx.drive_root, rel))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with out_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    f.close()
                    out_path.unlink(missing_ok=True)
                    raise ValueError(f"Download exceeded max_bytes={max_bytes}")
                f.write(chunk)

    rel_saved = str(out_path.relative_to(ctx.drive_root.resolve()))
    return (
        "OK: downloaded file\n"
        f"- path: {rel_saved}\n"
        f"- size_bytes: {total}\n"
        f"- content_type: {content_type or 'unknown'}\n"
        "Next: call analyze_document(path='<path>', source='drive') to inspect it."
    )


def _extract_archive(
    ctx: ToolContext,
    path: str,
    source: str = "drive",
    output_dir: str = "",
    max_files: int = MAX_ARCHIVE_FILES,
    overwrite: bool = False,
) -> str:
    archive_path = _resolve_document_path(ctx, path, source)
    if archive_path.suffix.lower() != ".zip":
        raise ValueError("Only .zip archives are supported")
    max_files = _clean_limit(max_files, MAX_ARCHIVE_FILES, 1, 500)

    base_name = _safe_filename(archive_path.stem, fallback="archive")
    rel_dir = safe_relpath(output_dir) if output_dir else str(Path("extracted") / base_name)
    out_root = _safe_output_path(ctx.drive_root, rel_dir)
    if out_root.exists() and not overwrite:
        out_root = _dedupe_path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    extracted: List[str] = []
    skipped: List[str] = []
    total = 0
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if len(extracted) >= max_files:
                skipped.append(f"... stopped after max_files={max_files}")
                break
            safe_name = _zip_member_safe_name(info.filename)
            if not safe_name:
                skipped.append(f"unsafe: {info.filename}")
                continue
            if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                skipped.append(f"oversized: {safe_name}")
                continue
            total += int(info.file_size or 0)
            if total > MAX_ARCHIVE_TOTAL_BYTES:
                skipped.append(f"archive total exceeded {MAX_ARCHIVE_TOTAL_BYTES} bytes")
                break
            target = (out_root / safe_name).resolve()
            try:
                target.relative_to(out_root.resolve())
            except ValueError:
                skipped.append(f"unsafe: {info.filename}")
                continue
            if target.exists() and not overwrite:
                target = _dedupe_path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
            extracted.append(str(target.relative_to(ctx.drive_root.resolve())))

    lines = [
        "OK: archive extracted",
        f"- archive: {path}",
        f"- output_dir: {out_root.relative_to(ctx.drive_root.resolve())}",
        f"- extracted_files: {len(extracted)}",
    ]
    if extracted:
        lines.append("")
        lines.append("## Extracted files")
        lines.extend(f"- {item}" for item in extracted[:100])
    if skipped:
        lines.append("")
        lines.append("## Skipped")
        lines.extend(f"- {item}" for item in skipped[:50])
    return "\n".join(lines)


def _analyze_document(
    ctx: ToolContext,
    path: str,
    source: str = "drive",
    analysis_type: str = "summary",
    question: str = "",
    max_chars: int = DEFAULT_MAX_CHARS,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_slides: int = DEFAULT_MAX_SLIDES,
    max_archive_files: int = MAX_ARCHIVE_FILES,
) -> str:
    analysis_type = _normalize_analysis_type(analysis_type)
    source = (source or "drive").strip().lower()
    max_chars = _clean_limit(max_chars, DEFAULT_MAX_CHARS, 2_000, 100_000)
    max_pages = _clean_limit(max_pages, DEFAULT_MAX_PAGES, 1, 200)
    max_slides = _clean_limit(max_slides, DEFAULT_MAX_SLIDES, 1, 300)
    max_archive_files = _clean_limit(max_archive_files, MAX_ARCHIVE_FILES, 1, 200)

    document_path = _resolve_document_path(ctx, path, source)
    extracted = _extract_document(
        document_path,
        max_pages=max_pages,
        max_slides=max_slides,
        max_archive_files=max_archive_files,
    )
    return _format_result(document_path, source, extracted, analysis_type, question or "", max_chars)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("analyze_document", {
            "name": "analyze_document",
            "description": (
                "Extract text and structure from PDF, PPTX, DOCX, and text-like files for analysis. "
                "Use it before summarizing documents, critiquing presentations, answering questions "
                "about uploaded files, or extracting action items. Default source is the user's Drive workspace."
            ),
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "Document path relative to the selected source root."},
                "source": {
                    "type": "string",
                    "enum": ["drive", "repo"],
                    "default": "drive",
                    "description": "Read from user Drive workspace by default. Repo source is admin-only.",
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["summary", "critique", "extract_tasks", "answer_question", "raw"],
                    "default": "summary",
                    "description": "How the caller intends to analyze the extracted content.",
                },
                "question": {
                    "type": "string",
                    "description": "Question to answer when analysis_type is answer_question.",
                },
                "max_chars": {
                    "type": "integer",
                    "default": DEFAULT_MAX_CHARS,
                    "description": "Maximum characters returned to the LLM.",
                },
                "max_pages": {
                    "type": "integer",
                    "default": DEFAULT_MAX_PAGES,
                    "description": "Maximum PDF pages to extract.",
                },
                "max_slides": {
                    "type": "integer",
                    "default": DEFAULT_MAX_SLIDES,
                    "description": "Maximum PPTX slides to extract.",
                },
                "max_archive_files": {
                    "type": "integer",
                    "default": MAX_ARCHIVE_FILES,
                    "description": "Maximum supported files to analyze inside a ZIP archive.",
                },
            }, "required": ["path"]},
        }, _analyze_document, timeout_sec=60),
        ToolEntry("extract_archive", {
            "name": "extract_archive",
            "description": (
                "Safely extract a ZIP archive from the user's Drive workspace. "
                "Use when the user needs files unpacked for later inspection or when a ZIP contains multiple reports."
            ),
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "ZIP path relative to the selected source root."},
                "source": {
                    "type": "string",
                    "enum": ["drive", "repo"],
                    "default": "drive",
                    "description": "Read archive from user Drive by default. Repo source is admin-only.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Optional output directory relative to user Drive. Defaults to extracted/<archive-name>.",
                },
                "max_files": {
                    "type": "integer",
                    "default": MAX_ARCHIVE_FILES,
                    "description": "Maximum archive entries to extract.",
                },
                "overwrite": {
                    "type": "boolean",
                    "default": False,
                    "description": "Overwrite existing extracted files instead of creating deduplicated names.",
                },
            }, "required": ["path"]},
        }, _extract_archive, timeout_sec=60),
        ToolEntry("download_url_to_drive", {
            "name": "download_url_to_drive",
            "description": (
                "Download a public HTTP/HTTPS file URL into the user's Drive workspace. "
                "Use for direct PDF/ZIP/DOCX/PPTX links that browser automation cannot read because they start a download."
            ),
            "parameters": {"type": "object", "properties": {
                "url": {"type": "string", "description": "Public HTTP/HTTPS URL to download."},
                "output_path": {
                    "type": "string",
                    "description": "Optional relative output path or directory in user Drive. Defaults to downloads/YYYY-MM-DD/<filename>.",
                },
                "max_bytes": {
                    "type": "integer",
                    "default": MAX_FILE_BYTES,
                    "description": "Maximum download size in bytes.",
                },
            }, "required": ["url"]},
        }, _download_url_to_drive, timeout_sec=90),
    ]
