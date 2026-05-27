"""Document analysis tools.

Extracts text and structure from user-provided documents so the agent can
summarize, critique, answer questions, or turn them into tasks.
"""

from __future__ import annotations

import re
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.utils import clip_text, safe_relpath


DEFAULT_MAX_CHARS = 30_000
DEFAULT_MAX_PAGES = 25
DEFAULT_MAX_SLIDES = 80
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_XML_PART_BYTES = 10 * 1024 * 1024

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl",
    ".xml", ".html", ".htm", ".log", ".py", ".js", ".ts", ".css",
}

ANALYSIS_TYPES = {"summary", "critique", "extract_tasks", "answer_question", "raw"}


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


def _extract_document(path: Path, max_pages: int, max_slides: int) -> ExtractedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path, max_pages)
    if suffix == ".pptx":
        return _extract_pptx(path, max_slides)
    if suffix == ".docx":
        return _extract_docx(path)
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
            "Unsupported document type. Supported: PDF, PPTX, DOCX, TXT, MD, CSV, JSON, HTML, XML, code files.",
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


def _analyze_document(
    ctx: ToolContext,
    path: str,
    source: str = "drive",
    analysis_type: str = "summary",
    question: str = "",
    max_chars: int = DEFAULT_MAX_CHARS,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_slides: int = DEFAULT_MAX_SLIDES,
) -> str:
    analysis_type = _normalize_analysis_type(analysis_type)
    source = (source or "drive").strip().lower()
    max_chars = _clean_limit(max_chars, DEFAULT_MAX_CHARS, 2_000, 100_000)
    max_pages = _clean_limit(max_pages, DEFAULT_MAX_PAGES, 1, 200)
    max_slides = _clean_limit(max_slides, DEFAULT_MAX_SLIDES, 1, 300)

    document_path = _resolve_document_path(ctx, path, source)
    extracted = _extract_document(document_path, max_pages=max_pages, max_slides=max_slides)
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
            }, "required": ["path"]},
        }, _analyze_document, timeout_sec=60),
    ]
