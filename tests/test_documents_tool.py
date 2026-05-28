import pathlib
import zipfile

import pytest

from ouroboros.tools.documents import _analyze_document, _extract_archive, _parse_page_ranges
from ouroboros.tools.registry import ToolContext


def _ctx(
    repo_dir: pathlib.Path,
    drive_root: pathlib.Path,
    role: str = "admin",
    progress=None,
) -> ToolContext:
    repo_dir.mkdir(parents=True, exist_ok=True)
    drive_root.mkdir(parents=True, exist_ok=True)
    return ToolContext(
        repo_dir=repo_dir,
        drive_root=drive_root,
        user_role=role,
        emit_progress_fn=progress or (lambda _: None),
    )


def test_analyze_document_reads_text_from_drive(tmp_path):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    (drive / "brief.md").write_text("# Brief\n\nLaunch OpenAI support.\n", encoding="utf-8")

    result = _analyze_document(
        _ctx(repo, drive),
        path="brief.md",
        analysis_type="summary",
    )

    assert "# Document Analysis" in result
    assert "type: text" in result
    assert "Launch OpenAI support." in result
    assert "Summarize the document" in result


def test_analyze_document_extracts_pptx_slides_and_notes(tmp_path):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    pptx = drive / "deck.pptx"
    slide_xml = """
    <p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
           xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
      <p:cSld><p:spTree>
        <p:sp><p:txBody>
          <a:p><a:r><a:t>Quarterly Plan</a:t></a:r></a:p>
          <a:p><a:r><a:t>Ship document analysis</a:t></a:r></a:p>
        </p:txBody></p:sp>
      </p:spTree></p:cSld>
    </p:sld>
    """.strip()
    notes_xml = """
    <p:notes xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
             xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
      <p:cSld><p:spTree>
        <p:sp><p:txBody>
          <a:p><a:r><a:t>Mention multi-user mode.</a:t></a:r></a:p>
        </p:txBody></p:sp>
      </p:spTree></p:cSld>
    </p:notes>
    """.strip()
    with zipfile.ZipFile(pptx, "w") as zf:
        zf.writestr("ppt/slides/slide1.xml", slide_xml)
        zf.writestr("ppt/notesSlides/notesSlide1.xml", notes_xml)

    result = _analyze_document(
        _ctx(repo, drive),
        path="deck.pptx",
        analysis_type="critique",
    )

    assert "type: pptx" in result
    assert "Slide 1" in result
    assert "Quarterly Plan" in result
    assert "Ship document analysis" in result
    assert "Speaker notes" in result
    assert "Mention multi-user mode." in result
    assert "Critique the document" in result


def test_analyze_document_extracts_supported_files_inside_zip(tmp_path):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    archive = drive / "reports.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("report.md", "# Report\n\nImportant issuer disclosure.")
        zf.writestr("notes/info.txt", "SPV list and INN references")
        zf.writestr("image.bin", b"\x00\x01")

    result = _analyze_document(
        _ctx(repo, drive),
        path="reports.zip",
        analysis_type="answer_question",
        question="What does the archive contain?",
    )

    assert "type: zip" in result
    assert "Archive contents" in result
    assert "report.md / Text" in result
    assert "Important issuer disclosure." in result
    assert "notes/info.txt / Text" in result
    assert "SPV list and INN references" in result


def test_analyze_document_extracts_requested_pdf_page_ranges(tmp_path, monkeypatch):
    class FakePage:
        def __init__(self, number: int):
            self.number = number

        def extract_text(self):
            return f"Text from page {self.number}"

    class FakeReader:
        is_encrypted = False

        def __init__(self, _path: str):
            self.pages = [FakePage(idx) for idx in range(1, 11)]

    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    (drive / "report.pdf").write_bytes(b"%PDF fake")
    monkeypatch.setattr(
        "ouroboros.tools.documents._try_import_pdf_reader",
        lambda: (FakeReader, "fakepdf"),
    )

    result = _analyze_document(
        _ctx(repo, drive),
        path="report.pdf",
        analysis_type="answer_question",
        question="What is on the target pages?",
        page_ranges="3-4,7",
        max_pages=10,
    )

    assert "pages_extracted: 3-4,7" in result
    assert "Page 3" in result
    assert "Text from page 3" in result
    assert "Page 4" in result
    assert "Page 7" in result
    assert "Page 1" not in result
    assert "Text from page 2" not in result


def test_parse_page_ranges_caps_and_skips_out_of_range():
    pages, warnings = _parse_page_ranges("2-4,9,12", total_pages=10, max_pages=3)

    assert pages == [2, 3, 4]
    assert any("Only the first 3 selected pages" in warning for warning in warnings)
    assert any("Skipped pages outside document length: 12" in warning for warning in warnings)


def test_analyze_document_emits_progress_for_zip(tmp_path):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    archive = drive / "reports.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("report.md", "# Report\n\nImportant issuer disclosure.")

    progress = []
    result = _analyze_document(
        _ctx(repo, drive, progress=progress.append),
        path="reports.zip",
        analysis_type="answer_question",
        question="What does the archive contain?",
    )

    assert "type: zip" in result
    assert any("Открыл файл" in item for item in progress)
    assert any("Открыл ZIP" in item for item in progress)
    assert any("Нашёл в ZIP" in item for item in progress)
    assert any("заверш" in item.lower() for item in progress)


def test_extract_archive_writes_safe_files_to_drive(tmp_path):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    archive = drive / "reports.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "no escape")
        zf.writestr("nested/report.md", "safe")

    result = _extract_archive(
        _ctx(repo, drive),
        path="reports.zip",
        output_dir="unzipped/reports",
    )

    assert "OK: archive extracted" in result
    assert (drive / "unzipped" / "reports" / "escape.txt").exists()
    assert (drive / "unzipped" / "reports" / "nested" / "report.md").read_text() == "safe"


def test_user_cannot_read_repo_documents_through_document_tool(tmp_path):
    repo = tmp_path / "repo"
    drive = tmp_path / "drive"
    repo.mkdir()
    (repo / "secret.md").write_text("admin-only", encoding="utf-8")

    with pytest.raises(PermissionError):
        _analyze_document(
            _ctx(repo, drive, role="user"),
            path="secret.md",
            source="repo",
        )
