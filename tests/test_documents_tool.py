import pathlib
import zipfile

import pytest

from ouroboros.tools.documents import _analyze_document, _extract_archive, _index_document, _parse_page_ranges, _search_document
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


def test_analyze_document_ocr_reads_image_from_drive(tmp_path, monkeypatch):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    image_bytes = b"\xff\xd8financial forecast photo"
    (drive / "forecast.jpg").write_bytes(image_bytes)
    captured = {}

    class FakeClient:
        def vision_query(self, **kwargs):
            captured.update(kwargs)
            return ("| Metric | 2026 |\n| Revenue | 100 |", {"prompt_tokens": 10, "completion_tokens": 5})

    monkeypatch.setattr("ouroboros.tools.documents._get_llm_client", lambda: FakeClient())
    monkeypatch.setattr("ouroboros.tools.documents._get_image_ocr_model", lambda: "vision-test")

    result = _analyze_document(
        _ctx(repo, drive),
        path="forecast.jpg",
        analysis_type="answer_question",
        question="Проверь прогноз",
    )

    assert "type: jpg" in result
    assert "extractor: vlm_ocr" in result
    assert "mime_type: image/jpeg" in result
    assert "Image OCR" in result
    assert "Revenue | 100" in result
    assert "Answer this question using only the extracted document content: Проверь прогноз" in result
    assert captured["model"] == "vision-test"
    assert captured["images"][0]["mime"] == "image/jpeg"


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


def test_analyze_document_adds_pdf_outline_navigation_map(tmp_path, monkeypatch):
    class FakePage:
        def __init__(self, number: int):
            self.number = number

        def extract_text(self):
            return f"Body text from page {self.number}"

    class FakeDestination:
        def __init__(self, title: str, page_index: int):
            self.title = title
            self.page_index = page_index

    class FakeReader:
        is_encrypted = False
        outline = [
            FakeDestination("Executive summary", 0),
            [FakeDestination("Credit risk factors", 11)],
        ]

        def __init__(self, _path: str):
            self.pages = [FakePage(idx) for idx in range(1, 21)]

        def get_destination_page_number(self, item):
            return item.page_index

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
        question="Where are the risk factors?",
        max_pages=3,
    )

    assert "toc_source: outline" in result
    assert "PDF navigation map" in result
    assert "p. 12: Credit risk factors" in result
    assert "call analyze_document again with page_ranges" in result


def test_analyze_document_detects_text_table_of_contents_in_pdf(tmp_path, monkeypatch):
    class FakePage:
        def __init__(self, text: str):
            self.text = text

        def extract_text(self):
            return self.text

    class FakeReader:
        is_encrypted = False

        def __init__(self, _path: str):
            toc = (
                "Содержание\n"
                "1. Введение ........ 3\n"
                "2. Риски проекта ........ 17\n"
                "3. Финансовая модель ........ 25\n"
                "4. Приложения ........ 38"
            )
            self.pages = [FakePage(toc)] + [FakePage(f"Body page {idx}") for idx in range(2, 41)]

    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    (drive / "memorandum.pdf").write_bytes(b"%PDF fake")
    monkeypatch.setattr(
        "ouroboros.tools.documents._try_import_pdf_reader",
        lambda: (FakeReader, "fakepdf"),
    )

    result = _analyze_document(
        _ctx(repo, drive),
        path="memorandum.pdf",
        analysis_type="answer_question",
        question="Есть ли в документе риски проекта?",
        max_pages=5,
    )

    assert "toc_source: front_matter_text" in result
    assert "toc_entries: 4" in result
    assert "ref p. 17: 2. Риски проекта" in result
    assert "may need a small offset from PDF page numbers" in result


def test_search_document_finds_pdf_matches_and_suggests_page_ranges(tmp_path, monkeypatch):
    class FakePage:
        def __init__(self, text: str):
            self.text = text

        def extract_text(self):
            return self.text

    class FakeDestination:
        def __init__(self, title: str, page_index: int):
            self.title = title
            self.page_index = page_index

    class FakeReader:
        is_encrypted = False
        outline = [FakeDestination("Финансовые ковенанты", 9)]

        def __init__(self, _path: str):
            pages = [FakePage(f"Body page {idx}") for idx in range(1, 21)]
            pages[9] = FakePage("Финансовые ковенанты: DSCR должен быть не ниже 1.20.")
            self.pages = pages

        def get_destination_page_number(self, item):
            return item.page_index

    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    (drive / "finance.pdf").write_bytes(b"%PDF fake")
    monkeypatch.setattr(
        "ouroboros.tools.documents._try_import_pdf_reader",
        lambda: (FakeReader, "fakepdf"),
    )

    result = _search_document(
        _ctx(repo, drive),
        path="finance.pdf",
        query="ковенанты DSCR",
        max_pages=20,
    )

    assert "# Document Search" in result
    assert "Navigation hits" in result
    assert "p. 10: Финансовые ковенанты" in result
    assert "Page 10" in result
    assert "DSCR должен быть не ниже 1.20" in result
    assert "page_ranges='10-11'" in result


def test_search_document_finds_docx_paragraph_matches(tmp_path):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    docx = drive / "credit.docx"
    document_xml = """
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p><w:r><w:t>Executive summary</w:t></w:r></w:p>
        <w:p><w:r><w:t>Borrower must maintain DSCR covenant above 1.20.</w:t></w:r></w:p>
        <w:p><w:r><w:t>Reporting package is due quarterly.</w:t></w:r></w:p>
      </w:body>
    </w:document>
    """.strip()
    with zipfile.ZipFile(docx, "w") as zf:
        zf.writestr("word/document.xml", document_xml)

    result = _search_document(
        _ctx(repo, drive),
        path="credit.docx",
        query="DSCR covenant",
    )

    assert "type: docx" in result
    assert "Paragraph 2" in result
    assert "Borrower must maintain DSCR covenant above 1.20" in result
    assert "DOCX does not store final rendered page numbers" in result


def test_index_document_caches_docx_and_search_reuses_index(tmp_path):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    docx = drive / "indexed-credit.docx"
    document_xml = """
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p>
          <w:pPr><w:pStyle w:val="Heading1"/></w:pPr>
          <w:r><w:t>Financial Covenants</w:t></w:r>
        </w:p>
        <w:p><w:r><w:t>Borrower must maintain DSCR covenant above 1.20.</w:t></w:r></w:p>
      </w:body>
    </w:document>
    """.strip()
    with zipfile.ZipFile(docx, "w") as zf:
        zf.writestr("word/document.xml", document_xml)

    ctx = _ctx(repo, drive)
    indexed = _index_document(ctx, path="indexed-credit.docx")

    assert "# Document Index" in indexed
    assert "document_indexes/" in indexed
    assert "Financial Covenants" in indexed
    assert "financial_terms" in indexed

    result = _search_document(
        ctx,
        path="indexed-credit.docx",
        query="DSCR covenant",
    )

    assert "index_used: true" in result
    assert "index_path: document_indexes/" in result
    assert "Paragraph 2" in result
    assert "Borrower must maintain DSCR covenant above 1.20" in result


def test_index_document_calibrates_pdf_toc_offset_and_search_shows_section(tmp_path, monkeypatch):
    class FakePage:
        def __init__(self, text: str):
            self.text = text

        def extract_text(self):
            return self.text

    class FakeReader:
        is_encrypted = False
        outline = []

        def __init__(self, _path: str):
            pages = [FakePage(f"Body page {idx}") for idx in range(1, 25)]
            pages[0] = FakePage(
                "Contents\n"
                "Financial Covenants ........ 10\n"
                "Events of Default ........ 15\n"
                "Risk Factors ........ 20"
            )
            pages[11] = FakePage("Financial Covenants\nBorrower must maintain DSCR covenant above 1.20.")
            pages[16] = FakePage("Events of Default\nCross-default applies after a payment default.")
            pages[21] = FakePage("Risk Factors\nMarket risk and liquidity risk are material.")
            self.pages = pages

    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    (drive / "toc-offset.pdf").write_bytes(b"%PDF fake")
    monkeypatch.setattr(
        "ouroboros.tools.documents._try_import_pdf_reader",
        lambda: (FakeReader, "fakepdf"),
    )

    ctx = _ctx(repo, drive)
    indexed = _index_document(ctx, path="toc-offset.pdf")

    assert "toc_calibration" in indexed
    assert "pdf p. 12 / printed p. 10: Financial Covenants" in indexed

    result = _search_document(
        ctx,
        path="toc-offset.pdf",
        query="DSCR covenant",
    )

    assert "index_used: true" in result
    assert "Page 12 - Financial Covenants" in result
    assert "Borrower must maintain DSCR covenant above 1.20" in result
    assert "page_ranges='12-13'" in result


def test_search_document_finds_pptx_slide_and_speaker_notes(tmp_path):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    pptx = drive / "board.pptx"
    slide1_xml = """
    <p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
           xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
      <p:cSld><p:spTree><p:sp><p:txBody>
        <a:p><a:r><a:t>Executive Summary</a:t></a:r></a:p>
        <a:p><a:r><a:t>Revenue is stable.</a:t></a:r></a:p>
      </p:txBody></p:sp></p:spTree></p:cSld>
    </p:sld>
    """.strip()
    slide2_xml = """
    <p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
           xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
      <p:cSld><p:spTree><p:sp><p:txBody>
        <a:p><a:r><a:t>Risk Update</a:t></a:r></a:p>
        <a:p><a:r><a:t>Liquidity covenant monitoring remains active.</a:t></a:r></a:p>
      </p:txBody></p:sp></p:spTree></p:cSld>
    </p:sld>
    """.strip()
    notes2_xml = """
    <p:notes xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
             xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
      <p:cSld><p:spTree><p:sp><p:txBody>
        <a:p><a:r><a:t>Mention DSCR covenant threshold of 1.20.</a:t></a:r></a:p>
      </p:txBody></p:sp></p:spTree></p:cSld>
    </p:notes>
    """.strip()
    with zipfile.ZipFile(pptx, "w") as zf:
        zf.writestr("ppt/slides/slide1.xml", slide1_xml)
        zf.writestr("ppt/slides/slide2.xml", slide2_xml)
        zf.writestr("ppt/notesSlides/notesSlide2.xml", notes2_xml)

    result = _search_document(
        _ctx(repo, drive),
        path="board.pptx",
        query="DSCR covenant",
    )

    assert "type: pptx" in result
    assert "index_used: temporary" in result
    assert "Slide 2 - Risk Update" in result
    assert "Mention DSCR covenant threshold of 1.20" in result


def test_index_document_caches_pptx_and_search_reuses_index(tmp_path):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    pptx = drive / "indexed-board.pptx"
    slide_xml = """
    <p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
           xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
      <p:cSld><p:spTree><p:sp><p:txBody>
        <a:p><a:r><a:t>Capital Structure</a:t></a:r></a:p>
        <a:p><a:r><a:t>Debt maturity and redemption profile.</a:t></a:r></a:p>
      </p:txBody></p:sp></p:spTree></p:cSld>
    </p:sld>
    """.strip()
    with zipfile.ZipFile(pptx, "w") as zf:
        zf.writestr("ppt/slides/slide1.xml", slide_xml)

    ctx = _ctx(repo, drive)
    indexed = _index_document(ctx, path="indexed-board.pptx")

    assert "type: pptx" in indexed
    assert "slide 1: Capital Structure" in indexed

    result = _search_document(
        ctx,
        path="indexed-board.pptx",
        query="debt maturity",
    )

    assert "index_used: true" in result
    assert "Slide 1 - Capital Structure" in result
    assert "Debt maturity and redemption profile" in result


def test_search_document_suggests_pptx_slide_ranges_and_analyze_uses_them(tmp_path):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    pptx = drive / "large-board.pptx"
    with zipfile.ZipFile(pptx, "w") as zf:
        for idx, (title, body) in enumerate([
            ("Executive Summary", "Revenue is stable."),
            ("Operations", "Churn reduction remains on track."),
            ("Covenant Monitoring", "DSCR covenant threshold is 1.20."),
            ("Appendix", "Definitions and backup tables."),
        ], start=1):
            slide_xml = f"""
            <p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                   xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
              <p:cSld><p:spTree><p:sp><p:txBody>
                <a:p><a:r><a:t>{title}</a:t></a:r></a:p>
                <a:p><a:r><a:t>{body}</a:t></a:r></a:p>
              </p:txBody></p:sp></p:spTree></p:cSld>
            </p:sld>
            """.strip()
            zf.writestr(f"ppt/slides/slide{idx}.xml", slide_xml)

    ctx = _ctx(repo, drive)
    search = _search_document(
        ctx,
        path="large-board.pptx",
        query="DSCR covenant",
    )

    assert "Slide 3 - Covenant Monitoring" in search
    assert "slide_ranges='3-4'" in search

    analysis = _analyze_document(
        ctx,
        path="large-board.pptx",
        analysis_type="answer_question",
        question="What is the DSCR threshold?",
        slide_ranges="3",
        max_slides=1,
    )

    assert "slides_extracted: 3" in analysis
    assert "Slide 3" in analysis
    assert "DSCR covenant threshold is 1.20" in analysis
    assert "Slide 1" not in analysis
    assert "Revenue is stable" not in analysis


def test_cached_pptx_search_respects_slide_ranges(tmp_path):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    pptx = drive / "indexed-large-board.pptx"
    slide1_xml = """
    <p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
           xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
      <p:cSld><p:spTree><p:sp><p:txBody>
        <a:p><a:r><a:t>Executive Summary</a:t></a:r></a:p>
        <a:p><a:r><a:t>Revenue is stable.</a:t></a:r></a:p>
      </p:txBody></p:sp></p:spTree></p:cSld>
    </p:sld>
    """.strip()
    slide2_xml = """
    <p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
           xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
      <p:cSld><p:spTree><p:sp><p:txBody>
        <a:p><a:r><a:t>Breach Monitoring</a:t></a:r></a:p>
        <a:p><a:r><a:t>Material covenant breach remediation is open.</a:t></a:r></a:p>
      </p:txBody></p:sp></p:spTree></p:cSld>
    </p:sld>
    """.strip()
    with zipfile.ZipFile(pptx, "w") as zf:
        zf.writestr("ppt/slides/slide1.xml", slide1_xml)
        zf.writestr("ppt/slides/slide2.xml", slide2_xml)

    ctx = _ctx(repo, drive)
    _index_document(ctx, path="indexed-large-board.pptx")

    narrowed_out = _search_document(
        ctx,
        path="indexed-large-board.pptx",
        query="breach remediation",
        slide_ranges="1",
    )
    assert "index_used: true" in narrowed_out
    assert "slides_searched: 1" in narrowed_out
    assert "[No lexical matches found" in narrowed_out
    assert "Breach Monitoring" not in narrowed_out

    narrowed_in = _search_document(
        ctx,
        path="indexed-large-board.pptx",
        query="breach remediation",
        slide_ranges="2",
    )
    assert "slides_searched: 2" in narrowed_in
    assert "Slide 2 - Breach Monitoring" in narrowed_in
    assert "Material covenant breach remediation is open" in narrowed_in


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
