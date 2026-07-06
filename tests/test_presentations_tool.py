import pathlib
import zipfile

from ouroboros.tools.documents import _analyze_document
from ouroboros.tools.presentation_editing import _edit_presentation, _inspect_presentation_for_edit
from ouroboros.tools.presentations import PPTX_MIME_TYPE, _create_presentation
from ouroboros.tools.registry import ToolContext


def _ctx(repo_dir: pathlib.Path, drive_root: pathlib.Path, progress=None) -> ToolContext:
    repo_dir.mkdir(parents=True, exist_ok=True)
    drive_root.mkdir(parents=True, exist_ok=True)
    return ToolContext(
        repo_dir=repo_dir,
        drive_root=drive_root,
        current_chat_id=123,
        current_user_id=456,
        user_role="user",
        emit_progress_fn=progress or (lambda _: None),
    )


def test_create_presentation_writes_pptx_and_queues_delivery(tmp_path):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    progress = []

    result = _create_presentation(
        _ctx(repo, drive, progress=progress.append),
        title="Launch Plan",
        subtitle="Q3 execution",
        slides=[
            {
                "title": "Priorities",
                "body": "Focus the launch around measurable outcomes.",
                "bullets": ["Finalize narrative", "Prepare demo", "Align sales"],
                "speaker_notes": "Mention the launch date and owner.",
            },
            {
                "layout": "two_column",
                "title": "Risks and mitigations",
                "left_title": "Risks",
                "left_bullets": ["Late demo", "Unclear ICP"],
                "right_title": "Mitigations",
                "right_bullets": ["Daily review", "Customer interviews"],
            },
        ],
        output_path="presentations/launch-plan.pptx",
        send_to_chat=True,
    )

    pptx = drive / "presentations" / "launch-plan.pptx"
    assert "OK: presentation created" in result
    assert pptx.exists()
    assert (drive / "presentations" / "launch-plan.notes.md").exists()
    assert any("Creating presentation" in item for item in progress)

    with zipfile.ZipFile(pptx) as zf:
        names = set(zf.namelist())
        assert "[Content_Types].xml" in names
        assert "ppt/presentation.xml" in names
        assert "ppt/slides/slide1.xml" in names
        assert "ppt/slides/slide2.xml" in names
        assert "ppt/slides/slide3.xml" in names

    analysis = _analyze_document(
        _ctx(repo, drive),
        path="presentations/launch-plan.pptx",
        analysis_type="raw",
        max_slides=10,
    )
    assert "Launch Plan" in analysis
    assert "Finalize narrative" in analysis
    assert "Customer interviews" in analysis


def test_create_presentation_queues_send_document_event(tmp_path):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    ctx = _ctx(repo, drive)

    _create_presentation(
        ctx,
        title="Board Update",
        slides=[{"title": "Summary", "bullets": ["Revenue up", "Costs stable"]}],
        output_path="board-update.pptx",
    )

    assert len(ctx.pending_events) == 1
    event = ctx.pending_events[0]
    assert event["type"] == "send_document"
    assert event["chat_id"] == 123
    assert event["path"] == "board-update.pptx"
    assert event["filename"] == "board-update.pptx"
    assert event["mime_type"] == PPTX_MIME_TYPE


def _write_editable_pptx(path: pathlib.Path) -> None:
    slide_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      <p:sp>
        <p:nvSpPr><p:cNvPr id="2" name="Title Box"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
        <p:spPr/>
        <p:txBody>
          <a:bodyPr/><a:lstStyle/>
          <a:p><a:r><a:rPr lang="en-US" sz="2400"><a:latin typeface="Arial"/></a:rPr><a:t>Old title</a:t></a:r></a:p>
        </p:txBody>
      </p:sp>
      <p:graphicFrame>
        <p:nvGraphicFramePr><p:cNvPr id="4" name="Table 1"/><p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr>
        <p:xfrm><a:off x="0" y="0"/><a:ext cx="3000000" cy="600000"/></p:xfrm>
        <a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table">
          <a:tbl>
            <a:tblPr/><a:tblGrid><a:gridCol w="3000000"/></a:tblGrid>
            <a:tr h="370840"><a:tc><a:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>Cell old</a:t></a:r></a:p></a:txBody><a:tcPr/></a:tc></a:tr>
          </a:tbl>
        </a:graphicData></a:graphic>
      </p:graphicFrame>
      <p:pic>
        <p:nvPicPr><p:cNvPr id="5" name="Picture 1"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>
        <p:blipFill><a:blip r:embed="rId2"/></p:blipFill><p:spPr/>
      </p:pic>
    </p:spTree>
  </p:cSld>
</p:sld>
"""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        zf.writestr("ppt/slides/slide1.xml", slide_xml)
        zf.writestr("ppt/slides/_rels/slide1.xml.rels", "<Relationships/>")
        zf.writestr("ppt/media/image1.png", b"fake-png-data")
        zf.writestr("ppt/theme/theme1.xml", "<theme/>")


def _slide_xml(text: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr/>
    <p:sp>
      <p:nvSpPr><p:cNvPr id="2" name="Title Box"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
      <p:spPr/>
      <p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:txBody>
    </p:sp>
  </p:spTree></p:cSld>
</p:sld>
"""


def _write_relationship_ordered_pptx(path: pathlib.Path) -> None:
    presentation_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <p:sldIdLst>
    <p:sldId id="257" r:id="rId5"/>
    <p:sldId id="258" r:id="rId6"/>
  </p:sldIdLst>
</p:presentation>
"""
    presentation_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide2.xml"/>
  <Relationship Id="rId6" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
</Relationships>
"""
    slide2_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide9.xml"/>
</Relationships>
"""
    notes_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:notes xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
         xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree>
    <p:sp><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>Speaker note old</a:t></a:r></a:p></p:txBody></p:sp>
  </p:spTree></p:cSld>
</p:notes>
"""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        zf.writestr("ppt/presentation.xml", presentation_xml)
        zf.writestr("ppt/_rels/presentation.xml.rels", presentation_rels)
        zf.writestr("ppt/slides/slide1.xml", _slide_xml("Physical first"))
        zf.writestr("ppt/slides/slide2.xml", _slide_xml("Logical first"))
        zf.writestr("ppt/slides/_rels/slide2.xml.rels", slide2_rels)
        zf.writestr("ppt/notesSlides/notesSlide9.xml", notes_xml)


def test_inspect_presentation_for_edit_reports_existing_targets(tmp_path):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    _write_editable_pptx(drive / "deck.pptx")

    result = _inspect_presentation_for_edit(
        _ctx(repo, drive),
        path="deck.pptx",
        search_text="Old title",
    )

    assert "slides: 1" in result
    assert "pictures: 1" in result
    assert "tables: 1" in result
    assert "slide 1 paragraph 1" in result
    assert "shape id=2 name=Title Box" in result
    assert "table=1 id=4 name=Table 1" in result


def test_edit_presentation_preserves_source_objects_and_queues_delivery(tmp_path):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    _write_editable_pptx(drive / "deck.pptx")
    ctx = _ctx(repo, drive)

    result = _edit_presentation(
        ctx,
        path="deck.pptx",
        operations=[
            {
                "type": "replace_text",
                "search_text": "Old title",
                "replacement_text": "New title",
                "confirmed": True,
            },
            {
                "type": "set_table_cell",
                "slide": 1,
                "table_index": 1,
                "row": 1,
                "col": 1,
                "text": "Updated cell",
                "confirmed": True,
            },
            {
                "type": "apply_text_style",
                "slide": 1,
                "shape_id": 2,
                "font_face": "Aptos",
                "font_size_pt": 28,
                "color": "#123456",
                "bold": True,
                "confirmed": True,
            },
        ],
        output_path="edited/deck-edited.pptx",
    )

    edited = drive / "edited" / "deck-edited.pptx"
    assert "OK: PowerPoint presentation edited" in result
    assert edited.exists()
    assert len(ctx.pending_events) == 1
    assert ctx.pending_events[0]["path"] == "edited/deck-edited.pptx"
    assert ctx.pending_events[0]["mime_type"] == PPTX_MIME_TYPE

    with zipfile.ZipFile(edited) as zf:
        assert zf.read("ppt/media/image1.png") == b"fake-png-data"
        assert zf.read("ppt/theme/theme1.xml") == b"<theme/>"
        slide_xml = zf.read("ppt/slides/slide1.xml").decode("utf-8")

    assert "New title" in slide_xml
    assert "Old title" not in slide_xml
    assert "Updated cell" in slide_xml
    assert "Cell old" not in slide_xml
    assert 'typeface="Aptos"' in slide_xml
    assert 'val="123456"' in slide_xml


def test_edit_presentation_uses_presentation_order_and_notes_relationships(tmp_path):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    _write_relationship_ordered_pptx(drive / "ordered.pptx")
    ctx = _ctx(repo, drive)

    inspect = _inspect_presentation_for_edit(ctx, path="ordered.pptx", search_text="Logical first")
    assert "slide 1: title='Logical first'" in inspect
    assert "slide 2: title='Physical first'" in inspect

    result = _edit_presentation(
        ctx,
        path="ordered.pptx",
        operations=[
            {
                "type": "replace_text",
                "slide": 1,
                "search_text": "Logical first",
                "replacement_text": "Updated logical",
                "confirmed": True,
            },
            {
                "type": "replace_text",
                "slide": 1,
                "include_notes": True,
                "search_text": "Speaker note old",
                "replacement_text": "Speaker note new",
                "confirmed": True,
            },
            {
                "type": "apply_text_style",
                "font_face": "Aptos",
                "confirmed": True,
            },
        ],
        output_path="edited/ordered-edited.pptx",
    )

    assert "OK: PowerPoint presentation edited" in result
    with zipfile.ZipFile(drive / "edited" / "ordered-edited.pptx") as zf:
        slide1 = zf.read("ppt/slides/slide1.xml").decode("utf-8")
        slide2 = zf.read("ppt/slides/slide2.xml").decode("utf-8")
        notes = zf.read("ppt/notesSlides/notesSlide9.xml").decode("utf-8")

    assert "Physical first" in slide1
    assert "Updated logical" in slide2
    assert "Logical first" not in slide2
    assert 'typeface="Aptos"' in slide1
    assert 'typeface="Aptos"' in slide2
    assert "Speaker note new" in notes
    assert "Speaker note old" not in notes
