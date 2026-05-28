import pathlib
import zipfile

from ouroboros.tools.documents import _analyze_document
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
