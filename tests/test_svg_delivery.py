import pathlib
import time

from ouroboros.agent import Env, OuroborosAgent
from ouroboros.svg_delivery import SVG_MIME_TYPE, extract_svg_attachments


def test_extract_svg_attachments_writes_named_files_and_removes_code(tmp_path):
    text = """Bot result: 2 SVG files.

1A.svg
```html
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">
  <rect width="10" height="10" fill="white"/>
</svg>
```

1B.svg
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">
  <circle cx="5" cy="5" r="4"/>
</svg>
```
"""

    cleaned, attachments = extract_svg_attachments(text, tmp_path, task_id="task-1")

    assert "```" not in cleaned
    assert "<svg" not in cleaned
    assert "Файл отправлен вложением: `1A.svg`" in cleaned
    assert "Файл отправлен вложением: `1B.svg`" in cleaned
    assert [item.filename for item in attachments] == ["1A.svg", "1B.svg"]
    assert (tmp_path / attachments[0].rel_path).read_text(encoding="utf-8").startswith("<svg")
    assert (tmp_path / attachments[1].rel_path).read_text(encoding="utf-8").startswith("<svg")


def test_agent_emit_task_results_queues_svg_documents(tmp_path):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    drive.mkdir()
    repo.mkdir()
    agent = OuroborosAgent(Env(repo_dir=repo, drive_root=drive), user_id=456, user_role="user")

    agent._pending_events = []
    agent._current_chat_id = 123
    agent._current_user_id = 456
    text = """Bot result.

coin.svg
```html
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">
  <path d="M1 1h8v8H1z"/>
</svg>
```
"""

    agent._emit_task_results(
        {"id": "task-svg", "chat_id": 123, "type": "chat"},
        text,
        usage={},
        llm_trace={"tool_calls": []},
        start_time=time.time(),
        drive_logs=drive / "logs",
    )

    message_events = [event for event in agent._pending_events if event["type"] == "send_message"]
    document_events = [event for event in agent._pending_events if event["type"] == "send_document"]
    assert len(message_events) == 1
    assert "```" not in message_events[0]["text"]
    assert len(document_events) == 1
    assert document_events[0]["filename"] == "coin.svg"
    assert document_events[0]["mime_type"] == SVG_MIME_TYPE
    assert (drive / document_events[0]["path"]).read_text(encoding="utf-8").startswith("<svg")
