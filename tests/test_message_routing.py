"""
Tests for v6 message routing: single-consumer delivery,
per-task mailbox, and forward_to_worker tool.

Run: pytest tests/test_message_routing.py -v
"""

import json
import importlib
import pathlib
import sys
import os
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestOwnerInjectPerTask(unittest.TestCase):
    """Test per-task mailbox in owner_inject.py."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.drive_root = pathlib.Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_write_creates_per_task_file(self):
        from ouroboros.owner_inject import write_owner_message, _mailbox_path
        write_owner_message(self.drive_root, "hello", task_id="abc123", msg_id="m1")
        path = _mailbox_path(self.drive_root, "abc123")
        self.assertTrue(path.exists())
        content = path.read_text()
        entry = json.loads(content.strip())
        self.assertEqual(entry["text"], "hello")
        self.assertEqual(entry["msg_id"], "m1")

    def test_drain_reads_only_own_task(self):
        from ouroboros.owner_inject import write_owner_message, drain_owner_messages
        write_owner_message(self.drive_root, "for task A", task_id="taskA", msg_id="m1")
        write_owner_message(self.drive_root, "for task B", task_id="taskB", msg_id="m2")

        msgs_a = drain_owner_messages(self.drive_root, task_id="taskA")
        msgs_b = drain_owner_messages(self.drive_root, task_id="taskB")

        self.assertEqual(msgs_a, ["for task A"])
        self.assertEqual(msgs_b, ["for task B"])

    def test_drain_dedup_with_seen_ids(self):
        from ouroboros.owner_inject import write_owner_message, drain_owner_messages
        write_owner_message(self.drive_root, "msg1", task_id="t1", msg_id="id1")
        write_owner_message(self.drive_root, "msg2", task_id="t1", msg_id="id2")

        seen = set()
        first_read = drain_owner_messages(self.drive_root, task_id="t1", seen_ids=seen)
        self.assertEqual(len(first_read), 2)
        self.assertEqual(seen, {"id1", "id2"})

        write_owner_message(self.drive_root, "msg3", task_id="t1", msg_id="id3")
        second_read = drain_owner_messages(self.drive_root, task_id="t1", seen_ids=seen)
        self.assertEqual(second_read, ["msg3"])
        self.assertIn("id3", seen)

    def test_cleanup_removes_file(self):
        from ouroboros.owner_inject import write_owner_message, cleanup_task_mailbox, _mailbox_path
        write_owner_message(self.drive_root, "hello", task_id="t1", msg_id="m1")
        path = _mailbox_path(self.drive_root, "t1")
        self.assertTrue(path.exists())

        cleanup_task_mailbox(self.drive_root, "t1")
        self.assertFalse(path.exists())

    def test_drain_nonexistent_task_returns_empty(self):
        from ouroboros.owner_inject import drain_owner_messages
        msgs = drain_owner_messages(self.drive_root, task_id="nonexistent")
        self.assertEqual(msgs, [])

    def test_messages_not_cleared_on_read(self):
        """Messages persist after read (append-only). Only cleanup removes them."""
        from ouroboros.owner_inject import write_owner_message, drain_owner_messages, _mailbox_path
        write_owner_message(self.drive_root, "persistent", task_id="t1", msg_id="m1")

        drain_owner_messages(self.drive_root, task_id="t1")

        path = _mailbox_path(self.drive_root, "t1")
        self.assertTrue(path.exists())
        self.assertIn("persistent", path.read_text())


class TestForwardToWorkerTool(unittest.TestCase):
    """Test that forward_to_worker tool is registered."""

    def test_tool_registered(self):
        from ouroboros.tools.registry import ToolRegistry
        registry = ToolRegistry(
            repo_dir=pathlib.Path("/tmp"),
            drive_root=pathlib.Path("/tmp"),
        )
        tools = registry.available_tools()
        self.assertIn("forward_to_worker", tools)


class TestDirectChatImageRouting(unittest.TestCase):
    """Test image task payload routing for Telegram photo edits."""

    def test_handle_chat_direct_preserves_image_note_and_image_context(self):
        with mock.patch.dict(sys.modules, {"requests": mock.MagicMock()}):
            workers = importlib.import_module("supervisor.workers")

        class FakeAgent:
            def __init__(self):
                self.task = None

            def handle_task(self, task):
                self.task = task
                return []

        fake_agent = FakeAgent()
        fake_queue = mock.MagicMock()
        text = (
            "\n\n[Telegram image saved]\n"
            "- path: uploads/2026-06-03/42_telegram_photo_42.jpg\n"
            "- filename: telegram_photo_42.jpg\n"
            "- mime_type: image/jpeg\n"
            "- size_bytes: 123\n"
            "Use edit_image(path='<path>', prompt='<requested edit>') when the user asks to modify this image."
        )

        with mock.patch.object(workers, "_get_chat_agent", return_value=fake_agent), \
                mock.patch.object(workers, "get_event_q", return_value=fake_queue):
            workers.handle_chat_direct(
                123,
                text,
                image_data=("base64-image", "image/jpeg", "убери фон"),
                user_id=456,
                user_role="user",
                drive_root=pathlib.Path("/tmp/drive"),
            )

        self.assertIsNotNone(fake_agent.task)
        self.assertIn("uploads/2026-06-03/42_telegram_photo_42.jpg", fake_agent.task["text"])
        self.assertEqual(fake_agent.task["image_base64"], "base64-image")
        self.assertEqual(fake_agent.task["image_mime"], "image/jpeg")
        self.assertEqual(fake_agent.task["image_caption"], "убери фон")

    def test_handle_chat_direct_scopes_returned_events_to_task_user(self):
        with mock.patch.dict(sys.modules, {"requests": mock.MagicMock()}):
            workers = importlib.import_module("supervisor.workers")

        class FakeAgent:
            def handle_task(self, task):
                return [
                    {
                        "type": "send_message",
                        "chat_id": 999,
                        "user_id": 1,
                        "user_role": "admin",
                        "drive_root": "/tmp/admin-root",
                        "text": "wrong chat",
                    },
                    {"type": "llm_usage", "usage": {"cost": 0.1}},
                ]

        class FakeQueue:
            def __init__(self):
                self.events = []

            def put(self, event):
                self.events.append(event)

        fake_queue = FakeQueue()
        drive_root = pathlib.Path("/tmp/ouroboros-user-456")

        with mock.patch.object(workers, "_get_chat_agent", return_value=FakeAgent()), \
                mock.patch.object(workers, "get_event_q", return_value=fake_queue):
            workers.handle_chat_direct(
                123,
                "hello",
                user_id=456,
                user_role="user",
                drive_root=drive_root,
            )

        self.assertEqual(len(fake_queue.events), 2)
        message_event = fake_queue.events[0]
        usage_event = fake_queue.events[1]
        self.assertEqual(message_event["chat_id"], 123)
        self.assertEqual(message_event["user_id"], 456)
        self.assertEqual(message_event["user_role"], "user")
        self.assertEqual(message_event["drive_root"], str(drive_root))
        self.assertEqual(usage_event["user_id"], 456)
        self.assertEqual(usage_event["user_role"], "user")
        self.assertEqual(usage_event["drive_root"], str(drive_root))


class TestQueueTimeoutRouting(unittest.TestCase):
    """Test task-specific supervisor notices do not leak to owner chat."""

    def test_soft_timeout_for_user_task_goes_to_task_chat(self):
        with mock.patch.dict(sys.modules, {"requests": mock.MagicMock()}):
            from supervisor import queue

            sent = []
            old_running = dict(queue.RUNNING)
            old_soft = queue.SOFT_TIMEOUT_SEC
            old_hard = queue.HARD_TIMEOUT_SEC
            old_drive = queue.DRIVE_ROOT
            drive_root = pathlib.Path("/tmp/ouroboros-timeout-test")
            try:
                queue.RUNNING.clear()
                queue.RUNNING["task1"] = {
                    "task": {
                        "id": "task1",
                        "type": "task",
                        "chat_id": 222,
                        "user_id": 333,
                        "user_role": "user",
                        "drive_root": str(drive_root / "users" / "333"),
                    },
                    "started_at": 100.0,
                    "last_heartbeat_at": 100.0,
                    "soft_sent": False,
                }
                queue.SOFT_TIMEOUT_SEC = 10
                queue.HARD_TIMEOUT_SEC = 1000
                queue.DRIVE_ROOT = drive_root

                with mock.patch.object(queue, "load_state", return_value={"owner_chat_id": 111}), \
                        mock.patch.object(queue.time, "time", return_value=120.0), \
                        mock.patch.object(queue, "send_with_budget", side_effect=lambda *args, **kwargs: sent.append((args, kwargs))):
                    queue.enforce_task_timeouts()
            finally:
                queue.RUNNING.clear()
                queue.RUNNING.update(old_running)
                queue.SOFT_TIMEOUT_SEC = old_soft
                queue.HARD_TIMEOUT_SEC = old_hard
                queue.DRIVE_ROOT = old_drive

        self.assertEqual(len(sent), 1)
        args, kwargs = sent[0]
        self.assertEqual(args[0], 222)
        self.assertEqual(kwargs["log_user_id"], 333)
        self.assertEqual(kwargs["log_drive_root"], (drive_root / "users" / "333").resolve())


if __name__ == "__main__":
    unittest.main()
