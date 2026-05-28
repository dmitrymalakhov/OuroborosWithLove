import json
import os
import pathlib
import sys
import tempfile
import types
import unittest


class TestMultiUserWorkspace(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.drive_root = pathlib.Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_normal_user_gets_isolated_memory_root(self):
        from supervisor.users import ensure_user_workspace

        root, created, rec = ensure_user_workspace(
            self.drive_root,
            user_id=123,
            chat_id=456,
            from_user={"username": "alice"},
            role="user",
        )

        self.assertTrue(created)
        self.assertEqual(root, (self.drive_root / "users" / "123").resolve())
        self.assertEqual(rec["role"], "user")
        self.assertEqual(rec["access_status"], "approved")
        self.assertTrue((root / "memory" / "scratchpad.md").exists())
        self.assertTrue((root / "memory" / "identity.md").exists())
        self.assertTrue((root / "logs" / "chat.jsonl").exists())

    def test_admin_keeps_global_root(self):
        from supervisor.users import ensure_user_workspace

        root, created, rec = ensure_user_workspace(
            self.drive_root,
            user_id=1,
            chat_id=2,
            role="admin",
            use_global_root=True,
        )

        self.assertTrue(created)
        self.assertEqual(root, self.drive_root)
        self.assertEqual(rec["role"], "admin")
        self.assertEqual(rec["access_status"], "approved")
        self.assertTrue((self.drive_root / "memory" / "identity.md").exists())

    def test_new_user_access_request_stays_pending_without_workspace(self):
        from supervisor.users import request_user_access, user_access_status

        rec, created, should_notify = request_user_access(
            self.drive_root,
            user_id=123,
            chat_id=456,
            from_user={"username": "alice"},
        )

        self.assertTrue(created)
        self.assertTrue(should_notify)
        self.assertEqual(user_access_status(rec), "pending")
        self.assertFalse((self.drive_root / "users" / "123" / "memory").exists())

    def test_legacy_user_without_access_status_keeps_access(self):
        from supervisor.users import request_user_access, user_access_status

        users_path = self.drive_root / "state" / "users.json"
        users_path.parent.mkdir(parents=True)
        users_path.write_text(json.dumps({
            "users": {
                "123": {
                    "user_id": 123,
                    "chat_id": 456,
                    "role": "user",
                    "drive_root": str(self.drive_root / "users" / "123"),
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            }
        }), encoding="utf-8")

        rec, created, should_notify = request_user_access(
            self.drive_root,
            user_id=123,
            chat_id=456,
            from_user={"username": "alice"},
        )

        self.assertFalse(created)
        self.assertFalse(should_notify)
        self.assertEqual(user_access_status(rec), "approved")

    def test_approve_pending_user_updates_status(self):
        from supervisor.users import (
            list_user_records,
            request_user_access,
            set_user_access_status,
            user_access_status,
        )

        request_user_access(self.drive_root, user_id=123, chat_id=456)

        changes = set_user_access_status(
            self.drive_root,
            [123],
            "approved",
            decided_by=1,
        )

        self.assertEqual(changes[0]["old_status"], "pending")
        self.assertEqual(changes[0]["status"], "approved")
        approved = list_user_records(self.drive_root, access_status="approved")
        self.assertEqual([rec["user_id"] for rec in approved], [123])
        self.assertEqual(user_access_status(approved[0]), "approved")


class TestMultiUserTools(unittest.TestCase):
    def test_user_role_cannot_see_admin_only_tools(self):
        from ouroboros.tools.registry import ToolContext, ToolRegistry

        registry = ToolRegistry(repo_dir=pathlib.Path("/tmp"), drive_root=pathlib.Path("/tmp"))
        registry.set_context(ToolContext(
            repo_dir=pathlib.Path("/tmp"),
            drive_root=pathlib.Path("/tmp/user"),
            shared_drive_root=pathlib.Path("/tmp"),
            current_user_id=123,
            user_role="user",
        ))

        tools = set(registry.available_tools())
        self.assertNotIn("repo_write_commit", tools)
        self.assertNotIn("run_shell", tools)
        self.assertNotIn("request_restart", tools)
        self.assertIn("update_scratchpad", tools)
        self.assertIn("chat_history", tools)
        self.assertIn("analyze_document", tools)

    def test_admin_role_can_see_admin_tools(self):
        from ouroboros.tools.registry import ToolContext, ToolRegistry

        registry = ToolRegistry(repo_dir=pathlib.Path("/tmp"), drive_root=pathlib.Path("/tmp"))
        registry.set_context(ToolContext(
            repo_dir=pathlib.Path("/tmp"),
            drive_root=pathlib.Path("/tmp"),
            current_user_id=1,
            user_role="admin",
        ))

        tools = set(registry.available_tools())
        self.assertIn("repo_write_commit", tools)
        self.assertIn("run_shell", tools)
        self.assertIn("request_restart", tools)

    def test_self_mod_disabled_hides_self_mod_tools_even_for_admin(self):
        from ouroboros.tools.registry import ToolContext, ToolRegistry

        old_value = os.environ.get("OUROBOROS_DISABLE_SELF_MODIFICATION")
        old_branch = os.environ.get("OUROBOROS_BRANCH_DEV")
        os.environ["OUROBOROS_DISABLE_SELF_MODIFICATION"] = "1"
        os.environ["OUROBOROS_BRANCH_DEV"] = "main"
        try:
            registry = ToolRegistry(repo_dir=pathlib.Path("/tmp"), drive_root=pathlib.Path("/tmp"))
            registry.set_context(ToolContext(
                repo_dir=pathlib.Path("/tmp"),
                drive_root=pathlib.Path("/tmp"),
                current_user_id=1,
                user_role="admin",
            ))

            tools = set(registry.available_tools())
            self.assertNotIn("repo_write_commit", tools)
            self.assertNotIn("repo_commit_push", tools)
            self.assertNotIn("claude_code_edit", tools)
            self.assertNotIn("request_restart", tools)
            self.assertIn("run_shell", tools)
            self.assertIn("analyze_document", tools)
        finally:
            if old_value is None:
                os.environ.pop("OUROBOROS_DISABLE_SELF_MODIFICATION", None)
            else:
                os.environ["OUROBOROS_DISABLE_SELF_MODIFICATION"] = old_value
            if old_branch is None:
                os.environ.pop("OUROBOROS_BRANCH_DEV", None)
            else:
                os.environ["OUROBOROS_BRANCH_DEV"] = old_branch

    def test_self_mod_disabled_blocks_branch_switch_shell_command(self):
        from ouroboros.tools.registry import ToolContext
        from ouroboros.tools.shell import _run_shell

        old_value = os.environ.get("OUROBOROS_DISABLE_SELF_MODIFICATION")
        old_branch = os.environ.get("OUROBOROS_BRANCH_DEV")
        os.environ["OUROBOROS_DISABLE_SELF_MODIFICATION"] = "1"
        os.environ["OUROBOROS_BRANCH_DEV"] = "main"
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            try:
                result = _run_shell(
                    ToolContext(repo_dir=root, drive_root=root, user_role="admin"),
                    ["git", "checkout", "ouroboros"],
                )
                self.assertIn("BRANCH_SWITCH_BLOCKED", result)
                self.assertIn("main", result)
                self.assertIn("ouroboros", result)
            finally:
                if old_value is None:
                    os.environ.pop("OUROBOROS_DISABLE_SELF_MODIFICATION", None)
                else:
                    os.environ["OUROBOROS_DISABLE_SELF_MODIFICATION"] = old_value
                if old_branch is None:
                    os.environ.pop("OUROBOROS_BRANCH_DEV", None)
                else:
                    os.environ["OUROBOROS_BRANCH_DEV"] = old_branch


class TestMultiUserEvents(unittest.TestCase):
    def test_schedule_task_preserves_user_scope(self):
        from supervisor.events import _handle_schedule_task

        with tempfile.TemporaryDirectory() as tmp:
            drive_root = pathlib.Path(tmp)
            enqueued = []
            sent = []
            ctx = types.SimpleNamespace(
                DRIVE_ROOT=drive_root,
                load_state=lambda: {"owner_chat_id": 111, "owner_id": 1},
                enqueue_task=enqueued.append,
                send_with_budget=lambda *args, **kwargs: sent.append((args, kwargs)),
                persist_queue_snapshot=lambda reason="": None,
            )

            old_queue = sys.modules.get("supervisor.queue")
            sys.modules["supervisor.queue"] = types.SimpleNamespace(PENDING=[], RUNNING={})
            try:
                _handle_schedule_task({
                    "type": "schedule_task",
                    "description": "do scoped work",
                    "task_id": "task1",
                    "chat_id": 222,
                    "user_id": 333,
                    "user_role": "user",
                    "drive_root": str(drive_root / "users" / "333"),
                }, ctx)
            finally:
                if old_queue is None:
                    sys.modules.pop("supervisor.queue", None)
                else:
                    sys.modules["supervisor.queue"] = old_queue

            self.assertEqual(len(enqueued), 1)
            task = enqueued[0]
            self.assertEqual(task["chat_id"], 222)
            self.assertEqual(task["user_id"], 333)
            self.assertEqual(task["user_role"], "user")
            self.assertEqual(task["drive_root"], str((drive_root / "users" / "333").resolve()))
            self.assertTrue(sent)


class TestMultiUserPrivacy(unittest.TestCase):
    def test_review_collection_skips_user_workspaces(self):
        from ouroboros.review import collect_sections

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            repo = root / "repo"
            drive = root / "drive"
            repo.mkdir()
            (repo / "main.py").write_text("print('ok')\n", encoding="utf-8")
            (drive / "memory").mkdir(parents=True)
            (drive / "memory" / "identity.md").write_text("admin identity\n", encoding="utf-8")
            (drive / "users" / "123" / "memory").mkdir(parents=True)
            (drive / "users" / "123" / "memory" / "identity.md").write_text("private user identity\n", encoding="utf-8")

            sections, _stats = collect_sections(repo, drive)
            paths = {path for path, _content in sections}
            contents = "\n".join(content for _path, content in sections)

            self.assertIn("repo/main.py", paths)
            self.assertIn("drive/memory/identity.md", paths)
            self.assertNotIn("private user identity", contents)

    def test_public_budget_line_hides_spend_and_git_ref(self):
        import sys
        import types

        old_requests = sys.modules.get("requests")
        sys.modules["requests"] = types.SimpleNamespace()
        try:
            from supervisor.telegram import _format_budget_line

            line = _format_budget_line({
                "spent_usd": 12.34,
                "current_branch": "ouroboros",
                "current_sha": "abcdef123456",
            }, public=True)
        finally:
            if old_requests is None:
                sys.modules.pop("requests", None)
            else:
                sys.modules["requests"] = old_requests

        self.assertIn("shared pool active", line)
        self.assertNotIn("12.34", line)
        self.assertNotIn("ouroboros", line)
        self.assertNotIn("abcdef", line)


if __name__ == "__main__":
    unittest.main()
