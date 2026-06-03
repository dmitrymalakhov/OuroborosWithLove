"""Smoke tests for VLM (Vision Language Model) support."""

import base64
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import pathlib

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestLLMVisionQuery(unittest.TestCase):
    """Test LLMClient.vision_query() message format."""

    def test_vision_query_url_format(self):
        """vision_query builds correct message format for URL images."""
        from ouroboros.llm import LLMClient

        client = LLMClient(api_key="test-key")

        captured_messages = []

        def mock_chat(messages, model, tools=None, reasoning_effort="low", max_tokens=1024, tool_choice="auto"):
            captured_messages.extend(messages)
            return {"content": "I see a test image."}, {"prompt_tokens": 10, "completion_tokens": 5}

        client.chat = mock_chat

        text, usage = client.vision_query(
            prompt="What do you see?",
            images=[{"url": "https://example.com/test.png"}],
            model="anthropic/claude-sonnet-4.6",
        )

        self.assertEqual(text, "I see a test image.")
        self.assertEqual(len(captured_messages), 1)
        content = captured_messages[0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(len(content), 2)
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[0]["text"], "What do you see?")
        self.assertEqual(content[1]["type"], "image_url")
        self.assertIn("url", content[1]["image_url"])
        self.assertEqual(content[1]["image_url"]["url"], "https://example.com/test.png")

    def test_vision_query_base64_format(self):
        """vision_query builds correct data URI for base64 images."""
        from ouroboros.llm import LLMClient

        client = LLMClient(api_key="test-key")
        captured_messages = []

        def mock_chat(messages, model, tools=None, reasoning_effort="low", max_tokens=1024, tool_choice="auto"):
            captured_messages.extend(messages)
            return {"content": "Base64 image description."}, {}

        client.chat = mock_chat

        fake_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        text, _ = client.vision_query(
            prompt="Describe this.",
            images=[{"base64": fake_b64, "mime": "image/png"}],
        )

        self.assertEqual(text, "Base64 image description.")
        content = captured_messages[0]["content"]
        image_part = content[1]
        self.assertTrue(image_part["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertIn(fake_b64, image_part["image_url"]["url"])

    def test_vision_query_multiple_images(self):
        """vision_query handles multiple images in one call."""
        from ouroboros.llm import LLMClient

        client = LLMClient(api_key="test-key")
        captured_messages = []

        def mock_chat(messages, model, tools=None, reasoning_effort="low", max_tokens=1024, tool_choice="auto"):
            captured_messages.extend(messages)
            return {"content": "Two images."}, {}

        client.chat = mock_chat

        client.vision_query(
            prompt="Compare these images.",
            images=[
                {"url": "https://example.com/img1.png"},
                {"url": "https://example.com/img2.png"},
            ],
        )

        content = captured_messages[0]["content"]
        self.assertEqual(len(content), 3)  # text + 2 images

    def test_vision_query_empty_images(self):
        """vision_query works with no images (just text)."""
        from ouroboros.llm import LLMClient

        client = LLMClient(api_key="test-key")

        def mock_chat(messages, model, tools=None, reasoning_effort="low", max_tokens=1024, tool_choice="auto"):
            return {"content": "Text only."}, {}

        client.chat = mock_chat

        text, _ = client.vision_query(prompt="Hello", images=[])
        self.assertEqual(text, "Text only.")


class TestAnalyzeScreenshotTool(unittest.TestCase):
    """Test the analyze_screenshot tool."""

    def _make_ctx(self, with_screenshot=True):
        from ouroboros.tools.registry import ToolContext, BrowserState
        ctx = MagicMock(spec=ToolContext)
        ctx.browser_state = BrowserState()
        ctx.event_queue = None
        ctx.task_id = "test-task"
        ctx.current_task_type = "task"
        if with_screenshot:
            ctx.browser_state.last_screenshot_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        else:
            ctx.browser_state.last_screenshot_b64 = None
        return ctx

    def test_no_screenshot_returns_warning(self):
        """analyze_screenshot returns warning when no screenshot available."""
        from ouroboros.tools.vision import _analyze_screenshot

        ctx = self._make_ctx(with_screenshot=False)
        result = _analyze_screenshot(ctx, prompt="What do you see?")
        self.assertIn("⚠️", result)
        self.assertIn("screenshot", result.lower())

    def test_analyze_screenshot_calls_vlm(self):
        """analyze_screenshot calls VLM with the screenshot base64."""
        from ouroboros.tools.vision import _analyze_screenshot

        ctx = self._make_ctx(with_screenshot=True)

        with patch("ouroboros.tools.vision._get_llm_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.vision_query.return_value = ("Beautiful UI.", {"prompt_tokens": 100, "completion_tokens": 20})
            mock_get_client.return_value = mock_client

            result = _analyze_screenshot(ctx, prompt="Describe the UI.")

        self.assertEqual(result, "Beautiful UI.")
        mock_client.vision_query.assert_called_once()
        call_kwargs = mock_client.vision_query.call_args
        # Check that base64 image was passed
        images = call_kwargs[1].get("images") or call_kwargs[0][1]
        self.assertEqual(len(images), 1)
        self.assertIn("base64", images[0])


class TestVlmQueryTool(unittest.TestCase):
    """Test the vlm_query tool."""

    def _make_ctx(self):
        from ouroboros.tools.registry import ToolContext, BrowserState
        ctx = MagicMock(spec=ToolContext)
        ctx.browser_state = BrowserState()
        ctx.event_queue = None
        ctx.task_id = "test-task"
        ctx.current_task_type = "task"
        return ctx

    def test_vlm_query_requires_image(self):
        """vlm_query returns error when no image provided."""
        from ouroboros.tools.vision import _vlm_query

        ctx = self._make_ctx()
        result = _vlm_query(ctx, prompt="What is this?")
        self.assertIn("⚠️", result)

    def test_vlm_query_with_url(self):
        """vlm_query calls VLM with URL image."""
        from ouroboros.tools.vision import _vlm_query

        ctx = self._make_ctx()

        with patch("ouroboros.tools.vision._get_llm_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.vision_query.return_value = ("A logo.", {})
            mock_get_client.return_value = mock_client

            result = _vlm_query(ctx, prompt="What is the logo?", image_url="https://example.com/logo.png")

        self.assertEqual(result, "A logo.")
        call_kwargs = mock_client.vision_query.call_args
        images = call_kwargs[1].get("images") or call_kwargs[0][1]
        self.assertEqual(images[0]["url"], "https://example.com/logo.png")

    def test_vlm_query_tool_registered(self):
        """vlm_query and analyze_screenshot tools are properly registered."""
        import pathlib
        from ouroboros.tools.registry import ToolRegistry

        registry = ToolRegistry(
            repo_dir=pathlib.Path("/tmp"),
            drive_root=pathlib.Path("/tmp"),
        )
        tools = registry.available_tools()
        self.assertIn("analyze_screenshot", tools, "analyze_screenshot must be registered")
        self.assertIn("vlm_query", tools, "vlm_query must be registered")


class TestEditImageTool(unittest.TestCase):
    """Test the edit_image tool."""

    def _make_ctx(self):
        from ouroboros.tools.registry import ToolContext

        tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(tmp.name)
        repo = root / "repo"
        drive = root / "drive"
        repo.mkdir()
        drive.mkdir()
        ctx = ToolContext(
            repo_dir=repo,
            drive_root=drive,
            current_chat_id=123,
            current_user_id=456,
            user_role="user",
            emit_progress_fn=lambda _text: None,
        )
        return tmp, ctx

    def test_edit_image_rejects_invalid_inputs(self):
        from ouroboros.tools.vision import _edit_image

        tmp, ctx = self._make_ctx()
        self.addCleanup(tmp.cleanup)

        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            result = _edit_image(ctx, path="missing.png", prompt="make it brighter")
        self.assertIn("OPENAI_API_KEY", result)
        self.assertEqual(ctx.pending_events, [])

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = _edit_image(ctx, path="missing.png", prompt="make it brighter")
        self.assertIn("not found", result)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = _edit_image(ctx, path="../outside.png", prompt="make it brighter")
        self.assertIn("Path traversal", result)

        (ctx.drive_root / "note.txt").write_text("not an image", encoding="utf-8")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = _edit_image(ctx, path="note.txt", prompt="make it brighter")
        self.assertIn("Unsupported image type", result)

        (ctx.drive_root / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = _edit_image(ctx, path="image.png", prompt="")
        self.assertIn("prompt is required", result)

    def test_edit_image_calls_openai_saves_output_and_queues_photo(self):
        from ouroboros.tools.vision import _edit_image

        tmp, ctx = self._make_ctx()
        self.addCleanup(tmp.cleanup)
        (ctx.drive_root / "uploads").mkdir()
        (ctx.drive_root / "uploads" / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\nsource")

        edited_b64 = base64.b64encode(b"edited-image-bytes").decode("ascii")
        response = MagicMock()
        response.data = [MagicMock(b64_json=edited_b64)]
        response.usage = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
        mock_client = MagicMock()
        mock_client.images.edit.return_value = response

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OUROBOROS_IMAGE_MODEL": "gpt-image-2"}), \
                patch("ouroboros.tools.vision._get_openai_image_client", return_value=mock_client):
            result = _edit_image(
                ctx,
                path="uploads/photo.png",
                prompt="remove the background",
                size="1024x1024",
                quality="high",
                output_format="png",
            )

        self.assertIn("OK: image edited", result)
        self.assertIn("telegram_delivery: queued", result)
        call_kwargs = mock_client.images.edit.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gpt-image-2")
        self.assertEqual(call_kwargs["prompt"], "remove the background")
        self.assertEqual(call_kwargs["size"], "1024x1024")
        self.assertEqual(call_kwargs["quality"], "high")
        self.assertEqual(call_kwargs["output_format"], "png")

        generated = list((ctx.drive_root / "generated" / "images").glob("*/*_edited.png"))
        self.assertEqual(len(generated), 1)
        self.assertEqual(generated[0].read_bytes(), b"edited-image-bytes")
        self.assertEqual(len(ctx.pending_events), 1)
        event = ctx.pending_events[0]
        self.assertEqual(event["type"], "send_photo")
        self.assertEqual(event["chat_id"], 123)
        self.assertEqual(event["image_base64"], edited_b64)
        self.assertEqual(event["mime_type"], "image/png")

    def test_edit_image_tool_registered(self):
        from ouroboros.tools.registry import ToolRegistry

        registry = ToolRegistry(
            repo_dir=pathlib.Path("/tmp"),
            drive_root=pathlib.Path("/tmp"),
        )
        tools = registry.available_tools()
        self.assertIn("edit_image", tools, "edit_image must be registered")


if __name__ == "__main__":
    unittest.main()
