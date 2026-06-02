"""
Ouroboros — LLM client.

The only module that communicates with the main LLM API.
Contract: chat(), default_model(), available_models(), add_usage().
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENROUTER_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_OPENAI_MODEL = "gpt-5.2"
DEFAULT_OPENAI_CODE_MODEL = "gpt-5.2-codex"
DEFAULT_LIGHT_MODEL = "google/gemini-3-pro-preview"
DEFAULT_OPENAI_LIGHT_MODEL = "gpt-5.4-mini"


def normalize_llm_provider(value: Optional[str] = None) -> str:
    """Normalize LLM provider config to a supported provider id."""
    raw = str(value if value is not None else os.environ.get("OUROBOROS_LLM_PROVIDER", "openrouter"))
    provider = raw.strip().lower().replace("-", "_")
    if provider in ("openai", "openai_api", "native_openai"):
        return "openai"
    return "openrouter"


def default_main_model(provider: Optional[str] = None) -> str:
    """Return provider-aware default model."""
    return DEFAULT_OPENAI_MODEL if normalize_llm_provider(provider) == "openai" else DEFAULT_OPENROUTER_MODEL


def default_code_model(provider: Optional[str] = None) -> str:
    """Return provider-aware default code-capable model."""
    return DEFAULT_OPENAI_CODE_MODEL if normalize_llm_provider(provider) == "openai" else DEFAULT_OPENROUTER_MODEL


def default_light_model(provider: Optional[str] = None) -> str:
    """Return provider-aware default lightweight model."""
    return DEFAULT_OPENAI_LIGHT_MODEL if normalize_llm_provider(provider) == "openai" else DEFAULT_LIGHT_MODEL


def default_fallback_models(provider: Optional[str] = None) -> str:
    """Return provider-aware fallback chain as a comma-separated config string."""
    if normalize_llm_provider(provider) == "openai":
        return "gpt-5.2,gpt-4.1"
    return "google/gemini-2.5-pro-preview,openai/o3,anthropic/claude-sonnet-4.6"


def normalize_reasoning_effort(value: str, default: str = "medium") -> str:
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    v = str(value or "").strip().lower()
    return v if v in allowed else default


def reasoning_rank(value: str) -> int:
    order = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5}
    return int(order.get(str(value or "").strip().lower(), 3))


def add_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Accumulate usage from one LLM call into a running total."""
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens"):
        total[k] = int(total.get(k) or 0) + int(usage.get(k) or 0)
    if usage.get("cost"):
        total["cost"] = float(total.get("cost") or 0) + float(usage["cost"])


def fetch_openrouter_pricing() -> Dict[str, Tuple[float, float, float]]:
    """
    Fetch current pricing from OpenRouter API.

    Returns dict of {model_id: (input_per_1m, cached_per_1m, output_per_1m)}.
    Returns empty dict on failure.
    """
    import logging
    log = logging.getLogger("ouroboros.llm")

    try:
        import requests
    except ImportError:
        log.warning("requests not installed, cannot fetch pricing")
        return {}

    try:
        url = "https://openrouter.ai/api/v1/models"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()

        data = resp.json()
        models = data.get("data", [])

        # Prefixes we care about
        prefixes = ("anthropic/", "openai/", "google/", "meta-llama/", "x-ai/", "qwen/")

        pricing_dict = {}
        for model in models:
            model_id = model.get("id", "")
            if not model_id.startswith(prefixes):
                continue

            pricing = model.get("pricing", {})
            if not pricing or not pricing.get("prompt"):
                continue

            # OpenRouter pricing is in dollars per token (raw values)
            raw_prompt = float(pricing.get("prompt", 0))
            raw_completion = float(pricing.get("completion", 0))
            raw_cached_str = pricing.get("input_cache_read")
            raw_cached = float(raw_cached_str) if raw_cached_str else None

            # Convert to per-million tokens
            prompt_price = round(raw_prompt * 1_000_000, 4)
            completion_price = round(raw_completion * 1_000_000, 4)
            if raw_cached is not None:
                cached_price = round(raw_cached * 1_000_000, 4)
            else:
                cached_price = round(prompt_price * 0.1, 4)  # fallback: 10% of prompt

            # Sanity check: skip obviously wrong prices
            if prompt_price > 1000 or completion_price > 1000:
                log.warning(f"Skipping {model_id}: prices seem wrong (prompt={prompt_price}, completion={completion_price})")
                continue

            pricing_dict[model_id] = (prompt_price, cached_price, completion_price)

        log.info(f"Fetched pricing for {len(pricing_dict)} models from OpenRouter")
        return pricing_dict

    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning(f"Failed to fetch OpenRouter pricing: {e}")
        return {}


class LLMClient:
    """Provider-aware OpenAI SDK wrapper. All main LLM calls go through this class."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        self._provider = normalize_llm_provider(provider)
        if self._provider == "openai":
            self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            self._base_url = base_url or os.environ.get("OUROBOROS_LLM_BASE_URL", "") or OPENAI_BASE_URL
        else:
            self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
            self._base_url = base_url or os.environ.get("OUROBOROS_LLM_BASE_URL", "") or OPENROUTER_BASE_URL
        self._client = None

    @property
    def provider(self) -> str:
        return self._provider

    def supports_model(self, model: str) -> bool:
        """Return whether this provider can call the configured model id."""
        model = str(model or "").strip()
        if not model:
            return False
        if self._provider == "openai":
            return "/" not in model or model.startswith("openai/")
        return True

    def _api_model(self, model: str) -> str:
        model = str(model or "").strip()
        if self._provider == "openai":
            if model.startswith("openai/"):
                return model.split("/", 1)[1]
            if "/" in model:
                raise ValueError(
                    "OUROBOROS_LLM_PROVIDER=openai only supports OpenAI model ids; "
                    f"got {model!r}. Use a native id like 'gpt-5.2' or 'openai/gpt-5.2'."
                )
        return model

    def _supports_openai_reasoning_effort(self, model: str) -> bool:
        api_model = self._api_model(model).lower()
        return api_model.startswith("o") or api_model.startswith("gpt-5")

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs: Dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._provider == "openrouter":
                kwargs["default_headers"] = {
                    "HTTP-Referer": "https://colab.research.google.com/",
                    "X-Title": "Ouroboros",
                }
            self._client = OpenAI(**kwargs)
        return self._client

    def _fetch_generation_cost(self, generation_id: str) -> Optional[float]:
        """Fetch cost from OpenRouter Generation API as fallback."""
        if self._provider != "openrouter":
            return None
        try:
            import requests
            url = f"{self._base_url.rstrip('/')}/generation?id={generation_id}"
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
            # Generation might not be ready yet — retry once after short delay
            time.sleep(0.5)
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
        except Exception:
            log.debug("Failed to fetch generation cost from OpenRouter", exc_info=True)
            pass
        return None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 16384,
        tool_choice: str = "auto",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single LLM call. Returns: (response_message_dict, usage_dict with cost)."""
        client = self._get_client()
        effort = normalize_reasoning_effort(reasoning_effort)
        api_model = self._api_model(model)

        if self._provider == "openai":
            kwargs: Dict[str, Any] = {
                "model": api_model,
                "messages": messages,
                "max_completion_tokens": max_tokens,
            }
            if self._supports_openai_reasoning_effort(model):
                kwargs["reasoning_effort"] = effort
        else:
            extra_body: Dict[str, Any] = {
                "reasoning": {"effort": effort, "exclude": True},
            }

            # Pin Anthropic models to Anthropic provider for prompt caching
            if model.startswith("anthropic/"):
                extra_body["provider"] = {
                    "order": ["Anthropic"],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                }

            kwargs = {
                "model": api_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "extra_body": extra_body,
            }
        if tools:
            tools_with_cache = [t for t in tools]  # shallow copy
            if self._provider == "openrouter" and tools_with_cache:
                # Add cache_control to last tool for Anthropic prompt caching.
                # This caches all tool schemas (they never change between calls).
                last_tool = {**tools_with_cache[-1]}  # copy last tool
                last_tool["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
                tools_with_cache[-1] = last_tool
            kwargs["tools"] = tools_with_cache
            kwargs["tool_choice"] = tool_choice

        resp = client.chat.completions.create(**kwargs)
        resp_dict = resp.model_dump()
        usage = resp_dict.get("usage") or {}
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}

        # Extract cached_tokens from prompt_tokens_details if available
        if not usage.get("cached_tokens"):
            prompt_details = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict) and prompt_details.get("cached_tokens"):
                usage["cached_tokens"] = int(prompt_details["cached_tokens"])

        # Extract cache_write_tokens from prompt_tokens_details if available
        # OpenRouter: "cache_write_tokens"
        # Native Anthropic: "cache_creation_tokens" or "cache_creation_input_tokens"
        if not usage.get("cache_write_tokens"):
            prompt_details_for_write = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details_for_write, dict):
                cache_write = (prompt_details_for_write.get("cache_write_tokens")
                              or prompt_details_for_write.get("cache_creation_tokens")
                              or prompt_details_for_write.get("cache_creation_input_tokens"))
                if cache_write:
                    usage["cache_write_tokens"] = int(cache_write)

        # Ensure cost is present in usage (OpenRouter includes it, but fallback if missing)
        if not usage.get("cost"):
            gen_id = resp_dict.get("id") or ""
            if gen_id:
                cost = self._fetch_generation_cost(gen_id)
                if cost is not None:
                    usage["cost"] = cost

        return msg, usage

    def vision_query(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        model: str = "anthropic/claude-sonnet-4.6",
        max_tokens: int = 1024,
        reasoning_effort: str = "low",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Send a vision query to an LLM. Lightweight — no tools, no loop.

        Args:
            prompt: Text instruction for the model
            images: List of image dicts. Each dict must have either:
                - {"url": "https://..."} — for URL images
                - {"base64": "<b64>", "mime": "image/png"} — for base64 images
            model: VLM-capable model ID
            max_tokens: Max response tokens
            reasoning_effort: Effort level

        Returns:
            (text_response, usage_dict)
        """
        # Build multipart content
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            if "url" in img:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img["url"]},
                })
            elif "base64" in img:
                mime = img.get("mime", "image/png")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img['base64']}"},
                })
            else:
                log.warning("vision_query: skipping image with unknown format: %s", list(img.keys()))

        messages = [{"role": "user", "content": content}]
        response_msg, usage = self.chat(
            messages=messages,
            model=model,
            tools=None,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        text = response_msg.get("content") or ""
        return text, usage

    def default_model(self) -> str:
        """Return the single default model from env. LLM switches via tool if needed."""
        return os.environ.get("OUROBOROS_MODEL", "").strip() or default_main_model(self._provider)

    def available_models(self) -> List[str]:
        """Return list of available models from env (for switch_model tool schema)."""
        main = os.environ.get("OUROBOROS_MODEL", "").strip() or default_main_model(self._provider)
        code = os.environ.get("OUROBOROS_MODEL_CODE", "")
        light = os.environ.get("OUROBOROS_MODEL_LIGHT", "")
        models = []
        seen_api_models = set()
        for candidate in (main, code, light):
            candidate = str(candidate or "").strip()
            if not candidate or not self.supports_model(candidate):
                continue
            api_model = self._api_model(candidate)
            if api_model in seen_api_models:
                continue
            seen_api_models.add(api_model)
            models.append(candidate)
        return models
