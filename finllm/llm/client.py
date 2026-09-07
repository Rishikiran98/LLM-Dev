"""LLM clients behind one small interface so the analyst layer is testable offline.

``ClaudeClient`` talks to the Claude API through the official ``anthropic`` SDK
(``pip install "finllm[llm]"``). ``FakeLLMClient`` returns canned responses for
unit tests and demos that must not hit the network.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Protocol, TypeVar

from pydantic import BaseModel

from finllm.config import LLMSettings

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    def complete(self, *, system: str, user: str) -> str:
        """Return plain text for a single-turn prompt."""
        ...

    def complete_structured(self, *, system: str, user: str, schema: type[T]) -> T:
        """Return an instance of ``schema`` parsed from the model's response."""
        ...


class ClaudeClient:
    """Thin wrapper over ``anthropic.Anthropic``.

    Credentials are resolved by the SDK (``ANTHROPIC_API_KEY`` or an
    ``ant auth login`` profile). Thinking is adaptive and the effort level
    comes from settings.
    """

    def __init__(self, settings: LLMSettings | None = None, client=None) -> None:
        self.settings = settings or LLMSettings()
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    'The Claude client needs the anthropic SDK: pip install "finllm[llm]"'
                ) from exc
            client = anthropic.Anthropic()
        self._client = client

    def _common(self) -> dict:
        return {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.settings.effort},
        }

    @staticmethod
    def _check_refusal(response) -> None:
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise RuntimeError(f"Claude declined the request (category={category})")

    def complete(self, *, system: str, user: str) -> str:
        response = self._client.messages.create(
            system=system,
            messages=[{"role": "user", "content": user}],
            **self._common(),
        )
        self._check_refusal(response)
        return "".join(block.text for block in response.content if block.type == "text")

    def complete_structured(self, *, system: str, user: str, schema: type[T]) -> T:
        response = self._client.messages.parse(
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
            **self._common(),
        )
        self._check_refusal(response)
        parsed = response.parsed_output
        if parsed is None:  # pragma: no cover - defensive
            raise RuntimeError("Claude returned no parseable structured output")
        return parsed


class FakeLLMClient:
    """Deterministic stand-in for tests. Records every prompt it receives."""

    def __init__(
        self,
        text: str = "FAKE ANSWER",
        structured: Callable[[type[BaseModel], str], BaseModel] | None = None,
    ) -> None:
        self.text = text
        self._structured = structured
        self.calls: list[dict[str, str]] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        return self.text

    def complete_structured(self, *, system: str, user: str, schema: type[T]) -> T:
        self.calls.append({"system": system, "user": user})
        if self._structured is not None:
            return self._structured(schema, user)  # type: ignore[return-value]
        # Build a minimal valid instance from the schema's example, if any.
        example = schema.model_config.get("json_schema_extra", {}).get("example")
        if example is None:
            raise ValueError(f"No fake response configured for {schema.__name__}")
        return schema.model_validate(json.loads(json.dumps(example)))
