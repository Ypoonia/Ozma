from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, Optional, Tuple

from doc_analyse.classifiers.base import (
    DEFAULT_SYSTEM_PROMPT,
    BaseClassifier,
    ClassifierDependencyError,
    ClassifierMessage,
    ensure_api_key,
    require_text_response,
)


class AnthropicClassifier(BaseClassifier):
    provider_name = "anthropic"
    default_model = "claude-3-5-haiku-latest"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        request_options: Optional[Dict[str, Any]] = None,
        **client_options: Any,
    ) -> None:
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
        self.api_key = api_key
        self._client = client
        self.client_options = client_options
        self.request_options = request_options or {}

    def _complete(self, messages: Sequence[ClassifierMessage]) -> str:
        client = self._get_client()
        system_prompt, user_messages = _split_anthropic_messages(messages)
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": user_messages,
            **self.request_options,
        }
        if system_prompt:
            params["system"] = system_prompt

        response = client.messages.create(**params)
        return _anthropic_response_text(response)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        options = dict(self.client_options)
        ensure_api_key("Anthropic", ("ANTHROPIC_API_KEY",), self.api_key, options)

        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ClassifierDependencyError(
                "AnthropicClassifier requires the 'anthropic' package. "
                "Install with: pip install anthropic"
            ) from exc

        self._client = Anthropic(**options)
        return self._client


def _split_anthropic_messages(
    messages: Sequence[ClassifierMessage],
) -> Tuple[str, Sequence[Dict[str, str]]]:
    system_parts = []
    user_messages = []

    for message in messages:
        if message.role == "system":
            system_parts.append(message.content)
        elif message.role in {"user", "assistant"}:
            user_messages.append({"role": message.role, "content": message.content})
        else:
            user_messages.append({"role": "user", "content": message.content})

    return "\n\n".join(system_parts), user_messages


def _anthropic_response_text(response: Any) -> str:
    # Anthropic returns content as blocks; only text blocks are usable classifier output.
    text_parts = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            text_parts.append(text)

    return require_text_response("Anthropic", "".join(text_parts))
