from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, Optional

from doc_analyse.classifiers.base import (
    DEFAULT_SYSTEM_PROMPT,
    BaseClassifier,
    ClassifierDependencyError,
    ClassifierMessage,
    ensure_api_key,
    require_text_response,
)


class OpenAIClassifier(BaseClassifier):
    provider_name = "openai"
    default_model = "gpt-4o-mini"

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
        payload = [{"role": message.role, "content": message.content} for message in messages]

        if hasattr(client, "responses"):
            response = client.responses.create(
                model=self.model,
                input=payload,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                **self.request_options,
            )
            return _openai_response_text(response)

        response = client.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **self.request_options,
        )
        return _openai_chat_response_text(response)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        options = dict(self.client_options)
        ensure_api_key("OpenAI", ("OPENAI_API_KEY",), self.api_key, options)

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ClassifierDependencyError(
                "OpenAIClassifier requires the 'openai' package. Install with: pip install openai"
            ) from exc

        self._client = OpenAI(**options)
        return self._client


def _openai_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    # The Responses API nests text in output items; normalize all text parts together.
    text_parts = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if isinstance(text, str) and text.strip():
                text_parts.append(text)

    return require_text_response("OpenAI", "".join(text_parts))


def _openai_chat_response_text(response: Any) -> str:
    try:
        text = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        text = None

    return require_text_response("OpenAI", text)
