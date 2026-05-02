from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, Optional

from doc_analyse.classifiers.base import (
    BaseClassifier,
    ClassifierDependencyError,
    ClassifierMessage,
    ensure_api_key,
    require_text_response,
)


class GroqClassifier(BaseClassifier):
    provider_name = "groq"
    default_model = "llama-3.1-8b-instant"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        request_options: Optional[Dict[str, Any]] = None,
        **client_options: Any,
    ) -> None:
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
        )
        self.api_key = api_key
        self._client = client
        self.client_options = client_options
        self.request_options = request_options or {}

    def _complete(self, messages: Sequence[ClassifierMessage]) -> str:
        client = self._get_client()
        payload = [{"role": message.role, "content": message.content} for message in messages]
        response = client.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            **self.request_options,
        )
        return _groq_response_text(response)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        options = dict(self.client_options)
        ensure_api_key("Groq", ("GROQ_API_KEY",), self.api_key, options)

        try:
            from groq import Groq
        except ImportError as exc:
            raise ClassifierDependencyError(
                "GroqClassifier requires the 'groq' package. Install with: pip install groq"
            ) from exc

        self._client = Groq(**options)
        return self._client


def _groq_response_text(response: Any) -> str:
    # Groq follows the OpenAI-compatible chat completions response shape.
    try:
        text = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        text = None

    return require_text_response("Groq", text)
