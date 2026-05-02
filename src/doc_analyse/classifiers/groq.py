from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, Optional

from doc_analyse.classifiers.base import (
    DEFAULT_SYSTEM_PROMPT,
    BaseClassifier,
    ClassifierDependencyError,
    ClassifierMessage,
)


class GroqClassifier(BaseClassifier):
    provider_name = "groq"
    default_model = "llama-3.1-8b-instant"

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
        response = client.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            **self.request_options,
        )
        return response.choices[0].message.content or ""

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from groq import Groq
        except ImportError as exc:
            raise ClassifierDependencyError(
                "GroqClassifier requires the 'groq' package. Install with: pip install groq"
            ) from exc

        options = dict(self.client_options)
        if self.api_key:
            options["api_key"] = self.api_key

        self._client = Groq(**options)
        return self._client
