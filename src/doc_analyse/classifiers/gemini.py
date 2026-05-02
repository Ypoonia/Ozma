from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from doc_analyse.classifiers.base import (
    BaseClassifier,
    ClassifierDependencyError,
    ClassifierMessage,
    DEFAULT_SYSTEM_PROMPT,
    render_messages_for_single_prompt,
)


class GeminiClassifier(BaseClassifier):
    provider_name = "gemini"
    default_model = "gemini-2.0-flash"

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
        prompt = render_messages_for_single_prompt(messages)
        config = self._build_generation_config()
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
            **self.request_options,
        )
        return getattr(response, "text", "") or ""

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from google import genai
        except ImportError as exc:
            raise ClassifierDependencyError(
                "GeminiClassifier requires the 'google-genai' package. Install with: pip install google-genai"
            ) from exc

        options = dict(self.client_options)
        if self.api_key:
            options["api_key"] = self.api_key

        self._client = genai.Client(**options)
        return self._client

    def _build_generation_config(self) -> Any:
        try:
            from google.genai import types
        except ImportError as exc:
            raise ClassifierDependencyError(
                "GeminiClassifier requires the 'google-genai' package. Install with: pip install google-genai"
            ) from exc

        return types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            response_mime_type="application/json",
        )
