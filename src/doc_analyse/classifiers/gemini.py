from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, Optional

from doc_analyse.classifiers.base import (
    BaseClassifier,
    ClassifierDependencyError,
    ClassifierMessage,
    ensure_api_key,
    render_messages_for_single_prompt,
    require_text_response,
)


class GeminiClassifier(BaseClassifier):
    provider_name = "gemini"
    default_model = "gemini-2.0-flash"

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
        prompt = render_messages_for_single_prompt(messages)
        config = self._build_generation_config()
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
            **self.request_options,
        )
        return _gemini_response_text(response)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        options = dict(self.client_options)
        ensure_api_key("Gemini", ("GEMINI_API_KEY", "GOOGLE_API_KEY"), self.api_key, options)

        try:
            from google import genai
        except ImportError as exc:
            raise ClassifierDependencyError(
                "GeminiClassifier requires the 'google-genai' package. "
                "Install with: pip install google-genai"
            ) from exc

        self._client = genai.Client(**options)
        return self._client

    def _build_generation_config(self) -> Any:
        try:
            from google.genai import types
        except ImportError as exc:
            raise ClassifierDependencyError(
                "GeminiClassifier requires the 'google-genai' package. "
                "Install with: pip install google-genai"
            ) from exc

        return types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            response_mime_type="application/json",
        )


def _gemini_response_text(response: Any) -> str:
    # Gemini exposes the generated text as a top-level convenience property.
    return require_text_response("Gemini", getattr(response, "text", None))
