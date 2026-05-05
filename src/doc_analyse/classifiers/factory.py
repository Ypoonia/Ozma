from __future__ import annotations

import os
from typing import Any, Dict, Type

from doc_analyse.classifiers.anthropic import AnthropicClassifier
from doc_analyse.classifiers.base import BaseClassifier
from doc_analyse.classifiers.gemini import GeminiClassifier
from doc_analyse.classifiers.openai import OpenAIClassifier

PROVIDERS: Dict[str, Type[BaseClassifier]] = {
    "anthropic": AnthropicClassifier,
    "claude": AnthropicClassifier,
    "codex": OpenAIClassifier,
    "gemini": GeminiClassifier,
    "google": GeminiClassifier,
    "openai": OpenAIClassifier,
}

PROVIDER_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "codex": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def build_classifier(provider: str, **kwargs: Any) -> BaseClassifier:
    key = provider.strip().lower()
    classifier_type = PROVIDERS.get(key)
    if classifier_type is None:
        available = ", ".join(sorted(PROVIDERS))
        raise ValueError(
            f"Unknown classifier provider '{provider}'. Available providers: {available}"
        )

    return classifier_type(**kwargs)


def classifier_from_env(prefix: str = "DOC_ANALYSE_LLM", **kwargs: Any) -> BaseClassifier:
    provider = os.getenv(f"{prefix}_PROVIDER", "openai").strip().lower()
    model = os.getenv(f"{prefix}_MODEL")
    api_key = os.getenv(f"{prefix}_API_KEY") or _provider_api_key(provider)

    classifier_kwargs: Dict[str, Any] = dict(kwargs)
    if model and "model" not in classifier_kwargs:
        classifier_kwargs["model"] = model
    if api_key and "api_key" not in classifier_kwargs:
        classifier_kwargs["api_key"] = api_key

    return build_classifier(provider, **classifier_kwargs)


def _provider_api_key(provider: str) -> str:
    env_name = PROVIDER_API_KEY_ENV.get(provider)
    if env_name is None:
        return ""

    if provider in {"gemini", "google"}:
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")

    return os.getenv(env_name, "")
