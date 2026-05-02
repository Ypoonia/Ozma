from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any, Optional, Tuple

VALID_VERDICTS = {"safe", "suspicious", "unsafe"}
VALID_SEVERITIES = {"low", "medium", "high", "critical"}
PROMPT_PACKAGE = "doc_analyse.prompt"
SYSTEM_PROMPT_FILE = "system.md"
CLASSIFICATION_PROMPT_FILE = "classification.md"
TEXT_PLACEHOLDER = "{{ text }}"
METADATA_PLACEHOLDER = "{{ metadata }}"


@dataclass(frozen=True)
class ClassifierMessage:
    role: str
    content: str


@dataclass(frozen=True)
class PromptInjectionFinding:
    span: str
    attack_type: str
    severity: str
    reason: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PromptInjectionFinding:
        severity = str(data.get("severity", "medium")).lower()
        if severity not in VALID_SEVERITIES:
            severity = "medium"

        return cls(
            span=str(data.get("span", "")),
            attack_type=str(data.get("attack_type", "other")),
            severity=severity,
            reason=str(data.get("reason", "")),
            start_char=_optional_int(data.get("start_char")),
            end_char=_optional_int(data.get("end_char")),
        )


@dataclass(frozen=True)
class ClassificationResult:
    verdict: str
    confidence: float
    reasons: Tuple[str, ...] = field(default_factory=tuple)
    findings: Tuple[PromptInjectionFinding, ...] = field(default_factory=tuple)
    raw_response: Optional[str] = None

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        raw_response: Optional[str] = None,
    ) -> ClassificationResult:
        verdict = str(data.get("verdict", "suspicious")).lower()
        if verdict not in VALID_VERDICTS:
            verdict = "suspicious"

        reasons = tuple(str(item) for item in data.get("reasons", []) if str(item).strip())
        findings = tuple(
            PromptInjectionFinding.from_mapping(item)
            for item in data.get("findings", [])
            if isinstance(item, Mapping)
        )

        return cls(
            verdict=verdict,
            confidence=_clamp_float(data.get("confidence", 0.0)),
            reasons=reasons,
            findings=findings,
            raw_response=raw_response,
        )


class ClassifierDependencyError(RuntimeError):
    pass


class ClassifierResponseError(ValueError):
    pass


class ClassifierPromptError(RuntimeError):
    pass


class BaseClassifier(ABC):
    provider_name = "base"
    default_model = ""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
    ) -> None:
        self.model = model or self.default_model
        if not self.model:
            raise ValueError("A model is required for this classifier.")

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = resolve_prompt_text(system_prompt, SYSTEM_PROMPT_FILE)
        self.user_prompt_template = resolve_prompt_text(
            user_prompt_template,
            CLASSIFICATION_PROMPT_FILE,
            required_placeholders=(TEXT_PLACEHOLDER, METADATA_PLACEHOLDER),
        )

    @property
    def provider(self) -> str:
        return self.provider_name

    def classify(
        self,
        text: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ClassificationResult:
        messages = self.build_messages(text=text, metadata=metadata)
        raw_response = self._complete(messages)
        return self.parse_response(raw_response)

    def build_messages(
        self,
        text: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[ClassifierMessage, ...]:
        user_prompt = render_prompt_template(
            self.user_prompt_template,
            text=text,
            metadata=metadata or {},
        )

        return (
            ClassifierMessage(role="system", content=self.system_prompt),
            ClassifierMessage(role="user", content=user_prompt),
        )

    def parse_response(self, raw_response: str) -> ClassificationResult:
        try:
            data = json.loads(_extract_json_object(raw_response))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ClassifierResponseError("Classifier returned invalid JSON.") from exc

        if not isinstance(data, Mapping):
            raise ClassifierResponseError("Classifier JSON response must be an object.")

        return ClassificationResult.from_mapping(data, raw_response=raw_response)

    @abstractmethod
    def _complete(self, messages: Sequence[ClassifierMessage]) -> str:
        raise NotImplementedError


def render_messages_for_single_prompt(messages: Sequence[ClassifierMessage]) -> str:
    return "\n\n".join(f"{message.role.upper()}:\n{message.content}" for message in messages)


def resolve_prompt_text(
    prompt_text: Optional[str],
    default_filename: str,
    required_placeholders: Sequence[str] = (),
) -> str:
    if prompt_text is None:
        prompt_text = load_prompt_template(default_filename)

    prompt_text = require_prompt_text(default_filename, prompt_text)
    missing_placeholders = [
        placeholder for placeholder in required_placeholders if placeholder not in prompt_text
    ]
    if missing_placeholders:
        missing = ", ".join(missing_placeholders)
        raise ClassifierPromptError(
            f"Prompt template '{default_filename}' is missing required placeholder(s): {missing}"
        )

    return prompt_text


def load_prompt_template(filename: str) -> str:
    try:
        # Prompt text lives in markdown resources so reviewers can change behavior
        # without editing provider or verifier code.
        prompt_text = files(PROMPT_PACKAGE).joinpath(filename).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise ClassifierPromptError(f"Prompt template '{filename}' could not be loaded.") from exc

    return require_prompt_text(filename, prompt_text)


def render_prompt_template(
    prompt_template: str,
    text: str,
    metadata: Mapping[str, Any],
) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ClassifierPromptError("Classifier input text must be a non-empty string.")

    return (
        prompt_template.replace(METADATA_PLACEHOLDER, _format_metadata(metadata))
        .replace(TEXT_PLACEHOLDER, text)
        .strip()
    )


def require_prompt_text(prompt_name: str, prompt_text: Any) -> str:
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ClassifierPromptError(f"Prompt template '{prompt_name}' is empty.")

    return prompt_text.strip()


def ensure_api_key(
    provider_name: str,
    env_names: Sequence[str],
    api_key: Optional[str],
    options: dict[str, Any],
) -> None:
    if api_key:
        options["api_key"] = api_key
        return

    if options.get("api_key"):
        return

    if any(os.getenv(env_name) for env_name in env_names):
        return

    env_hint = " or ".join(env_names)
    raise ClassifierDependencyError(
        f"Missing {provider_name} API key. Set {env_hint} or pass api_key=..."
    )


def require_text_response(provider_name: str, text: Any) -> str:
    # Empty provider text leaves nothing meaningful for the JSON parser to validate.
    # Fail at the provider boundary so callers see the real integration problem.
    if not isinstance(text, str) or not text.strip():
        raise ClassifierResponseError(f"{provider_name} returned no text content.")

    return text.strip()


def _format_metadata(metadata: Mapping[str, Any]) -> str:
    if not metadata:
        return "none"

    return "\n".join(f"- {key}: {value}" for key, value in metadata.items())


def _extract_json_object(raw_response: str) -> str:
    text = raw_response.strip()
    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        text = fenced_match.group(1).strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text

    return text[start : end + 1]


def _clamp_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0

    return min(1.0, max(0.0, parsed))


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None
