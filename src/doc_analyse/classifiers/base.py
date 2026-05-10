from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from doc_analyse.classifiers.config import resolve_generation_config
from doc_analyse.prompt.loader import (
    PromptTemplateError,
    load_default_classification_prompt,
    load_default_system_prompt,
    render_classification_prompt,
)

logger = logging.getLogger(__name__)

VALID_VERDICTS = {"safe", "suspicious", "unsafe"}
VALID_SEVERITIES = {"low", "medium", "high", "critical"}


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


ClassifierPromptError = PromptTemplateError


class BaseClassifier(ABC):
    """Provider-neutral classifier contract used by verifier/workers.

    Concrete providers own SDK setup, request options, and response extraction.
    Callers should depend on this class so verifier logic never branches on
    OpenAI, Anthropic, Gemini, Groq, or any future provider.
    """

    provider_name = "base"
    default_model = ""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
    ) -> None:
        self.model = model or self.default_model
        if not self.model:
            raise ValueError("A model is required for this classifier.")

        generation_config = resolve_generation_config(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.temperature = generation_config.temperature
        self.max_tokens = generation_config.max_tokens
        self.system_prompt = load_default_system_prompt(system_prompt)
        self.user_prompt_template = load_default_classification_prompt(user_prompt_template)

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
        user_prompt = render_classification_prompt(
            self.user_prompt_template,
            text=text,
            metadata=metadata or {},
        )

        return (
            ClassifierMessage(role="system", content=self.system_prompt),
            ClassifierMessage(role="user", content=user_prompt),
        )

    def parse_response(self, raw_response: str) -> ClassificationResult:
        extracted = _extract_json_object(raw_response)

        # Try direct parse first
        try:
            data = json.loads(extracted)
        except (TypeError, json.JSONDecodeError):
            # Truncation fallback: try to recover partial JSON from truncated responses
            logger.warning(
                "classifier.parse_failed_primary",
                extra={
                    "event": "classifier.parse_failed_primary",
                    "provider": self.provider,
                    "model": self.model,
                    "reason": "invalid_json_trying_truncation_fallback",
                    "response_char_count": (
                        len(raw_response) if isinstance(raw_response, str) else None
                    ),
                },
            )
            data = _try_parse_with_truncation_fallback(extracted)

        if data is None:
            logger.warning(
                "classifier.parse_failed",
                extra={
                    "event": "classifier.parse_failed",
                    "provider": self.provider,
                    "model": self.model,
                    "reason": "invalid_json",
                    "response_char_count": (
                        len(raw_response) if isinstance(raw_response, str) else None
                    ),
                    "exception_type": "JSONDecodeError",
                },
            )
            raise ClassifierResponseError("Classifier returned invalid JSON.")

        if not isinstance(data, Mapping):
            logger.warning(
                "classifier.parse_failed",
                extra={
                    "event": "classifier.parse_failed",
                    "provider": self.provider,
                    "model": self.model,
                    "reason": "non_object_json",
                    "response_type": type(data).__name__,
                },
            )
            raise ClassifierResponseError("Classifier JSON response must be an object.")

        result = ClassificationResult.from_mapping(data, raw_response=raw_response)
        logger.debug(
            "classifier.parse_succeeded",
            extra={
                "event": "classifier.parse_succeeded",
                "provider": self.provider,
                "model": self.model,
                "verdict": result.verdict,
                "confidence": result.confidence,
                "reason_count": len(result.reasons),
                "finding_count": len(result.findings),
            },
        )
        return result

    @abstractmethod
    def _complete(self, messages: Sequence[ClassifierMessage]) -> str:
        """Return raw provider text for the normalized classifier messages."""
        raise NotImplementedError


def render_messages_for_single_prompt(messages: Sequence[ClassifierMessage]) -> str:
    return "\n\n".join(f"{message.role.upper()}:\n{message.content}" for message in messages)


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


def _try_parse_with_truncation_fallback(text: str) -> dict[str, Any] | None:
    """Try to parse JSON, handling truncation gracefully.

    MiniMax sometimes truncates long responses mid-string, leaving unclosed strings
    and arrays. This attempts to recover by stripping trailing incomplete strings
    and unclosed containers.
    """
    import json

    # First try: normal parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Second try: strip trailing incomplete strings (unclosed " at end)
    # Common pattern: '"reason": "Document contains...' (cut off mid-string)
    try:
        truncated = text
        for _ in range(5):
            try:
                return json.loads(truncated)
            except json.JSONDecodeError as exc:
                if exc.pos is None:
                    break
                pos = exc.pos
                truncated = truncated[:pos].rstrip()
                m = re.search(r'[^\\]"[,\s]*$', truncated)
                if m:
                    truncated = truncated[:m.end() - 1].rstrip() + '"}'
                else:
                    break
    except Exception:
        pass

    # Third try: handle unclosed arrays/objects at the end
    repaired = _force_close_unclosed_containers(text)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Fourth try: strip character by character from the end
    # This handles cases where a string is unclosed
    for _ in range(10):
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as exc:
            if exc.pos is None:
                break
            repaired = repaired[:exc.pos].rstrip()
            if repaired.endswith(","):
                repaired = repaired[:-1].rstrip() + "}"
            elif repaired.endswith("}"):
                repaired = repaired[:-1].rstrip()
            elif repaired.endswith("]"):
                repaired = repaired[:-1].rstrip()
            elif repaired.endswith('"'):
                repaired = repaired[:-1].rstrip() + '"'
            else:
                break

    return None


def _force_close_unclosed_containers(text: str) -> str:
    """Close any unclosed [ and { at the end of truncated JSON.

    Uses a stack-based approach to track open containers, then closes
    them from innermost to outermost (reverse of opening order).
    """
    stack = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in {"{", "["}:
            stack.append(ch)
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()

    if not stack:
        return text

    # Build the closing sequence: innermost first
    # stack=['{', '['] → reversed=['[', '{'] → closers='}]'
    # stack=['['] → reversed=['['] → closers=']'
    # stack=['{'] → reversed=['{'] → closers='}'
    closers = ""
    for opener in reversed(stack):
        closers += "}" if opener == "{" else "]"

    # Simply append the closers without stripping any existing content.
    # The text typically ends with a complete finding object '...end_char": N}'
    # and the top-level array [ and object { are unclosed.
    # Appending '}]' (close array, close top-level) is correct.
    # We do NOT strip the trailing } — stripping cuts into numeric data.
    return text.rstrip() + closers


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
