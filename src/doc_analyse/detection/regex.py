from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable, Optional, Pattern, Union

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in broken runtime installs.
    yaml = None

from doc_analyse.detection.base import BaseDetector
from doc_analyse.detection.models import DetectionFinding
from doc_analyse.ingestion.models import TextChunk

DEFAULT_REGEX_FLAGS = re.IGNORECASE | re.MULTILINE
DEFAULT_REGEX_GLOSSARY = "glossary.yaml"
_GLOSSARY_PACKAGE = "doc_analyse.detection"
_REQUIRED_RULE_FIELDS = ("rule_id", "pattern", "category", "severity", "reason")


class RegexGlossaryError(ValueError):
    """Raised when a regex glossary file is missing, invalid, or incomplete."""


@dataclass(frozen=True)
class RegexRule:
    rule_id: str
    pattern: Pattern[str]
    category: str
    severity: str
    reason: str

    @classmethod
    def compile(
        cls,
        rule_id: str,
        pattern: str,
        category: str,
        severity: str,
        reason: str,
        flags: int = DEFAULT_REGEX_FLAGS,
    ) -> RegexRule:
        return cls(
            rule_id=rule_id,
            pattern=re.compile(pattern, flags=flags),
            category=category,
            severity=severity,
            reason=reason,
        )


class RegexDetector(BaseDetector):
    """Pure-regex cheap detector for prompt-injection-like evidence."""

    def __init__(self, rules: Optional[Iterable[RegexRule]] = None) -> None:
        self.rules = DEFAULT_REGEX_RULES if rules is None else tuple(rules)

    @classmethod
    def from_glossary(cls, path: Union[str, Path]) -> RegexDetector:
        """Build a detector from a project-specific YAML glossary."""
        return cls(rules=compile_regex_rules(load_regex_rule_definitions(path)))

    def detect(self, chunk: TextChunk) -> tuple[DetectionFinding, ...]:
        if not isinstance(chunk.text, str) or not chunk.text.strip():
            return ()

        findings = []
        seen = set()
        for rule in self.rules:
            for match in rule.pattern.finditer(chunk.text):
                span = _matched_text(match)
                if not span.strip():
                    continue

                start_char = chunk.start_char + match.start()
                end_char = chunk.start_char + match.end()
                key = (rule.rule_id, start_char, end_char, span)
                if key in seen:
                    continue

                seen.add(key)
                findings.append(
                    DetectionFinding(
                        span=span,
                        category=rule.category,
                        severity=rule.severity,
                        reason=rule.reason,
                        start_char=start_char,
                        end_char=end_char,
                        source=chunk.source,
                        rule_id=rule.rule_id,
                        metadata=dict(chunk.metadata),
                    )
                )

        return tuple(sorted(findings, key=lambda finding: (finding.start_char, finding.end_char)))


@dataclass(frozen=True)
class RegexRuleDefinition:
    rule_id: str
    pattern: str
    category: str
    severity: str
    reason: str


def load_regex_rule_definitions(
    path: Optional[Union[str, Path]] = None,
) -> tuple[RegexRuleDefinition, ...]:
    """Load regex rule definitions from YAML.

    Passing no path loads the packaged default glossary. Passing a path lets
    applications extend or replace the built-in rules without changing code.
    """
    source = DEFAULT_REGEX_GLOSSARY if path is None else str(path)
    return parse_regex_glossary(_read_glossary_text(path), source=source)


def parse_regex_glossary(text: str, *, source: str = "<memory>") -> tuple[RegexRuleDefinition, ...]:
    if yaml is None:
        raise RegexGlossaryError("PyYAML is required to load regex glossary files.")

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RegexGlossaryError(f"Invalid regex glossary YAML in {source}: {exc}") from exc

    if not isinstance(data, Mapping):
        raise RegexGlossaryError(f"Regex glossary {source} must be a mapping with a 'rules' list.")

    rules = data.get("rules")
    if not isinstance(rules, list) or not rules:
        raise RegexGlossaryError(f"Regex glossary {source} must define a non-empty 'rules' list.")

    return tuple(
        _rule_definition_from_mapping(rule, index=index, source=source)
        for index, rule in enumerate(rules, start=1)
    )


def compile_regex_rules(rules: Iterable[RegexRuleDefinition]) -> tuple[RegexRule, ...]:
    compiled_rules = []
    for rule in rules:
        try:
            compiled_rules.append(
                RegexRule.compile(
                    rule_id=rule.rule_id,
                    pattern=rule.pattern,
                    category=rule.category,
                    severity=rule.severity,
                    reason=rule.reason,
                )
            )
        except re.error as exc:
            raise RegexGlossaryError(
                f"Invalid regex pattern for rule '{rule.rule_id}': {exc}"
            ) from exc

    return tuple(compiled_rules)


def _read_glossary_text(path: Optional[Union[str, Path]]) -> str:
    try:
        if path is None:
            return (
                files(_GLOSSARY_PACKAGE)
                .joinpath(DEFAULT_REGEX_GLOSSARY)
                .read_text(encoding="utf-8")
            )
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        source = DEFAULT_REGEX_GLOSSARY if path is None else str(path)
        raise RegexGlossaryError(f"Could not read regex glossary {source}: {exc}") from exc


def _rule_definition_from_mapping(
    rule: Any,
    *,
    index: int,
    source: str,
) -> RegexRuleDefinition:
    if not isinstance(rule, Mapping):
        raise RegexGlossaryError(f"Rule #{index} in {source} must be a mapping.")

    values = {}
    for field in _REQUIRED_RULE_FIELDS:
        value = rule.get(field)
        if not isinstance(value, str) or not value.strip():
            raise RegexGlossaryError(
                f"Rule #{index} in {source} must define a non-empty string '{field}'."
            )
        values[field] = value.strip()

    return RegexRuleDefinition(
        rule_id=values["rule_id"],
        pattern=values["pattern"],
        category=values["category"],
        severity=values["severity"],
        reason=values["reason"],
    )


DEFAULT_REGEX_RULE_DEFINITIONS = load_regex_rule_definitions()
DEFAULT_REGEX_RULES = compile_regex_rules(DEFAULT_REGEX_RULE_DEFINITIONS)


def _matched_text(match: re.Match[str]) -> str:
    group: Union[str, None] = match.group(0)
    return group or ""
