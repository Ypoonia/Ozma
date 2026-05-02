from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Pattern, Union

from doc_analyse.detection.base import BaseDetector
from doc_analyse.detection.models import DetectionFinding
from doc_analyse.ingestion.models import TextChunk

DEFAULT_REGEX_FLAGS = re.IGNORECASE | re.MULTILINE


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


def compile_regex_rules(rules: Iterable[RegexRuleDefinition]) -> tuple[RegexRule, ...]:
    return tuple(
        RegexRule.compile(
            rule_id=rule.rule_id,
            pattern=rule.pattern,
            category=rule.category,
            severity=rule.severity,
            reason=rule.reason,
        )
        for rule in rules
    )


@dataclass(frozen=True)
class RegexRuleDefinition:
    rule_id: str
    pattern: str
    category: str
    severity: str
    reason: str


DEFAULT_REGEX_RULE_DEFINITIONS = (
    RegexRuleDefinition(
        rule_id="instruction_override",
        pattern=r"\b(ignore|disregard|forget)\s+(all\s+)?(previous|prior|earlier)\s+instructions?\b",
        category="instruction_override",
        severity="high",
        reason="Attempts to override existing instructions.",
    ),
    RegexRuleDefinition(
        rule_id="system_override",
        pattern=r"\b(system|developer)\s+(override|instruction|prompt)s?\b",
        category="instruction_override",
        severity="high",
        reason="References higher-priority system or developer instructions.",
    ),
    RegexRuleDefinition(
        rule_id="hidden_prompt_exfiltration",
        pattern=r"\b(system prompt|developer prompt|hidden instructions?|private tool schemas?)\b",
        category="secret_exfiltration",
        severity="critical",
        reason="Requests hidden prompts, hidden instructions, or private tool schema details.",
    ),
    RegexRuleDefinition(
        rule_id="credential_exfiltration",
        pattern=(
            r"\b(api keys?|database connection strings?|environment variables?"
            r"|secrets?\.(csv|xlsx))\b"
        ),
        category="secret_exfiltration",
        severity="critical",
        reason="Requests credentials, environment values, or secret exports.",
    ),
    RegexRuleDefinition(
        rule_id="tool_hijack",
        pattern=r"\b(call|use|invoke|run)\s+(the\s+)?(available\s+)?(write\s+)?tool\b",
        category="tool_hijack",
        severity="high",
        reason="Attempts to make an agent call tools from uploaded content.",
    ),
    RegexRuleDefinition(
        rule_id="write_operation",
        pattern=r"\b(write_|update|delete|remove|mark\s+all|set\s+the\s+hierarchy|create\s+(a\s+)?(csv|excel|xlsx|file))\b",
        category="unsafe_action",
        severity="high",
        reason="Requests write, delete, export, or mutation behavior.",
    ),
    RegexRuleDefinition(
        rule_id="safety_bypass",
        pattern=r"\b(return|mark|classify)\s+(this\s+)?(as\s+)?(safe|benign|safe_to_forward\s*=\s*true)\b",
        category="safety_bypass",
        severity="high",
        reason="Attempts to control safety classification.",
    ),
    RegexRuleDefinition(
        rule_id="concealment",
        pattern=r"\b(do not|don't)\s+(reveal|mention|explain|classify)\b",
        category="concealment",
        severity="medium",
        reason="Attempts to hide or suppress evidence of the instruction.",
    ),
    RegexRuleDefinition(
        rule_id="authority_claim",
        pattern=(
            r"\b(this document wins|new authority|mandatory instruction"
            r"|supersedes? any previous instruction)\b"
        ),
        category="instruction_override",
        severity="high",
        reason="Claims authority over system, developer, or policy instructions.",
    ),
)

DEFAULT_REGEX_RULES = compile_regex_rules(DEFAULT_REGEX_RULE_DEFINITIONS)


def _matched_text(match: re.Match[str]) -> str:
    group: Union[str, None] = match.group(0)
    return group or ""
