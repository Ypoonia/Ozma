from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class DetectionFinding:
    """Local evidence found before any LLM validation is called."""

    span: str
    category: str
    severity: str
    reason: str
    start_char: int
    end_char: int
    source: str
    rule_id: str
    requires_llm_validation: bool = False
    score: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.end_char - self.start_char
