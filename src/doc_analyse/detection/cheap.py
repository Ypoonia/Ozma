from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from doc_analyse.detection.models import DetectionFinding


class CheapRouterError(ValueError):
    pass


# Routing decisions
DECISION_SAFE = "safe"
DECISION_REVIEW = "review"
DECISION_HOLD = "hold"
CHEAP_DECISIONS = {DECISION_SAFE, DECISION_REVIEW, DECISION_HOLD}

# Severity weights for YARA score calculation
_SEVERITY_WEIGHTS: Mapping[str, float] = {
    "critical": 40.0,
    "high": 25.0,
    "medium": 10.0,
    "low": 5.0,
}

# Default thresholds
_YARA_REVIEW_THRESHOLD = 15.0
_YARA_HOLD_THRESHOLD = 40.0
_PG_REVIEW_THRESHOLD = 0.40
_PG_HOLD_THRESHOLD = 0.75
_YARA_WEIGHT = 0.50
_PG_WEIGHT = 0.50


@dataclass(frozen=True)
class YaraEvidence:
    """YARA match evidence — routing signal source, not a finding."""

    rule_id: str
    category: str
    severity: str
    span: str
    start_char: int
    end_char: int

    @classmethod
    def from_finding(cls, f: DetectionFinding) -> YaraEvidence:
        return cls(
            rule_id=f.rule_id,
            category=f.category,
            severity=f.severity,
            span=f.span,
            start_char=f.start_char,
            end_char=f.end_char,
        )


@dataclass(frozen=True)
class CheapChunkDecision:
    """Output of Layer 1 cheap routing — source of truth for Layer 2 routing."""

    decision: str  # "safe" | "review" | "hold"
    risk_score: float  # 0–100
    pg_score: float  # 0.0–1.0
    yara_score: float  # 0–100
    findings: Sequence[YaraEvidence] = field(default_factory=tuple)
    reason: str = ""

    def requires_layer2(self) -> bool:
        return self.decision in {DECISION_REVIEW, DECISION_HOLD}


class CheapRouter:
    """Combines YARA evidence and Prompt Guard score into a routing decision.

    Decision logic:
      - HOLD if either signal is very strong (YARA > 40 or PG > 0.75)
      - REVIEW if combined risk > 20
      - SAFE otherwise
    """

    def __init__(
        self,
        yara_review_threshold: float = _YARA_REVIEW_THRESHOLD,
        yara_hold_threshold: float = _YARA_HOLD_THRESHOLD,
        pg_review_threshold: float = _PG_REVIEW_THRESHOLD,
        pg_hold_threshold: float = _PG_HOLD_THRESHOLD,
        yara_weight: float = _YARA_WEIGHT,
        pg_weight: float = _PG_WEIGHT,
    ) -> None:
        if not 0 <= yara_weight <= 1 or not 0 <= pg_weight <= 1:
            raise ValueError("Weights must be between 0 and 1")
        if not 0 <= yara_review_threshold <= yara_hold_threshold:
            raise ValueError("yara_review_threshold must be <= yara_hold_threshold")
        if not 0 <= pg_review_threshold <= pg_hold_threshold:
            raise ValueError("pg_review_threshold must be <= pg_hold_threshold")

        self.yara_review_threshold = yara_review_threshold
        self.yara_hold_threshold = yara_hold_threshold
        self.pg_review_threshold = pg_review_threshold
        self.pg_hold_threshold = pg_hold_threshold
        self.yara_weight = yara_weight
        self.pg_weight = pg_weight

    def route(
        self,
        yara_findings: Sequence[DetectionFinding],
        pg_score: float,
    ) -> CheapChunkDecision:
        evidence = tuple(YaraEvidence.from_finding(f) for f in yara_findings)
        yara_score = _compute_yara_score(evidence)
        pg_score = max(0.0, min(1.0, pg_score))

        yara_strong = yara_score >= self.yara_hold_threshold
        yara_moderate = yara_score >= self.yara_review_threshold
        pg_strong = pg_score >= self.pg_hold_threshold
        pg_moderate = pg_score >= self.pg_review_threshold

        risk_score = (yara_score * self.yara_weight) + (pg_score * 100 * self.pg_weight)

        if yara_strong or pg_strong:
            decision = DECISION_HOLD
            reason = _build_reason(decision, yara_score, pg_score, evidence, yara_strong, pg_strong)
        elif yara_moderate or pg_moderate or risk_score >= 20.0:
            decision = DECISION_REVIEW
            reason = _build_reason(decision, yara_score, pg_score, evidence, yara_moderate, pg_moderate)
        else:
            decision = DECISION_SAFE
            reason = f"YARA={yara_score:.0f}, PG={pg_score:.2f} — both signals weak."

        return CheapChunkDecision(
            decision=decision,
            risk_score=risk_score,
            pg_score=pg_score,
            yara_score=yara_score,
            findings=evidence,
            reason=reason,
        )


def _compute_yara_score(evidence: Sequence[YaraEvidence]) -> float:
    if not evidence:
        return 0.0
    score = sum(_SEVERITY_WEIGHTS.get(e.severity, 10.0) for e in evidence)
    return min(100.0, score)


def _build_reason(
    decision: str,
    yara_score: float,
    pg_score: float,
    evidence: Sequence[YaraEvidence],
    yara_signal: bool,
    pg_signal: bool,
) -> str:
    parts = [f"YARA={yara_score:.0f}, PG={pg_score:.2f}"]
    if evidence:
        rule_ids = sorted({e.rule_id for e in evidence})
        parts.append(f"YARA hits: {', '.join(rule_ids)}")
    if pg_signal:
        parts.append("PG strong")
    return " | ".join(parts)
