"""Data structures for Layer 1 cheap detection routing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from doc_analyse.detection.models import DetectionFinding

# Routing decisions
DECISION_SAFE = "safe"
DECISION_REVIEW = "review"
DECISION_HOLD = "hold"
_CHUNK_DECISIONS = frozenset({DECISION_SAFE, DECISION_REVIEW, DECISION_HOLD})
logger = logging.getLogger(__name__)

# Severity weights for YARA score
_SEVERITY_WEIGHTS: Mapping[str, float] = {
    "critical": 40.0,
    "high": 25.0,
    "medium": 10.0,
    "low": 5.0,
}


# Category-combination routing rules.
# Each entry is (required_categories, pg_gate, decision).
# pg_gate None = decision fires regardless of pg_score.
# pg_gate numeric = fires only when pg_score < gate (for "alone" downgrade rules).
#
# Note: a previous `(frozenset({"topic_mention"}), 0.10, DECISION_SAFE)` entry
# existed to downgrade meta-discussion documents to SAFE, but no YARA rule in
# default.yara ever produced the "topic_mention" category, so the rule was
# unreachable in production. Removed in favour of aligning config with the
# fail-closed direction set by the Layer 1 floor. The "exact match" / pg_gate
# code path in _check_category_combination_rules is kept intact so the
# downgrade semantic can be reintroduced cheaply if a topic_mention YARA rule
# is added later.
_CATEGORY_COMBINATION_RULES: tuple[tuple[frozenset[str], float | None, str], ...] = (
    # tool_hijack + instruction_override → always HOLD
    (frozenset({"tool_hijack", "instruction_override"}), None, DECISION_HOLD),
    # secret_exfiltration + instruction_override → always HOLD
    # covers hidden_prompt_exfiltration (category=secret_exfiltration) + instruction_override
    (frozenset({"secret_exfiltration", "instruction_override"}), None, DECISION_HOLD),
    # secret_exfiltration + tool_hijack → always HOLD
    (frozenset({"secret_exfiltration", "tool_hijack"}), None, DECISION_HOLD),
)


def _check_category_combination_rules(
    evidence: Sequence[YaraEvidence],
    pg_score: float,
) -> str | None:
    """Check category-combination rules first. Returns decision or None."""
    categories = {e.category for e in evidence}
    for required_categories, pg_gate, decision in _CATEGORY_COMBINATION_RULES:
        # Exact equality for "alone" rules (topic_mention), subset for combo rules
        if required_categories.issubset(categories):
            if len(required_categories) == len(categories):
                # Exact match: single-category "alone" rules
                if pg_gate is None:
                    return decision
                if pg_gate == 0.10 and pg_score < 0.10:
                    return decision
            else:
                # Superset match: multi-category combo rules
                if pg_gate is None:
                    return decision
    return None


@dataclass(frozen=True)
class YaraEvidence:
    rule_id: str
    category: str
    severity: str
    span: str
    start_char: int
    end_char: int
    weight: float = 0.0  # YARA rule weight (0 if not set)
    route_hint: str = "evidence"

    @classmethod
    def from_finding(cls, f: DetectionFinding) -> YaraEvidence:
        if isinstance(f, YaraEvidence):
            return f
        metadata = f.metadata or {}
        # Router reads raw YARA weight from metadata["yara_weight"].
        # yara.py stores normalized score in f.score; raw weight is in metadata.
        raw_weight = float(metadata.get("yara_weight", 0.0))
        return cls(
            rule_id=f.rule_id,
            category=f.category,
            severity=f.severity,
            span=f.span,
            start_char=f.start_char,
            end_char=f.end_char,
            weight=raw_weight,
            route_hint=str(metadata.get("route_hint", "evidence")),
        )


@dataclass(frozen=True)
class CheapChunkDecision:
    decision: str  # "safe" | "review" | "hold"
    risk_score: float  # 0-100
    pg_score: float  # 0.0-1.0
    yara_score: float  # 0-100
    findings: Sequence[YaraEvidence] = field(default=())
    reason: str = ""

    def requires_layer2(self) -> bool:
        return self.decision in {DECISION_REVIEW, DECISION_HOLD}


# Default thresholds
_YARA_REVIEW_THRESHOLD = 15.0
_YARA_HOLD_THRESHOLD = 40.0
_PG_REVIEW_THRESHOLD = 0.40
_PG_HOLD_THRESHOLD = 0.75
_YARA_WEIGHT = 0.50
_PG_WEIGHT = 0.50


class CheapRouter:
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
        if yara_weight == 0 and pg_weight == 0:
            raise ValueError("At least one of yara_weight or pg_weight must be non-zero")
        if not 0 <= yara_review_threshold <= yara_hold_threshold:
            raise ValueError("yara_review_threshold must be <= yara_hold_threshold")
        if not 0 <= pg_review_threshold <= pg_hold_threshold:
            raise ValueError("pg_review_threshold must be <= pg_hold_threshold")
        if yara_hold_threshold > 100.0:
            raise ValueError("yara_hold_threshold cannot exceed 100.0 (YARA score max)")
        if pg_hold_threshold > 1.0:
            raise ValueError("pg_hold_threshold cannot exceed 1.0 (PG score max)")

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
        # Accept YaraEvidence directly (from tests) or DetectionFinding (from production)
        evidence: tuple[YaraEvidence, ...] = tuple(
            f if isinstance(f, YaraEvidence) else YaraEvidence.from_finding(f)
            for f in yara_findings
        )
        raw_pg_score = pg_score
        pg_score = max(0.0, min(1.0, pg_score))
        if pg_score != raw_pg_score:
            logger.debug(
                "cheap_router_pg_score_clamped",
                extra={"input_pg_score": raw_pg_score, "pg_score": pg_score},
            )

        # Category-combination rules override numeric scoring
        combo_decision = _check_category_combination_rules(evidence, pg_score)
        if combo_decision is not None:
            yara_score = _compute_yara_score(evidence)
            risk_score = self._compute_risk_score(yara_score, pg_score)
            reason = _build_reason(yara_score, pg_score, evidence, False, False)
            _log_route_decision(
                decision=combo_decision,
                risk_score=risk_score,
                pg_score=pg_score,
                yara_score=yara_score,
                evidence=evidence,
                route_reason="category_combination",
            )
            return CheapChunkDecision(
                decision=combo_decision,
                risk_score=risk_score,
                pg_score=pg_score,
                yara_score=yara_score,
                findings=evidence,
                reason=reason,
            )

        # Collect advisory route hints — floor for final decision, not authoritative
        has_hold_hint = any(e.route_hint == "hold" for e in evidence)
        has_review_hint = any(e.route_hint == "review" for e in evidence)

        yara_score = _compute_yara_score(evidence)

        yara_strong = (
            self.yara_weight > 0
            and yara_score >= self.yara_hold_threshold
        )
        yara_moderate = (
            self.yara_weight > 0
            and yara_score >= self.yara_review_threshold
        )
        pg_strong = (
            self.pg_weight > 0
            and pg_score >= self.pg_hold_threshold
        )
        pg_moderate = (
            self.pg_weight > 0
            and pg_score >= self.pg_review_threshold
        )

        risk_score = self._compute_risk_score(yara_score, pg_score)

        if yara_strong or pg_strong:
            decision = DECISION_HOLD
            route_reason = "strong_signal"
            reason = _build_reason(yara_score, pg_score, evidence, yara_strong, pg_strong)
        elif yara_moderate or pg_moderate or risk_score >= 20.0 or has_hold_hint or has_review_hint:
            decision = DECISION_REVIEW
            route_reason = "moderate_signal_or_hint"
            reason = _build_reason(yara_score, pg_score, evidence, yara_moderate, pg_moderate)
        elif risk_score >= 10.0:
            decision = DECISION_REVIEW
            route_reason = "risk_score_floor"
            reason = _build_reason(yara_score, pg_score, evidence, False, False)
        else:
            decision = DECISION_SAFE
            route_reason = "weak_signals"
            reason = f"YARA={yara_score:.0f}, PG={pg_score:.2f} — both signals weak."

        _log_route_decision(
            decision=decision,
            risk_score=risk_score,
            pg_score=pg_score,
            yara_score=yara_score,
            evidence=evidence,
            route_reason=route_reason,
            has_hold_hint=has_hold_hint,
            has_review_hint=has_review_hint,
            yara_strong=yara_strong,
            yara_moderate=yara_moderate,
            pg_strong=pg_strong,
            pg_moderate=pg_moderate,
        )
        return CheapChunkDecision(
            decision=decision,
            risk_score=risk_score,
            pg_score=pg_score,
            yara_score=yara_score,
            findings=evidence,
            reason=reason,
        )

    def _compute_risk_score(self, yara_score: float, pg_score: float) -> float:
        """Blend YARA (0-100) and PG (0-1) signals, clamped to the 0-100
        range documented on ``CheapChunkDecision.risk_score``.

        With user-supplied weights summing above 1.0 (each individually
        capped at 1.0), the unclamped sum can exceed 100; the threshold
        ladder (>= 20.0, >= 10.0) and the dataclass docstring both treat
        ``risk_score`` as 0-100, so we clamp here rather than scattering
        ``min(100.0, ...)`` over every comparison.
        """
        blended = (yara_score * self.yara_weight) + (pg_score * 100.0 * self.pg_weight)
        return min(100.0, max(0.0, blended))


def _log_route_decision(
    *,
    decision: str,
    risk_score: float,
    pg_score: float,
    yara_score: float,
    evidence: Sequence[YaraEvidence],
    route_reason: str,
    has_hold_hint: bool = False,
    has_review_hint: bool = False,
    yara_strong: bool = False,
    yara_moderate: bool = False,
    pg_strong: bool = False,
    pg_moderate: bool = False,
) -> None:
    payload = {
        "decision": decision,
        "route_reason": route_reason,
        "risk_score": round(risk_score, 2),
        "pg_score": round(pg_score, 4),
        "yara_score": round(yara_score, 2),
        "yara_findings": len(evidence),
        "yara_categories": sorted({e.category for e in evidence}),
        "has_hold_hint": has_hold_hint,
        "has_review_hint": has_review_hint,
        "yara_strong": yara_strong,
        "yara_moderate": yara_moderate,
        "pg_strong": pg_strong,
        "pg_moderate": pg_moderate,
    }
    if decision == DECISION_SAFE:
        logger.debug("cheap_router_decision", extra=payload)
    else:
        logger.info("cheap_router_decision", extra=payload)


def _compute_yara_score(evidence: Sequence[YaraEvidence]) -> float:
    if not evidence:
        return 0.0
    # Deduplicate by category: only count the highest-weight finding per category.
    # This prevents repeated-match inflation when the same rule fires multiple times.
    best_by_category: dict[str, float] = {}
    for e in evidence:
        w = e.weight if e.weight > 0 else _SEVERITY_WEIGHTS.get(e.severity.strip().lower(), 10.0)
        if w > best_by_category.get(e.category, 0.0):
            best_by_category[e.category] = w
    return min(100.0, sum(best_by_category.values()))


def _build_reason(
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
    if yara_signal:
        parts.append("YARA strong")
    if pg_signal:
        parts.append("PG strong")
    return " | ".join(parts)
