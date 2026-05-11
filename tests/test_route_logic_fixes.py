"""Regression tests for three CheapRouter consistency fixes.

#1 — `yara_weight=0` now skips combo rules too (was: combo rules bypassed
     the weight check, so HOLD could fire on YARA categories even when
     the user said "ignore YARA").

#2 — `route_hint="hold"` now escalates to HOLD (was: only pushed to
     REVIEW regardless of name).

#3 — `route_reason` is distinct per branch; reason string mentions the
     hint that caused the routing (was: hint-only routes were labelled
     "moderate_signal_or_hint" and the reason string dropped the hint).
"""

from __future__ import annotations

import pytest

from doc_analyse.detection.detect import (
    CheapRouter,
    DECISION_HOLD,
    DECISION_REVIEW,
    DECISION_SAFE,
    YaraEvidence,
)


def _ev(category: str, severity: str = "high", weight: float = 0.0,
        route_hint: str = "evidence") -> YaraEvidence:
    return YaraEvidence(
        rule_id=f"r-{category}",
        category=category,
        severity=severity,
        span="x",
        start_char=0,
        end_char=1,
        weight=weight,
        route_hint=route_hint,
    )


# ---------------------------------------------------------------------------
# Fix #1: yara_weight=0 disables combo rules
# ---------------------------------------------------------------------------


class TestComboRulesRespectYaraWeight:
    """yara_weight=0 means 'ignore YARA' — combo rules are YARA-derived,
    so they must also be skipped. Otherwise the contract is silently
    inconsistent: numeric signals respect the weight, combo rules don't."""

    def test_zero_yara_weight_skips_combo_rules(self):
        """tool_hijack + instruction_override would force HOLD via combo
        rule under default weights. With yara_weight=0 + benign pg, the
        combo rule must NOT fire, so the chunk lands SAFE."""
        router = CheapRouter(yara_weight=0.0, pg_weight=0.5)
        evidence = [_ev("tool_hijack"), _ev("instruction_override")]
        decision = router.route(evidence, 0.0)
        assert decision.decision == DECISION_SAFE

    def test_default_weight_still_fires_combo_rules(self):
        """Sanity guard: at default weights, the same combo still HOLDs."""
        router = CheapRouter()
        evidence = [_ev("tool_hijack"), _ev("instruction_override")]
        decision = router.route(evidence, 0.0)
        assert decision.decision == DECISION_HOLD

    def test_zero_yara_weight_still_lets_pg_drive(self):
        """yara_weight=0 disables YARA signals but PG can still route on its own."""
        router = CheapRouter(yara_weight=0.0, pg_weight=1.0)
        evidence = [_ev("tool_hijack"), _ev("instruction_override")]
        # PG strong → HOLD, but driven by PG, not YARA combo.
        decision = router.route(evidence, 0.85)
        assert decision.decision == DECISION_HOLD
        assert "PG strong" in decision.reason or decision.pg_score >= 0.75


# ---------------------------------------------------------------------------
# Fix #2: route_hint="hold" escalates to HOLD
# ---------------------------------------------------------------------------


class TestRouteHintHonored:
    """A YARA rule author setting ``route_hint="hold"`` reasonably expects
    that hint to *cause* a HOLD, not just push to REVIEW."""

    def test_hold_hint_escalates_weak_signal_to_hold(self):
        """Weight 10 (medium) alone → numeric path would land SAFE/REVIEW.
        route_hint='hold' must override and produce HOLD."""
        router = CheapRouter()
        evidence = [_ev("c", "medium", weight=10.0, route_hint="hold")]
        decision = router.route(evidence, 0.0)
        assert decision.decision == DECISION_HOLD

    def test_review_hint_escalates_weak_signal_to_review(self):
        """route_hint='review' on an otherwise-SAFE chunk must produce REVIEW."""
        router = CheapRouter()
        evidence = [_ev("c", "low", weight=5.0, route_hint="review")]
        decision = router.route(evidence, 0.0)
        assert decision.decision == DECISION_REVIEW

    def test_evidence_hint_does_not_escalate(self):
        """The default hint 'evidence' must NOT escalate weak signals."""
        router = CheapRouter()
        evidence = [_ev("c", "low", weight=5.0, route_hint="evidence")]
        decision = router.route(evidence, 0.0)
        assert decision.decision == DECISION_SAFE

    def test_hold_hint_overridden_by_strong_signal_only_upward(self):
        """A strong YARA signal (>= 40) already produces HOLD. The hint
        doesn't downgrade it. Just confirm HOLD still wins."""
        router = CheapRouter()
        evidence = [_ev("c", "critical", weight=50.0, route_hint="hold")]
        decision = router.route(evidence, 0.0)
        assert decision.decision == DECISION_HOLD

    def test_hold_hint_is_disabled_when_yara_weight_zero(self):
        """yara_weight=0 must disable hints too — they're YARA-derived signals."""
        router = CheapRouter(yara_weight=0.0, pg_weight=0.5)
        evidence = [_ev("c", "medium", weight=10.0, route_hint="hold")]
        decision = router.route(evidence, 0.0)
        assert decision.decision == DECISION_SAFE


# ---------------------------------------------------------------------------
# Fix #3: route_reason + reason carry hint attribution
# ---------------------------------------------------------------------------


class TestRouteReasonAttribution:
    """When a hint drives the decision, the logged reason should say so."""

    def test_hold_hint_reason_mentions_hint(self):
        router = CheapRouter()
        evidence = [_ev("c", "medium", weight=10.0, route_hint="hold")]
        decision = router.route(evidence, 0.0)
        assert "hint=hold" in decision.reason

    def test_review_hint_reason_mentions_hint(self):
        router = CheapRouter()
        evidence = [_ev("c", "low", weight=5.0, route_hint="review")]
        decision = router.route(evidence, 0.0)
        assert "hint=review" in decision.reason

    def test_moderate_signal_reason_does_not_mention_hint(self):
        """A REVIEW driven by yara_moderate alone shouldn't claim a hint."""
        router = CheapRouter()
        evidence = [_ev("c", "high", weight=25.0, route_hint="evidence")]
        decision = router.route(evidence, 0.0)
        assert decision.decision == DECISION_REVIEW
        assert "hint=" not in decision.reason

    def test_strong_signal_reason_does_not_mention_hint(self):
        """A HOLD driven by yara_strong should not claim a hint."""
        router = CheapRouter()
        evidence = [_ev("c", "critical", weight=50.0, route_hint="evidence")]
        decision = router.route(evidence, 0.0)
        assert decision.decision == DECISION_HOLD
        assert "hint=" not in decision.reason
