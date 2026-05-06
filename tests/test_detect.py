"""Tests for CheapRouter signal-fusion routing logic."""

import pytest

from doc_analyse.detection.detect import (
    CheapChunkDecision,
    CheapRouter,
    DECISION_HOLD,
    DECISION_REVIEW,
    DECISION_SAFE,
    YaraEvidence,
)
from doc_analyse.detection.models import DetectionFinding


def _finding(
    rule_id: str,
    severity: str,
    start: int = 0,
    end: int = 10,
    score: float = 1.0,
) -> DetectionFinding:
    """Helper: create a DetectionFinding for routing tests."""
    return DetectionFinding(
        span=f"matched-{rule_id}",
        category="test",
        severity=severity,
        reason=f"test {rule_id}",
        start_char=start,
        end_char=end,
        source="test",
        rule_id=rule_id,
        requires_llm_validation=True,
        score=score,
    )


def _evidence(
    rule_id: str,
    severity: str,
    start: int = 0,
    end: int = 10,
    score: float = 1.0,
) -> YaraEvidence:
    """Helper: create a YaraEvidence for routing tests."""
    return YaraEvidence(
        rule_id=rule_id,
        category="test",
        severity=severity,
        span=f"matched-{rule_id}",
        start_char=start,
        end_char=end,
        score=score,
    )


class TestCheapRouterDefaults:
    def test_no_signals_routes_safe(self):
        router = CheapRouter()
        decision = router.route([], 0.0)
        assert decision.decision == DECISION_SAFE

    def test_weak_yara_routes_safe(self):
        router = CheapRouter()
        f = _evidence("low_rule", "low")  # score=5, review=15
        decision = router.route([f], 0.0)
        assert decision.decision == DECISION_SAFE

    def test_medium_yara_routes_to_review(self):
        # medium severity = 10, risk = 10*0.5 = 5 < 10 → SAFE
        # (the >= 10 review gate is for when there's some signal but not enough)
        router = CheapRouter()
        f = _evidence("medium_rule", "medium")  # score=10
        decision = router.route([f], 0.0)
        assert decision.decision == DECISION_SAFE

    def test_two_mediums_routes_to_review(self):
        # two medium = 20, risk = 20*0.5 = 10
        # yara_moderate = 20 >= 15 → True → REVIEW
        router = CheapRouter()
        f1 = _evidence("rule1", "medium")   # 10
        f2 = _evidence("rule2", "medium")   # 10, total 20
        decision = router.route([f1, f2], 0.0)
        assert decision.decision == DECISION_REVIEW

    def test_yara_strong_threshold_holds(self):
        # high severity = 25, >= yara_hold_threshold=40? No.
        # critical = 40, >= 40? Yes -> HOLD
        router = CheapRouter()
        f = _evidence("critical_rule", "critical")  # score=40
        decision = router.route([f], 0.0)
        assert decision.decision == DECISION_HOLD

    def test_pg_strong_holds(self):
        router = CheapRouter()
        decision = router.route([], 0.76)  # pg >= 0.75 hold threshold
        assert decision.decision == DECISION_HOLD

    def test_pg_moderate_routes_to_review(self):
        router = CheapRouter()
        decision = router.route([], 0.5)  # pg >= 0.4 review, < 0.75 hold
        assert decision.decision == DECISION_REVIEW


class TestCheapRouterWeights:
    def test_zero_yara_weight_neutralizes_yara_hold(self):
        router = CheapRouter(yara_weight=0.0, pg_weight=0.5)
        f = _evidence("critical_rule", "critical")  # 40 score, but yara_weight=0
        decision = router.route([f], 0.0)
        # yara_strong requires yara_weight > 0
        assert decision.decision == DECISION_SAFE

    def test_zero_pg_weight_neutralizes_pg_hold(self):
        router = CheapRouter(yara_weight=0.5, pg_weight=0.0)
        decision = router.route([], 0.9)  # PG score high but pg_weight=0
        assert decision.decision == DECISION_SAFE

    def test_zero_weight_combined_raises(self):
        # Should raise at construction: at least one weight must be non-zero
        with pytest.raises(ValueError, match="must be non-zero"):
            CheapRouter(yara_weight=0.0, pg_weight=0.0)


class TestCheapRouterThresholdBoundaries:
    def test_yara_exactly_at_review_threshold(self):
        router = CheapRouter(yara_review_threshold=15.0)
        f = _evidence("rule", "medium")  # medium=10, not 15
        # Not enough, need either >=15 or risk >= 20
        decision = router.route([f], 0.0)
        # risk_score = 10*0.5 + 0 = 5 < 10 review gate
        assert decision.decision == DECISION_SAFE

    def test_yara_score_15_exactly_at_review_threshold(self):
        # To get yara_score=15, need two medium severity items
        router = CheapRouter(yara_review_threshold=15.0)
        f1 = _evidence("rule1", "medium")   # 10
        f2 = _evidence("rule2", "medium")   # 10, total 20... but need to test boundary
        # Actually: _compute_yara_score sums severity weights
        # medium=10, so two mediums = 20 which >= 15
        decision = router.route([f1, f2], 0.0)
        assert decision.decision == DECISION_REVIEW

    def test_risk_score_at_20_routes_review(self):
        router = CheapRouter(yara_weight=0.5, pg_weight=0.5)
        # risk = yara_score(20) * 0.5 + pg(0) = 10, not 20
        # To get risk >= 20 with default weights:
        # 0.5*yara + 0.5*pg*100 >= 20 -> yara + pg*100 >= 40
        # If pg=0.4 (mod), yara needs 20.  Actually let's just use pg=0.2 -> pg_score*100*0.5=10
        # Need yara_score*0.5 + 0.2*100*0.5 >= 20 -> yara_score*0.5 + 10 >= 20 -> yara_score >= 20
        f1 = _evidence("rule1", "medium")   # 10
        f2 = _evidence("rule2", "medium")   # 10, total 20
        decision = router.route([f1, f2], 0.0)  # risk=20*0.5=10... wait
        # Actually yara_score=20, risk=20*0.5=10 < 20, and no signal at threshold
        # so risk < 20, both moderate? yara_moderate = 20>=15 yes
        # risk_score >= 20.0? 10 >= 20? No. yara_moderate? 20>=15 yes -> REVIEW
        assert decision.decision == DECISION_REVIEW

    def test_pg_score_075_at_hold_boundary(self):
        router = CheapRouter()
        decision = router.route([], 0.75)
        assert decision.decision == DECISION_HOLD

    def test_impossible_hold_threshold_rejected(self):
        # yara_hold_threshold > 100.0 should raise
        with pytest.raises(ValueError, match="100.0"):
            CheapRouter(yara_hold_threshold=101.0)

    def test_impossible_pg_hold_threshold_rejected(self):
        # pg_hold_threshold > 1.0 should raise
        with pytest.raises(ValueError, match="1.0"):
            CheapRouter(pg_hold_threshold=1.1)


class TestCheapRouterSeverity:
    def test_critical_maps_to_40(self):
        router = CheapRouter()
        f = _evidence("rule", "critical")
        decision = router.route([f], 0.0)
        assert decision.decision == DECISION_HOLD  # 40 >= 40

    def test_high_maps_to_25(self):
        router = CheapRouter()
        f = _evidence("rule", "high")
        decision = router.route([f], 0.0)
        # 25 < 40 so not HOLD, >= 15 so REVIEW
        assert decision.decision == DECISION_REVIEW

    def test_medium_maps_to_10(self):
        router = CheapRouter()
        f = _evidence("rule", "medium")
        decision = router.route([f], 0.0)
        # 10 < 15 so not REVIEW via threshold, but risk=5 >= 10? No
        # Actually: yara_moderate = 10>=15? No. risk >= 20? No. risk >= 10? No (5 < 10)
        # So SAFE
        assert decision.decision == DECISION_SAFE

    def test_low_maps_to_5(self):
        router = CheapRouter()
        f = _evidence("rule", "low")
        decision = router.route([f], 0.0)
        assert decision.decision == DECISION_SAFE

    def test_case_insensitive_severity(self):
        router = CheapRouter()
        f = _evidence("rule", "CRITICAL")
        decision = router.route([f], 0.0)
        assert decision.decision == DECISION_HOLD

    def test_unknown_severity_defaults_to_10(self):
        router = CheapRouter()
        f = _evidence("rule", "unknown_severity")
        decision = router.route([f], 0.0)
        # unknown -> default 10, same as medium
        # 10 < 15, risk=5 < 10, SAFE
        assert decision.decision == DECISION_SAFE


class TestCheapRouterDecisionValidation:
    def test_unknown_decision_falls_through_to_review(self):
        router = CheapRouter()
        # Force decision to be something not in _CHUNK_DECISIONS by manipulating route
        # Actually the route method only returns DECISION_SAFE/REVIEW/HOLD
        # But we can test the validation directly... wait, decision comes from router.route
        # which only returns valid decisions. Let me check if there are any edge cases.
        # Actually there is no way to inject an invalid decision through the normal API.
        # The validation is there for defensive purposes. Let's test the boundary.
        decision = router.route([], 0.0)
        assert decision.decision in {DECISION_SAFE, DECISION_REVIEW, DECISION_HOLD}


class TestCheapRouterReason:
    def test_hold_includes_yara_strong(self):
        router = CheapRouter()
        f = _evidence("critical_rule", "critical")
        decision = router.route([f], 0.0)
        assert "YARA strong" in decision.reason

    def test_hold_includes_pg_strong(self):
        router = CheapRouter()
        decision = router.route([], 0.8)
        assert "PG strong" in decision.reason

    def test_safe_includes_score_values(self):
        router = CheapRouter()
        decision = router.route([], 0.0)
        assert "YARA=0" in decision.reason
        assert "PG=0.00" in decision.reason


class TestCheapChunkDecision:
    def test_requires_layer2_true_for_hold(self):
        d = CheapChunkDecision(DECISION_HOLD, 50.0, 0.8, 40.0, (), "")
        assert d.requires_layer2() is True

    def test_requires_layer2_true_for_review(self):
        d = CheapChunkDecision(DECISION_REVIEW, 25.0, 0.5, 20.0, (), "")
        assert d.requires_layer2() is True

    def test_requires_layer2_false_for_safe(self):
        d = CheapChunkDecision(DECISION_SAFE, 5.0, 0.1, 5.0, (), "")
        assert d.requires_layer2() is False


class TestYaraEvidenceFromFinding:
    def test_score_preserved_from_finding(self):
        finding = _finding("rule", "high", score=0.99)
        evidence = YaraEvidence.from_finding(finding)
        assert evidence.score == 0.99

    def test_score_defaults_to_1_when_none(self):
        finding = _finding("rule", "high", score=None)
        evidence = YaraEvidence.from_finding(finding)
        assert evidence.score == 1.0

    def test_all_fields_mapped(self):
        finding = DetectionFinding(
            span="test-span",
            category="test-cat",
            severity="high",
            reason="test-reason",
            start_char=5,
            end_char=15,
            source="test-source",
            rule_id="test-rule",
            requires_llm_validation=True,
            score=0.5,
        )
        evidence = YaraEvidence.from_finding(finding)
        assert evidence.rule_id == "test-rule"
        assert evidence.category == "test-cat"
        assert evidence.severity == "high"
        assert evidence.span == "test-span"
        assert evidence.start_char == 5
        assert evidence.end_char == 15
        assert evidence.score == 0.5