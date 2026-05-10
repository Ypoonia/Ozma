r"""Tests for the Layer 1 floor on Layer 2 verdicts.

A Layer 1 HOLD or REVIEW must NEVER be downgraded to SAFE by Layer 2 —
that would defeat the cheap-detector signal and let an attacker who
fools the LLM bypass the entire pipeline.

The matrix this file locks in:

    Layer 1 \ Layer 2     SAFE        SUSPICIOUS   UNSAFE
    ------------------    ----------  -----------  ---------
    SAFE  (not routed)    SAFE        n/a          n/a
    REVIEW                SUSPICIOUS  SUSPICIOUS   UNSAFE
    HOLD                  SUSPICIOUS  SUSPICIOUS   UNSAFE

The SUSPICIOUS-floor for routed chunks is the security invariant.
Layer 2 can only upgrade routed chunks (SUSPICIOUS → UNSAFE), never
downgrade them (SUSPICIOUS → SAFE).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_analyse.classifiers import ClassificationResult
from doc_analyse.detection.detect import (
    CheapChunkDecision,
    DECISION_HOLD,
    DECISION_REVIEW,
    DECISION_SAFE,
)
from doc_analyse.orchestration import (
    DocumentOrchestrator,
    _apply_layer1_floor,
    VERDICT_SAFE,
    VERDICT_SUSPICIOUS,
    VERDICT_UNSAFE,
)
from doc_analyse.workers import WorkerOutcome, WorkerResult

from tests.test_orchestration import FakeRouter, FakeYara


# ---------------------------------------------------------------------------
# Unit: _apply_layer1_floor
# ---------------------------------------------------------------------------


class TestApplyLayer1Floor:
    """Direct tests of the floor helper — every cell of the verdict matrix."""

    @pytest.mark.parametrize(
        "layer1,layer2,expected",
        [
            # Layer 1 HOLD: SAFE blocked, SUSPICIOUS preserved, UNSAFE preserved
            (DECISION_HOLD, VERDICT_SAFE, VERDICT_SUSPICIOUS),
            (DECISION_HOLD, VERDICT_SUSPICIOUS, VERDICT_SUSPICIOUS),
            (DECISION_HOLD, VERDICT_UNSAFE, VERDICT_UNSAFE),
            # Layer 1 REVIEW: SAFE blocked, SUSPICIOUS preserved, UNSAFE preserved
            (DECISION_REVIEW, VERDICT_SAFE, VERDICT_SUSPICIOUS),
            (DECISION_REVIEW, VERDICT_SUSPICIOUS, VERDICT_SUSPICIOUS),
            (DECISION_REVIEW, VERDICT_UNSAFE, VERDICT_UNSAFE),
            # Layer 1 SAFE: defensive — Layer 2 verdict accepted unchanged
            # (in normal flow, SAFE chunks never reach Layer 2 at all)
            (DECISION_SAFE, VERDICT_SAFE, VERDICT_SAFE),
            (DECISION_SAFE, VERDICT_SUSPICIOUS, VERDICT_SUSPICIOUS),
            (DECISION_SAFE, VERDICT_UNSAFE, VERDICT_UNSAFE),
        ],
    )
    def test_floor_matrix(self, layer1, layer2, expected):
        assert _apply_layer1_floor(layer1, layer2) == expected


# ---------------------------------------------------------------------------
# Integration: orchestration end-to-end
# ---------------------------------------------------------------------------


class _Layer2VerdictPool:
    """Test pool that returns a fixed Layer 2 verdict for every routed chunk."""

    def __init__(self, verdict: str):
        self.verdict = verdict

    def classify_chunks_with_outcomes(self, chunks):
        outcomes = []
        for idx, chunk in enumerate(chunks):
            outcomes.append(
                WorkerOutcome(
                    chunk=chunk,
                    chunk_index=idx,
                    result=WorkerResult(
                        chunk=chunk,
                        classification=ClassificationResult(
                            verdict=self.verdict,
                            confidence=0.9,
                            reasons=("test",),
                        ),
                    ),
                )
            )
        return tuple(outcomes)

    def close(self):
        pass


def _orchestrator_with_pool(pool, *, hold_on_risk: bool):
    return DocumentOrchestrator(
        yara=FakeYara(),
        pg=None,
        router=FakeRouter(hold_on_risk=hold_on_risk),
        worker_pool=pool,
    )


@pytest.fixture
def risky_doc(tmp_path: Path) -> Path:
    path = tmp_path / "risky.txt"
    # FakeYara flags "RISKY-INSTRUCTION"; FakeRouter then routes to Layer 2.
    path.write_text("RISKY-INSTRUCTION goes here.", encoding="utf-8")
    return path


class TestOrchestrationEndToEndFloor:
    """The bug this PR fixes: a YARA-flagged chunk that the LLM mislabels
    "safe" must NOT short-circuit to SAFE. It stays at SUSPICIOUS so the
    cheap-detector signal is never silently overridden."""

    def test_hold_plus_layer2_safe_pinned_to_suspicious(self, risky_doc: Path):
        """**THIS IS THE BUG FIX.** Without _apply_layer1_floor, this
        chunk's final_verdict would be SAFE, and the document verdict
        would be SAFE — letting the YARA hit through entirely."""
        pool = _Layer2VerdictPool(verdict=VERDICT_SAFE)
        orchestrator = _orchestrator_with_pool(pool, hold_on_risk=True)

        result = orchestrator.analyze_path(risky_doc)

        flagged = [cr for cr in result.chunk_results if cr.routed_to_llm]
        assert len(flagged) == 1
        assert flagged[0].cheap_decision.decision == DECISION_HOLD
        assert flagged[0].llm_classification is not None
        assert flagged[0].llm_classification.verdict == "safe"
        # Critical assertion: Layer 2's SAFE did NOT win.
        assert flagged[0].final_verdict == VERDICT_SUSPICIOUS
        assert result.verdict == VERDICT_SUSPICIOUS

    def test_review_plus_layer2_safe_pinned_to_suspicious(self, risky_doc: Path):
        pool = _Layer2VerdictPool(verdict=VERDICT_SAFE)
        orchestrator = _orchestrator_with_pool(pool, hold_on_risk=False)

        result = orchestrator.analyze_path(risky_doc)

        flagged = [cr for cr in result.chunk_results if cr.routed_to_llm]
        assert len(flagged) == 1
        assert flagged[0].cheap_decision.decision == DECISION_REVIEW
        assert flagged[0].final_verdict == VERDICT_SUSPICIOUS
        assert result.verdict == VERDICT_SUSPICIOUS

    def test_hold_plus_layer2_unsafe_upgrades_correctly(self, risky_doc: Path):
        """Layer 2 CAN still upgrade — the floor only blocks downgrade."""
        pool = _Layer2VerdictPool(verdict=VERDICT_UNSAFE)
        orchestrator = _orchestrator_with_pool(pool, hold_on_risk=True)

        result = orchestrator.analyze_path(risky_doc)

        flagged = [cr for cr in result.chunk_results if cr.routed_to_llm]
        assert flagged[0].final_verdict == VERDICT_UNSAFE
        assert result.verdict == VERDICT_UNSAFE

    def test_review_plus_layer2_unsafe_upgrades_correctly(self, risky_doc: Path):
        pool = _Layer2VerdictPool(verdict=VERDICT_UNSAFE)
        orchestrator = _orchestrator_with_pool(pool, hold_on_risk=False)

        result = orchestrator.analyze_path(risky_doc)

        flagged = [cr for cr in result.chunk_results if cr.routed_to_llm]
        assert flagged[0].final_verdict == VERDICT_UNSAFE

    def test_hold_plus_layer2_suspicious_stays_suspicious(self, risky_doc: Path):
        pool = _Layer2VerdictPool(verdict=VERDICT_SUSPICIOUS)
        orchestrator = _orchestrator_with_pool(pool, hold_on_risk=True)

        result = orchestrator.analyze_path(risky_doc)

        flagged = [cr for cr in result.chunk_results if cr.routed_to_llm]
        assert flagged[0].final_verdict == VERDICT_SUSPICIOUS

    def test_safe_layer1_chunk_stays_safe_not_routed(self, tmp_path: Path):
        """A Layer 1 SAFE chunk never reaches Layer 2 — sanity check that
        the floor doesn't accidentally fire on non-routed chunks."""
        path = tmp_path / "safe.txt"
        path.write_text("Nothing risky here.", encoding="utf-8")
        # Pool would say "unsafe" if asked, but we should never call it.
        pool = _Layer2VerdictPool(verdict=VERDICT_UNSAFE)
        orchestrator = _orchestrator_with_pool(pool, hold_on_risk=True)

        result = orchestrator.analyze_path(path)

        assert all(not cr.routed_to_llm for cr in result.chunk_results)
        assert all(cr.final_verdict == VERDICT_SAFE for cr in result.chunk_results)
        assert result.verdict == VERDICT_SAFE


class TestFloorRespectsExistingFailureContract:
    """The floor must not interfere with the per-chunk failure isolation
    introduced in PR #41 — failed routed chunks already pin to SUSPICIOUS
    via the elif-no-llm branch, and the floor should not change that."""

    def test_layer2_failed_chunk_still_suspicious(self, risky_doc: Path):
        class FailingPool:
            def classify_chunks_with_outcomes(self, chunks):
                return tuple(
                    WorkerOutcome(
                        chunk=chunk,
                        chunk_index=idx,
                        error="provider down",
                        error_type="RuntimeError",
                    )
                    for idx, chunk in enumerate(chunks)
                )

            def close(self):
                pass

        orchestrator = _orchestrator_with_pool(FailingPool(), hold_on_risk=True)
        result = orchestrator.analyze_path(risky_doc)

        flagged = [cr for cr in result.chunk_results if cr.routed_to_llm]
        assert flagged[0].llm_failed
        assert flagged[0].final_verdict == VERDICT_SUSPICIOUS


# ---------------------------------------------------------------------------
# Defensive: malformed Layer 2 verdicts go through normalize → SUSPICIOUS,
# then the floor leaves them unchanged. Confirm the chain holds.
# ---------------------------------------------------------------------------


class TestNormalizeAndFloorChain:
    def test_unknown_verdict_normalizes_to_suspicious_then_floor_passes_through(
        self, risky_doc: Path
    ):
        pool = _Layer2VerdictPool(verdict="garbage-not-a-real-verdict")
        orchestrator = _orchestrator_with_pool(pool, hold_on_risk=True)

        result = orchestrator.analyze_path(risky_doc)

        flagged = [cr for cr in result.chunk_results if cr.routed_to_llm]
        # _normalize_verdict turns "garbage" into "suspicious"; floor leaves it.
        assert flagged[0].final_verdict == VERDICT_SUSPICIOUS
