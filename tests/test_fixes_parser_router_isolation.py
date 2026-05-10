"""Tests for the four fixes on branch fix/parser-router-error-isolation.

1. _force_close_unclosed_containers + truncation parser hardening
2. Layer 2 JSON parse failures are retryable
3. CheapRouter.risk_score clamped to [0, 100] regardless of user weights
4. Per-chunk Layer 2 failure isolation + identification on ChunkAnalysisResult
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from doc_analyse.classifiers import ClassificationResult, ClassifierResponseError
from doc_analyse.classifiers.base import (
    _force_close_unclosed_containers,
    _try_parse_with_truncation_fallback,
)
from doc_analyse.detection.detect import CheapRouter, YaraEvidence, DECISION_HOLD
from doc_analyse.ingestion.models import TextChunk
from doc_analyse.orchestration import DocumentOrchestrator
from doc_analyse.workers import (
    ClassifierWorkerPool,
    StatelessClassifierWorker,
    WorkerOutcome,
    WorkerResult,
)
from doc_analyse.workers.pool import _is_retryable

from tests.test_orchestration import FakeRouter, FakeYara


# ---------------------------------------------------------------------------
# Bug 1 — truncation parser
# ---------------------------------------------------------------------------


class TestForceCloseUnclosedContainers:
    def test_complete_object_unchanged(self):
        assert _force_close_unclosed_containers('{"a": 1}') == '{"a": 1}'

    def test_unclosed_string_is_closed(self):
        repaired = _force_close_unclosed_containers('{"a": "incomp')
        assert json.loads(repaired) == {"a": "incomp"}

    def test_unclosed_array(self):
        repaired = _force_close_unclosed_containers('{"a": [1, 2, 3')
        assert json.loads(repaired) == {"a": [1, 2, 3]}

    def test_hanging_colon_supplies_null(self):
        # Previously: produced "{a:}" which is invalid; the new repair adds
        # null so the closer yields parseable JSON.
        repaired = _force_close_unclosed_containers('{"a":')
        assert json.loads(repaired) == {"a": None}

    def test_hanging_comma_dropped(self):
        repaired = _force_close_unclosed_containers('{"a": 1,')
        assert json.loads(repaired) == {"a": 1}

    def test_truncated_mid_escape_drops_dangling_backslash(self):
        # Previously: closing " would be consumed as the escaped char.
        repaired = _force_close_unclosed_containers('{"a": "val\\')
        assert json.loads(repaired) == {"a": "val"}

    def test_nested_truncation(self):
        repaired = _force_close_unclosed_containers('[{"a":1},{"b":2')
        assert json.loads(repaired) == [{"a": 1}, {"b": 2}]


class TestTruncationFallback:
    def test_real_truncated_classifier_response(self):
        truncated = (
            '{"verdict":"unsafe","confidence":0.9,'
            '"findings":[{"span":"hi","severity":"high","reason":"because'
        )
        parsed = _try_parse_with_truncation_fallback(truncated)
        assert parsed is not None
        assert parsed["verdict"] == "unsafe"
        assert parsed["findings"][0]["span"] == "hi"

    def test_garbage_tail_recovered_via_char_strip(self):
        parsed = _try_parse_with_truncation_fallback('{"a": 1}garbage')
        assert parsed == {"a": 1}

    def test_array_root_returns_none(self):
        # Classifier responses must be objects; arrays are rejected.
        assert _try_parse_with_truncation_fallback("[1,2,3]") is None

    @pytest.mark.parametrize(
        "adversarial",
        [
            None,
            123,
            "}}}}}",
            "{{{{{",
            "[" * 10_000,  # depth-bomb that crashes json.loads with RecursionError
            '"' * 1000,
            '\\' * 100,
            "\x00\x01\x02",
            "",
        ],
    )
    def test_never_raises_on_adversarial_input(self, adversarial):
        # The function must NEVER raise — a parser bug must not crash callers.
        # Returns None on failure.
        result = _try_parse_with_truncation_fallback(adversarial)
        assert result is None or isinstance(result, dict)


# ---------------------------------------------------------------------------
# Bug 2 — JSON parse failures are retryable
# ---------------------------------------------------------------------------


class TestIsRetryable:
    def test_classifier_response_error_is_retryable(self):
        # ClassifierResponseError subclasses ValueError, so without the special
        # case it would be treated as non-retryable. Verify the special case.
        assert _is_retryable(ClassifierResponseError("invalid JSON")) is True

    def test_json_decode_error_is_retryable(self):
        exc = json.JSONDecodeError("Expecting value", "doc", 0)
        assert _is_retryable(exc) is True

    def test_plain_value_error_remains_non_retryable(self):
        # Other ValueErrors (e.g. empty chunk) are real input bugs — don't retry.
        assert _is_retryable(ValueError("Worker chunk text must be ...")) is False

    def test_type_error_non_retryable(self):
        assert _is_retryable(TypeError("wrong type")) is False

    def test_runtime_error_retryable(self):
        assert _is_retryable(RuntimeError("network flake")) is True


class TestRetryDrivesThroughTransientJSONFailure:
    """End-to-end: a classifier that returns invalid JSON twice then valid
    JSON should succeed via worker retry."""

    def test_transient_json_failure_recovers_via_retry(self, monkeypatch):
        from doc_analyse.workers import pool as pool_module

        # Eliminate retry sleeps so the test is fast.
        monkeypatch.setattr(pool_module, "RETRY_DELAYS", (0, 0, 0))

        attempts = {"n": 0}

        class FlakeyClassifier:
            def classify(self, text, metadata=None):
                attempts["n"] += 1
                if attempts["n"] < 3:
                    raise ClassifierResponseError("Classifier returned invalid JSON.")
                return ClassificationResult(
                    verdict="safe", confidence=0.9, reasons=("ok",)
                )

        worker = StatelessClassifierWorker(
            classifier_factory=lambda: FlakeyClassifier()
        )
        chunk = TextChunk(text="hello", source="doc.txt", start_char=0, end_char=5)

        with ClassifierWorkerPool(worker=worker, max_workers=1) as pool:
            results = pool.classify_chunks((chunk,))

        assert attempts["n"] == 3
        assert results[0].classification.verdict == "safe"


# ---------------------------------------------------------------------------
# Bug 3 — risk_score clamped to [0, 100]
# ---------------------------------------------------------------------------


class TestRiskScoreClamping:
    def test_max_signals_with_max_weights_clamps_to_100(self):
        # Both weights at 1.0 + max signals naively yields 200; must clamp.
        router = CheapRouter(yara_weight=1.0, pg_weight=1.0)
        ev = YaraEvidence(
            rule_id="r",
            category="c",
            severity="critical",
            span="s",
            start_char=0,
            end_char=10,
            weight=100.0,
        )
        decision = router.route([ev], 1.0)
        assert decision.risk_score <= 100.0
        assert decision.risk_score == 100.0
        assert decision.decision == DECISION_HOLD

    def test_default_weights_unchanged(self):
        # Weights summing to 1.0 already produce ≤ 100 — clamp is a no-op.
        router = CheapRouter()  # 0.5 + 0.5
        ev = YaraEvidence(
            rule_id="r",
            category="c",
            severity="critical",
            span="s",
            start_char=0,
            end_char=10,
            weight=100.0,
        )
        decision = router.route([ev], 1.0)
        # 100*0.5 + 1.0*100*0.5 = 100. Equal, not less.
        assert decision.risk_score == 100.0

    def test_no_signals_zero_risk(self):
        router = CheapRouter(yara_weight=1.0, pg_weight=1.0)
        decision = router.route([], 0.0)
        assert decision.risk_score == 0.0


# ---------------------------------------------------------------------------
# Bug 4 — Per-chunk Layer 2 failure isolation
# ---------------------------------------------------------------------------


class TestClassifyChunksWithOutcomes:
    def test_all_success_returns_in_input_order(self):
        class GoodClassifier:
            def classify(self, text, metadata=None):
                return ClassificationResult(
                    verdict="safe", confidence=0.9, reasons=("ok",)
                )

        worker = StatelessClassifierWorker(classifier_factory=lambda: GoodClassifier())
        chunks = tuple(
            TextChunk(text=f"c{i}", source="doc.txt", start_char=i, end_char=i + 2)
            for i in range(3)
        )

        with ClassifierWorkerPool(worker=worker, max_workers=2) as pool:
            outcomes = pool.classify_chunks_with_outcomes(chunks)

        assert len(outcomes) == 3
        assert all(o.succeeded for o in outcomes)
        assert [o.chunk.text for o in outcomes] == ["c0", "c1", "c2"]
        assert [o.chunk_index for o in outcomes] == [0, 1, 2]

    def test_per_chunk_exception_isolated_siblings_continue(self, monkeypatch):
        from doc_analyse.workers import pool as pool_module
        # No retry sleeps.
        monkeypatch.setattr(pool_module, "RETRY_DELAYS", (0, 0, 0))

        class MixedClassifier:
            def classify(self, text, metadata=None):
                if text == "bad":
                    raise RuntimeError("provider had a bad day")
                return ClassificationResult(
                    verdict="safe", confidence=0.9, reasons=()
                )

        worker = StatelessClassifierWorker(
            classifier_factory=lambda: MixedClassifier()
        )
        chunks = (
            TextChunk(text="good-1", source="d", start_char=0, end_char=6),
            TextChunk(text="bad", source="d", start_char=7, end_char=10),
            TextChunk(text="good-2", source="d", start_char=11, end_char=17),
        )

        with ClassifierWorkerPool(worker=worker, max_workers=2) as pool:
            outcomes = pool.classify_chunks_with_outcomes(chunks)

        assert len(outcomes) == 3
        assert outcomes[0].succeeded
        assert not outcomes[1].succeeded
        assert outcomes[1].error_type == "RuntimeError"
        assert "provider had a bad day" in outcomes[1].error
        assert outcomes[2].succeeded

    def test_per_chunk_timeout_isolated(self, monkeypatch):
        from doc_analyse.workers import pool as pool_module
        # Short timeout for fast test; concurrency lets siblings run in parallel.
        monkeypatch.setattr(pool_module, "CHUNK_TIMEOUT", 0.4)

        hang_event = threading.Event()

        class HangOnBad:
            def classify(self, text, metadata=None):
                if text == "hung":
                    hang_event.wait()
                return ClassificationResult(
                    verdict="safe", confidence=0.9, reasons=()
                )

        worker = StatelessClassifierWorker(classifier_factory=lambda: HangOnBad())
        chunks = (
            TextChunk(text="good", source="d", start_char=0, end_char=4),
            TextChunk(text="hung", source="d", start_char=5, end_char=9),
        )
        pool = ClassifierWorkerPool(worker=worker, max_workers=2)
        try:
            outcomes = pool.classify_chunks_with_outcomes(chunks)
            assert outcomes[0].succeeded, "good chunk should not be poisoned by hung sibling"
            assert not outcomes[1].succeeded
            assert outcomes[1].error_type == "TimeoutError"
        finally:
            hang_event.set()
            pool.close()

    def test_empty_input_returns_empty_tuple(self):
        class _Cls:
            def classify(self, text, metadata=None):
                return ClassificationResult(verdict="safe", confidence=1.0)

        worker = StatelessClassifierWorker(classifier_factory=lambda: _Cls())
        with ClassifierWorkerPool(worker=worker) as pool:
            assert pool.classify_chunks_with_outcomes(()) == ()


class TestOrchestrationIsolatesLayer2Failures:
    """Layer 2 failures are captured per-chunk and surfaced on the result."""

    def test_one_chunk_fails_others_succeed(self, tmp_path: Path):
        path = tmp_path / "doc.txt"
        # Three risky chunks, all routed to Layer 2.
        path.write_text(
            "RISKY-INSTRUCTION first occurrence here.\n"
            "RISKY-INSTRUCTION second occurrence in middle.\n"
            "RISKY-INSTRUCTION third occurrence at end.",
            encoding="utf-8",
        )

        class IsolatingPool:
            """Per-chunk: chunk 1 fails, chunks 0 and 2 succeed."""

            calls = 0

            def classify_chunks_with_outcomes(self, chunks):
                outcomes = []
                for idx, chunk in enumerate(chunks):
                    if idx == 1:
                        outcomes.append(
                            WorkerOutcome(
                                chunk=chunk,
                                chunk_index=idx,
                                error="provider returned junk",
                                error_type="ClassifierResponseError",
                            )
                        )
                    else:
                        outcomes.append(
                            WorkerOutcome(
                                chunk=chunk,
                                chunk_index=idx,
                                result=WorkerResult(
                                    chunk=chunk,
                                    classification=ClassificationResult(
                                        verdict="unsafe",
                                        confidence=0.9,
                                        reasons=("worker",),
                                    ),
                                ),
                            )
                        )
                return tuple(outcomes)

            def close(self):
                self.closed = True

        from doc_analyse.ingestion import TextChunker
        chunker = TextChunker(chunk_size=50, chunk_overlap=0)
        orchestrator = DocumentOrchestrator(
            yara=FakeYara(),
            pg=None,
            router=FakeRouter(hold_on_risk=True),
            worker_pool=IsolatingPool(),
        )

        result = orchestrator.analyze_path(path, chunker=chunker)

        # Document-level reasons must surface BOTH unsafe chunks AND the failure.
        assert any("Layer 2 failed on chunk indices" in r for r in result.reasons), (
            f"Expected failure surface in reasons; got {result.reasons}"
        )
        assert result.verdict == "unsafe"  # the two unsafe chunks dominate

        # The failed chunk is identifiable on the per-chunk result.
        failing = [cr for cr in result.chunk_results if cr.llm_failed]
        assert len(failing) == 1
        assert failing[0].llm_error_type == "ClassifierResponseError"
        assert failing[0].llm_classification is None
        # Fail-closed: failed routed chunk stays at SUSPICIOUS, never SAFE.
        assert failing[0].final_verdict == "suspicious"

    def test_legacy_pool_failure_marks_all_routed_chunks(self, tmp_path: Path):
        """A legacy pool exposing only classify_chunks raising mid-batch
        records the failure on every routed chunk via the backward-compat path."""
        path = tmp_path / "doc.txt"
        path.write_text(
            "RISKY-INSTRUCTION one.\nRISKY-INSTRUCTION two.",
            encoding="utf-8",
        )

        class LegacyFailingPool:
            def classify_chunks(self, chunks):
                raise RuntimeError("legacy pool aborted the whole batch")

            def close(self):
                pass

        from doc_analyse.ingestion import TextChunker
        chunker = TextChunker(chunk_size=30, chunk_overlap=0)
        orchestrator = DocumentOrchestrator(
            yara=FakeYara(),
            pg=None,
            router=FakeRouter(hold_on_risk=True),
            worker_pool=LegacyFailingPool(),
        )

        result = orchestrator.analyze_path(path, chunker=chunker)

        # Document analysis must complete (not crash) and surface failure.
        assert any("Layer 2 failed on chunk indices" in r for r in result.reasons)
        # Both routed chunks recorded the same legacy-batch failure.
        failing = [cr for cr in result.chunk_results if cr.llm_failed]
        assert len(failing) >= 1
        for cr in failing:
            assert cr.llm_error_type == "RuntimeError"
            assert "legacy pool aborted" in cr.llm_error
            assert cr.final_verdict == "suspicious"
