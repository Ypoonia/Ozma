"""Tests for the remaining review issues fixed on this branch.

- P0 #5: prompt-template metadata injection (single-pass substitution)
- P1 #6: DocumentAnalysisResult.findings property aggregates Layer 1 evidence
- P1 #7: PG synthetic findings span the whole chunk, not zero-width
- P1 #9: dead validation branch in CheapRouter is gone (already covered by
         existing test_detect.py; nothing extra needed here)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_analyse.classifiers import ClassificationResult
from doc_analyse.detection.detect import DECISION_HOLD, DECISION_REVIEW
from doc_analyse.detection.models import DetectionFinding
from doc_analyse.ingestion.models import TextChunk
from doc_analyse.orchestration import DocumentOrchestrator, run_layer1
from doc_analyse.prompt.loader import (
    METADATA_PLACEHOLDER,
    TEXT_PLACEHOLDER,
    render_classification_prompt,
)
from doc_analyse.workers import WorkerOutcome, WorkerResult

from tests.test_orchestration import FakeRouter, FakeYara


# ---------------------------------------------------------------------------
# P0 #5 — prompt-template metadata injection
# ---------------------------------------------------------------------------


class TestPromptTemplateMetadataInjection:
    """Single-pass substitution: an attacker-controlled metadata value
    cannot smuggle the chunk text into the metadata slot, and special
    regex characters in metadata values survive unchanged."""

    TEMPLATE = (
        "## Metadata\n"
        + METADATA_PLACEHOLDER
        + "\n\n## Text\n"
        + TEXT_PLACEHOLDER
        + "\n"
    )

    def test_text_appears_exactly_once_when_metadata_contains_text_placeholder(self):
        text = "SECRET-CHUNK-TEXT-MUST-NOT-DUPLICATE"
        metadata = {"source": TEXT_PLACEHOLDER}  # adversarial filename
        out = render_classification_prompt(self.TEMPLATE, text, metadata)
        # Without the fix, text appears twice (in Metadata and Text sections).
        assert out.count(text) == 1, (
            f"template injection: text duplicated. Got:\n{out}"
        )
        # The literal placeholder string survives in the metadata section.
        assert TEXT_PLACEHOLDER in out

    def test_metadata_value_with_metadata_placeholder_does_not_recurse(self):
        metadata = {"source": METADATA_PLACEHOLDER}
        out = render_classification_prompt(self.TEMPLATE, "ok", metadata)
        # The literal {{ metadata }} survives — single pass means no recursion.
        assert METADATA_PLACEHOLDER in out
        # Sanity: text still appears exactly once.
        assert out.count("ok") == 1

    def test_regex_backreferences_in_metadata_survive_verbatim(self):
        # re.sub callback returns are NOT interpreted as backreferences,
        # but lock that property in.
        metadata = {"crafted": r"\1\g<1>$0"}
        out = render_classification_prompt(self.TEMPLATE, "ok", metadata)
        assert r"\1\g<1>$0" in out

    def test_normal_substitution_still_works(self):
        out = render_classification_prompt(
            self.TEMPLATE, "hello", {"source_id": "doc.txt"}
        )
        assert "hello" in out
        assert "source_id: doc.txt" in out
        assert TEXT_PLACEHOLDER not in out
        assert METADATA_PLACEHOLDER not in out

    def test_substitution_only_matches_canonical_whitespace_form(self):
        """The regex matches the same literal forms as the old .replace()
        chain — `{{ text }}` with exactly one space inside the braces. A
        non-canonical form like `{{text}}` (no spaces) is NOT substituted,
        keeping substitution in lockstep with the literal-containment
        check used by ``resolve_prompt_text``."""
        nonstandard = "Body: {{text}}"  # no spaces — not the canonical form
        out = render_classification_prompt(nonstandard, "hello", {})
        # Non-canonical placeholder is left untouched (matches old behavior).
        assert "{{text}}" in out
        assert "hello" not in out


# ---------------------------------------------------------------------------
# P1 #6 — DocumentAnalysisResult.findings aggregator
# ---------------------------------------------------------------------------


class _PassthroughPool:
    """Returns Layer 2 verdicts that don't override Layer 1."""

    def classify_chunks_with_outcomes(self, chunks):
        return tuple(
            WorkerOutcome(
                chunk=c,
                chunk_index=i,
                result=WorkerResult(
                    chunk=c,
                    classification=ClassificationResult(
                        verdict="suspicious", confidence=0.9, reasons=("ok",)
                    ),
                ),
            )
            for i, c in enumerate(chunks)
        )

    def close(self):
        pass


class TestFindingsAggregatorAtDocumentLevel:
    """Use the real CheapRouter: FakeRouter returns ``findings=()`` so it
    cannot exercise the aggregation path. The real router populates
    ``decision.findings`` from the YARA evidence, which is what production
    relies on."""

    def test_findings_property_returns_all_layer1_findings_in_order(
        self, tmp_path: Path
    ):
        from doc_analyse.detection.detect import CheapRouter
        from doc_analyse.ingestion import TextChunker

        path = tmp_path / "doc.txt"
        path.write_text(
            "RISKY-INSTRUCTION first.\n"
            "Plain text here.\n"
            "RISKY-INSTRUCTION second.",
            encoding="utf-8",
        )
        chunker = TextChunker(chunk_size=30, chunk_overlap=0)
        orchestrator = DocumentOrchestrator(
            yara=FakeYara(),
            pg=None,
            router=CheapRouter(),
            worker_pool=_PassthroughPool(),
        )

        result = orchestrator.analyze_path(path, chunker=chunker)
        findings = result.findings

        # Two RISKY-INSTRUCTION chunks → two YARA findings.
        assert len(findings) == 2
        assert all(isinstance(f, DetectionFinding) for f in findings)
        assert all(f.rule_id == "risk_marker" for f in findings)
        # Order matches chunk-then-finding order.
        assert findings[0].start_char < findings[1].start_char

    def test_findings_empty_for_clean_document(self, tmp_path: Path):
        from doc_analyse.detection.detect import CheapRouter

        path = tmp_path / "clean.txt"
        path.write_text("Nothing risky here at all.", encoding="utf-8")
        orchestrator = DocumentOrchestrator(
            yara=FakeYara(),
            pg=None,
            router=CheapRouter(),
            worker_pool=_PassthroughPool(),
        )
        result = orchestrator.analyze_path(path)
        assert result.findings == ()


# ---------------------------------------------------------------------------
# P1 #7 — PG synthetic finding span covers the chunk
# ---------------------------------------------------------------------------


class _StubYara:
    """No YARA findings, so the PG-only synthetic-finding path fires."""

    def detect(self, chunk):
        return ()


class _ScoringRouter:
    """Routes any non-zero PG to HOLD so the synthetic finding is created."""

    def route(self, yara_findings, pg_score):
        from doc_analyse.detection.detect import CheapChunkDecision
        if pg_score > 0:
            return CheapChunkDecision(
                decision=DECISION_HOLD,
                risk_score=80.0,
                pg_score=pg_score,
                yara_score=0.0,
                findings=(),
                reason="PG flagged",
            )
        return CheapChunkDecision(
            decision="safe", risk_score=0.0, pg_score=pg_score,
            yara_score=0.0, findings=(), reason="ok",
        )


class _ScoringPG:
    """Stub PG that returns a fixed malicious score for any input."""

    def __init__(self, score):
        self._score = score

    def load(self):
        # PromptGuardDetector.load() returns a callable that takes text.
        # The orchestrator calls _normalise_scores on the result, so the
        # callable must return the raw HF-pipeline shape.
        score = self._score
        def _call(text):
            return [{"label": "malicious", "score": score}]
        return _call


class TestPGSyntheticFindingSpansChunk:
    def test_pg_only_finding_spans_whole_chunk_not_zero_width(self):
        chunk = TextChunk(
            text="content body that PG evaluated end-to-end",
            source="d",
            start_char=100,
            end_char=141,  # 100 + len(text)
        )
        decision, findings = run_layer1(
            chunk=chunk,
            yara=_StubYara(),
            pg=_ScoringPG(0.85),  # strong signal → HOLD
            router=_ScoringRouter(),
        )

        # We expect one synthetic PG finding (no YARA findings to compete).
        pg_findings = [f for f in findings if f.rule_id == "prompt_guard"]
        assert len(pg_findings) == 1
        f = pg_findings[0]
        assert f.start_char == 100
        # The bug: previously end_char was chunk.start_char (100). Now it
        # spans the whole chunk.
        assert f.end_char == 141
        assert f.end_char > f.start_char, "PG finding must not be zero-width"

    def test_pg_error_finding_also_spans_whole_chunk(self):
        """When PG raises, _pg_error_finding is emitted for routed chunks
        and must also span the whole chunk."""

        class _BrokenPG:
            def load(self):
                def _call(text):
                    raise RuntimeError("PG broke")
                return _call

        chunk = TextChunk(
            text="content where PG itself failed",
            source="d",
            start_char=50,
            end_char=80,
        )
        decision, findings = run_layer1(
            chunk=chunk,
            yara=_StubYara(),
            pg=_BrokenPG(),
            router=_ScoringRouter(),
        )

        # Decision should fail-closed to REVIEW (not SAFE) because PG errored.
        assert decision.decision in {DECISION_REVIEW, DECISION_HOLD}, (
            f"PG error must not leave decision SAFE; got {decision.decision}"
        )

        err_findings = [f for f in findings if f.rule_id == "prompt_guard_error"]
        assert len(err_findings) == 1
        f = err_findings[0]
        assert f.start_char == 50
        assert f.end_char == 80
        assert f.end_char > f.start_char


# ---------------------------------------------------------------------------
# P1 #9 — dead branch removal (sanity: existing tests still pass)
# ---------------------------------------------------------------------------


class TestDecisionAlwaysInValidSet:
    """The provably-unreachable validation branch was removed; assert the
    invariant it claimed to defend still holds via the constructive code."""

    def test_router_always_returns_a_known_decision(self):
        from doc_analyse.detection.detect import (
            CheapRouter,
            DECISION_SAFE,
            DECISION_REVIEW,
            DECISION_HOLD,
            YaraEvidence,
        )
        router = CheapRouter()
        valid = {DECISION_SAFE, DECISION_REVIEW, DECISION_HOLD}

        # Walk through a representative spread of inputs.
        test_inputs = [
            ([], 0.0),
            ([], 0.5),
            ([], 1.0),
            (
                [
                    YaraEvidence(
                        rule_id="r", category="c", severity="critical",
                        span="s", start_char=0, end_char=1, weight=100.0,
                    )
                ],
                0.0,
            ),
        ]
        for findings, pg in test_inputs:
            assert router.route(findings, pg).decision in valid
