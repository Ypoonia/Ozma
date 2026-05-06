from pathlib import Path

from doc_analyse.classifiers import ClassificationResult
from doc_analyse.detection import BaseDetector
from doc_analyse.detection.cheap import (
    CHEAP_DECISIONS,
    CheapChunkDecision,
    CheapRouter,
    DECISION_HOLD,
    DECISION_REVIEW,
    DECISION_SAFE,
    YaraEvidence,
)
from doc_analyse.detection.detect import CheapDetector, CheapResult
from doc_analyse.ingestion import TextChunker
from doc_analyse.ingestion.models import TextChunk
from doc_analyse.orchestration import DocumentOrchestrator, analyze_document_path
from doc_analyse.workers import WorkerResult


class FakeLayer1Detector(CheapDetector):
    """Fake CheapDetector for testing — routes based on whether text contains RISKY-INSTRUCTION."""

    def __init__(self, hold_on_risk: bool = True) -> None:
        super().__init__()
        self.hold_on_risk = hold_on_risk

    def detect(self, chunk: TextChunk) -> CheapResult:
        if "RISKY-INSTRUCTION" in chunk.text:
            decision = DECISION_HOLD if self.hold_on_risk else DECISION_REVIEW
            idx = chunk.text.index("RISKY-INSTRUCTION")
            evidence = (
                YaraEvidence(
                    rule_id="risk_marker",
                    category="instruction_override",
                    severity="high",
                    span="RISKY-INSTRUCTION",
                    start_char=chunk.start_char + idx,
                    end_char=chunk.start_char + idx + 17,
                ),
            )
            cheap_decision = CheapChunkDecision(
                decision=decision,
                risk_score=50.0,
                pg_score=0.0,
                yara_score=40.0,
                findings=evidence,
                reason="RISKY-INSTRUCTION found.",
            )
        else:
            cheap_decision = CheapChunkDecision(
                decision=DECISION_SAFE,
                risk_score=0.0,
                pg_score=0.0,
                yara_score=0.0,
                findings=(),
                reason="No risk signals.",
            )

        return CheapResult(chunk=chunk, decision=cheap_decision)


class FakeLayer1ReviewDetector(CheapDetector):
    """Fake CheapDetector that returns REVIEW (ambiguous) for RISKY-INSTRUCTION."""

    def detect(self, chunk: TextChunk) -> CheapResult:
        if "RISKY-INSTRUCTION" in chunk.text:
            cheap_decision = CheapChunkDecision(
                decision=DECISION_REVIEW,
                risk_score=25.0,
                pg_score=0.0,
                yara_score=20.0,
                findings=(),
                reason="Ambiguous.",
            )
        else:
            cheap_decision = CheapChunkDecision(
                decision=DECISION_SAFE,
                risk_score=0.0,
                pg_score=0.0,
                yara_score=0.0,
                findings=(),
                reason="No risk signals.",
            )

        return CheapResult(chunk=chunk, decision=cheap_decision)


class FakeLayer1NoFindingsDetector(CheapDetector):
    """Fake CheapDetector that always returns findings but SAFE decision."""

    def detect(self, chunk: TextChunk) -> CheapResult:
        if "FLAGGED-BUT-NO-LLM" in chunk.text:
            idx = chunk.text.index("FLAGGED-BUT-NO-LLM")
            evidence = (
                YaraEvidence(
                    rule_id="noise_marker",
                    category="noise",
                    severity="low",
                    span="FLAGGED-BUT-NO-LLM",
                    start_char=chunk.start_char + idx,
                    end_char=chunk.start_char + idx + 17,
                ),
            )
            cheap_decision = CheapChunkDecision(
                decision=DECISION_SAFE,
                risk_score=5.0,
                pg_score=0.0,
                yara_score=5.0,
                findings=evidence,
                reason="Low severity hits only.",
            )
        else:
            cheap_decision = CheapChunkDecision(
                decision=DECISION_SAFE,
                risk_score=0.0,
                pg_score=0.0,
                yara_score=0.0,
                findings=(),
                reason="No risk signals.",
            )

        return CheapResult(chunk=chunk, decision=cheap_decision)


class FakeWorkerPool:
    def __init__(self):
        self.calls = []
        self.closed = False

    def classify_chunks(self, chunks):
        chunk_list = tuple(chunks)
        self.calls.append(chunk_list)

        results = []
        for chunk in chunk_list:
            verdict = "unsafe" if "RISKY-INSTRUCTION" in chunk.text else "safe"
            results.append(
                WorkerResult(
                    chunk=chunk,
                    classification=ClassificationResult(
                        verdict=verdict,
                        confidence=0.9,
                        reasons=("worker",),
                    ),
                )
            )

        return tuple(results)

    def close(self):
        self.closed = True


def test_orchestrator_end_to_end_with_index_traceability(tmp_path: Path):
    path = tmp_path / "doc.txt"
    path.write_text(
        "Normal section.\n"
        "RISKY-INSTRUCTION appears in this chunk and should route to llm.\n"
        "Normal ending.",
        encoding="utf-8",
    )
    chunker = TextChunker(chunk_size=60, chunk_overlap=0)
    pool = FakeWorkerPool()
    orchestrator = DocumentOrchestrator(
        detector=FakeLayer1Detector(),
        worker_pool=pool,
    )

    result = orchestrator.analyze_path(path, chunker=chunker)

    assert result.verdict == "unsafe"
    assert len(pool.calls) == 1
    routed_chunks = pool.calls[0]
    assert len(routed_chunks) == 1
    routed_index = routed_chunks[0].metadata["chunk_index"]
    routed_result = result.chunk_result(routed_index)
    assert routed_result.routed_to_llm is True
    assert routed_result.llm_classification is not None
    assert routed_result.llm_classification.verdict == "unsafe"
    assert (
        result.chunk_text(routed_index)
        == result.ingested_document.text[
            routed_result.chunk.start_char : routed_result.chunk.end_char
        ]
    )


def test_orchestrator_skips_worker_pool_when_no_cheap_findings(tmp_path: Path):
    path = tmp_path / "safe.txt"
    path.write_text("No risk markers in this document.", encoding="utf-8")
    pool = FakeWorkerPool()
    orchestrator = DocumentOrchestrator(
        detector=FakeLayer1Detector(),
        worker_pool=pool,
    )

    result = orchestrator.analyze_path(path)

    assert result.verdict == "safe"
    assert pool.calls == []
    assert all(chunk_result.routed_to_llm is False for chunk_result in result.chunk_results)


def test_orchestrator_routes_only_requires_llm_when_configured(tmp_path: Path):
    path = tmp_path / "mixed.txt"
    path.write_text("FLAGGED-BUT-NO-LLM appears here.", encoding="utf-8")
    pool = FakeWorkerPool()
    orchestrator = DocumentOrchestrator(
        detector=FakeLayer1NoFindingsDetector(),
        worker_pool=pool,
    )

    result = orchestrator.analyze_path(path)

    assert pool.calls == []
    assert result.verdict == "safe"
    assert any(chunk_result.cheap_findings for chunk_result in result.chunk_results)


def test_orchestrator_context_manager_closes_worker_pool(tmp_path: Path):
    path = tmp_path / "safe.txt"
    path.write_text("safe", encoding="utf-8")
    pool = FakeWorkerPool()
    pool.closed = False

    with DocumentOrchestrator(detector=FakeLayer1Detector(), worker_pool=pool) as orchestrator:
        result = orchestrator.analyze_path(path)

    assert result.verdict == "safe"
    assert pool.closed is True


def test_analyze_document_path_can_close_worker_pool(tmp_path: Path):
    path = tmp_path / "safe.txt"
    path.write_text("safe", encoding="utf-8")
    pool = FakeWorkerPool()
    pool.closed = False

    result = analyze_document_path(
        path,
        detector=FakeLayer1Detector(),
        worker_pool=pool,
        close_worker_pool=True,
    )

    assert result.verdict == "safe"
    assert pool.closed is True


def test_orchestrator_normalizes_unknown_llm_verdict_to_suspicious(tmp_path: Path):
    path = tmp_path / "doc.txt"
    path.write_text("RISKY-INSTRUCTION", encoding="utf-8")

    class UnknownVerdictPool(FakeWorkerPool):
        def classify_chunks(self, chunks):
            chunk = tuple(chunks)[0]
            return (
                WorkerResult(
                    chunk=chunk,
                    classification=ClassificationResult(
                        verdict="typo-unsafe",
                        confidence=0.6,
                        reasons=("unknown",),
                    ),
                ),
            )

    orchestrator = DocumentOrchestrator(
        detector=FakeLayer1Detector(),
        worker_pool=UnknownVerdictPool(),
    )

    result = orchestrator.analyze_path(path)

    assert result.verdict == "suspicious"
    assert result.chunk_results[0].final_verdict == "suspicious"
