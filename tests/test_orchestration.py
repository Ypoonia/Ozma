from pathlib import Path

from doc_analyse.classifiers import ClassificationResult
from doc_analyse.detection import DetectionFinding, YaraDetector
from doc_analyse.detection.detect import (
    CheapRouter,
    DECISION_HOLD,
    DECISION_REVIEW,
    DECISION_SAFE,
    YaraEvidence,
)
from doc_analyse.ingestion import TextChunker
from doc_analyse.ingestion.models import TextChunk
from doc_analyse.orchestration import (
    DocumentOrchestrator,
    analyze_document_path,
)
from doc_analyse.workers import WorkerResult


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


class FakeYara:
    """YaraDetector stand-in that flags only RISKY-INSTRUCTION."""

    def __init__(self, findings=None):
        self._findings = findings or ()

    def detect(self, chunk: TextChunk):
        if "RISKY-INSTRUCTION" not in chunk.text:
            return ()

        idx = chunk.text.index("RISKY-INSTRUCTION")
        return (
            DetectionFinding(
                span="RISKY-INSTRUCTION",
                category="instruction_override",
                severity="high",
                reason="Risk marker for test.",
                rule_id="risk_marker",
                start_char=chunk.start_char + idx,
                end_char=chunk.start_char + idx + 17,
                source=chunk.source,
                requires_llm_validation=True,
            ),
        )


class FakeRouter:
    """Router that returns HOLD for RISKY-INSTRUCTION, SAFE otherwise."""

    def __init__(self, hold_on_risk=True):
        self.hold_on_risk = hold_on_risk

    def route(self, yara_findings, pg_score):
        from doc_analyse.detection.detect import CheapChunkDecision

        if yara_findings and self.hold_on_risk:
            return CheapChunkDecision(
                decision=DECISION_HOLD,
                risk_score=50.0,
                pg_score=0.0,
                yara_score=40.0,
                findings=(),
                reason="RISKY-INSTRUCTION found.",
            )
        elif yara_findings and not self.hold_on_risk:
            return CheapChunkDecision(
                decision=DECISION_REVIEW,
                risk_score=25.0,
                pg_score=0.0,
                yara_score=20.0,
                findings=(),
                reason="RISKY-INSTRUCTION found.",
            )
        return CheapChunkDecision(
            decision=DECISION_SAFE,
            risk_score=0.0,
            pg_score=0.0,
            yara_score=0.0,
            findings=(),
            reason="No risk signals.",
        )


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
        yara=FakeYara(),
        pg=None,
        router=FakeRouter(hold_on_risk=True),
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


def test_orchestrator_skips_worker_pool_when_no_cheap_findings(tmp_path: Path):
    path = tmp_path / "safe.txt"
    path.write_text("No risk markers in this document.", encoding="utf-8")
    pool = FakeWorkerPool()
    orchestrator = DocumentOrchestrator(
        yara=FakeYara(),
        pg=None,
        router=FakeRouter(hold_on_risk=True),
        worker_pool=pool,
    )

    result = orchestrator.analyze_path(path)

    assert result.verdict == "safe"
    assert pool.calls == []
    assert all(not cr.routed_to_llm for cr in result.chunk_results)


def test_orchestrator_routes_review_chunks_to_layer2(tmp_path: Path):
    path = tmp_path / "doc.txt"
    path.write_text("RISKY-INSTRUCTION", encoding="utf-8")
    pool = FakeWorkerPool()
    orchestrator = DocumentOrchestrator(
        yara=FakeYara(),
        pg=None,
        router=FakeRouter(hold_on_risk=False),  # REVIEW, not HOLD
        worker_pool=pool,
    )

    result = orchestrator.analyze_path(path)

    # REVIEW still goes to Layer 2
    assert len(pool.calls) == 1
    assert result.chunk_results[0].routed_to_llm is True


def test_orchestrator_context_manager_closes_worker_pool(tmp_path: Path):
    path = tmp_path / "safe.txt"
    path.write_text("safe", encoding="utf-8")
    pool = FakeWorkerPool()
    pool.closed = False

    with DocumentOrchestrator(
        yara=FakeYara(),
        pg=None,
        router=FakeRouter(),
        worker_pool=pool,
    ) as orchestrator:
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
        yara=FakeYara(),
        pg=None,
        router=FakeRouter(),
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
        yara=FakeYara(),
        pg=None,
        router=FakeRouter(),
        worker_pool=UnknownVerdictPool(),
    )

    result = orchestrator.analyze_path(path)

    assert result.verdict == "suspicious"
    assert result.chunk_results[0].final_verdict == "suspicious"
