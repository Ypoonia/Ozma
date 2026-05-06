"""Document-level orchestration for ingestion, cheap detection, and LLM validation.

Primary flow:
1) ingest and chunk one document
2) run Layer 1 cheap detection (YARA + Prompt Guard + CheapRouter) on all chunks
3) route chunks by CheapChunkDecision: SAFE skips Layer 2, REVIEW/HOLD go to Layer 2
4) aggregate chunk results into a final document verdict
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

from doc_analyse.classifiers import ClassificationResult
from doc_analyse.classifiers.base import VALID_VERDICTS
from doc_analyse.detection.base import BaseDetector
from doc_analyse.detection.cheap import CheapChunkDecision
from doc_analyse.detection.detect import CheapDetector
from doc_analyse.detection.models import DetectionFinding
from doc_analyse.ingestion import ConverterRegistry, IngestedDocument, TextChunker, ingest_document
from doc_analyse.ingestion.models import TextChunk
from doc_analyse.workers import ClassifierWorkerPool

VERDICT_SAFE = "safe"
VERDICT_SUSPICIOUS = "suspicious"
VERDICT_UNSAFE = "unsafe"

_ORCHESTRATION_VERDICTS = {VERDICT_SAFE, VERDICT_SUSPICIOUS, VERDICT_UNSAFE}
if _ORCHESTRATION_VERDICTS - VALID_VERDICTS:  # pragma: no cover - import-time invariant guard
    raise RuntimeError(
        "Orchestration verdict constants must stay aligned with classifier verdicts."
    )


@dataclass(frozen=True)
class ChunkAnalysisResult:
    chunk_index: int
    chunk: TextChunk
    cheap_findings: tuple[DetectionFinding, ...]
    cheap_decision: CheapChunkDecision
    routed_to_llm: bool
    llm_classification: Optional[ClassificationResult]
    final_verdict: str


@dataclass(frozen=True)
class DocumentAnalysisResult:
    ingested_document: IngestedDocument
    chunk_results: tuple[ChunkAnalysisResult, ...]
    verdict: str
    reasons: tuple[str, ...]
    unmapped_cheap_findings: tuple[DetectionFinding, ...] = ()

    def chunk_result(self, chunk_index: int) -> ChunkAnalysisResult:
        return self.chunk_results[chunk_index]

    def chunk_text(self, chunk_index: int) -> str:
        chunk = self.chunk_results[chunk_index].chunk
        return self.ingested_document.text[chunk.start_char : chunk.end_char]


@dataclass
class DocumentOrchestrator:
    """Orchestrates cheap Layer 1 + LLM Layer 2 analysis of a document."""

    detector: CheapDetector
    worker_pool: ClassifierWorkerPool
    _closed: bool = False

    def __enter__(self) -> DocumentOrchestrator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        close_pool = getattr(self.worker_pool, "close", None)
        if callable(close_pool):
            close_pool()
        self._closed = True

    def analyze_path(
        self,
        path: Union[str, Path],
        *,
        registry: Optional[ConverterRegistry] = None,
        chunker: Optional[TextChunker] = None,
    ) -> DocumentAnalysisResult:
        ingested = ingest_document(path, registry=registry, chunker=chunker)
        return self.analyze_ingested(ingested)

    def analyze_ingested(self, ingested: IngestedDocument) -> DocumentAnalysisResult:
        chunks = ingested.chunks

        # Layer 1: cheap detection on all chunks
        cheap_results = self.detector.detect_many(chunks)
        routed_indices = tuple(
            index
            for index, cr in enumerate(cheap_results)
            if cr.decision.requires_layer2()
        )

        # Layer 2: LLM validation on REVIEW/HOLD chunks only
        llm_results = self._validate_routed_chunks(chunks, routed_indices)

        chunk_results = tuple(
            self._build_chunk_result(
                chunk_index=index,
                chunk=chunks[index],
                cheap_decision=cheap_results[index].decision,
                llm_classification=llm_results.get(index),
            )
            for index in range(len(chunks))
        )

        verdict, reasons = _aggregate_document_verdict(chunk_results)
        return DocumentAnalysisResult(
            ingested_document=ingested,
            chunk_results=chunk_results,
            verdict=verdict,
            reasons=reasons,
        )

    def _validate_routed_chunks(
        self,
        chunks: tuple[TextChunk, ...],
        chunk_indices: Iterable[int],
    ) -> dict[int, ClassificationResult]:
        indices = tuple(chunk_indices)
        if not indices:
            return {}

        routed_chunks = tuple(chunks[index] for index in indices)
        worker_results = self.worker_pool.classify_chunks(routed_chunks)
        return {
            chunk_index: worker_result.classification
            for chunk_index, worker_result in zip(indices, worker_results)
        }

    def _build_chunk_result(
        self,
        *,
        chunk_index: int,
        chunk: TextChunk,
        cheap_decision: CheapChunkDecision,
        llm_classification: Optional[ClassificationResult],
    ) -> ChunkAnalysisResult:
        # Convert YaraEvidence back to DetectionFinding for compatibility
        findings = tuple(
            DetectionFinding(
                span=e.span,
                category=e.category,
                severity=e.severity,
                reason=f"[YARA] {e.rule_id} — {e.category} ({e.severity})",
                start_char=e.start_char,
                end_char=e.end_char,
                source=chunk.source,
                rule_id=e.rule_id,
                requires_llm_validation=cheap_decision.decision in {"review", "hold"},
            )
            for e in cheap_decision.findings
        )

        if llm_classification is not None:
            final_verdict = _normalize_verdict(llm_classification.verdict)
        elif cheap_decision.decision in {"hold", "review"}:
            final_verdict = VERDICT_SUSPICIOUS
        else:
            final_verdict = VERDICT_SAFE

        return ChunkAnalysisResult(
            chunk_index=chunk_index,
            chunk=chunk,
            cheap_findings=findings,
            cheap_decision=cheap_decision,
            routed_to_llm=llm_classification is not None,
            llm_classification=llm_classification,
            final_verdict=final_verdict,
        )


def _aggregate_document_verdict(
    chunk_results: tuple[ChunkAnalysisResult, ...],
) -> tuple[str, tuple[str, ...]]:
    unsafe_indices = tuple(
        r.chunk_index for r in chunk_results if r.final_verdict == VERDICT_UNSAFE
    )
    suspicious_indices = tuple(
        r.chunk_index for r in chunk_results if r.final_verdict == VERDICT_SUSPICIOUS
    )

    if unsafe_indices:
        return VERDICT_UNSAFE, (f"Unsafe chunk indices: {list(unsafe_indices)}",)
    if suspicious_indices:
        return VERDICT_SUSPICIOUS, (f"Suspicious chunk indices: {list(suspicious_indices)}",)
    return VERDICT_SAFE, ("No suspicious or unsafe chunks detected.",)


def analyze_document_path(
    path: Union[str, Path],
    *,
    detector: CheapDetector,
    worker_pool: ClassifierWorkerPool,
    registry: Optional[ConverterRegistry] = None,
    chunker: Optional[TextChunker] = None,
    close_worker_pool: bool = False,
) -> DocumentAnalysisResult:
    orchestrator = DocumentOrchestrator(
        detector=detector,
        worker_pool=worker_pool,
    )
    if close_worker_pool:
        with orchestrator:
            return orchestrator.analyze_path(path, registry=registry, chunker=chunker)

    return orchestrator.analyze_path(path, registry=registry, chunker=chunker)


def _normalize_verdict(raw_verdict: str) -> str:
    verdict = str(raw_verdict).strip().lower()
    if verdict in _ORCHESTRATION_VERDICTS:
        return verdict
    return VERDICT_SUSPICIOUS
