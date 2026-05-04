from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

from doc_analyse.classifiers import ClassificationResult
from doc_analyse.detection import BaseDetector, DetectionFinding
from doc_analyse.ingestion import ConverterRegistry, IngestedDocument, TextChunker, ingest_document
from doc_analyse.ingestion.models import TextChunk
from doc_analyse.workers import ClassifierWorkerPool


@dataclass(frozen=True)
class ChunkAnalysisResult:
    chunk_index: int
    chunk: TextChunk
    cheap_findings: tuple[DetectionFinding, ...]
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

    def chunk_result(self, index: int) -> ChunkAnalysisResult:
        return self.chunk_results[index]

    def chunk_text(self, index: int) -> str:
        chunk = self.chunk_results[index].chunk
        return self.ingested_document.text[chunk.start_char : chunk.end_char]


@dataclass
class DocumentOrchestrator:
    detector: BaseDetector
    worker_pool: ClassifierWorkerPool
    route_all_flagged_chunks: bool = True

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
        cheap_findings = self.detector.detect_many(chunks)
        cheap_by_index, unmapped = _index_findings_by_chunk(chunks, cheap_findings)
        routed_indices = tuple(
            index
            for index, findings in cheap_by_index.items()
            if self._should_route_to_llm(findings)
        )

        llm_results = self._validate_routed_chunks(chunks, routed_indices)
        chunk_results = tuple(
            self._build_chunk_result(
                chunk_index=index,
                chunk=chunk,
                cheap_findings=cheap_by_index[index],
                llm_classification=llm_results.get(index),
            )
            for index, chunk in enumerate(chunks)
        )

        verdict, reasons = _aggregate_document_verdict(chunk_results)
        return DocumentAnalysisResult(
            ingested_document=ingested,
            chunk_results=chunk_results,
            verdict=verdict,
            reasons=reasons,
            unmapped_cheap_findings=unmapped,
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
        cheap_findings: tuple[DetectionFinding, ...],
        llm_classification: Optional[ClassificationResult],
    ) -> ChunkAnalysisResult:
        if llm_classification is not None:
            final_verdict = llm_classification.verdict
        elif cheap_findings:
            final_verdict = "suspicious"
        else:
            final_verdict = "safe"

        return ChunkAnalysisResult(
            chunk_index=chunk_index,
            chunk=chunk,
            cheap_findings=cheap_findings,
            routed_to_llm=llm_classification is not None,
            llm_classification=llm_classification,
            final_verdict=final_verdict,
        )

    def _should_route_to_llm(self, findings: tuple[DetectionFinding, ...]) -> bool:
        if not findings:
            return False

        if self.route_all_flagged_chunks:
            return True

        return any(finding.requires_llm_validation for finding in findings)


def _index_findings_by_chunk(
    chunks: tuple[TextChunk, ...],
    findings: tuple[DetectionFinding, ...],
) -> tuple[dict[int, tuple[DetectionFinding, ...]], tuple[DetectionFinding, ...]]:
    indexed = {index: [] for index in range(len(chunks))}
    unmapped = []

    for finding in findings:
        chunk_index = _resolve_chunk_index(chunks, finding)
        if chunk_index is None:
            unmapped.append(finding)
            continue

        indexed[chunk_index].append(finding)

    return (
        {index: tuple(values) for index, values in indexed.items()},
        tuple(unmapped),
    )


def _resolve_chunk_index(chunks: tuple[TextChunk, ...], finding: DetectionFinding) -> Optional[int]:
    metadata_index = finding.metadata.get("chunk_index")
    if isinstance(metadata_index, int) and 0 <= metadata_index < len(chunks):
        chunk = chunks[metadata_index]
        if chunk.source == finding.source:
            return metadata_index

    for index, chunk in enumerate(chunks):
        if chunk.source != finding.source:
            continue
        if finding.start_char >= chunk.start_char and finding.end_char <= chunk.end_char:
            return index

    return None


def _aggregate_document_verdict(
    chunk_results: tuple[ChunkAnalysisResult, ...],
) -> tuple[str, tuple[str, ...]]:
    unsafe_indices = tuple(
        result.chunk_index for result in chunk_results if result.final_verdict == "unsafe"
    )
    suspicious_indices = tuple(
        result.chunk_index for result in chunk_results if result.final_verdict == "suspicious"
    )

    if unsafe_indices:
        return "unsafe", (f"Unsafe chunk indices: {list(unsafe_indices)}",)
    if suspicious_indices:
        return "suspicious", (f"Suspicious chunk indices: {list(suspicious_indices)}",)
    return "safe", ("No suspicious or unsafe chunks detected.",)


def analyze_document_path(
    path: Union[str, Path],
    *,
    detector: BaseDetector,
    worker_pool: ClassifierWorkerPool,
    registry: Optional[ConverterRegistry] = None,
    chunker: Optional[TextChunker] = None,
    route_all_flagged_chunks: bool = True,
) -> DocumentAnalysisResult:
    orchestrator = DocumentOrchestrator(
        detector=detector,
        worker_pool=worker_pool,
        route_all_flagged_chunks=route_all_flagged_chunks,
    )
    return orchestrator.analyze_path(path, registry=registry, chunker=chunker)
