from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable, Optional

from doc_analyse.detection.models import DetectionFinding
from doc_analyse.ingestion.models import TextChunk


class BaseDetector(ABC):
    """Provider-free detector contract for cheap local evidence collection."""

    @abstractmethod
    def detect(self, chunk: TextChunk) -> tuple[DetectionFinding, ...]:
        raise NotImplementedError

    def detect_many(self, chunks: Iterable[TextChunk]) -> tuple[DetectionFinding, ...]:
        findings = []
        for chunk in chunks:
            findings.extend(self.detect(chunk))

        return self._finalize_findings(findings)

    def _finalize_findings(
        self, findings: Iterable[DetectionFinding]
    ) -> tuple[DetectionFinding, ...]:
        deduped = []
        seen = set()
        for finding in findings:
            key = self._finding_key(finding)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(finding)

        return tuple(
            sorted(
                deduped,
                key=lambda finding: (finding.start_char, finding.end_char, finding.rule_id),
            )
        )

    @staticmethod
    def _finding_key(finding: DetectionFinding) -> tuple[str, int, int, str]:
        return (
            finding.rule_id,
            finding.start_char,
            finding.end_char,
            finding.span,
        )

    def _build_finding(
        self,
        *,
        chunk: TextChunk,
        span: str,
        category: str,
        severity: str,
        reason: str,
        rule_id: str,
        start_char: int,
        end_char: int,
        score: Optional[float] = None,
        requires_llm_validation: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> DetectionFinding:
        resolved_metadata = dict(chunk.metadata)
        if metadata:
            resolved_metadata.update(metadata)
        if requires_llm_validation:
            # Backward compatibility while callers migrate to the first-class field.
            resolved_metadata["requires_llm_validation"] = True

        return DetectionFinding(
            span=span,
            category=category,
            severity=severity,
            reason=reason,
            start_char=start_char,
            end_char=end_char,
            source=chunk.source,
            rule_id=rule_id,
            requires_llm_validation=requires_llm_validation,
            score=score,
            metadata=resolved_metadata,
        )


class ParallelDetector(BaseDetector):
    """Runs cheap detectors independently for stateless worker-style fanout."""

    def __init__(
        self, detectors: Iterable[BaseDetector], max_workers: Optional[int] = None
    ) -> None:
        self.detectors = tuple(detectors)
        self.max_workers = max_workers

    def detect(self, chunk: TextChunk) -> tuple[DetectionFinding, ...]:
        return self.detect_many((chunk,))

    def detect_many(self, chunks: Iterable[TextChunk]) -> tuple[DetectionFinding, ...]:
        chunk_list = tuple(chunks)
        if not self.detectors or not chunk_list:
            return ()

        findings = []
        with ThreadPoolExecutor(max_workers=self.max_workers or len(self.detectors)) as executor:
            futures = {
                executor.submit(detector.detect, chunk): (detector, chunk)
                for chunk in chunk_list
                for detector in self.detectors
            }
            for future in as_completed(futures):
                detector, chunk = futures[future]
                try:
                    findings.extend(future.result())
                except Exception as exc:
                    findings.append(_detector_error_finding(chunk, detector, exc))

        return self._finalize_findings(findings)


def _detector_error_finding(
    chunk: TextChunk,
    detector: BaseDetector,
    exc: Exception,
) -> DetectionFinding:
    detector_name = detector.__class__.__name__
    return detector._build_finding(
        chunk=chunk,
        span=chunk.text,
        category="detector_error",
        severity="medium",
        reason=f"{detector_name} failed; send this chunk to LLM validation.",
        rule_id=f"{detector_name.lower()}_error",
        start_char=chunk.start_char,
        end_char=chunk.end_char,
        requires_llm_validation=True,
        metadata={
            "detector": detector_name,
            "error": str(exc),
        },
    )
