from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Optional

from doc_analyse.detection.models import DetectionFinding
from doc_analyse.ingestion.models import TextChunk


class BaseDetector(ABC):
    """Provider-free detector contract for cheap local evidence collection."""

    @abstractmethod
    def detect(self, chunk: TextChunk) -> tuple[DetectionFinding, ...]:
        raise NotImplementedError

    def detect_many(self, chunks: Iterable[TextChunk]) -> tuple[DetectionFinding, ...]:
        findings = []
        seen = set()
        for chunk in chunks:
            for finding in self.detect(chunk):
                key = (
                    finding.rule_id,
                    finding.start_char,
                    finding.end_char,
                    finding.span,
                )
                if key in seen:
                    continue

                seen.add(key)
                findings.append(finding)

        return tuple(findings)


class ParallelDetector(BaseDetector):
    """Runs cheap detectors independently for stateless worker-style fanout."""

    def __init__(
        self, detectors: Iterable[BaseDetector], max_workers: Optional[int] = None
    ) -> None:
        self.detectors = tuple(detectors)
        self.max_workers = max_workers

    def detect(self, chunk: TextChunk) -> tuple[DetectionFinding, ...]:
        if not self.detectors:
            return ()

        findings = []
        with ThreadPoolExecutor(max_workers=self.max_workers or len(self.detectors)) as executor:
            futures = {
                executor.submit(detector.detect, chunk): detector for detector in self.detectors
            }
            for future in as_completed(futures):
                detector = futures[future]
                try:
                    findings.extend(future.result())
                except Exception as exc:
                    findings.append(_detector_error_finding(chunk, detector, exc))

        return tuple(
            sorted(
                self._dedupe(findings),
                key=lambda finding: (finding.start_char, finding.end_char, finding.rule_id),
            )
        )

    def _dedupe(self, findings: Iterable[DetectionFinding]) -> tuple[DetectionFinding, ...]:
        deduped = []
        seen = set()
        for finding in findings:
            key = (
                finding.rule_id,
                finding.start_char,
                finding.end_char,
                finding.span,
            )
            if key in seen:
                continue

            seen.add(key)
            deduped.append(finding)

        return tuple(deduped)


def _detector_error_finding(
    chunk: TextChunk,
    detector: BaseDetector,
    exc: Exception,
) -> DetectionFinding:
    detector_name = detector.__class__.__name__
    return DetectionFinding(
        span=chunk.text,
        category="detector_error",
        severity="medium",
        reason=f"{detector_name} failed; send this chunk to LLM validation.",
        start_char=chunk.start_char,
        end_char=chunk.end_char,
        source=chunk.source,
        rule_id=f"{detector_name.lower()}_error",
        metadata={
            **dict(chunk.metadata),
            "detector": detector_name,
            "error": str(exc),
            "requires_llm_validation": True,
        },
    )
