from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

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
