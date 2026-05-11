from __future__ import annotations

from abc import ABC, abstractmethod
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
    def _finding_key(finding: DetectionFinding) -> tuple[str, str, int, int]:
        return (
            finding.rule_id,
            finding.source,
            finding.start_char,
            finding.end_char,
        )

    @staticmethod
    def _build_finding(
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
