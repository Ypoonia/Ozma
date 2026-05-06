"""Per-chunk signal fusion: YARA evidence + Prompt Guard score + CheapRouter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from doc_analyse.detection.base import BaseDetector
from doc_analyse.detection.cheap import CheapChunkDecision, CheapRouter
from doc_analyse.detection.models import DetectionFinding
from doc_analyse.detection.normalize import normalize_for_detection
from doc_analyse.detection.prompt_guard import PromptGuardDetector, _normalise_scores
from doc_analyse.detection.yara import YaraDetector
from doc_analyse.ingestion.models import TextChunk


@dataclass(frozen=True)
class CheapResult:
    """Result of a single chunk through Layer 1 cheap detection."""

    chunk: TextChunk
    decision: CheapChunkDecision

    @property
    def yara_findings(self) -> tuple[DetectionFinding, ...]:
        return tuple(
            DetectionFinding(
                span=e.span,
                category=e.category,
                severity=e.severity,
                reason="",
                start_char=e.start_char,
                end_char=e.end_char,
                source=self.chunk.source,
                rule_id=e.rule_id,
                requires_llm_validation=False,
            )
            for e in self.decision.findings
        )


class CheapDetector:
    """Fuses YARA evidence + Prompt Guard score through CheapRouter.

    This is the new Layer 1 design. It replaces ParallelDetector as the
    primary cheap detection entry point.

    Usage:
        detector = CheapDetector()
        for result in detector.detect_chunks(chunks):
            if result.decision.requires_layer2():
                send_to_layer2(result.chunk)
    """

    def __init__(
        self,
        yara_detector: Optional[YaraDetector] = None,
        prompt_guard: Optional[PromptGuardDetector] = None,
        router: Optional[CheapRouter] = None,
        normalize: bool = True,
    ) -> None:
        self._yara = yara_detector or YaraDetector()
        self._pg = prompt_guard
        self._router = router or CheapRouter()
        self._normalize = normalize

    def detect(self, chunk: TextChunk) -> CheapResult:
        """Run Layer 1 on a single chunk, return CheapResult with routing decision."""
        normalized = normalize_for_detection(chunk.text) if self._normalize else chunk.text

        # YARA evidence
        yara_findings = self._yara.detect(chunk)

        # Prompt Guard score
        pg_score = self._pg_score(normalized)

        # Router decision
        decision = self._router.route(yara_findings, pg_score)

        return CheapResult(chunk=chunk, decision=decision)

    def detect_many(self, chunks: Iterable[TextChunk]) -> tuple[CheapResult, ...]:
        """Run Layer 1 on multiple chunks."""
        return tuple(self.detect(chunk) for chunk in chunks)

    def _pg_score(self, normalized_text: str) -> float:
        if self._pg is None:
            return 0.0

        try:
            # Call the PG pipeline directly to get the raw malicious score,
            # bypassing PromptGuardDetector.detect() which filters by threshold.
            # This preserves sub-threshold scores (e.g. 0.40-0.49) for the router.
            raw_scores = _normalise_scores(self._pg.load()(normalized_text))
            return raw_scores.get("malicious", 0.0)
        except Exception:
            # Prompt Guard failed (missing deps, etc.) — treat as no signal
            return 0.0


class Layer2Classifier:
    """Wraps a classifier for Layer 2 validation of REVIEW/HOLD chunks."""

    def __init__(self, classifier_factory: Any) -> None:
        self._factory = classifier_factory

    def validate(self, chunk: TextChunk) -> Any:
        """Call the LLM classifier on a chunk. Returns ClassificationResult."""
        classifier = self._factory()
        return classifier.classify(
            text=chunk.text,
            metadata={
                "source": chunk.source,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            },
        )