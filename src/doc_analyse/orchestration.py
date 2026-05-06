"""Document-level orchestration for ingestion, cheap detection, and LLM validation.

Primary flow:
1) ingest and chunk one document
2) run Layer 1 cheap detection on each chunk (visible loop below)
3) send REVIEW/HOLD chunks to Layer 2 LLM validator
4) aggregate chunk results into a final document verdict
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

from doc_analyse.classifiers import ClassificationResult
from doc_analyse.classifiers.base import VALID_VERDICTS
from doc_analyse.detection import YaraDetector
from doc_analyse.detection.detect import (
    DECISION_HOLD,
    DECISION_REVIEW,
    DECISION_SAFE,
    YaraEvidence,
)
from doc_analyse.detection.detect import CheapChunkDecision, CheapRouter
from doc_analyse.detection.models import DetectionFinding
from doc_analyse.detection.normalize import normalize_for_detection
from doc_analyse.detection.prompt_guard import PromptGuardDetector, _normalise_scores
from doc_analyse.ingestion import ConverterRegistry, IngestedDocument, TextChunker, ingest_document
from doc_analyse.ingestion.chunking import _build_byte_to_char
from doc_analyse.ingestion.models import TextChunk
from doc_analyse.workers import ClassifierWorkerPool

VERDICT_SAFE = "safe"
VERDICT_SUSPICIOUS = "suspicious"
VERDICT_UNSAFE = "unsafe"

_ORCHESTRATION_VERDICTS = {VERDICT_SAFE, VERDICT_SUSPICIOUS, VERDICT_UNSAFE}
if _ORCHESTRATION_VERDICTS - VALID_VERDICTS:  # pragma: no cover
    raise RuntimeError(
        "Orchestration verdict constants must stay aligned with classifier verdicts."
    )


# ---------------------------------------------------------------------------
# Per-chunk Layer 1 — this is the core cheap detection loop
# ---------------------------------------------------------------------------

def _build_normalized_to_original_map(original: str, normalized: str) -> list[int]:
    """Map each normalized character position to its original character position.

    Returns a list where result[i] = original character index of normalized[i],
    or -1 if normalized[i] was stripped (ZW char). The last entry maps to len(original).
    This is the inverse mapping of what _build_byte_to_char does for bytes.
    """
    norm_to_orig: list[int] = []
    orig_idx = 0
    for _ in normalized:
        # Skip any zero-width or stripped chars in original
        while orig_idx < len(original) and original[orig_idx] in {'\u200b', '\u200c', '\u200d', '\ufeff', '\u00ad', ' '}:
            orig_idx += 1
        if orig_idx < len(original):
            norm_to_orig.append(orig_idx)
            orig_idx += 1
        else:
            norm_to_orig.append(-1)
    norm_to_orig.append(len(original))  # sentinel for end-of-string
    return norm_to_orig


def _adjust_finding_coords(
    finding: DetectionFinding,
    normalized_chunk_start: int,
    norm_to_orig: list[int],
) -> DetectionFinding:
    """Map a YARA finding from normalized-text coords back to original document coords.

    YARA gives us (start_char, end_char) in the normalized text's coordinate space.
    We need to translate these to original document coordinates.

    norm_to_orig[i] = original character position of normalized[i].
    For a match of `span_len` chars at normalized position `norm_relative_start`:
      orig_start = norm_to_orig[norm_relative_start]
      orig_end   = norm_to_orig[norm_relative_start + span_len - 1] + 1
    """
    norm_start_relative = finding.start_char - normalized_chunk_start
    span_len = len(finding.span)

    if 0 <= norm_start_relative < len(norm_to_orig) - 1:
        orig_start = norm_to_orig[norm_start_relative]
        if norm_start_relative + span_len - 1 < len(norm_to_orig) - 1:
            orig_end = norm_to_orig[norm_start_relative + span_len - 1] + 1
        else:
            orig_end = finding.end_char  # fallback
    else:
        orig_start = finding.start_char
        orig_end = finding.end_char

    return DetectionFinding(
        span=finding.span,
        category=finding.category,
        severity=finding.severity,
        reason=finding.reason,
        start_char=orig_start,
        end_char=orig_end,
        source=finding.source,
        rule_id=finding.rule_id,
        requires_llm_validation=finding.requires_llm_validation,
        score=finding.score,
        metadata=finding.metadata,
    )


def run_layer1(
    chunk: TextChunk,
    yara: YaraDetector,
    pg: Optional[PromptGuardDetector],
    router: CheapRouter,
) -> tuple[CheapChunkDecision, tuple[DetectionFinding, ...]]:
    """Run Layer 1 cheap detection on one chunk.

    Step-by-step:
      1. normalize the chunk text
      2. build normalized→original position map for accurate offset translation
      3. run YARA on the normalized text  (byte_to_char rebuilt for consistent offsets)
      4. map YARA findings back to original document coordinates
      5. run Prompt Guard on the normalized text  (raw score, not thresholded findings)
      6. router combines both signals into a decision
      7. return decision + YARA evidence as DetectionFindings
    """
    # 1. normalize
    normalized = normalize_for_detection(chunk.text)

    # 2. build normalized→original position map
    norm_to_orig = _build_normalized_to_original_map(chunk.text, normalized)

    # 3. YARA on normalized text — byte_to_char for normalized text
    normalized_chunk = TextChunk(
        text=normalized,
        source=chunk.source,
        start_char=chunk.start_char,
        end_char=chunk.start_char + len(normalized),
        metadata={"byte_to_char": _build_byte_to_char(normalized)},
    )
    raw_yara_findings = yara.detect(normalized_chunk)

    # 4. map YARA finding offsets from normalized coords → original coords
    normalized_chunk_start = normalized_chunk.start_char
    yara_findings = tuple(
        _adjust_finding_coords(f, normalized_chunk_start, norm_to_orig)
        for f in raw_yara_findings
    )

    # 5. Prompt Guard — raw score (sub-threshold scores preserved)
    pg_score, pg_error = _pg_raw_score(normalized, pg)

    # 6. router decision
    decision = router.route(yara_findings, pg_score)

    # 7. build findings — YARA evidence plus PG-only finding if no YARA hits
    findings: list[DetectionFinding] = []
    requires_llm = decision.decision in {DECISION_REVIEW, DECISION_HOLD}

    for e in decision.findings:
        # e.start_char and e.end_char are already in original coords (from _adjust_finding_coords)
        findings.append(DetectionFinding(
            span=e.span,
            category=e.category,
            severity=e.severity,
            reason=f"[YARA] {e.rule_id} — {e.category} ({e.severity})",
            start_char=e.start_char,
            end_char=e.end_char,
            source=chunk.source,
            rule_id=e.rule_id,
            requires_llm_validation=requires_llm,
            score=e.score,
        ))

    # PG-only hold/review: create synthetic finding so evidence is never empty
    if pg_score > 0 and not yara_findings and decision.decision in {DECISION_REVIEW, DECISION_HOLD}:
        findings.append(DetectionFinding(
            span="",
            category="prompt_guard_signal",
            severity="high",
            reason=f"[PG] score={pg_score:.3f} — {'strong' if pg_score >= 0.75 else 'moderate'} signal",
            start_char=chunk.start_char,
            end_char=chunk.start_char,
            source=chunk.source,
            rule_id="prompt_guard",
            requires_llm_validation=requires_llm,
            score=pg_score,
        ))

    return decision, tuple(findings)


def _pg_raw_score(text: str, pg: Optional[PromptGuardDetector]) -> tuple[float, Optional[str]]:
    """Return (raw PG malicious score, error_message) for text.

    Unlike PromptGuardDetector.detect() which filters by threshold and returns findings,
    this returns the raw score so the router sees all signals including sub-threshold.
    Returns (0.0, None) when PG is not configured. Returns (0.0, err_msg) on failure so
    the caller can decide whether to treat 0.0 as safe or uncertain.
    """
    if pg is None:
        return 0.0, None

    try:
        scores = _normalise_scores(pg.load()(text))
        return scores.get("malicious", 0.0), None
    except Exception as exc:
        return 0.0, str(exc)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

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

    def chunk_result(self, chunk_index: int) -> ChunkAnalysisResult:
        return self.chunk_results[chunk_index]

    def chunk_text(self, chunk_index: int) -> str:
        chunk = self.chunk_results[chunk_index].chunk
        return self.ingested_document.text[chunk.start_char:chunk.end_char]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class DocumentOrchestrator:
    """Orchestrates cheap Layer 1 + LLM Layer 2 analysis of a document."""

    yara: YaraDetector
    pg: Optional[PromptGuardDetector]
    router: CheapRouter
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

        # -------------------------------------------------------------------
        # Layer 1 — cheap detection on every chunk (visible loop)
        # -------------------------------------------------------------------
        chunk_decisions: list[CheapChunkDecision] = []
        chunk_findings: list[tuple[DetectionFinding, ...]] = []

        for chunk in chunks:
            decision, findings = run_layer1(chunk, self.yara, self.pg, self.router)
            chunk_decisions.append(decision)
            chunk_findings.append(findings)

        # -------------------------------------------------------------------
        # Route — only REVIEW/HOLD go to Layer 2
        # -------------------------------------------------------------------
        routed_indices = [
            idx for idx, d in enumerate(chunk_decisions)
            if d.requires_layer2()
        ]
        routed_indices_set = set(routed_indices)

        # -------------------------------------------------------------------
        # Layer 2 — LLM validation on routed chunks
        # -------------------------------------------------------------------
        llm_results: dict[int, ClassificationResult] = {}
        if routed_indices:
            routed_chunks = tuple(chunks[i] for i in routed_indices)
            worker_results = self.worker_pool.classify_chunks(routed_chunks)
            for chunk_index, worker_result in zip(routed_indices, worker_results):
                llm_results[chunk_index] = worker_result.classification

        # -------------------------------------------------------------------
        # Build per-chunk results
        # -------------------------------------------------------------------
        chunk_results = []
        for idx, chunk in enumerate(chunks):
            decision = chunk_decisions[idx]
            findings = chunk_findings[idx]
            llm = llm_results.get(idx)

            if llm is not None:
                final_verdict = _normalize_verdict(llm.verdict)
            elif decision.decision in {DECISION_HOLD, DECISION_REVIEW}:
                final_verdict = VERDICT_SUSPICIOUS
            else:
                final_verdict = VERDICT_SAFE

            chunk_results.append(ChunkAnalysisResult(
                chunk_index=idx,
                chunk=chunk,
                cheap_findings=findings,
                cheap_decision=decision,
                routed_to_llm=idx in routed_indices_set,
                llm_classification=llm,
                final_verdict=final_verdict,
            ))

        verdict, reasons = _aggregate_document_verdict(tuple(chunk_results))
        return DocumentAnalysisResult(
            ingested_document=ingested,
            chunk_results=tuple(chunk_results),
            verdict=verdict,
            reasons=reasons,
        )


def _aggregate_document_verdict(
    chunk_results: tuple[ChunkAnalysisResult, ...],
) -> tuple[str, tuple[str, ...]]:
    unsafe_indices = [r.chunk_index for r in chunk_results if r.final_verdict == VERDICT_UNSAFE]
    suspicious_indices = [r.chunk_index for r in chunk_results if r.final_verdict == VERDICT_SUSPICIOUS]

    if unsafe_indices:
        return VERDICT_UNSAFE, (f"Unsafe chunk indices: {unsafe_indices}",)
    if suspicious_indices:
        return VERDICT_SUSPICIOUS, (f"Suspicious chunk indices: {suspicious_indices}",)
    return VERDICT_SAFE, ("No suspicious or unsafe chunks detected.",)


def _normalize_verdict(raw_verdict: str) -> str:
    verdict = str(raw_verdict).strip().lower()
    if verdict in _ORCHESTRATION_VERDICTS:
        return verdict
    return VERDICT_SUSPICIOUS


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------

def analyze_document_path(
    path: Union[str, Path],
    *,
    yara: Optional[YaraDetector] = None,
    pg: Optional[PromptGuardDetector] = None,
    router: Optional[CheapRouter] = None,
    worker_pool: ClassifierWorkerPool,
    registry: Optional[ConverterRegistry] = None,
    chunker: Optional[TextChunker] = None,
    close_worker_pool: bool = False,
) -> DocumentAnalysisResult:
    orchestrator = DocumentOrchestrator(
        yara=yara or YaraDetector(),
        pg=pg,
        router=router or CheapRouter(),
        worker_pool=worker_pool,
    )
    if close_worker_pool:
        with orchestrator:
            return orchestrator.analyze_path(path, registry=registry, chunker=chunker)
    return orchestrator.analyze_path(path, registry=registry, chunker=chunker)


def build_orchestrator(
    *,
    yara: Optional[YaraDetector] = None,
    pg: Optional[PromptGuardDetector] = None,
    router: Optional[CheapRouter] = None,
    worker_pool: ClassifierWorkerPool,
) -> DocumentOrchestrator:
    """Build a DocumentOrchestrator with sensible defaults.

    Usage:
        orchestrator = build_orchestrator(worker_pool=my_pool)
        result = orchestrator.analyze_path("document.pdf")

    Or with full customization:
        orchestrator = build_orchestrator(
            yara=YaraDetector(),
            pg=PromptGuardDetector(),
            router=CheapRouter(yara_weight=0.7, pg_weight=0.3),
            worker_pool=my_pool,
        )
    """
    return DocumentOrchestrator(
        yara=yara or YaraDetector(),
        pg=pg,
        router=router or CheapRouter(),
        worker_pool=worker_pool,
    )
