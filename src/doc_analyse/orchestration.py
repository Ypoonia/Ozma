"""Document-level orchestration for ingestion, cheap detection, and LLM validation.

Primary flow:
1) ingest and chunk one document
2) run Layer 1 cheap detection on each chunk (visible loop below)
3) send REVIEW/HOLD chunks to Layer 2 LLM validator
4) aggregate chunk results into a final document verdict
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from doc_analyse.classifiers import ClassificationResult
from doc_analyse.classifiers.base import VALID_VERDICTS
from doc_analyse.detection import YaraDetector
from doc_analyse.detection.detect import (
    DECISION_HOLD,
    DECISION_REVIEW,
    DECISION_SAFE,
    CheapChunkDecision,
    CheapRouter,
    YaraEvidence,
)
from doc_analyse.detection.models import DetectionFinding
from doc_analyse.detection.normalize import normalize_for_detection
from doc_analyse.detection.prompt_guard import PromptGuardDetector, _normalise_scores
from doc_analyse.ingestion import ConverterRegistry, IngestedDocument, TextChunker, ingest_document
from doc_analyse.ingestion.models import TextChunk
from doc_analyse.workers import ClassifierWorkerPool

logger = logging.getLogger(__name__)

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

def run_layer1(
    chunk: TextChunk,
    yara: YaraDetector,
    pg: Optional[PromptGuardDetector],
    router: CheapRouter,
) -> tuple[CheapChunkDecision, tuple[DetectionFinding, ...]]:
    """Run Layer 1 cheap detection on one chunk."""
    result = _run_layer1(chunk, yara, pg, router)
    return result.decision, result.findings


@dataclass(frozen=True)
class _Layer1Run:
    decision: CheapChunkDecision
    findings: tuple[DetectionFinding, ...]
    pg_error: Optional[str]


def _run_layer1(
    chunk: TextChunk,
    yara: YaraDetector,
    pg: Optional[PromptGuardDetector],
    router: CheapRouter,
) -> _Layer1Run:
    """Run Layer 1 cheap detection on one chunk.

    Step-by-step:
      1. run YARA on the raw chunk text   (byte_to_char from TextChunker is correct for raw text)
      2. normalize the chunk text         (Prompt Guard needs NFKC-clean input)
      3. run Prompt Guard on the normalized text  (raw score, not thresholded findings)
      4. router combines both signals into a decision
      5. return decision + YARA evidence as DetectionFindings
    """
    # 1. YARA on raw text — offsets are correct, no coordinate remapping needed
    yara_findings = yara.detect(chunk)

    # 2. normalize for Prompt Guard
    normalized = normalize_for_detection(chunk.text)

    # 3. Prompt Guard — raw score (sub-threshold scores preserved)
    pg_score, pg_error = _pg_raw_score(normalized, pg)

    # 4. router decision
    decision = router.route(yara_findings, pg_score)
    decision = _fail_closed_pg_decision(decision, pg_error)

    # 5. build findings — YARA evidence plus PG-only finding if no YARA hits
    findings: list[DetectionFinding] = []
    requires_llm = decision.decision in {DECISION_REVIEW, DECISION_HOLD}
    original_yara_findings = {
        _yara_evidence_key(f): f
        for f in yara_findings
    }

    for e in decision.findings:
        findings.append(_build_yara_finding(
            evidence=e,
            original=original_yara_findings.get(_yara_evidence_key(e)),
            chunk=chunk,
            requires_llm=requires_llm,
        ))

    if pg_error and requires_llm:
        findings.append(_pg_error_finding(chunk, requires_llm=requires_llm))

    # PG-only hold/review: create synthetic finding so evidence is never empty
    if pg_score > 0 and not yara_findings and decision.decision in {DECISION_REVIEW, DECISION_HOLD}:
        signal = "strong" if pg_score >= 0.75 else "moderate"
        # PG evaluates the whole chunk, so the finding spans the whole chunk.
        # Earlier code anchored end_char at chunk.start_char (zero-width),
        # which made UIs render a hairline at the chunk boundary instead of
        # highlighting the chunk PG actually scored.
        findings.append(DetectionFinding(
            span="",
            category="prompt_guard_signal",
            severity="high",
            reason=f"[PG] score={pg_score:.3f} — {signal} signal",
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            source=chunk.source,
            rule_id="prompt_guard",
            requires_llm_validation=requires_llm,
            score=pg_score,
        ))

    return _Layer1Run(decision=decision, findings=tuple(findings), pg_error=pg_error)


def _fail_closed_pg_decision(
    decision: CheapChunkDecision,
    pg_error: Optional[str],
) -> CheapChunkDecision:
    if pg_error is None or decision.decision != DECISION_SAFE:
        return decision

    reason = "Prompt Guard failed; routing to review."
    if decision.reason:
        reason = f"{decision.reason} | {reason}"

    return CheapChunkDecision(
        decision=DECISION_REVIEW,
        risk_score=decision.risk_score,
        pg_score=decision.pg_score,
        yara_score=decision.yara_score,
        findings=decision.findings,
        reason=reason,
    )


def _yara_evidence_key(
    evidence: Union[DetectionFinding, YaraEvidence],
) -> tuple[str, int, int, str]:
    return (
        evidence.rule_id,
        evidence.start_char,
        evidence.end_char,
        evidence.span,
    )


def _build_yara_finding(
    *,
    evidence: Union[DetectionFinding, YaraEvidence],
    original: Optional[DetectionFinding],
    chunk: TextChunk,
    requires_llm: bool,
) -> DetectionFinding:
    source = original or evidence
    metadata = dict(original.metadata) if original is not None else {}
    raw_weight = _yara_weight(evidence, original)
    metadata.setdefault("detector", "YaraDetector")
    metadata.setdefault("yara_rule", evidence.rule_id)
    metadata.setdefault("yara_weight", raw_weight)
    metadata.setdefault("route_hint", _yara_route_hint(evidence, original))

    score = source.score if isinstance(source, DetectionFinding) else None
    if score is None:
        score = min(1.0, raw_weight / 100.0) if raw_weight > 0 else None

    return DetectionFinding(
        span=source.span,
        category=source.category,
        severity=source.severity,
        reason=source.reason if isinstance(source, DetectionFinding) else (
            f"[YARA] {source.rule_id} — {source.category} ({source.severity})"
        ),
        start_char=source.start_char,
        end_char=source.end_char,
        source=source.source if isinstance(source, DetectionFinding) else chunk.source,
        rule_id=source.rule_id,
        requires_llm_validation=requires_llm,
        score=score,
        metadata=metadata,
    )


def _yara_weight(
    evidence: Union[DetectionFinding, YaraEvidence],
    original: Optional[DetectionFinding],
) -> float:
    if isinstance(evidence, YaraEvidence):
        return evidence.weight

    score_source = original if original is not None else evidence
    metadata = dict(evidence.metadata or {})
    if original is not None:
        metadata.update(original.metadata or {})

    raw_weight = metadata.get("yara_weight")
    if raw_weight is not None:
        try:
            return float(raw_weight)
        except (TypeError, ValueError):
            pass

    if isinstance(score_source, DetectionFinding) and score_source.score is not None:
        return max(0.0, min(100.0, score_source.score * 100.0))

    return 0.0


def _yara_route_hint(
    evidence: Union[DetectionFinding, YaraEvidence],
    original: Optional[DetectionFinding],
) -> str:
    if isinstance(evidence, YaraEvidence):
        return evidence.route_hint
    if original is not None:
        return str((original.metadata or {}).get("route_hint", "evidence"))
    return str((evidence.metadata or {}).get("route_hint", "evidence"))


def _pg_error_finding(chunk: TextChunk, *, requires_llm: bool) -> DetectionFinding:
    # PG ran (or tried to) on the whole chunk, so the error finding spans the
    # whole chunk — same rationale as the PG signal finding above.
    return DetectionFinding(
        span="",
        category="prompt_guard_error",
        severity="high",
        reason="Prompt Guard failed; routed to Layer 2 for review.",
        start_char=chunk.start_char,
        end_char=chunk.end_char,
        source=chunk.source,
        rule_id="prompt_guard_error",
        requires_llm_validation=requires_llm,
        metadata={
            "detector": "PromptGuardDetector",
            "fail_closed": True,
        },
    )


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
    # Populated when Layer 2 was attempted but failed for this chunk (timeout,
    # exhausted retries, malformed response, …). Sibling chunks are unaffected;
    # this chunk's verdict is held at SUSPICIOUS rather than downgraded to SAFE.
    llm_error: Optional[str] = None
    llm_error_type: Optional[str] = None

    @property
    def llm_failed(self) -> bool:
        return self.llm_error is not None


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

    @property
    def findings(self) -> tuple[DetectionFinding, ...]:
        """All Layer 1 findings across all chunks, in chunk-then-finding order.

        Convenience accessor so callers can iterate the document's
        Layer 1 evidence without walking ``chunk_results`` themselves.
        Findings are ``DetectionFinding`` objects carrying ``rule_id``,
        ``span``, ``severity``, ``start_char`` / ``end_char``, and
        ``metadata`` — enough to render or audit per-document.
        """
        return tuple(
            finding
            for chunk_result in self.chunk_results
            for finding in chunk_result.cheap_findings
        )


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
        logger.info(
            "analyze_start",
            extra={
                "event": "doc_analyse.analyze_start",
                "chunk_count": len(chunks),
                "text_length": len(ingested.text),
            },
        )

        # -------------------------------------------------------------------
        # Layer 1 — cheap detection on every chunk (visible loop)
        # -------------------------------------------------------------------
        chunk_decisions: list[CheapChunkDecision] = []
        chunk_findings: list[tuple[DetectionFinding, ...]] = []

        for idx, chunk in enumerate(chunks):
            layer1 = _run_layer1(chunk, self.yara, self.pg, self.router)
            decision = layer1.decision
            findings = layer1.findings
            chunk_decisions.append(decision)
            chunk_findings.append(findings)
            log_extra = {
                "event": "doc_analyse.layer1_decision",
                "chunk_index": idx,
                "decision": decision.decision,
                "requires_layer2": decision.requires_layer2(),
                "risk_score": decision.risk_score,
                "yara_score": decision.yara_score,
                "pg_score": decision.pg_score,
                "finding_count": len(findings),
            }
            if layer1.pg_error is not None:
                log_extra["pg_error"] = layer1.pg_error
                log_extra["pg_failed_closed"] = True
            logger.info("layer1_decision", extra=log_extra)

        # -------------------------------------------------------------------
        # Route — only REVIEW/HOLD go to Layer 2
        # -------------------------------------------------------------------
        routed_indices = [
            idx for idx, d in enumerate(chunk_decisions)
            if d.requires_layer2()
        ]
        routed_indices_set = set(routed_indices)
        logger.info(
            "routed_indices",
            extra={
                "event": "doc_analyse.routed_indices",
                "routed_count": len(routed_indices),
                "routed_indices": routed_indices,
            },
        )

        # -------------------------------------------------------------------
        # Layer 2 — LLM validation on routed chunks (per-chunk error isolation)
        # -------------------------------------------------------------------
        llm_results: dict[int, ClassificationResult] = {}
        llm_errors: dict[int, tuple[str, str]] = {}
        layer2_result_summaries = []
        logger.info(
            "layer2_workers",
            extra={
                "event": "doc_analyse.layer2_workers",
                "submitted_chunk_count": len(routed_indices),
                "worker_count": getattr(self.worker_pool, "max_workers", None),
            },
        )
        if routed_indices:
            routed_chunks = tuple(chunks[i] for i in routed_indices)
            llm_results, llm_errors = _run_layer2(
                self.worker_pool, routed_chunks, routed_indices
            )
            for chunk_index, classification in llm_results.items():
                layer2_result_summaries.append({
                    "chunk_index": chunk_index,
                    "verdict": classification.verdict,
                    "confidence": classification.confidence,
                })
        logger.info(
            "layer2_results",
            extra={
                "event": "doc_analyse.layer2_results",
                "result_count": len(layer2_result_summaries),
                "failure_count": len(llm_errors),
                "results": layer2_result_summaries,
                "failed_chunk_indices": sorted(llm_errors.keys()),
            },
        )

        # -------------------------------------------------------------------
        # Build per-chunk results
        # -------------------------------------------------------------------
        chunk_results = []
        for idx, chunk in enumerate(chunks):
            decision = chunk_decisions[idx]
            findings = chunk_findings[idx]
            llm = llm_results.get(idx)
            error_pair = llm_errors.get(idx)

            if llm is not None:
                final_verdict = _apply_layer1_floor(
                    decision.decision, _normalize_verdict(llm.verdict)
                )
            elif decision.decision in {DECISION_HOLD, DECISION_REVIEW}:
                # Layer 1 routed this chunk for LLM review. Whether Layer 2
                # was skipped (no LLM result) or failed (error_pair set), we
                # fail-closed at SUSPICIOUS — never downgrade a routed chunk
                # to SAFE just because the LLM was unavailable.
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
                llm_error=(error_pair[0] if error_pair else None),
                llm_error_type=(error_pair[1] if error_pair else None),
            ))

        verdict, reasons = _aggregate_document_verdict(tuple(chunk_results))
        logger.info(
            "aggregate_verdict",
            extra={
                "event": "doc_analyse.aggregate_verdict",
                "verdict": verdict,
                "reasons": reasons,
            },
        )
        result = DocumentAnalysisResult(
            ingested_document=ingested,
            chunk_results=tuple(chunk_results),
            verdict=verdict,
            reasons=reasons,
        )
        logger.info(
            "analyze_end",
            extra={
                "event": "doc_analyse.analyze_end",
                "chunk_count": len(result.chunk_results),
                "verdict": result.verdict,
            },
        )
        return result


def _run_layer2(
    worker_pool: ClassifierWorkerPool,
    routed_chunks: tuple[TextChunk, ...],
    routed_indices: list[int],
) -> tuple[dict[int, ClassificationResult], dict[int, tuple[str, str]]]:
    """Run Layer 2 classification with per-chunk error isolation.

    Returns ``(results_by_index, errors_by_index)``. Per-chunk failures
    populate ``errors_by_index[idx] = (message, error_type)`` instead of
    bubbling up — so a single bad chunk cannot poison the whole document.

    Prefers the worker pool's ``classify_chunks_with_outcomes`` (per-chunk
    isolation). Falls back to ``classify_chunks`` (whole-batch raise) for
    legacy pools — still better than crashing the document, since we now
    record the failure on every routed chunk so callers can identify it.
    """
    results: dict[int, ClassificationResult] = {}
    errors: dict[int, tuple[str, str]] = {}
    if not routed_chunks:
        return results, errors

    outcomes_method = getattr(worker_pool, "classify_chunks_with_outcomes", None)
    if outcomes_method is not None:
        outcomes = outcomes_method(routed_chunks)
        for chunk_index, outcome in zip(routed_indices, outcomes):
            if outcome.result is not None:
                results[chunk_index] = outcome.result.classification
            else:
                error_type = outcome.error_type or "WorkerPoolError"
                error_msg = outcome.error or "Unknown Layer 2 failure."
                errors[chunk_index] = (error_msg, error_type)
                logger.warning(
                    "layer2_chunk_failed",
                    extra={
                        "event": "doc_analyse.layer2_chunk_failed",
                        "chunk_index": chunk_index,
                        "error_type": error_type,
                        "error": error_msg,
                    },
                )
        return results, errors

    # Backward-compat path: legacy worker pool only exposes classify_chunks,
    # which raises on the first failure. We catch the exception here so the
    # document analysis still completes and the failure is recorded on every
    # routed chunk (we cannot tell which one failed without per-chunk outcomes).
    try:
        worker_results = worker_pool.classify_chunks(routed_chunks)
    except Exception as exc:
        msg = str(exc)
        type_name = type(exc).__name__
        logger.warning(
            "layer2_batch_failed_legacy_pool",
            extra={
                "event": "doc_analyse.layer2_batch_failed_legacy_pool",
                "error_type": type_name,
                "error": msg,
                "failed_chunk_indices": list(routed_indices),
            },
        )
        for chunk_index in routed_indices:
            errors[chunk_index] = (msg, type_name)
        return results, errors

    for chunk_index, worker_result in zip(routed_indices, worker_results):
        results[chunk_index] = worker_result.classification
    return results, errors


def _aggregate_document_verdict(
    chunk_results: tuple[ChunkAnalysisResult, ...],
) -> tuple[str, tuple[str, ...]]:
    unsafe_indices = [r.chunk_index for r in chunk_results if r.final_verdict == VERDICT_UNSAFE]
    suspicious_indices = [
        r.chunk_index
        for r in chunk_results
        if r.final_verdict == VERDICT_SUSPICIOUS
    ]
    failed_indices = [r.chunk_index for r in chunk_results if r.llm_failed]

    reasons: list[str] = []
    if unsafe_indices:
        reasons.append(f"Unsafe chunk indices: {unsafe_indices}")
    if suspicious_indices:
        reasons.append(f"Suspicious chunk indices: {suspicious_indices}")
    if failed_indices:
        reasons.append(f"Layer 2 failed on chunk indices: {failed_indices}")

    if unsafe_indices:
        return VERDICT_UNSAFE, tuple(reasons)
    if suspicious_indices:
        return VERDICT_SUSPICIOUS, tuple(reasons)
    if not reasons:
        reasons.append("No suspicious or unsafe chunks detected.")
    return VERDICT_SAFE, tuple(reasons)


def _normalize_verdict(raw_verdict: str) -> str:
    verdict = str(raw_verdict).strip().lower()
    if verdict in _ORCHESTRATION_VERDICTS:
        return verdict
    return VERDICT_SUSPICIOUS


def _apply_layer1_floor(layer1_decision: str, layer2_verdict: str) -> str:
    """Enforce that Layer 2 cannot downgrade a routed chunk's verdict to SAFE.

    Layer 1 HOLD/REVIEW means at least one cheap detector (YARA rule or
    Prompt Guard signal) flagged the chunk. If Layer 2 returns ``safe``,
    that represents disagreement — not a green light. We pin the routed
    chunk at SUSPICIOUS minimum so the cheap-detector signal is never
    silently overridden, while still letting Layer 2 upgrade to UNSAFE.

    Without this floor, an attacker who can craft a chunk that triggers
    YARA but persuades the LLM it is benign (or a flaky LLM that hallucinates
    "safe") could exfiltrate past the entire pipeline. The asymmetry is
    intentional: cheap detectors are conservative and the LLM is fallible,
    so the LLM gets to upgrade ('actually unsafe') but not downgrade
    ('actually safe — please ignore the YARA hit').

    Layer 1 SAFE chunks are not routed to Layer 2 at all, so this function
    is only reached when ``layer1_decision`` is HOLD or REVIEW. The
    ``layer1_decision == SAFE`` branch is defensive — if a future caller
    routes SAFE chunks for any reason, we accept Layer 2's verdict
    unmodified.
    """
    if layer1_decision in {DECISION_HOLD, DECISION_REVIEW}:
        if layer2_verdict == VERDICT_SAFE:
            return VERDICT_SUSPICIOUS
    return layer2_verdict


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
