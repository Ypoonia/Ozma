from __future__ import annotations

import logging
from threading import Lock
from typing import Any, Iterable, Mapping, Optional

from doc_analyse.detection.base import BaseDetector
from doc_analyse.detection.models import DetectionFinding
from doc_analyse.ingestion.models import TextChunk

DEFAULT_PROMPT_GUARD_MODEL = "meta-llama/Llama-Prompt-Guard-2-86M"
DEFAULT_MALICIOUS_THRESHOLD = 0.80
DEFAULT_UNCERTAIN_THRESHOLD = 0.50
logger = logging.getLogger(__name__)


class PromptGuardDependencyError(RuntimeError):
    pass


class PromptGuardDetector(BaseDetector):
    """Cheap ML detector backed by Meta Llama Prompt Guard 2.

    The default behavior is to initialize the Hugging Face pipeline during
    detector construction so worker fanout does not pay the cold-start cost on
    the first chunk.
    """

    def __init__(
        self,
        model: str = DEFAULT_PROMPT_GUARD_MODEL,
        classifier: Optional[Any] = None,
        malicious_threshold: float = DEFAULT_MALICIOUS_THRESHOLD,
        uncertain_threshold: float = DEFAULT_UNCERTAIN_THRESHOLD,
        device: int = -1,
        eager_load: bool = True,
        **pipeline_options: Any,
    ) -> None:
        if not 0.0 <= uncertain_threshold <= malicious_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= uncertain_threshold <= malicious_threshold <= 1."
            )

        self.model = model
        self._classifier = classifier
        self.malicious_threshold = malicious_threshold
        self.uncertain_threshold = uncertain_threshold
        self.device = device
        self.eager_load = eager_load
        self.pipeline_options = pipeline_options
        self._load_lock = Lock()

        if self._classifier is None and self.eager_load:
            self.load()

    def detect(self, chunk: TextChunk) -> tuple[DetectionFinding, ...]:
        if not isinstance(chunk.text, str) or not chunk.text.strip():
            return ()

        scores = _normalise_scores(self.load()(chunk.text))
        malicious_score = scores.get("malicious", 0.0)
        pg_metadata = {"detector": "PromptGuardDetector", "model": self.model}
        log_payload = {
            "chunk_start": chunk.start_char,
            "chunk_end": chunk.end_char,
            "malicious_score": malicious_score,
            "has_benign_score": "benign" in scores,
        }

        if malicious_score >= self.malicious_threshold:
            logger.info(
                "prompt_guard_detection_finding",
                extra={**log_payload, "category": "prompt_guard_malicious"},
            )
            return (
                self._build_finding(
                    chunk=chunk,
                    span=chunk.text,
                    category="prompt_guard_malicious",
                    severity="high",
                    reason="Prompt Guard classified this chunk as malicious.",
                    rule_id="prompt_guard",
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    requires_llm_validation=True,
                    score=malicious_score,
                    metadata=pg_metadata,
                ),
            )

        if malicious_score >= self.uncertain_threshold:
            logger.info(
                "prompt_guard_detection_finding",
                extra={**log_payload, "category": "prompt_guard_uncertain"},
            )
            return (
                self._build_finding(
                    chunk=chunk,
                    span=chunk.text,
                    category="prompt_guard_uncertain",
                    severity="medium",
                    reason="Prompt Guard score is uncertain enough to require LLM validation.",
                    rule_id="prompt_guard",
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    requires_llm_validation=True,
                    score=malicious_score,
                    metadata=pg_metadata,
                ),
            )

        logger.debug("prompt_guard_detection_no_findings", extra=log_payload)
        return ()

    def load(self) -> Any:
        """Return the cached classifier, building it once when needed."""
        if self._classifier is not None:
            return self._classifier

        with self._load_lock:
            if self._classifier is not None:
                return self._classifier
            try:
                self._classifier = self._build_classifier()
            except Exception:
                raise
            logger.info(
                "prompt_guard_classifier_loaded",
                extra={"device": self.device, "eager_load": self.eager_load},
            )
        return self._classifier

    def _build_classifier(self) -> Any:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise PromptGuardDependencyError(
                "PromptGuardDetector requires optional dependencies. "
                "Install with: pip install -e '.[prompt-guard]'"
            ) from exc

        try:
            return pipeline(
                "text-classification",
                model=self.model,
                top_k=None,
                truncation=True,
                max_length=512,
                device=self.device,
                **self.pipeline_options,
            )
        except Exception as exc:
            _log_pg_load_failure(exc)
            raise


def _log_pg_load_failure(exc: Exception) -> None:
    """Log a Prompt Guard load failure with actionable error message."""
    error_type = type(exc).__name__
    message = str(exc).lower()
    hint = "set HF_TOKEN environment variable"
    if "401" in message or "unauthorized" in message:
        hint = "Hugging Face token required for this gated model. Set: export HF_TOKEN=hf_..."
    elif "403" in message or "forbidden" in message:
        hint = "Hugging Face token missing or lacks model access. Set: export HF_TOKEN=hf_..."
    elif "connection" in message or "network" in message:
        hint = "Check internet connection or Hugging Face accessibility."
    elif "not found" in message or "does not exist" in message:
        hint = (
            "Model name may have changed. "
            "Check meta-llama/Llama-Prompt-Guard-2-86M availability."
        )
    logger.warning(
        "prompt_guard_classifier_load_failed",
        extra={"error_type": error_type, "error_message": str(exc), "hint": hint},
    )


def _normalise_scores(raw_output: Any) -> dict[str, float]:
    rows = tuple(_flatten_pipeline_output(raw_output))
    if not rows:
        logger.warning(
            "prompt_guard_score_extraction_failed",
            extra={
                "raw_output_type": type(raw_output).__name__,
                "row_count": 0,
                "reason": "empty_output",
            },
        )
        return {}

    scores = {}
    recognized_rows = 0
    for row in rows:
        if not isinstance(row, Mapping):
            continue

        label = str(row.get("label", "")).strip().lower()
        score = row.get("score")
        if not isinstance(score, (float, int)) or isinstance(score, bool):
            continue

        # Handle both descriptive labels (malicious/benign) and index labels (LABEL_1/LABEL_0).
        # LABEL_1 = malicious (injection probability), LABEL_0 = benign (safe probability).
        label_lower = label.lower()
        if label_lower in {"malicious", "jailbreak", "injection", "label_1"}:
            scores["malicious"] = max(scores.get("malicious", 0.0), float(score))
            recognized_rows += 1
        elif label_lower in {"benign", "safe", "label_0"}:
            scores["benign"] = max(scores.get("benign", 0.0), float(score))
            recognized_rows += 1

    if scores:
        logger.debug(
            "prompt_guard_score_extracted",
            extra={
                "row_count": len(rows),
                "recognized_rows": recognized_rows,
                "skipped_rows": len(rows) - recognized_rows,
                "score_labels": sorted(scores),
                "malicious_score": scores.get("malicious", 0.0),
            },
        )
    else:
        logger.warning(
            "prompt_guard_score_extraction_failed",
            extra={
                "raw_output_type": type(raw_output).__name__,
                "row_count": len(rows),
                "reason": "no_supported_scores",
            },
        )

    return scores


def _flatten_pipeline_output(raw_output: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(raw_output, Mapping):
        return (raw_output,)

    if isinstance(raw_output, list):
        if raw_output and isinstance(raw_output[0], list):
            return tuple(item for group in raw_output for item in group)

        return tuple(raw_output)

    return ()
