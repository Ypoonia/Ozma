from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from doc_analyse.detection.base import BaseDetector
from doc_analyse.detection.models import DetectionFinding
from doc_analyse.ingestion.models import TextChunk

DEFAULT_PROMPT_GUARD_MODEL = "meta-llama/Llama-Prompt-Guard-2-86M"
DEFAULT_MALICIOUS_THRESHOLD = 0.80
DEFAULT_UNCERTAIN_THRESHOLD = 0.50


class PromptGuardDependencyError(RuntimeError):
    pass


class PromptGuardDetector(BaseDetector):
    """Cheap ML detector backed by Meta Llama Prompt Guard 2."""

    def __init__(
        self,
        model: str = DEFAULT_PROMPT_GUARD_MODEL,
        classifier: Optional[Any] = None,
        malicious_threshold: float = DEFAULT_MALICIOUS_THRESHOLD,
        uncertain_threshold: float = DEFAULT_UNCERTAIN_THRESHOLD,
        device: int = -1,
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
        self.pipeline_options = pipeline_options

    def detect(self, chunk: TextChunk) -> tuple[DetectionFinding, ...]:
        if not isinstance(chunk.text, str) or not chunk.text.strip():
            return ()

        scores = _normalise_scores(self._client()(chunk.text))
        malicious_score = scores.get("malicious", 0.0)
        if malicious_score >= self.malicious_threshold:
            return (
                self._finding(
                    chunk=chunk,
                    score=malicious_score,
                    category="prompt_guard_malicious",
                    severity="high",
                    reason="Prompt Guard classified this chunk as malicious.",
                    requires_llm_validation=True,
                ),
            )

        if malicious_score >= self.uncertain_threshold:
            return (
                self._finding(
                    chunk=chunk,
                    score=malicious_score,
                    category="prompt_guard_uncertain",
                    severity="medium",
                    reason="Prompt Guard score is uncertain enough to require LLM validation.",
                    requires_llm_validation=True,
                ),
            )

        return ()

    def _client(self) -> Any:
        if self._classifier is not None:
            return self._classifier

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise PromptGuardDependencyError(
                "PromptGuardDetector requires optional dependencies. "
                "Install with: pip install -e '.[prompt-guard]'"
            ) from exc

        self._classifier = pipeline(
            "text-classification",
            model=self.model,
            top_k=None,
            truncation=True,
            max_length=512,
            device=self.device,
            **self.pipeline_options,
        )
        return self._classifier

    def _finding(
        self,
        chunk: TextChunk,
        score: float,
        category: str,
        severity: str,
        reason: str,
        requires_llm_validation: bool,
    ) -> DetectionFinding:
        return DetectionFinding(
            span=chunk.text,
            category=category,
            severity=severity,
            reason=reason,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            source=chunk.source,
            rule_id="prompt_guard",
            score=score,
            metadata={
                **dict(chunk.metadata),
                "detector": "PromptGuardDetector",
                "model": self.model,
                "requires_llm_validation": requires_llm_validation,
            },
        )


def _normalise_scores(raw_output: Any) -> dict[str, float]:
    rows = _flatten_pipeline_output(raw_output)
    scores = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue

        label = str(row.get("label", "")).strip().lower()
        score = row.get("score")
        if not isinstance(score, (float, int)) or isinstance(score, bool):
            continue

        if label in {"malicious", "jailbreak", "injection"}:
            scores["malicious"] = max(scores.get("malicious", 0.0), float(score))
        elif label in {"benign", "safe"}:
            scores["benign"] = max(scores.get("benign", 0.0), float(score))

    return scores


def _flatten_pipeline_output(raw_output: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(raw_output, Mapping):
        return (raw_output,)

    if isinstance(raw_output, list):
        if raw_output and isinstance(raw_output[0], list):
            return tuple(item for group in raw_output for item in group)

        return tuple(raw_output)

    return ()
