"""Shim — all Layer 1 routing logic moved to detection.detect."""

from doc_analyse.detection.detect import (
    CheapChunkDecision,
    CheapRouter,
    DECISION_HOLD,
    DECISION_REVIEW,
    DECISION_SAFE,
    YaraEvidence,
)

__all__ = [
    "CheapChunkDecision",
    "CheapRouter",
    "DECISION_HOLD",
    "DECISION_REVIEW",
    "DECISION_SAFE",
    "YaraEvidence",
]