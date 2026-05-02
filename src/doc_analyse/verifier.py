from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from doc_analyse.classifiers.base import BaseClassifier, ClassificationResult


@dataclass
class DocumentVerifier:
    classifier: BaseClassifier

    def verify_text(
        self,
        text: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ClassificationResult:
        return self.classifier.classify(text=text, metadata=metadata)
