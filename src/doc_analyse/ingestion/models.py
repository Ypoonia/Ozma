from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class DocumentSegment:
    """A traceable section inside the normalized document text."""

    text: str
    start_char: int
    end_char: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedDocument:
    """Normalized document representation shared by ingestion and chunking."""

    text: str
    source: str
    mime_type: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    segments: tuple[DocumentSegment, ...] = field(default_factory=tuple)

    @property
    def source_path(self) -> Path:
        return Path(self.source)


@dataclass(frozen=True)
class TextChunk:
    """Chunk text plus offsets into the normalized document."""

    text: str
    source: str
    start_char: int
    end_char: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.end_char - self.start_char


@dataclass(frozen=True)
class IngestedDocument:
    """Result returned by the high-level ingestion pipeline."""

    document: ExtractedDocument
    chunks: tuple[TextChunk, ...]

    @property
    def text(self) -> str:
        return self.document.text

    @property
    def source(self) -> str:
        return self.document.source
