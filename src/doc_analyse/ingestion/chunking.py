from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from doc_analyse.ingestion.models import ExtractedDocument, TextChunk

DEFAULT_CHUNK_SIZE = 1500  # Must fit in PromptGuard's 512-token window (512 * ~3 chars/token - overhead)
DEFAULT_CHUNK_OVERLAP = 200


@dataclass(frozen=True)
class TextChunker:
    """Splits normalized document text while preserving source offsets."""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

    def chunk(self, document: ExtractedDocument) -> tuple[TextChunk, ...]:
        """Return chunks that can be traced back to the normalized document text."""
        text = document.text
        if not isinstance(text, str) or not text.strip():
            raise ValueError("document text must be a non-empty string.")

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            end = _move_end_to_boundary(text, start, end)
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        source=document.source,
                        start_char=start,
                        end_char=end,
                        metadata=_chunk_metadata(document.metadata, len(chunks)),
                    )
                )

            if end >= text_length:
                break

            start = max(end - self.chunk_overlap, start + 1)
            start = _move_start_to_boundary(text, start)

        return tuple(chunks)


def chunk_document(
    document: ExtractedDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[TextChunk, ...]:
    return TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap).chunk(document)


def _move_end_to_boundary(text: str, start: int, end: int) -> int:
    # Prefer natural boundaries so cheap detectors and classifiers get readable spans.
    if end >= len(text):
        return end

    boundary = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
    if boundary <= start:
        return end

    return boundary


def _move_start_to_boundary(text: str, start: int) -> int:
    while start < len(text) and text[start].isspace():
        start += 1

    return start


def _chunk_metadata(document_metadata: Mapping[str, Any], index: int) -> dict[str, Any]:
    metadata = dict(document_metadata)
    metadata["chunk_index"] = index
    return metadata
