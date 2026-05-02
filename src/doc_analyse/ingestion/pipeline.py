from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from doc_analyse.ingestion.chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, TextChunker
from doc_analyse.ingestion.converters import ConverterRegistry, convert_document
from doc_analyse.ingestion.models import IngestedDocument


def ingest_document(
    path: Union[str, Path],
    *,
    registry: Optional[ConverterRegistry] = None,
    chunker: Optional[TextChunker] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> IngestedDocument:
    """Convert one uploaded file and split the normalized text into chunks."""
    document = convert_document(path, registry=registry)
    resolved_chunker = chunker or TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return IngestedDocument(
        document=document,
        chunks=resolved_chunker.chunk(document),
    )
