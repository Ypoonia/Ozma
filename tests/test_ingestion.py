from pathlib import Path

import pytest

import doc_analyse
import doc_analyse.ingestion as ingestion
from doc_analyse.ingestion import (
    ConverterRegistry,
    DocumentConversionError,
    MarkItDownDocumentConverter,
    TextChunker,
    UnsupportedDocumentError,
    chunk_document,
    convert_document,
    ingest_document,
)
from doc_analyse.ingestion.models import ExtractedDocument


class FakeMarkItDown:
    def __init__(self, text: str) -> None:
        self.text = text

    def convert(self, path: str):
        return type("MarkItDownResult", (), {"text_content": self.text, "source": path})()


def test_convert_document_reads_text_file(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text("First line\nSecond line", encoding="utf-8")

    document = convert_document(path)

    assert document.text == "First line\nSecond line"
    assert document.source == str(path)
    assert document.mime_type == "text/plain"
    assert document.metadata["converter"] == "text"
    assert document.metadata["extension"] == ".txt"
    assert len(document.segments) == 1
    assert document.segments[0].start_char == 0
    assert document.segments[0].end_char == len(document.text)


def test_base_document_converter_is_public_extension_point():
    assert "BaseDocumentConverter" in ingestion.__all__
    assert ingestion.BaseDocumentConverter is doc_analyse.BaseDocumentConverter


def test_convert_document_rejects_missing_or_unsupported_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        convert_document(tmp_path / "missing.txt")

    unsupported = tmp_path / "sample.bin"
    unsupported.write_bytes(b"binary")

    with pytest.raises(UnsupportedDocumentError):
        ConverterRegistry().convert(unsupported)


def test_convert_document_rejects_empty_text(tmp_path: Path):
    path = tmp_path / "empty.md"
    path.write_text("   ", encoding="utf-8")

    with pytest.raises(DocumentConversionError, match="convertible text"):
        convert_document(path)


def test_markitdown_converter_normalizes_any_upload_to_document(tmp_path: Path):
    path = tmp_path / "sample.docx"
    path.write_bytes(b"fake")
    converter = MarkItDownDocumentConverter(markitdown=FakeMarkItDown("# Title\n\nBody"))
    registry = ConverterRegistry(converters=[converter])

    document = convert_document(path, registry=registry)

    assert document.text == "# Title\n\nBody"
    assert document.metadata["converter"] == "markitdown"
    assert document.metadata["normalized_format"] == "markdown"
    assert document.segments[0].end_char == len(document.text)


def test_chunk_document_preserves_offsets_and_metadata():
    document = ExtractedDocument(
        text="alpha beta gamma delta epsilon",
        source="memory.txt",
        metadata={"extension": ".txt"},
    )

    chunks = chunk_document(document, chunk_size=12, chunk_overlap=2)

    assert [chunk.text for chunk in chunks] == ["alpha beta", "ta gamma", "ma delta", "ta epsilon"]
    assert [(chunk.start_char, chunk.end_char) for chunk in chunks] == [
        (0, 10),
        (8, 16),
        (14, 22),
        (20, 30),
    ]
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1, 2, 3]
    assert all(chunk.metadata["extension"] == ".txt" for chunk in chunks)


def test_chunker_rejects_invalid_settings():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=0)

    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, chunk_overlap=10)

    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, chunk_overlap=-1)


def test_ingest_document_extracts_and_chunks_path(tmp_path: Path):
    path = tmp_path / "sample.md"
    path.write_text("alpha beta gamma delta epsilon", encoding="utf-8")

    result = ingest_document(path, chunk_size=12, chunk_overlap=2)

    assert result.document.source == str(path)
    assert result.text == "alpha beta gamma delta epsilon"
    assert [chunk.metadata["chunk_index"] for chunk in result.chunks] == [0, 1, 2, 3]
