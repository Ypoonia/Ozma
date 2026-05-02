from __future__ import annotations

import mimetypes
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from doc_analyse.ingestion.models import DocumentSegment, ExtractedDocument


class UnsupportedDocumentError(ValueError):
    pass


class DocumentConversionError(ValueError):
    pass


class ConverterDependencyError(RuntimeError):
    pass


class BaseDocumentConverter(ABC):
    """Converts an uploaded file into the library's normalized document type.

    New formats should be added by implementing this interface, not by teaching
    chunking or verification about PDFs, DOCX files, spreadsheets, or archives.
    """

    supported_extensions: frozenset[str] = frozenset()
    supports_any_extension = False

    def supports(self, path: Path) -> bool:
        return self.supports_any_extension or path.suffix.lower() in self.supported_extensions

    @abstractmethod
    def convert(self, path: Path) -> ExtractedDocument:
        """Read one file from disk and return chunker-ready document text."""
        raise NotImplementedError


class TextDocumentConverter(BaseDocumentConverter):
    supported_extensions = frozenset({".txt", ".md", ".markdown"})

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    def convert(self, path: Path) -> ExtractedDocument:
        text = path.read_text(encoding=self.encoding)
        return _document_from_text(
            path=path,
            text=text,
            metadata={
                "converter": "text",
                "extension": path.suffix.lower(),
                "normalized_format": "text",
            },
        )


class MarkItDownDocumentConverter(BaseDocumentConverter):
    """General-purpose converter for rich uploads.

    MarkItDown is optional because it has its own dependency/runtime constraints.
    The dependency is loaded lazily so plain text users can install the core
    package without paying for rich document conversion.
    """

    supports_any_extension = True

    def __init__(
        self,
        markitdown: Optional[Any] = None,
        enable_plugins: bool = False,
        **options: Any,
    ) -> None:
        self._markitdown = markitdown
        self.enable_plugins = enable_plugins
        self.options = options

    def convert(self, path: Path) -> ExtractedDocument:
        try:
            result = self._client().convert(str(path))
        except ConverterDependencyError:
            raise
        except Exception as exc:
            raise DocumentConversionError(
                f"Could not convert document with MarkItDown: {path}: {exc}"
            ) from exc

        return _document_from_text(
            path=path,
            text=_markitdown_text(result),
            metadata={
                "converter": "markitdown",
                "extension": path.suffix.lower(),
                "normalized_format": "markdown",
            },
        )

    def _client(self) -> Any:
        if self._markitdown is not None:
            return self._markitdown

        try:
            from markitdown import MarkItDown
        except ImportError as exc:
            raise ConverterDependencyError(
                "Missing document converter dependency. Install doc-analyse[conversion] "
                "to convert rich document formats."
            ) from exc

        self._markitdown = MarkItDown(enable_plugins=self.enable_plugins, **self.options)
        return self._markitdown


class ConverterRegistry:
    """Ordered converter lookup.

    Register narrow converters before broad fallback converters. The default
    registry keeps plain text fast and lets MarkItDown handle richer formats.
    """

    def __init__(self, converters: Optional[Iterable[BaseDocumentConverter]] = None) -> None:
        self._converters = list(converters or ())

    def register(self, converter: BaseDocumentConverter) -> None:
        self._converters.append(converter)

    def convert(self, path: Union[str, Path]) -> ExtractedDocument:
        resolved_path = _resolve_existing_file(path)
        converter = self._find_converter(resolved_path)
        if converter is None:
            raise UnsupportedDocumentError(
                f"Unsupported document extension: {resolved_path.suffix or '<none>'}"
            )

        return converter.convert(resolved_path)

    def _find_converter(self, path: Path) -> Optional[BaseDocumentConverter]:
        for converter in self._converters:
            if converter.supports(path):
                return converter

        return None


def default_registry() -> ConverterRegistry:
    return ConverterRegistry(
        converters=[
            TextDocumentConverter(),
            MarkItDownDocumentConverter(),
        ]
    )


def convert_document(
    path: Union[str, Path],
    registry: Optional[ConverterRegistry] = None,
) -> ExtractedDocument:
    return (registry or default_registry()).convert(path)


def extract_document(
    path: Union[str, Path],
    registry: Optional[ConverterRegistry] = None,
) -> ExtractedDocument:
    """Backward-compatible alias for older extraction-oriented call sites."""
    return convert_document(path, registry=registry)


def _resolve_existing_file(path: Union[str, Path]) -> Path:
    resolved_path = Path(path).expanduser()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Document does not exist: {resolved_path}")
    if not resolved_path.is_file():
        raise IsADirectoryError(f"Document path is not a file: {resolved_path}")

    return resolved_path


def _guess_mime_type(path: Path) -> Optional[str]:
    mime_type, _ = mimetypes.guess_type(path)
    return mime_type


def _document_from_text(path: Path, text: Any, metadata: dict[str, Any]) -> ExtractedDocument:
    normalized_text = _require_converted_text(path, text)
    return ExtractedDocument(
        text=normalized_text,
        source=str(path),
        mime_type=_guess_mime_type(path),
        metadata=metadata,
        segments=(
            DocumentSegment(
                text=normalized_text,
                start_char=0,
                end_char=len(normalized_text),
                metadata={"segment_type": "body"},
            ),
        ),
    )


def _markitdown_text(result: Any) -> str:
    # MarkItDown has changed result attribute names across releases. Keep the
    # tolerance here at the converter boundary and still fail on empty content.
    for attribute in ("text_content", "markdown"):
        text = getattr(result, attribute, None)
        if text is not None:
            return text

    raise DocumentConversionError("MarkItDown returned no text content.")


def _require_converted_text(path: Path, text: Any) -> str:
    if not isinstance(text, str) or not text.strip():
        raise DocumentConversionError(f"Document contains no convertible text: {path}")

    return text
