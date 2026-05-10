from __future__ import annotations

import logging
from importlib.resources import files
from pathlib import Path
from typing import Any, Optional, Union

try:
    import yara
except ImportError:  # pragma: no cover
    yara = None  # type: ignore[assignment]

from doc_analyse.detection.base import BaseDetector
from doc_analyse.detection.models import DetectionFinding
from doc_analyse.ingestion.chunking import _build_byte_to_char
from doc_analyse.ingestion.models import TextChunk

DEFAULT_YARA_RULES_FILE = "default.yara"
_YARA_PACKAGE = "doc_analyse.detection"
logger = logging.getLogger(__name__)


class YaraGlossaryError(ValueError):
    """Raised when a YARA rules file is missing, invalid, or cannot be compiled."""


def _log_origin(origin: str) -> str:
    if origin.startswith("<") and origin.endswith(">"):
        return origin
    return Path(origin).name or origin


def _read_yara_source(path: Optional[Union[str, Path]]) -> str:
    try:
        if path is None:
            pkg_file = files(_YARA_PACKAGE).joinpath(DEFAULT_YARA_RULES_FILE)
            return pkg_file.read_text(encoding="utf-8")
        return Path(str(path)).read_text(encoding="utf-8")
    except OSError as exc:
        source = DEFAULT_YARA_RULES_FILE if path is None else str(path)
        raise YaraGlossaryError(f"Could not read YARA rules file '{source}': {exc}") from exc


def compile_yara_rules(source: str, *, origin: str = "<memory>") -> Any:
    """Compile YARA rules from source text."""
    log_origin = _log_origin(origin)
    if yara is None:
        logger.warning("yara_compile_unavailable", extra={"origin": log_origin})
        raise YaraGlossaryError(
            "yara-python is required. Install with: pip install 'doc-analyse[yara]'"
        )
    logger.debug(
        "yara_compile_start",
        extra={"origin": log_origin, "source_chars": len(source)},
    )
    try:
        compiled = yara.compile(source=source)
    except (yara.Error, yara.SyntaxError) as exc:
        logger.warning(
            "yara_compile_failed",
            extra={"origin": log_origin, "error_type": type(exc).__name__},
        )
        raise YaraGlossaryError(f"Failed to compile YARA rules from '{origin}': {exc}") from exc
    logger.info(
        "yara_compile_succeeded",
        extra={"origin": log_origin, "source_chars": len(source)},
    )
    return compiled


def _meta_bool(value: Any, default: bool) -> bool:
    """Parse a YARA metadata value as boolean.

    YARA metadata values come in as strings, ints, or bools depending on
    how they were declared in the .yara file. Handles the common cases.

    Default to `default` if value is None or unparseable.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "yes", "1", "on"):
            return True
        if s in ("false", "no", "0", "off"):
            return False
    return default


def _meta_float(value: Any, default: float) -> float:
    """Parse a YARA metadata value as float.

    Handles int, float, and numeric string representations.
    """
    if value is None:
        return default
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            pass
    return default


class YaraDetector(BaseDetector):
    """YARA-based cheap detector for prompt-injection-like evidence."""

    def __init__(self, compiled: Optional[Any] = None) -> None:
        self._compiled = compiled if compiled is not None else _DEFAULT_COMPILED_RULES

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> YaraDetector:
        """Build a detector from a standalone .yar file."""
        source = _read_yara_source(path)
        compiled = compile_yara_rules(source, origin=str(path))
        return cls(compiled=compiled)

    def detect(self, chunk: TextChunk) -> tuple[DetectionFinding, ...]:
        if not isinstance(chunk.text, str) or not chunk.text.strip():
            return ()
        if self._compiled is None:
            # `_DEFAULT_COMPILED_RULES` is None either because yara-python is not
            # installed (yara is None) or because the bundled rules failed to compile.
            # Distinguish the two cases so the error message is actionable.
            if yara is None:
                raise YaraGlossaryError(
                    "yara-python is required. Install with: pip install 'doc-analyse[yara]'"
                )
            raise YaraGlossaryError(
                "Default YARA rules failed to load. "
                "Reinstall the package or supply custom rules via from_file()."
            )

        encoded = chunk.text.encode("utf-8")
        byte_to_char = chunk.metadata.get("byte_to_char")
        if byte_to_char is None:
            byte_to_char = _build_byte_to_char(chunk.text)

        results = self._compiled.match(data=encoded)
        logger.debug(
            "yara_detection_matched",
            extra={
                "chunk_start": chunk.start_char,
                "chunk_end": chunk.end_char,
                "chunk_bytes": len(encoded),
                "matched_rules": len(results),
            },
        )

        findings = []
        for rule in results:
            rule_meta = dict(rule.meta)
            rule_id = str(rule_meta.get("rule_id", rule.rule))

            for string_match in rule.strings:
                for instance in string_match.instances:
                    matched_bytes = instance.matched_data
                    if isinstance(matched_bytes, bytes):
                        span_text = matched_bytes.decode("utf-8", errors="replace")
                    else:
                        span_text = matched_bytes
                    if not span_text.strip():
                        continue

                    char_offset = byte_to_char[instance.offset]
                    span_char_len = len(span_text)

                    # Read per-rule metadata
                    weight = _meta_float(rule_meta.get("weight"), 0.0)
                    route_hint = str(rule_meta.get("route_hint", "evidence"))
                    requires_llm = _meta_bool(rule_meta.get("requires_llm_validation"), False)

                    finding_metadata = {
                        "detector": "YaraDetector",
                        "yara_rule": rule_id,
                        "yara_weight": weight,
                        "route_hint": route_hint,
                    }

                    # Normalize YARA weight to 0.0-1.0 range for the generic score field.
                    # Raw weight lives in metadata["yara_weight"] for the router.
                    normalized_score = min(1.0, weight / 100.0) if weight > 0 else None

                    findings.append(
                        self._build_finding(
                            chunk=chunk,
                            span=span_text,
                            category=str(rule_meta.get("category", "unknown")),
                            severity=str(rule_meta.get("severity", "medium")),
                            reason=str(rule_meta.get("reason", "YARA rule matched.")),
                            rule_id=rule_id,
                            start_char=chunk.start_char + char_offset,
                            end_char=chunk.start_char + char_offset + span_char_len,
                            requires_llm_validation=requires_llm,
                            score=normalized_score,
                            metadata=finding_metadata,
                        )
                    )

        finalized = self._finalize_findings(findings)
        log_payload = {
            "chunk_start": chunk.start_char,
            "chunk_end": chunk.end_char,
            "matched_rules": len(results),
            "finding_count": len(finalized),
            "rule_count": len({finding.rule_id for finding in finalized}),
        }
        if finalized:
            logger.info("yara_detection_findings", extra=log_payload)
        else:
            logger.debug("yara_detection_no_findings", extra=log_payload)
        return finalized


# ---------------------------------------------------------------------------
# Module-level compiled rules
# ---------------------------------------------------------------------------

def _load_default_rules() -> Any:
    if yara is None:
        logger.debug(
            "yara_default_rules_skipped",
            extra={"origin": DEFAULT_YARA_RULES_FILE, "reason": "dependency_unavailable"},
        )
        return None
    try:
        source = files(_YARA_PACKAGE).joinpath(DEFAULT_YARA_RULES_FILE).read_text(encoding="utf-8")
        compiled = yara.compile(source=source)
    except Exception as exc:
        logger.warning(
            "yara_default_rules_load_failed",
            extra={
                "origin": DEFAULT_YARA_RULES_FILE,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        return None
    logger.info(
        "yara_default_rules_loaded",
        extra={"origin": DEFAULT_YARA_RULES_FILE, "source_chars": len(source)},
    )
    return compiled


_DEFAULT_COMPILED_RULES = _load_default_rules()
