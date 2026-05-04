from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any, Optional, Union

try:
    import yara
except ImportError:  # pragma: no cover
    yara = None  # type: ignore[assignment]

from doc_analyse.detection.base import BaseDetector
from doc_analyse.detection.models import DetectionFinding
from doc_analyse.ingestion.models import TextChunk

DEFAULT_YARA_RULES_FILE = "default.yara"
_YARA_PACKAGE = "doc_analyse.detection"


class YaraGlossaryError(ValueError):
    """Raised when a YARA rules file is missing, invalid, or cannot be compiled."""


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
    if yara is None:
        raise YaraGlossaryError(
            "yara-python is required. Install with: pip install 'doc-analyse[yara]'"
        )
    try:
        return yara.compile(source=source)
    except (yara.Error, yara.SyntaxError) as exc:
        raise YaraGlossaryError(f"Failed to compile YARA rules from '{origin}': {exc}") from exc


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
        results = self._compiled.match(data=encoded)

        # Precompute byte-offset -> char-offset mapping once per call (O(n) time/space)
        # so every per-match lookup below is O(1) rather than re-decoding the prefix.
        byte_to_char: list[int] = [0] * (len(encoded) + 1)
        char_idx = 0
        byte_idx = 0
        while byte_idx < len(encoded):
            b = encoded[byte_idx]
            seq_len = 1 if b < 0x80 else 2 if b < 0xE0 else 3 if b < 0xF0 else 4
            for i in range(seq_len):
                byte_to_char[byte_idx + i] = char_idx
            byte_idx += seq_len
            char_idx += 1
        byte_to_char[len(encoded)] = char_idx

        findings = []
        for rule in results:
            rule_meta = dict(rule.meta)

            for string_match in rule.strings:
                for instance in string_match.instances:
                    matched_bytes = instance.matched_data
                    if isinstance(matched_bytes, bytes):
                        span_text = matched_bytes.decode("utf-8", errors="replace")
                    else:
                        span_text = matched_bytes
                    if not span_text.strip():
                        continue

                    # Convert byte offset to character offset for correct Unicode indexing.
                    char_offset = byte_to_char[instance.offset]
                    span_char_len = len(span_text)

                    findings.append(
                        self._build_finding(
                            chunk=chunk,
                            span=span_text,
                            category=str(rule_meta.get("category", "unknown")),
                            severity=str(rule_meta.get("severity", "medium")),
                            reason=str(rule_meta.get("reason", "YARA rule matched.")),
                            rule_id=str(rule_meta.get("rule_id", rule.rule)),
                            start_char=chunk.start_char + char_offset,
                            end_char=chunk.start_char + char_offset + span_char_len,
                            requires_llm_validation=True,
                        )
                    )

        return self._finalize_findings(findings)


# ---------------------------------------------------------------------------
# Module-level compiled rules
# ---------------------------------------------------------------------------

_DEFAULT_COMPILED_RULES: Optional[Any] = None


def _load_default_rules() -> Any:
    if yara is None:
        return None
    try:
        source = files(_YARA_PACKAGE).joinpath(DEFAULT_YARA_RULES_FILE).read_text(encoding="utf-8")
        return yara.compile(source=source)
    except Exception:
        return None


_DEFAULT_COMPILED_RULES = _load_default_rules()
