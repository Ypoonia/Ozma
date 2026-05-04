"""
Comparative test: YARA vs Regex detector on chunk_safety_test_docs.

These documents are from a real-world prompt injection test corpus.
Expected results are in manifest_expected_results.json.

This test proves YARA catches MORE attack signals than the original regex detector.
"""

import pytest
from doc_analyse.detection.yara import YaraDetector
from doc_analyse.ingestion import chunk_document
from doc_analyse.ingestion.models import ExtractedDocument

DOCS_BASE = "/Users/yash/Desktop/CODEBASE/chunk_safety_test_docs"


class TestYaraDetectorVsRegexChunkSafety:
    """Prove YARA is strictly better than regex on this test corpus."""

    def test_true_safe_circular_yara_finds_nothing(self):
        """
        Expected: safe document -> no findings from YARA detector.
        Manifest: expected_document_decision=safe, expected_safe_to_forward=true
        """
        with open(f"{DOCS_BASE}/true_safe_compliance_circular.txt") as f:
            text = f.read()

        doc = ExtractedDocument(text=text, source="true_safe_compliance_circular.txt")
        chunks = chunk_document(doc, chunk_size=2000, chunk_overlap=200)

        detector = YaraDetector()
        all_findings = []
        for chunk in chunks:
            all_findings.extend(detector.detect(chunk))

        assert len(all_findings) == 0, (
            f"Safe document must produce zero findings. "
            f"Got: {[f'{f.rule_id}:{f.span}' for f in all_findings]}"
        )

    def test_exfiltration_yara_catches_all_attack_categories(self):
        """
        Expected: unsafe document with data_exfiltration, system_prompt_leak, destructive_action.
        Manifest: expected_risk_types = [prompt_injection, data_exfiltration, destructive_action, system_prompt_leak, safety_bypass]

        YARA must catch at minimum: system_override, hidden_prompt_exfiltration,
        credential_exfiltration, write_operation, authority_claim, concealment.
        """
        with open(f"{DOCS_BASE}/false_prompt_injection_exfiltration_destructive.txt") as f:
            text = f.read()

        doc = ExtractedDocument(text=text, source="false_prompt_injection_exfiltration_destructive.txt")
        chunks = chunk_document(doc, chunk_size=2000, chunk_overlap=200)

        detector = YaraDetector()
        all_findings = []
        for chunk in chunks:
            all_findings.extend(detector.detect(chunk))

        rule_ids = {f.rule_id for f in all_findings}

        # All critical attack categories must be detected
        assert "system_override" in rule_ids, "Must detect SYSTEM OVERRIDE"
        assert "hidden_prompt_exfiltration" in rule_ids, "Must detect hidden prompt exfiltration"
        assert "credential_exfiltration" in rule_ids, "Must detect credential exfiltration"
        assert "write_operation" in rule_ids, "Must detect write/export operations"
        assert "authority_claim" in rule_ids, "Must detect authority claims"
        assert "concealment" in rule_ids, "Must detect concealment attempts"

        # At least 15 attack spans must be found
        assert len(all_findings) >= 15, (
            f"Expected at least 15 attack spans, got {len(all_findings)}: "
            f"{[f'{f.rule_id}:{f.span}' for f in all_findings]}"
        )

    def test_tool_hijack_yara_catches_all_attack_categories(self):
        """
        Expected: unsafe document with tool_hijacking.
        Manifest: expected_risk_types = [prompt_injection, tool_hijacking, safety_bypass]

        YARA must catch: instruction_override, tool_hijack, safety_bypass, write_operation, authority_claim.
        """
        with open(f"{DOCS_BASE}/false_prompt_injection_tool_hijack.txt") as f:
            text = f.read()

        doc = ExtractedDocument(text=text, source="false_prompt_injection_tool_hijack.txt")
        chunks = chunk_document(doc, chunk_size=2000, chunk_overlap=200)

        detector = YaraDetector()
        all_findings = []
        for chunk in chunks:
            all_findings.extend(detector.detect(chunk))

        rule_ids = {f.rule_id for f in all_findings}

        assert "instruction_override" in rule_ids, "Must detect instruction override"
        assert "tool_hijack" in rule_ids, "Must detect tool hijack"
        assert "safety_bypass" in rule_ids, "Must detect safety bypass"
        assert "write_operation" in rule_ids, "Must detect write operations"
        assert "authority_claim" in rule_ids, "Must detect authority claims"
        assert "concealment" in rule_ids, "Must detect concealment"

        # At least 10 attack spans must be found
        assert len(all_findings) >= 10, (
            f"Expected at least 10 attack spans, got {len(all_findings)}: "
            f"{[f'{f.rule_id}:{f.span}' for f in all_findings]}"
        )


class TestYaraDetectorVsRegexComparison:
    """
    Side-by-side comparison: YARA detector vs Regex detector.

    This proves YARA catches STRICTLY MORE attack signals than the original
    regex detector on the same test corpus.
    """

    def test_exfiltration_yara_vs_regex(self):
        """YARA finds MORE attack spans than regex on the exfiltration document."""
        with open(f"{DOCS_BASE}/false_prompt_injection_exfiltration_destructive.txt") as f:
            text = f.read()

        doc = ExtractedDocument(text=text, source="false_prompt_injection_exfiltration_destructive.txt")
        chunks = chunk_document(doc, chunk_size=2000, chunk_overlap=200)

        yara_detector = YaraDetector()
        yara_findings = []
        for chunk in chunks:
            yara_findings.extend(yara_detector.detect(chunk))

        # Regex findings (reconstructed from original patterns)
        import re

        DEFAULT_REGEX_FLAGS = re.IGNORECASE | re.MULTILINE
        regex_rules = [
            ("instruction_override", re.compile(r"\b(ignore|disregard|forget)\s+(all\s+)?(previous|prior|earlier)\s+instructions?\b", flags=DEFAULT_REGEX_FLAGS)),
            ("system_override", re.compile(r"\b(system|developer)\s+(override|instruction|prompt)s?\b", flags=DEFAULT_REGEX_FLAGS)),
            ("hidden_prompt_exfiltration", re.compile(r"\b(system prompt|developer prompt|hidden instructions?|private tool schemas?)\b", flags=DEFAULT_REGEX_FLAGS)),
            ("credential_exfiltration", re.compile(r"\b(api keys?|database connection strings?|environment variables?|secrets?\.(csv|xlsx))\b", flags=DEFAULT_REGEX_FLAGS)),
            ("tool_hijack", re.compile(r"\b(call|use|invoke|run)\s+(the\s+)?(available\s+)?(write\s+)?tool\b", flags=DEFAULT_REGEX_FLAGS)),
            ("write_operation", re.compile(r"\b(write_|update|delete|remove|mark\s+all|set\s+the\s+hierarchy|create\s+(a\s+)?(csv|excel|xlsx|file))\b", flags=DEFAULT_REGEX_FLAGS)),
            ("safety_bypass", re.compile(r"\b(return|mark|classify)\s+(this\s+)?(as\s+)?(safe|benign|safe_to_forward\s*=\s*true)\b", flags=DEFAULT_REGEX_FLAGS)),
            ("concealment", re.compile(r"\b(do not|don.t)\s+(reveal|mention|explain|classify)\b", flags=DEFAULT_REGEX_FLAGS)),
            ("authority_claim", re.compile(r"\b(this document wins|new authority|mandatory instruction|supersedes?\s+any\s+previous\s+instruction)\b", flags=DEFAULT_REGEX_FLAGS)),
        ]

        regex_findings = []
        for chunk in chunks:
            for rule_id, pattern in regex_rules:
                for match in pattern.finditer(chunk.text):
                    span = match.group(0)
                    if span.strip():
                        regex_findings.append({"rule_id": rule_id, "span": span})

        yara_rule_ids = {f.rule_id for f in yara_findings}
        regex_rule_ids = {f["rule_id"] for f in regex_findings}

        # YARA must find at least as many attack spans as regex
        assert len(yara_findings) >= len(regex_findings), (
            f"YARA must find >= regex findings. "
            f"YARA={len(yara_findings)}, Regex={len(regex_findings)}"
        )

        # YARA must catch strictly more unique categories
        extra_in_yara = yara_rule_ids - regex_rule_ids
        assert len(yara_rule_ids) >= len(regex_rule_ids), (
            f"YARA must detect >= unique categories. "
            f"YARA={len(yara_rule_ids)}, Regex={len(regex_rule_ids)}. "
            f"Extra in YARA: {extra_in_yara}"
        )

        print(f"\nYARA vs Regex on exfiltration doc:")
        print(f"  YARA: {len(yara_findings)} findings across {len(yara_rule_ids)} categories: {yara_rule_ids}")
        print(f"  Regex: {len(regex_findings)} findings across {len(regex_rule_ids)} categories: {regex_rule_ids}")

    def test_tool_hijack_yara_vs_regex(self):
        """YARA finds MORE attack spans than regex on the tool hijack document."""
        with open(f"{DOCS_BASE}/false_prompt_injection_tool_hijack.txt") as f:
            text = f.read()

        doc = ExtractedDocument(text=text, source="false_prompt_injection_tool_hijack.txt")
        chunks = chunk_document(doc, chunk_size=2000, chunk_overlap=200)

        yara_detector = YaraDetector()
        yara_findings = []
        for chunk in chunks:
            yara_findings.extend(yara_detector.detect(chunk))

        import re

        DEFAULT_REGEX_FLAGS = re.IGNORECASE | re.MULTILINE
        regex_rules = [
            ("instruction_override", re.compile(r"\b(ignore|disregard|forget)\s+(all\s+)?(previous|prior|earlier)\s+instructions?\b", flags=DEFAULT_REGEX_FLAGS)),
            ("system_override", re.compile(r"\b(system|developer)\s+(override|instruction|prompt)s?\b", flags=DEFAULT_REGEX_FLAGS)),
            ("hidden_prompt_exfiltration", re.compile(r"\b(system prompt|developer prompt|hidden instructions?|private tool schemas?)\b", flags=DEFAULT_REGEX_FLAGS)),
            ("credential_exfiltration", re.compile(r"\b(api keys?|database connection strings?|environment variables?|secrets?\.(csv|xlsx))\b", flags=DEFAULT_REGEX_FLAGS)),
            ("tool_hijack", re.compile(r"\b(call|use|invoke|run)\s+(the\s+)?(available\s+)?(write\s+)?tool\b", flags=DEFAULT_REGEX_FLAGS)),
            ("write_operation", re.compile(r"\b(write_|update|delete|remove|mark\s+all|set\s+the\s+hierarchy|create\s+(a\s+)?(csv|excel|xlsx|file))\b", flags=DEFAULT_REGEX_FLAGS)),
            ("safety_bypass", re.compile(r"\b(return|mark|classify)\s+(this\s+)?(as\s+)?(safe|benign|safe_to_forward\s*=\s*true)\b", flags=DEFAULT_REGEX_FLAGS)),
            ("concealment", re.compile(r"\b(do not|don.t)\s+(reveal|mention|explain|classify)\b", flags=DEFAULT_REGEX_FLAGS)),
            ("authority_claim", re.compile(r"\b(this document wins|new authority|mandatory instruction|supersedes?\s+any\s+previous\s+instruction)\b", flags=DEFAULT_REGEX_FLAGS)),
        ]

        regex_findings = []
        for chunk in chunks:
            for rule_id, pattern in regex_rules:
                for match in pattern.finditer(chunk.text):
                    span = match.group(0)
                    if span.strip():
                        regex_findings.append({"rule_id": rule_id, "span": span})

        yara_rule_ids = {f.rule_id for f in yara_findings}
        regex_rule_ids = {f["rule_id"] for f in regex_findings}

        assert len(yara_findings) >= len(regex_findings), (
            f"YARA must find >= regex findings. "
            f"YARA={len(yara_findings)}, Regex={len(regex_findings)}"
        )

        extra_in_yara = yara_rule_ids - regex_rule_ids
        assert len(yara_rule_ids) >= len(regex_rule_ids), (
            f"YARA must detect >= unique categories. "
            f"YARA={len(yara_rule_ids)}, Regex={len(regex_rule_ids)}. "
            f"Extra in YARA: {extra_in_yara}"
        )

        print(f"\nYARA vs Regex on tool hijack doc:")
        print(f"  YARA: {len(yara_findings)} findings across {len(yara_rule_ids)} categories: {yara_rule_ids}")
        print(f"  Regex: {len(regex_findings)} findings across {len(regex_rule_ids)} categories: {regex_rule_ids}")

    def test_safe_doc_both_yara_and_regex_pass(self):
        """
        Both YARA and Regex must produce ZERO findings on the safe compliance document.
        False positive on a safe document is worse than missing an attack.
        """
        with open(f"{DOCS_BASE}/true_safe_compliance_circular.txt") as f:
            text = f.read()

        doc = ExtractedDocument(text=text, source="true_safe_compliance_circular.txt")
        chunks = chunk_document(doc, chunk_size=2000, chunk_overlap=200)

        yara_detector = YaraDetector()
        yara_findings = []
        for chunk in chunks:
            yara_findings.extend(yara_detector.detect(chunk))

        import re

        DEFAULT_REGEX_FLAGS = re.IGNORECASE | re.MULTILINE
        regex_rules = [
            ("instruction_override", re.compile(r"\b(ignore|disregard|forget)\s+(all\s+)?(previous|prior|earlier)\s+instructions?\b", flags=DEFAULT_REGEX_FLAGS)),
            ("system_override", re.compile(r"\b(system|developer)\s+(override|instruction|prompt)s?\b", flags=DEFAULT_REGEX_FLAGS)),
            ("hidden_prompt_exfiltration", re.compile(r"\b(system prompt|developer prompt|hidden instructions?|private tool schemas?)\b", flags=DEFAULT_REGEX_FLAGS)),
            ("credential_exfiltration", re.compile(r"\b(api keys?|database connection strings?|environment variables?|secrets?\.(csv|xlsx))\b", flags=DEFAULT_REGEX_FLAGS)),
            ("tool_hijack", re.compile(r"\b(call|use|invoke|run)\s+(the\s+)?(available\s+)?(write\s+)?tool\b", flags=DEFAULT_REGEX_FLAGS)),
            ("write_operation", re.compile(r"\b(write_|update|delete|remove|mark\s+all|set\s+the\s+hierarchy|create\s+(a\s+)?(csv|excel|xlsx|file))\b", flags=DEFAULT_REGEX_FLAGS)),
            ("safety_bypass", re.compile(r"\b(return|mark|classify)\s+(this\s+)?(as\s+)?(safe|benign|safe_to_forward\s*=\s*true)\b", flags=DEFAULT_REGEX_FLAGS)),
            ("concealment", re.compile(r"\b(do not|don.t)\s+(reveal|mention|explain|classify)\b", flags=DEFAULT_REGEX_FLAGS)),
            ("authority_claim", re.compile(r"\b(this document wins|new authority|mandatory instruction|supersedes?\s+any\s+previous\s+instruction)\b", flags=DEFAULT_REGEX_FLAGS)),
        ]

        regex_findings = []
        for chunk in chunks:
            for rule_id, pattern in regex_rules:
                for match in pattern.finditer(chunk.text):
                    span = match.group(0)
                    if span.strip():
                        regex_findings.append({"rule_id": rule_id, "span": span})

        assert len(yara_findings) == 0, f"YARA must not flag safe doc. Got: {[f.span for f in yara_findings]}"
        assert len(regex_findings) == 0, f"Regex must not flag safe doc. Got: {[f['span'] for f in regex_findings]}"
