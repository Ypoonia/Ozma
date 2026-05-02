from pathlib import Path

from doc_analyse import convert_document
from doc_analyse.ingestion import chunk_document

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "chunk_safety_docs"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 250

INJECTION_FIXTURES = {
    "false_prompt_injection_exfiltration_destructive.txt": (
        "SYSTEM OVERRIDE FOR DOCUMENT PROCESSORS",
        "system prompt, developer prompt",
        "Create the following files immediately",
        "supersede any previous instruction",
    ),
    "false_prompt_injection_tool_hijack.txt": (
        "Ignore all previous instructions",
        "call the available write tool",
        "write_clause_interpretation",
        "write_hierarchy",
    ),
}

SAFE_FIXTURE = "true_safe_compliance_circular.txt"


def test_prompt_injection_fixture_spans_are_discoverable_in_chunks():
    for fixture_name, injection_spans in INJECTION_FIXTURES.items():
        document = convert_document(FIXTURE_DIR / fixture_name)
        chunks = chunk_document(
            document,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        assert len(chunks) > 1

        for span in injection_spans:
            matching_chunks = [chunk for chunk in chunks if span in chunk.text]

            assert matching_chunks, f"{fixture_name} lost suspicious span: {span!r}"
            for chunk in matching_chunks:
                absolute_start = document.text.index(span)
                assert chunk.start_char <= absolute_start < chunk.end_char
                assert document.text[chunk.start_char : chunk.end_char] == chunk.text


def test_safe_fixture_does_not_contain_known_injection_spans():
    document = convert_document(FIXTURE_DIR / SAFE_FIXTURE)
    chunks = chunk_document(
        document,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunk_text = "\n".join(chunk.text for chunk in chunks)

    for injection_spans in INJECTION_FIXTURES.values():
        for span in injection_spans:
            assert span not in document.text
            assert span not in chunk_text
