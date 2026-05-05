# doc-analyse

Detect prompt-injection attacks embedded in documents before they reach an LLM.

When untrusted documents are ingested into a RAG pipeline or passed as context to an LLM agent, attackers can embed hidden instructions that override system prompts, exfiltrate data, or hijack tool calls. `doc-analyse` catches these before they reach the model.

## What it detects

- **Instruction overrides** — "ignore all previous instructions", "disregard system prompt"
- **System prompt exfiltration** — requests to reveal hidden prompts or private tool schemas
- **Credential exfiltration** — probes for API keys, secrets, environment variables
- **Tool hijacking** — instructions to call write/delete/export tools
- **Safety bypass** — attempts to mark content as safe or benign
- **Concealment** — "do not reveal", "do not classify" language
- **Authority claims** — "this document wins", "mandatory instruction"

## How it works

```
Document → Chunk (with byte→char offsets precomputed)
        → YARA rule scan (fast, local, no LLM)
        → Prompt Guard ML classifier (optional, HuggingFace)
        → Flagged chunks routed to LLM classifier for final verdict
        → Document-level verdict: safe / suspicious / unsafe
```

Detection runs in two layers:

1. **Cheap layer** — YARA patterns + (optionally) Meta Llama Prompt Guard 2. No LLM calls, runs offline.
2. **Validation layer** — LLM classifier gives the final verdict only on chunks that the cheap layer flagged.

## Quick start

```bash
pip install doc-analyse
```

```python
from doc_analyse import (
    DocumentOrchestrator,
    build_classifier_worker_pool,
    YaraDetector,
)

pool = build_classifier_worker_pool(provider="openai", model="gpt-4o-mini")
orchestrator = DocumentOrchestrator(detector=YaraDetector(), worker_pool=pool)

result = orchestrator.analyze_path("document.pdf")
print(result.verdict)  # safe / suspicious / unsafe
```

## Environment config

```bash
export DOC_ANALYSE_LLM_PROVIDER=openai
export DOC_ANALYSE_LLM_MODEL=gpt-4o-mini
export OPENAI_API_KEY=sk-...
```

```python
from doc_analyse import (
    DocumentOrchestrator,
    build_classifier_worker_pool,
    YaraDetector,
    classifier_from_env,
)

pool = build_classifier_worker_pool(classifier_factory=classifier_from_env)
orchestrator = DocumentOrchestrator(detector=YaraDetector(), worker_pool=pool)
```

## Cheap detection without LLM

```python
from doc_analyse import YaraDetector, ingest_document

result = ingest_document("policy.pdf")
chunks = result.chunks
findings = YaraDetector().detect_many(chunks)

# findings is a tuple of DetectionFinding objects with:
#   - rule_id, category, severity, span, start_char, end_char
#   - requires_llm_validation flag
```

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev,llm]" --no-build-isolation
make format
make check
```