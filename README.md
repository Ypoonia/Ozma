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

## Architecture

```
Document
  → converter (PDF/text extraction)
  → chunker (text → chunks with byte→char offsets)
  → YARA on raw chunk (byte-accurate pattern matching on original text)
  → normalize → Prompt Guard on normalized chunk (ML classifier)
  → CheapRouter (fuses YARA + PG signals)
  → SAFE / REVIEW / HOLD
  → Layer 2 (LLM classifier) only for REVIEW/HOLD chunks
  → document verdict aggregation
```

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

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev,llm]" --no-build-isolation
make format
make check
```
