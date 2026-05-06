# doc-analyse

Detect prompt-injection attacks embedded in documents before they reach an LLM.

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
