# Ozma

Detect prompt-injection attacks embedded in documents before they reach an LLM.

## Go usage

```go
package main

import "github.com/Ypoonia/Ozma"

func main() {
    ingested, err := ozma.IngestDocument("document.txt", nil, nil, ozma.DefaultChunkSize, ozma.DefaultChunkOverlap)
    if err != nil {
        panic(err)
    }

    yara, err := ozma.NewYaraDetector()
    if err != nil {
        panic(err)
    }

    decision, findings, err := ozma.RunLayer1(ingested.Chunks[0], yara, nil, ozma.NewCheapRouter())
    _, _ = decision, findings
    if err != nil {
        panic(err)
    }
}
```

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
