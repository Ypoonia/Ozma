/**
 * Tests for orchestration.
 * Ported from tests/test_orchestration.py
 */

import { describe, it, expect, vi } from "vitest";
import { writeFileSync, mkdirSync } from "fs";
import { join } from "path";
import {
  DocumentOrchestrator,
  DocumentOrchestratorAsync,
  analyze_document_path,
  VERDICT_SAFE,
  VERDICT_UNSAFE,
  VERDICT_SUSPICIOUS,
  DECISION_HOLD,
  DECISION_REVIEW,
  DECISION_SAFE,
} from "../src/orchestration/orchestrator.js";
import { createClassificationResult } from "../src/classifiers/base.js";
import { createDetectionFinding } from "../src/detection/models.js";
import { createTextChunk } from "../src/ingestion/models.js";
import { TextChunker } from "../src/ingestion/chunking.js";
import { YaraDetector } from "../src/detection/yara.js";
import { CheapRouter } from "../src/detection/detect.js";
import { WorkerResult } from "../src/workers/pool.js";

// ---------------------------------------------------------------------------
// Fakes
// ---------------------------------------------------------------------------

class FakeWorkerPool {
  calls: readonly (readonly createTextChunk[])[] = [];
  closed = false;

  classify_chunks(chunks: readonly createTextChunk[]): readonly WorkerResult[] {
    const chunkList = [...chunks] as readonly createTextChunk[];
    this.calls = [...this.calls, chunkList];

    return chunkList.map((chunk) => ({
      chunk,
      classification: createClassificationResult({
        verdict: chunk.text.includes("RISKY-INSTRUCTION") ? "unsafe" : "safe",
        confidence: 0.9,
        reasons: ["worker"],
        findings: [],
      }),
    }));
  }

  close(): void {
    this.closed = true;
  }
}

class FakeYara {
  constructor(private findings: readonly createDetectionFinding[] = []) {}

  detect(chunk: createTextChunk): readonly createDetectionFinding[] {
    if (!chunk.text.includes("RISKY-INSTRUCTION")) return [];

    const idx = chunk.text.indexOf("RISKY-INSTRUCTION");
    return [
      createDetectionFinding({
        span: "RISKY-INSTRUCTION",
        category: "instruction_override",
        severity: "high",
        reason: "Risk marker for test.",
        rule_id: "risk_marker",
        start_char: chunk.start_char + idx,
        end_char: chunk.start_char + idx + 17,
        source: chunk.source,
        requires_llm_validation: true,
      }),
    ];
  }
}

class FakeRouter {
  constructor(private holdOnRisk = true) {}

  route(yaraFindings: readonly createDetectionFinding[], pgScore: number) {
    if (yaraFindings.length > 0 && this.holdOnRisk) {
      return Object.freeze({
        decision: DECISION_HOLD,
        risk_score: 50.0,
        pg_score: 0.0,
        yara_score: 40.0,
        findings: [],
        reason: "RISKY-INSTRUCTION found.",
      });
    } else if (yaraFindings.length > 0 && !this.holdOnRisk) {
      return Object.freeze({
        decision: DECISION_REVIEW,
        risk_score: 25.0,
        pg_score: 0.0,
        yara_score: 20.0,
        findings: [],
        reason: "RISKY-INSTRUCTION found.",
      });
    }
    return Object.freeze({
      decision: DECISION_SAFE,
      risk_score: 0.0,
      pg_score: 0.0,
      yara_score: 0.0,
      findings: [],
      reason: "No risk signals.",
    });
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("test_orchestrator_end_to_end_with_index_traceability", () => {
  it("full pipeline with RISKY-INSTRUCTION routes to LLM and returns unsafe", () => {
    // Create temp file via Node.js
    const tmpDir = "/tmp/ozma-test-" + Date.now();
    mkdirSync(tmpDir, { recursive: true });
    const path = join(tmpDir, "doc.txt");
    writeFileSync(
      path,
      "Normal section.\nRISKY-INSTRUCTION appears in this chunk and should route to llm.\nNormal ending.",
      "utf-8",
    );

    const chunker = new TextChunker({ chunkSize: 60, chunkOverlap: 0 });
    const pool = new FakeWorkerPool();
    const orchestrator = new DocumentOrchestrator({
      yara: new FakeYara() as unknown as YaraDetector,
      pg: null,
      router: new FakeRouter() as unknown as CheapRouter,
      workerPool: pool as unknown as import("../src/workers/pool.js").ClassifierWorkerPool,
    });

    const result = orchestrator.analyze_path(path, { chunker });

    expect(result.verdict).toBe(VERDICT_UNSAFE);
    expect(pool.calls.length).toBe(1);
    const routedChunks = pool.calls[0]!;
    expect(routedChunks.length).toBe(1);
    const routedIndex = routedChunks[0]!.metadata["chunk_index"];
    expect(routedIndex).toBeDefined();
  });
});

describe("test_orchestrator_skips_worker_pool_when_no_cheap_findings", () => {
  it("safe document skips worker pool", () => {
    const tmpDir = "/tmp/ozma-test-" + Date.now();
    mkdirSync(tmpDir, { recursive: true });
    const path = join(tmpDir, "safe.txt");
    writeFileSync(path, "No risk markers in this document.", "utf-8");

    const pool = new FakeWorkerPool();
    const orchestrator = new DocumentOrchestrator({
      yara: new FakeYara() as unknown as YaraDetector,
      pg: null,
      router: new FakeRouter() as unknown as CheapRouter,
      workerPool: pool as unknown as import("../src/workers/pool.js").ClassifierWorkerPool,
    });

    const result = orchestrator.analyze_path(path);

    expect(result.verdict).toBe(VERDICT_SAFE);
    expect(pool.calls.length).toBe(0);
  });
});

describe("test_orchestrator_routes_review_chunks_to_layer2", () => {
  it("REVIEW chunks still route to Layer 2", () => {
    const tmpDir = "/tmp/ozma-test-" + Date.now();
    mkdirSync(tmpDir, { recursive: true });
    const path = join(tmpDir, "doc.txt");
    writeFileSync(path, "RISKY-INSTRUCTION", "utf-8");

    const pool = new FakeWorkerPool();
    const orchestrator = new DocumentOrchestrator({
      yara: new FakeYara() as unknown as YaraDetector,
      pg: null,
      router: new FakeRouter(false) as unknown as CheapRouter,
      workerPool: pool as unknown as import("../src/workers/pool.js").ClassifierWorkerPool,
    });

    const result = orchestrator.analyze_path(path);

    expect(pool.calls.length).toBe(1);
  });
});

describe("test_orchestrator_context_manager_closes_worker_pool", () => {
  it("context manager closes pool on exit", () => {
    const tmpDir = "/tmp/ozma-test-" + Date.now();
    mkdirSync(tmpDir, { recursive: true });
    const path = join(tmpDir, "safe.txt");
    writeFileSync(path, "safe", "utf-8");

    const pool = new FakeWorkerPool();
    const orchestrator = new DocumentOrchestrator({
      yara: new FakeYara() as unknown as YaraDetector,
      pg: null,
      router: new FakeRouter() as unknown as CheapRouter,
      workerPool: pool as unknown as import("../src/workers/pool.js").ClassifierWorkerPool,
    });

    orchestrator.close();

    expect(pool.closed).toBe(true);
  });
});

describe("test_orchestrator_normalizes_unknown_llm_verdict_to_suspicious", () => {
  it("unknown verdict normalizes to suspicious", () => {
    const tmpDir = "/tmp/ozma-test-" + Date.now();
    mkdirSync(tmpDir, { recursive: true });
    const path = join(tmpDir, "doc.txt");
    writeFileSync(path, "RISKY-INSTRUCTION", "utf-8");

    class UnknownVerdictPool extends FakeWorkerPool {
      classify_chunks(
        chunks: readonly createTextChunk[],
      ): readonly WorkerResult[] {
        const chunk = chunks[0]!;
        return [
          {
            chunk,
            classification: createClassificationResult({
              verdict: "typo-unsafe",
              confidence: 0.6,
              reasons: ["unknown"],
              findings: [],
            }),
          },
        ];
      }
    }

    const pool = new UnknownVerdictPool();
    const orchestrator = new DocumentOrchestrator({
      yara: new FakeYara() as unknown as YaraDetector,
      pg: null,
      router: new FakeRouter() as unknown as CheapRouter,
      workerPool: pool as unknown as import("../src/workers/pool.js").ClassifierWorkerPool,
    });

    const result = orchestrator.analyze_path(path);

    expect(result.verdict).toBe(VERDICT_SUSPICIOUS);
  });
});
