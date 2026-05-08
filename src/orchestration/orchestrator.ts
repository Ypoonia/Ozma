/**
 * Document-level orchestration for ingestion, cheap detection, and LLM validation.
 * Ported from doc_analyse/orchestration.py
 */

import { resolve } from "path";
import { BaseClassifier, ClassificationResult } from "../classifiers/base.js";
import { createClassificationResult } from "../classifiers/base.js";
import { YaraDetector } from "../detection/yara.js";
import { CheapRouter, DECISION_HOLD, DECISION_REVIEW, DECISION_SAFE } from "../detection/detect.js";
import { CheapChunkDecision } from "../detection/detect.js";
import { DetectionFinding, createDetectionFinding } from "../detection/models.js";
import { normalize_for_detection } from "../detection/normalize.js";
import { PromptGuardDetector } from "../detection/promptGuard.js";
import { IngestedDocument } from "../ingestion/models.js";
import { TextChunk } from "../ingestion/models.js";
import { TextChunker } from "../ingestion/chunking.js";
import { ingest_document } from "../ingestion/pipeline.js";
import { ConverterRegistry } from "../ingestion/converters.js";
import { ClassifierWorkerPool } from "../workers/pool.js";

export { DECISION_HOLD, DECISION_REVIEW, DECISION_SAFE };

export const VERDICT_SAFE = "safe";
export const VERDICT_SUSPICIOUS = "suspicious";
export const VERDICT_UNSAFE = "unsafe";

// ---------------------------------------------------------------------------
// Per-chunk Layer 1
// ---------------------------------------------------------------------------

export function run_layer1(
  chunk: TextChunk,
  yara: YaraDetector,
  pg: PromptGuardDetector | null,
  router: CheapRouter,
): [CheapChunkDecision, readonly DetectionFinding[]] {
  const yaraFindings = yara.detect(chunk);
  const normalized = normalize_for_detection(chunk.text);
  const pgScore = _pg_raw_score_sync(normalized, pg);
  const decision = router.route(yaraFindings, pgScore);
  return [decision, _build_layer1_findings(chunk, decision, yaraFindings, pgScore)];
}

export async function run_layer1_async(
  chunk: TextChunk,
  yara: YaraDetector,
  pg: PromptGuardDetector | null,
  router: CheapRouter,
): Promise<[CheapChunkDecision, readonly DetectionFinding[]]> {
  const yaraFindings = yara.detect(chunk);
  const normalized = normalize_for_detection(chunk.text);
  const pgScore = await _pg_raw_score_async(normalized, pg);
  const decision = router.route(yaraFindings, pgScore);
  const findings = _build_layer1_findings(chunk, decision, yaraFindings, pgScore);
  return [decision, findings];
}

function _pg_raw_score_sync(text: string, pg: PromptGuardDetector | null): number {
  if (!pg) return 0.0;
  try {
    return pg.raw_score_sync(text);
  } catch {
    return 0.0;
  }
}

async function _pg_raw_score_async(
  text: string,
  pg: PromptGuardDetector | null,
): Promise<number> {
  if (!pg) return 0.0;
  try {
    return await pg.raw_score(text);
  } catch {
    return 0.0;
  }
}

function _build_layer1_findings(
  chunk: TextChunk,
  decision: CheapChunkDecision,
  yaraFindings: readonly DetectionFinding[],
  pgScore: number,
): readonly DetectionFinding[] {
  const findings: DetectionFinding[] = [];
  const requiresLlm = decision.decision === DECISION_REVIEW || decision.decision === DECISION_HOLD;

  for (const e of decision.findings) {
    const normalizedScore = e.weight > 0 ? Math.min(1.0, e.weight / 100.0) : null;
    findings.push(
      createDetectionFinding({
        span: e.span,
        category: e.category,
        severity: e.severity,
        reason: `[YARA] ${e.rule_id} — ${e.category} (${e.severity})`,
        start_char: e.start_char,
        end_char: e.end_char,
        source: chunk.source,
        rule_id: e.rule_id,
        requires_llm_validation: requiresLlm,
        score: normalizedScore,
        metadata: Object.freeze({
          detector: "YaraDetector",
          yara_rule: e.rule_id,
          yara_weight: e.weight,
          route_hint: e.route_hint,
        }),
      }),
    );
  }

  if (pgScore > 0 && yaraFindings.length === 0 && (decision.decision === DECISION_REVIEW || decision.decision === DECISION_HOLD)) {
    findings.push(
      createDetectionFinding({
        span: "",
        category: "prompt_guard_signal",
        severity: "high",
        reason: `[PG] score=${pgScore.toFixed(3)} — ${pgScore >= 0.75 ? "strong" : "moderate"} signal`,
        start_char: chunk.start_char,
        end_char: chunk.start_char,
        source: chunk.source,
        rule_id: "prompt_guard",
        requires_llm_validation: requiresLlm,
        score: pgScore,
        metadata: Object.freeze({
          detector: "PromptGuardDetector",
          pg_malicious: pgScore,
        }),
      }),
    );
  }

  return Object.freeze(findings);
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

export interface ChunkAnalysisResult {
  readonly chunk_index: number;
  readonly chunk: TextChunk;
  readonly cheap_findings: readonly DetectionFinding[];
  readonly cheap_decision: CheapChunkDecision;
  readonly routed_to_llm: boolean;
  readonly llm_classification: ClassificationResult | null;
  readonly final_verdict: string;
}

export interface DocumentAnalysisResult {
  readonly ingested_document: IngestedDocument;
  readonly chunk_results: readonly ChunkAnalysisResult[];
  readonly verdict: string;
  readonly reasons: readonly string[];
  chunk_result(chunk_index: number): ChunkAnalysisResult;
  chunk_text(chunk_index: number): string;
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

export class DocumentOrchestrator {
  private readonly yara: YaraDetector;
  private readonly pg: PromptGuardDetector | null;
  private readonly router: CheapRouter;
  private readonly worker_pool: ClassifierWorkerPool;
  private _closed = false;

  constructor({
    yara,
    pg,
    router,
    workerPool,
  }: {
    yara: YaraDetector;
    pg: PromptGuardDetector | null;
    router: CheapRouter;
    workerPool: ClassifierWorkerPool;
  }) {
    this.yara = yara;
    this.pg = pg;
    this.router = router;
    this.worker_pool = workerPool;
  }

  close(): void {
    if (this._closed) return;
    this.worker_pool.close();
    this._closed = true;
  }

  analyze_path(
    path: string,
    {
      registry,
      chunker,
    }: {
      registry?: ConverterRegistry;
      chunker?: TextChunker;
    } = {},
  ): DocumentAnalysisResult {
    const resolvedPath = resolve(path);
    const ingested = ingest_document(resolvedPath, { registry, chunker });
    return this.analyze_ingested(ingested);
  }

  analyze_ingested(ingested: IngestedDocument): DocumentAnalysisResult {
    const chunks = ingested.chunks;

    // Layer 1 — cheap detection on every chunk
    const chunkDecisions: CheapChunkDecision[] = [];
    const chunkFindings: Array<readonly DetectionFinding[]> = [];

    for (const chunk of chunks) {
      const [decision, findings] = run_layer1(chunk, this.yara, this.pg, this.router);
      chunkDecisions.push(decision);
      chunkFindings.push(findings);
    }

    // Route — only REVIEW/HOLD go to Layer 2
    const routedIndices: number[] = [];
    for (let i = 0; i < chunkDecisions.length; i++) {
      const d = chunkDecisions[i]!;
      if (d.decision === DECISION_REVIEW || d.decision === DECISION_HOLD) {
        routedIndices.push(i);
      }
    }
    const routedIndicesSet = new Set(routedIndices);

    // Layer 2 — LLM validation
    const llmResults: Map<number, ClassificationResult> = new Map();
    if (routedIndices.length > 0) {
      const routedChunks = routedIndices.map((i) => chunks[i]!);
      const maybeWorkerResults = this.worker_pool.classify_chunks(
        routedChunks,
      ) as unknown;
      if (maybeWorkerResults instanceof Promise) {
        throw new Error(
          "Synchronous LLM classification requires a synchronous worker pool. Use DocumentOrchestratorAsync for async pools.",
        );
      }
      const workerResults = maybeWorkerResults as readonly {
        readonly classification: ClassificationResult;
      }[];
      for (let i = 0; i < routedIndices.length; i++) {
        const chunkIndex = routedIndices[i]!;
        const workerResult = workerResults[i]!;
        llmResults.set(chunkIndex, workerResult.classification);
      }
    }

    // Build per-chunk results
    const chunkResults: ChunkAnalysisResult[] = [];
    for (let idx = 0; idx < chunks.length; idx++) {
      const chunk = chunks[idx]!;
      const decision = chunkDecisions[idx]!;
      const findings = chunkFindings[idx]!;
      const llm = llmResults.get(idx) ?? null;

      let finalVerdict: string;
      if (llm !== null) {
        finalVerdict = _normalize_verdict(llm.verdict);
      } else if (decision.decision === DECISION_HOLD || decision.decision === DECISION_REVIEW) {
        finalVerdict = VERDICT_SUSPICIOUS;
      } else {
        finalVerdict = VERDICT_SAFE;
      }

      chunkResults.push(
        Object.freeze({
          chunk_index: idx,
          chunk,
          cheap_findings: findings,
          cheap_decision: decision,
          routed_to_llm: routedIndicesSet.has(idx),
          llm_classification: llm,
          final_verdict: finalVerdict,
        }),
      );
    }

    const [verdict, reasons] = _aggregate_document_verdict(chunkResults as unknown as readonly ChunkAnalysisResult[]);
    const frozenResults = Object.freeze(chunkResults);

    return Object.freeze({
      ingested_document: ingested,
      chunk_results: frozenResults,
      verdict,
      reasons: Object.freeze(reasons),
      chunk_result(idx: number) {
        return frozenResults[idx]!;
      },
      chunk_text(idx: number) {
        const c = frozenResults[idx]!.chunk;
        return ingested.text.substring(c.start_char, c.end_char);
      },
    });
  }
}

// ---------------------------------------------------------------------------
// Async orchestrator variant
// ---------------------------------------------------------------------------

export class DocumentOrchestratorAsync {
  private readonly yara: YaraDetector;
  private readonly pg: PromptGuardDetector | null;
  private readonly router: CheapRouter;
  private readonly worker_pool: ClassifierWorkerPool;
  private _closed = false;

  constructor({
    yara,
    pg,
    router,
    workerPool,
  }: {
    yara: YaraDetector;
    pg: PromptGuardDetector | null;
    router: CheapRouter;
    workerPool: ClassifierWorkerPool;
  }) {
    this.yara = yara;
    this.pg = pg;
    this.router = router;
    this.worker_pool = workerPool;
  }

  close(): void {
    if (this._closed) return;
    this.worker_pool.close();
    this._closed = true;
  }

  async analyze_path(
    path: string,
    {
      registry,
      chunker,
    }: {
      registry?: ConverterRegistry;
      chunker?: TextChunker;
    } = {},
  ): Promise<DocumentAnalysisResult> {
    const resolvedPath = resolve(path);
    const ingested = ingest_document(resolvedPath, { registry, chunker });
    return this.analyze_ingested(ingested);
  }

  async analyze_ingested(ingested: IngestedDocument): Promise<DocumentAnalysisResult> {
    const chunks = ingested.chunks;

    // Layer 1 — cheap detection on every chunk
    const chunkDecisions: CheapChunkDecision[] = [];
    const chunkFindings: Array<readonly DetectionFinding[]> = [];

    for (const chunk of chunks) {
      const [decision, findings] = await run_layer1_async(
        chunk,
        this.yara,
        this.pg,
        this.router,
      );
      chunkDecisions.push(decision);
      chunkFindings.push(findings);
    }

    // Route — only REVIEW/HOLD go to Layer 2
    const routedIndices: number[] = [];
    for (let i = 0; i < chunkDecisions.length; i++) {
      const d = chunkDecisions[i]!;
      if (d.decision === DECISION_REVIEW || d.decision === DECISION_HOLD) {
        routedIndices.push(i);
      }
    }
    const routedIndicesSet = new Set(routedIndices);

    // Layer 2 — LLM validation
    const llmResults: Map<number, ClassificationResult> = new Map();
    if (routedIndices.length > 0) {
      const routedChunks = routedIndices.map((i) => chunks[i]!);
      const workerResults = await this.worker_pool.classify_chunks(routedChunks);
      for (let i = 0; i < routedIndices.length; i++) {
        const chunkIndex = routedIndices[i]!;
        const workerResult = workerResults[i]!;
        llmResults.set(chunkIndex, workerResult.classification);
      }
    }

    // Build per-chunk results
    const chunkResults: ChunkAnalysisResult[] = [];
    for (let idx = 0; idx < chunks.length; idx++) {
      const chunk = chunks[idx]!;
      const decision = chunkDecisions[idx]!;
      const findings = chunkFindings[idx]!;
      const llm = llmResults.get(idx) ?? null;

      let finalVerdict: string;
      if (llm !== null) {
        finalVerdict = _normalize_verdict(llm.verdict);
      } else if (decision.decision === DECISION_HOLD || decision.decision === DECISION_REVIEW) {
        finalVerdict = VERDICT_SUSPICIOUS;
      } else {
        finalVerdict = VERDICT_SAFE;
      }

      chunkResults.push(
        Object.freeze({
          chunk_index: idx,
          chunk,
          cheap_findings: findings,
          cheap_decision: decision,
          routed_to_llm: routedIndicesSet.has(idx),
          llm_classification: llm,
          final_verdict: finalVerdict,
        }),
      );
    }

    const [verdict, reasons] = _aggregate_document_verdict(chunkResults as unknown as readonly ChunkAnalysisResult[]);
    const frozenResults = Object.freeze(chunkResults);

    return Object.freeze({
      ingested_document: ingested,
      chunk_results: frozenResults,
      verdict,
      reasons: Object.freeze(reasons),
      chunk_result(idx: number) {
        return frozenResults[idx]!;
      },
      chunk_text(idx: number) {
        const c = frozenResults[idx]!.chunk;
        return ingested.text.substring(c.start_char, c.end_char);
      },
    });
  }
}

// ---------------------------------------------------------------------------
// Aggregation helpers
// ---------------------------------------------------------------------------

function _aggregate_document_verdict(
  chunkResults: readonly ChunkAnalysisResult[],
): [string, readonly string[]] {
  const unsafeIndices = chunkResults
    .filter((r) => r.final_verdict === VERDICT_UNSAFE)
    .map((r) => r.chunk_index);

  const suspiciousIndices = chunkResults
    .filter((r) => r.final_verdict === VERDICT_SUSPICIOUS)
    .map((r) => r.chunk_index);

  if (unsafeIndices.length > 0) {
    return [VERDICT_UNSAFE, Object.freeze([`Unsafe chunk indices: ${unsafeIndices.join(", ")}`])];
  }
  if (suspiciousIndices.length > 0) {
    return [VERDICT_SUSPICIOUS, Object.freeze([`Suspicious chunk indices: ${suspiciousIndices.join(", ")}`])];
  }
  return [VERDICT_SAFE, Object.freeze(["No suspicious or unsafe chunks detected."])];
}

function _normalize_verdict(rawVerdict: string): string {
  const verdict = rawVerdict.trim().toLowerCase();
  if (verdict === VERDICT_SAFE || verdict === VERDICT_SUSPICIOUS || verdict === VERDICT_UNSAFE) {
    return verdict;
  }
  return VERDICT_SUSPICIOUS;
}

// ---------------------------------------------------------------------------
// Convenience builders
// ---------------------------------------------------------------------------

export function build_orchestrator({
  yara,
  pg,
  router,
  workerPool,
}: {
  yara?: YaraDetector;
  pg?: PromptGuardDetector | null;
  router?: CheapRouter;
  workerPool: ClassifierWorkerPool;
}): DocumentOrchestrator {
  return new DocumentOrchestrator({
    yara: yara ?? new YaraDetector(),
    pg: pg ?? null,
    router: router ?? new CheapRouter(),
    workerPool,
  });
}

export function build_orchestrator_async({
  yara,
  pg,
  router,
  workerPool,
}: {
  yara?: YaraDetector;
  pg?: PromptGuardDetector | null;
  router?: CheapRouter;
  workerPool: ClassifierWorkerPool;
}): DocumentOrchestratorAsync {
  return new DocumentOrchestratorAsync({
    yara: yara ?? new YaraDetector(),
    pg: pg ?? null,
    router: router ?? new CheapRouter(),
    workerPool,
  });
}

export async function analyze_document_path(
  path: string,
  opts?: {
    yara?: YaraDetector;
    pg?: PromptGuardDetector | null;
    router?: CheapRouter;
    workerPool?: ClassifierWorkerPool;
    registry?: ConverterRegistry;
    chunker?: TextChunker;
    close_worker_pool?: boolean;
  },
): Promise<DocumentAnalysisResult> {
  const { yara, pg, router, workerPool, registry, chunker, close_worker_pool = false } = opts ?? {};
  if (!workerPool) {
    throw new Error("analyze_document_path requires a workerPool argument.");
  }
  const orchestrator = build_orchestrator_async({
    yara: yara ?? new YaraDetector(),
    pg: pg ?? null,
    router: router ?? new CheapRouter(),
    workerPool,
  });

  try {
    return await orchestrator.analyze_path(path, { registry, chunker });
  } finally {
    if (close_worker_pool) {
      orchestrator.close();
    }
  }
}
