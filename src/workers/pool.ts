/**
 * Classifier worker pool with async concurrency.
 * Ported from doc_analyse/workers/pool.py
 */

import pLimit from "p-limit";
import { BaseClassifier, ClassificationResult } from "../classifiers/base.js";
import {
  build_classifier,
  classifier_from_env,
} from "../classifiers/factory.js";
import { TextChunk } from "../ingestion/models.js";
import { load_classifier_agent_prompt } from "../prompt/loader.js";

export const RETRY_DELAYS = [1000, 2000, 4000]; // ms
export const CHUNK_TIMEOUT_MS = 120_000; // 120 seconds
const MAX_CONCURRENT = 16;

export interface WorkerResult {
  readonly chunk: TextChunk;
  readonly classification: ClassificationResult;
}

export class WorkerPoolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "WorkerPoolError";
  }
}

// ---------------------------------------------------------------------------
// Stateless classifier worker
// ---------------------------------------------------------------------------

export class StatelessClassifierWorker {
  private readonly _classifier_factory: () => BaseClassifier;
  private _classifier_instance: BaseClassifier | null = null;

  constructor(classifierFactory: () => BaseClassifier) {
    this._classifier_factory = classifierFactory;
  }

  classify_chunk(chunk: TextChunk): WorkerResult {
    if (!chunk.text || !chunk.text.trim()) {
      throw new TypeError("Worker chunk text must be a non-empty string.");
    }

    const classification = this._classifier().classify(
      chunk.text,
      _chunk_classification_metadata(chunk),
    );
    return Object.freeze({
      chunk,
      classification,
    });
  }

  private _classifier(): BaseClassifier {
    if (this._classifier_instance !== null) {
      return this._classifier_instance;
    }

    this._classifier_instance = this._classifier_factory();
    return this._classifier_instance;
  }
}

// ---------------------------------------------------------------------------
// Non-retryable errors (mirror Python's _is_retryable)
// ---------------------------------------------------------------------------

function _is_retryable(exc: unknown): boolean {
  if (exc instanceof Error) {
    if (
      exc instanceof TypeError ||
      exc instanceof ReferenceError ||
      exc instanceof SyntaxError ||
      exc instanceof URIError
    ) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Retry logic
// ---------------------------------------------------------------------------

async function _sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function _classify_with_retry(
  worker: StatelessClassifierWorker,
  chunk: TextChunk,
): Promise<WorkerResult> {
  let lastError: unknown;

  try {
    return worker.classify_chunk(chunk);
  } catch (exc) {
    lastError = exc;
  }

  for (const delayMs of RETRY_DELAYS) {
    if (!_is_retryable(lastError)) throw lastError;
    await _sleep(delayMs);

    try {
      return worker.classify_chunk(chunk);
    } catch (exc) {
      lastError = exc;
    }
  }

  throw lastError;
}

// ---------------------------------------------------------------------------
// Classifier worker pool
// ---------------------------------------------------------------------------

export class ClassifierWorkerPool {
  private readonly worker: StatelessClassifierWorker;
  private readonly max_workers: number | undefined;
  private _closed = false;

  constructor(worker: StatelessClassifierWorker, maxWorkers?: number) {
    this.worker = worker;
    this.max_workers = maxWorkers;
  }

  close(): void {
    this._closed = true;
  }

  async classify_chunks(chunks: readonly TextChunk[]): Promise<readonly WorkerResult[]> {
    if (this._closed) throw new WorkerPoolError("Pool is closed.");
    if (chunks.length === 0) return [];

    const indexedChunks: Array<[number, TextChunk]> = chunks.map((c, i) => [i, c]);
    const resultsByIndex: Map<number, WorkerResult> = new Map();
    const limit = pLimit(this.max_workers ?? MAX_CONCURRENT);

    // Submit in batches for backpressure (same as Python)
    const batches: TextChunk[][] = [];
    for (let i = 0; i < indexedChunks.length; i += MAX_CONCURRENT) {
      batches.push(indexedChunks.slice(i, i + MAX_CONCURRENT).map(([, c]) => c));
    }

    const deadlineByChunk: Map<number, number> = new Map();
    let chunkIndex = 0;
    for (const batch of batches) {
      for (const chunk of batch) {
        deadlineByChunk.set(chunkIndex, Date.now() + CHUNK_TIMEOUT_MS);
        chunkIndex++;
      }
    }

    const tasks: Array<Promise<WorkerResult>> = [];

    for (const [index, chunk] of indexedChunks) {
      tasks.push(
        limit(async () => {
          const deadline = deadlineByChunk.get(index)!;
          const result = await _with_deadline(
            () => _classify_with_retry(this.worker, chunk),
            deadline,
            index,
          );
          return result;
        }),
      );
    }

    const settled = await Promise.allSettled(tasks);

    for (let i = 0; i < settled.length; i++) {
      const result = settled[i]!;
      if (result.status === "fulfilled") {
        resultsByIndex.set(i, result.value);
      } else {
        throw new WorkerPoolError(
          `Worker failed for chunk index ${i} after ${RETRY_DELAYS.length} retries: ${result.reason}`,
        );
      }
    }

    return indexedChunks.map(([i]) => resultsByIndex.get(i)!);
  }
}

async function _with_deadline(
  promiseFn: () => Promise<WorkerResult>,
  deadline: number,
  chunkIndex: number,
): Promise<WorkerResult> {
  const timeoutMs = Math.max(0, deadline - Date.now());

  const result = await Promise.race([
    promiseFn(),
    new Promise<WorkerResult>((_, reject) =>
      setTimeout(
        () => reject(new WorkerPoolError(`Chunk ${chunkIndex} timed out after ${CHUNK_TIMEOUT_MS}ms`)),
        timeoutMs,
      ),
    ),
  ]);

  return result;
}

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

export function build_stateless_classifier_factory(
  opts: {
    provider?: string | null;
    prefix?: string;
    system_prompt?: string | null;
    [key: string]: unknown;
  } = {},
): () => BaseClassifier {
  const {
    provider = null,
    prefix = "DOC_ANALYSE_LLM",
    system_prompt = null,
    ...classifier_kwargs
  } = opts;
  const workerSystemPrompt = load_classifier_agent_prompt(system_prompt ?? undefined);
  const baseKwargs: Record<string, unknown> = {
    ...classifier_kwargs,
    system_prompt: workerSystemPrompt,
  };

  if (provider !== null) {
    const normalizedProvider = String(provider).trim().toLowerCase();
    return () => build_classifier(normalizedProvider, baseKwargs);
  }

  return () => classifier_from_env(prefix, baseKwargs);
}

export function build_classifier_worker_pool(
  opts: {
    provider?: string | null;
    prefix?: string;
    system_prompt?: string | null;
    max_workers?: number | null;
    [key: string]: unknown;
  } = {},
): ClassifierWorkerPool {
  const { provider, prefix, system_prompt, max_workers, ...classifier_kwargs } = opts;

  const classifierFactory = build_stateless_classifier_factory({
    provider,
    prefix,
    system_prompt,
    ...classifier_kwargs,
  });

  const worker = new StatelessClassifierWorker(classifierFactory);
  return new ClassifierWorkerPool(worker, max_workers ?? undefined);
}

function _chunk_classification_metadata(
  chunk: TextChunk,
): Record<string, unknown> {
  const metadata: Record<string, unknown> = { ...chunk.metadata };
  metadata["source"] = chunk.source;
  metadata["start_char"] = chunk.start_char;
  metadata["end_char"] = chunk.end_char;
  return metadata;
}
