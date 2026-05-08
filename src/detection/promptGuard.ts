/**
 * Prompt Guard detector using a pluggable scorer.
 * Ported from doc_analyse/detection/prompt_guard.py
 */

import { BaseDetector } from "./base.js";
import { DetectionFinding, createDetectionFinding } from "./models.js";
import { TextChunk } from "../ingestion/models.js";
import { normalize_for_detection } from "./normalize.js";

export const DEFAULT_PROMPT_GUARD_MODEL =
  "meta-llama/Llama-Prompt-Guard-2-86M";
export const DEFAULT_MALICIOUS_THRESHOLD = 0.8;
export const DEFAULT_UNCERTAIN_THRESHOLD = 0.5;

export class PromptGuardDependencyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PromptGuardDependencyError";
  }
}

type ScoreRow = Readonly<Record<string, unknown>>;
export type PromptGuardScoreOutput =
  | { malicious?: number; benign?: number }
  | readonly ScoreRow[]
  | readonly (readonly ScoreRow[])[]
  | ScoreRow;

export interface PromptGuardClient {
  score(text: string): PromptGuardScoreOutput | Promise<PromptGuardScoreOutput>;
}

export class HuggingFacePromptGuardClient implements PromptGuardClient {
  private readonly apiKey: string;
  private readonly modelUrl: string;

  constructor(apiKey: string) {
    this.apiKey = apiKey;
    this.modelUrl =
      "https://api-inference.huggingface.co/models/meta-llama/Llama-Prompt-Guard-2-86M";
  }

  async score(text: string): Promise<{ malicious: number; benign: number }> {
    const response = await fetch(this.modelUrl, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ inputs: text }),
    });

    if (!response.ok) {
      throw new Error(
        `Prompt Guard API error: ${response.status} ${response.statusText}`,
      );
    }

    const data = (await response.json()) as number[][];
    const probs = data[0];
    if (!probs || probs[0] === undefined || probs[1] === undefined) {
      return { malicious: 0, benign: 1 };
    }

    return { malicious: probs[0], benign: probs[1] };
  }
}

const INJECTION_KEYWORDS = [
  /\b(ignore|disregard|forget)\s+(all\s+)?previous\s+instructions?\b/i,
  /\b(you\s+are\s+now|act\s+as|pretend\s+to\s+be)\b/i,
  /\b(safe|benign|harmless)\s+(to\s+)?(forward|process|execute)\b/i,
  /\{[\s\S]*system\s*[\s\S]*instruction[\s\S]*\}/i,
  /\<\|.*?\|>?\s*system\s*\</i,
];

export class HeuristicPromptGuardClient implements PromptGuardClient {
  score(text: string): { malicious: number; benign: number } {
    const normalized = normalize_for_detection(text);
    let score = 0;

    for (const regex of INJECTION_KEYWORDS) {
      if (regex.test(normalized)) {
        score += 0.15;
      }
    }

    return {
      malicious: Math.min(1.0, score),
      benign: Math.max(0.0, 1.0 - score),
    };
  }
}

export class PromptGuardDetector extends BaseDetector {
  private readonly client: PromptGuardClient | null;
  readonly malicious_threshold: number;
  readonly uncertain_threshold: number;
  private _available = true;

  constructor({
    client = null,
    maliciousThreshold = DEFAULT_MALICIOUS_THRESHOLD,
    uncertainThreshold = DEFAULT_UNCERTAIN_THRESHOLD,
  }: {
    client?: PromptGuardClient | null;
    maliciousThreshold?: number;
    uncertainThreshold?: number;
  } = {}) {
    super();

    if (
      uncertainThreshold < 0 ||
      maliciousThreshold < uncertainThreshold ||
      maliciousThreshold > 1
    ) {
      throw new Error(
        "thresholds must satisfy 0 <= uncertain_threshold <= malicious_threshold <= 1.",
      );
    }

    this.client = client;
    this.malicious_threshold = maliciousThreshold;
    this.uncertain_threshold = uncertainThreshold;
  }

  detect(chunk: TextChunk): readonly DetectionFinding[] {
    if (!chunk.text || !chunk.text.trim()) {
      return [];
    }

    const scores = this._score_sync(chunk.text);
    return this._findings_from_scores(chunk, scores.malicious, scores.benign);
  }

  async detect_async(chunk: TextChunk): Promise<readonly DetectionFinding[]> {
    if (!chunk.text || !chunk.text.trim()) {
      return [];
    }

    const scores = await this._score_async(chunk.text);
    return this._findings_from_scores(chunk, scores.malicious, scores.benign);
  }

  raw_score_sync(text: string): number {
    return this._score_sync(text).malicious;
  }

  async raw_score(text: string): Promise<number> {
    const scores = await this._score_async(text);
    return scores.malicious;
  }

  private _score_sync(text: string): { malicious: number; benign: number } {
    if (!this.client) {
      if (!this._available) {
        return { malicious: 0, benign: 1 };
      }
      throw new PromptGuardDependencyError(
        "Prompt Guard client not configured. Provide a client or install with API credentials.",
      );
    }

    try {
      const output = this.client.score(text);
      if (_is_promise_like(output)) {
        throw new PromptGuardDependencyError(
          "Prompt Guard client is asynchronous. Use detect_async() or DocumentOrchestratorAsync.",
        );
      }
      return _coerce_scores(output);
    } catch (err) {
      this._available = false;
      if (err instanceof PromptGuardDependencyError) {
        throw err;
      }
      throw new PromptGuardDependencyError(`Prompt Guard scoring failed: ${err}`);
    }
  }

  private async _score_async(
    text: string,
  ): Promise<{ malicious: number; benign: number }> {
    if (!this.client) {
      if (!this._available) {
        return { malicious: 0, benign: 1 };
      }
      throw new PromptGuardDependencyError(
        "Prompt Guard client not configured. Provide a client or install with API credentials.",
      );
    }

    try {
      const output = await this.client.score(text);
      return _coerce_scores(output);
    } catch (err) {
      this._available = false;
      throw new PromptGuardDependencyError(`Prompt Guard scoring failed: ${err}`);
    }
  }

  private _findings_from_scores(
    chunk: TextChunk,
    malicious: number,
    benign: number,
  ): readonly DetectionFinding[] {
    const metadata = Object.freeze({
      detector: "PromptGuardDetector",
      pg_malicious: malicious,
      pg_benign: benign,
      model: DEFAULT_PROMPT_GUARD_MODEL,
    });

    if (malicious >= this.malicious_threshold) {
      return Object.freeze([
        createDetectionFinding({
          span: chunk.text,
          category: "prompt_guard_malicious",
          severity: "high",
          reason: "Prompt Guard classified this chunk as malicious.",
          rule_id: "prompt_guard",
          start_char: chunk.start_char,
          end_char: chunk.end_char,
          source: chunk.source,
          requires_llm_validation: true,
          score: malicious,
          metadata,
        }),
      ]);
    }

    if (malicious >= this.uncertain_threshold) {
      return Object.freeze([
        createDetectionFinding({
          span: chunk.text,
          category: "prompt_guard_uncertain",
          severity: "medium",
          reason:
            "Prompt Guard score is uncertain enough to require LLM validation.",
          rule_id: "prompt_guard",
          start_char: chunk.start_char,
          end_char: chunk.end_char,
          source: chunk.source,
          requires_llm_validation: true,
          score: malicious,
          metadata,
        }),
      ]);
    }

    return [];
  }
}

function _is_promise_like(value: unknown): value is Promise<unknown> {
  return Boolean(
    value &&
      typeof value === "object" &&
      "then" in (value as Record<string, unknown>) &&
      typeof (value as { then?: unknown }).then === "function",
  );
}

function _coerce_scores(
  rawOutput: PromptGuardScoreOutput,
): { malicious: number; benign: number } {
  if (_is_direct_score_object(rawOutput)) {
    return {
      malicious: _clamp_score(rawOutput.malicious ?? 0),
      benign: _clamp_score(rawOutput.benign ?? 0),
    };
  }

  const normalized = _normalise_scores(rawOutput);
  return {
    malicious: normalized.malicious ?? 0,
    benign: normalized.benign ?? 0,
  };
}

function _is_direct_score_object(
  rawOutput: PromptGuardScoreOutput,
): rawOutput is { malicious?: number; benign?: number } {
  return Boolean(
    rawOutput &&
      typeof rawOutput === "object" &&
      !Array.isArray(rawOutput) &&
      ("malicious" in rawOutput || "benign" in rawOutput),
  );
}

function _clamp_score(value: unknown): number {
  const parsed = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(parsed)) {
    return 0;
  }
  return Math.max(0, Math.min(1, parsed));
}

export function _normalise_scores(
  rawOutput: PromptGuardScoreOutput,
): Record<string, number> {
  const rows = _flatten_pipeline_output(rawOutput);
  const scores: Record<string, number> = {};

  for (const row of rows) {
    const label = String(row.label ?? "").trim().toLowerCase();
    const score = row.score;
    if (typeof score !== "number") {
      continue;
    }

    if (label === "malicious" || label === "jailbreak" || label === "injection") {
      scores.malicious = Math.max(scores.malicious ?? 0, score);
    } else if (label === "benign" || label === "safe") {
      scores.benign = Math.max(scores.benign ?? 0, score);
    }
  }

  return scores;
}

function _flatten_pipeline_output(
  rawOutput: PromptGuardScoreOutput,
): readonly ScoreRow[] {
  if (Array.isArray(rawOutput)) {
    if (rawOutput.length > 0 && Array.isArray(rawOutput[0])) {
      return (rawOutput as readonly (readonly ScoreRow[])[]).flat();
    }
    return rawOutput as readonly ScoreRow[];
  }

  if (rawOutput && typeof rawOutput === "object" && !_is_direct_score_object(rawOutput)) {
    return [rawOutput as ScoreRow];
  }

  return [];
}
