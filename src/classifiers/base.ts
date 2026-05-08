/**
 * Base classifier contract and data models.
 * Ported from doc_analyse/classifiers/base.py
 */

import { resolve_generation_config } from "./config.js";
import {
  load_default_system_prompt,
  load_default_classification_prompt,
  render_classification_prompt,
  PromptTemplateError,
} from "../prompt/loader.js";

export const VALID_VERDICTS = new Set(["safe", "suspicious", "unsafe"]);
export const VALID_SEVERITIES = new Set(["low", "medium", "high", "critical"]);

export interface ClassifierMessage {
  readonly role: string;
  readonly content: string;
}

export interface PromptInjectionFinding {
  readonly span: string;
  readonly attack_type: string;
  readonly severity: string;
  readonly reason: string;
  readonly start_char: number | null;
  readonly end_char: number | null;
}

export interface ClassificationResult {
  readonly verdict: "safe" | "suspicious" | "unsafe";
  readonly confidence: number;
  readonly reasons: readonly string[];
  readonly findings: readonly PromptInjectionFinding[];
  readonly raw_response: string | null;
}

export class ClassifierDependencyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ClassifierDependencyError";
  }
}

export class ClassifierResponseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ClassifierResponseError";
  }
}

export class ClassifierPromptError extends PromptTemplateError {}

// ---------------------------------------------------------------------------
// JSON extraction helpers
// ---------------------------------------------------------------------------

export function extract_json_object(rawResponse: string): string {
  let text = rawResponse.trim();

  const fencedMatch = text.match(
    /```(?:json)?\s*([\s\S]*?)```/i,
  );
  if (fencedMatch) {
    text = fencedMatch[1]!.trim();
  }

  if (text.startsWith("{") && text.endsWith("}")) {
    return text;
  }

  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) {
    return text;
  }

  return text.substring(start, end + 1);
}

function clamp_float(value: unknown): number {
  let parsed: number;
  try {
    parsed = Number(value);
  } catch {
    return 0.0;
  }
  return Math.min(1.0, Math.max(0.0, parsed));
}

function optional_int(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  try {
    return parseInt(String(value), 10);
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Frozen result factories
// ---------------------------------------------------------------------------

export function createPromptInjectionFinding(
  data: Record<string, unknown>,
): PromptInjectionFinding {
  let severity = String(data["severity"] ?? "medium").toLowerCase();
  if (!VALID_SEVERITIES.has(severity)) severity = "medium";

  return Object.freeze({
    span: String(data["span"] ?? ""),
    attack_type: String(data["attack_type"] ?? "other"),
    severity,
    reason: String(data["reason"] ?? ""),
    start_char: optional_int(data["start_char"]),
    end_char: optional_int(data["end_char"]),
  });
}

export function createClassificationResult(
  data: Record<string, unknown>,
  rawResponse: string | null = null,
): ClassificationResult {
  let verdict = String(data["verdict"] ?? "suspicious").toLowerCase();
  if (!VALID_VERDICTS.has(verdict)) verdict = "suspicious";

  const reasons = (data["reasons"] as unknown[])
    .filter((r) => typeof r === "string" && r.trim())
    .map((r) => String(r));

  const findingsData = Array.isArray(data["findings"])
    ? data["findings"] as Record<string, unknown>[]
    : [];

  const findings = findingsData
    .filter((item) => typeof item === "object" && item !== null)
    .map(createPromptInjectionFinding);

  return Object.freeze({
    verdict: verdict as "safe" | "suspicious" | "unsafe",
    confidence: clamp_float(data["confidence"]),
    reasons: Object.freeze(reasons),
    findings: Object.freeze(findings),
    raw_response: rawResponse,
  });
}

// ---------------------------------------------------------------------------
// Base classifier
// ---------------------------------------------------------------------------

export abstract class BaseClassifier {
  abstract readonly provider_name: string;
  abstract readonly default_model: string;

  protected readonly model: string;
  protected readonly temperature: number;
  protected readonly max_tokens: number;
  protected readonly system_prompt: string;
  protected readonly user_prompt_template: string;

  constructor({
    model,
    temperature,
    max_tokens,
    system_prompt,
    user_prompt_template,
  }: {
    model?: string | null;
    temperature?: number | null;
    max_tokens?: number | null;
    system_prompt?: string | null;
    user_prompt_template?: string | null;
  }) {
    const defaultModel = (this as unknown as { default_model: string }).default_model;
    const resolvedModel = model ?? defaultModel;
    if (!resolvedModel) {
      throw new Error("A model is required for this classifier.");
    }

    const generationConfig = resolve_generation_config({ temperature, max_tokens });
    this.model = resolvedModel;
    this.temperature = generationConfig.temperature;
    this.max_tokens = generationConfig.max_tokens;
    this.system_prompt = load_default_system_prompt(system_prompt ?? undefined);
    this.user_prompt_template = load_default_classification_prompt(
      user_prompt_template ?? undefined,
    );
  }

  get provider(): string {
    return this.provider_name;
  }

  classify(
    text: string,
    metadata: Record<string, unknown> | null = null,
  ): ClassificationResult {
    const messages = this.build_messages(text, metadata);
    const rawResponse = this._complete(messages);
    return this.parse_response(rawResponse);
  }

  build_messages(
    text: string,
    metadata: Record<string, unknown> | null = null,
  ): readonly ClassifierMessage[] {
    const userPrompt = render_classification_prompt(
      this.user_prompt_template,
      text,
      metadata ?? {},
    );

    return Object.freeze([
      Object.freeze({ role: "system", content: this.system_prompt }),
      Object.freeze({ role: "user", content: userPrompt }),
    ]);
  }

  parse_response(rawResponse: string): ClassificationResult {
    let data: Record<string, unknown>;
    try {
      const jsonText = extract_json_object(rawResponse);
      data = JSON.parse(jsonText);
    } catch (err) {
      throw new ClassifierResponseError(
        "Classifier returned invalid JSON.",
      );
    }

    if (typeof data !== "object" || data === null || Array.isArray(data)) {
      throw new ClassifierResponseError(
        "Classifier JSON response must be an object.",
      );
    }

    return createClassificationResult(data, rawResponse);
  }

  abstract _complete(messages: readonly ClassifierMessage[]): string;
}

// ---------------------------------------------------------------------------
// Helper utilities
// ---------------------------------------------------------------------------

export function render_messages_for_single_prompt(
  messages: readonly ClassifierMessage[],
): string {
  return messages
    .map((m) => `${m.role.toUpperCase()}:\n${m.content}`)
    .join("\n\n");
}

export function ensure_api_key(
  providerName: string,
  envNames: readonly string[],
  apiKey: string | null,
  options: Record<string, unknown>,
): void {
  if (apiKey) {
    options["api_key"] = apiKey;
    return;
  }

  if (options["api_key"]) return;

  if (envNames.some((name) => process.env[name])) return;

  const envHint = envNames.join(" or ");
  throw new ClassifierDependencyError(
    `Missing ${providerName} API key. Set ${envHint} or pass api_key=...`,
  );
}

export function require_text_response(
  providerName: string,
  text: unknown,
): string {
  if (typeof text !== "string" || !text.trim()) {
    throw new ClassifierResponseError(
      `${providerName} returned no text content.`,
    );
  }
  return text.trim();
}
