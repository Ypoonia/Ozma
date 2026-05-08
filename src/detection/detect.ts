/**
 * Layer 1 cheap detection routing.
 * Ported from doc_analyse/detection/detect.py
 */

import { DetectionFinding } from "./models.js";

export const DECISION_SAFE = "safe";
export const DECISION_REVIEW = "review";
export const DECISION_HOLD = "hold";
const CHUNK_DECISIONS = new Set([DECISION_SAFE, DECISION_REVIEW, DECISION_HOLD]);

const SEVERITY_WEIGHTS: Readonly<Record<string, number>> = {
  critical: 40.0,
  high: 25.0,
  medium: 10.0,
  low: 5.0,
};

type Decision = typeof DECISION_SAFE | typeof DECISION_REVIEW | typeof DECISION_HOLD;

// Category-combination routing rules
const CATEGORY_COMBINATION_RULES: readonly (
  readonly [ReadonlySet<string>, number | null, Decision]
)[] = [
  [new Set(["tool_hijack", "instruction_override"]), null, DECISION_HOLD],
  [new Set(["secret_exfiltration", "instruction_override"]), null, DECISION_HOLD],
  [new Set(["secret_exfiltration", "tool_hijack"]), null, DECISION_HOLD],
  [new Set(["topic_mention"]), 0.1, DECISION_SAFE],
];

export interface YaraEvidence {
  readonly rule_id: string;
  readonly category: string;
  readonly severity: string;
  readonly span: string;
  readonly start_char: number;
  readonly end_char: number;
  readonly weight: number;
  readonly route_hint: string;
}

export interface CheapChunkDecision {
  readonly decision: Decision;
  readonly risk_score: number;
  readonly pg_score: number;
  readonly yara_score: number;
  readonly findings: readonly YaraEvidence[];
  readonly reason: string;
}

export function createYaraEvidence(props: {
  rule_id: string;
  category: string;
  severity: string;
  span: string;
  start_char: number;
  end_char: number;
  weight?: number;
  route_hint?: string;
}): YaraEvidence {
  return Object.freeze({
    rule_id: props.rule_id,
    category: props.category,
    severity: props.severity,
    span: props.span,
    start_char: props.start_char,
    end_char: props.end_char,
    weight: props.weight ?? 0.0,
    route_hint: props.route_hint ?? "evidence",
  });
}

export function yaraEvidenceFromFinding(f: DetectionFinding): YaraEvidence {
  const metadata = f.metadata ?? {};
  const rawWeight = typeof metadata["yara_weight"] === "number"
    ? (metadata["yara_weight"] as number)
    : 0.0;
  const routeHint = typeof metadata["route_hint"] === "string"
    ? (metadata["route_hint"] as string)
    : "evidence";

  return createYaraEvidence({
    rule_id: f.rule_id,
    category: f.category,
    severity: f.severity,
    span: f.span,
    start_char: f.start_char,
    end_char: f.end_char,
    weight: rawWeight,
    route_hint: routeHint,
  });
}

function _check_category_combination_rules(
  evidence: readonly YaraEvidence[],
  pgScore: number,
): Decision | null {
  const categories = new Set(evidence.map((e) => e.category));

  for (const [requiredCategories, pgGate, decision] of CATEGORY_COMBINATION_RULES) {
    const isSubset = [...requiredCategories].every((c) => categories.has(c));
    if (!isSubset) continue;

    if (requiredCategories.size === categories.size) {
      // Exact match: single-category "alone" rules
      if (pgGate === null) return decision;
      if (pgGate === 0.1 && pgScore < 0.1) return decision;
    } else {
      // Superset match: multi-category combo rules
      if (pgGate === null) return decision;
    }
  }
  return null;
}

function _compute_yara_score(evidence: readonly YaraEvidence[]): number {
  if (evidence.length === 0) return 0.0;

  const bestByCategory: Record<string, number> = {};
  for (const e of evidence) {
    const w = e.weight > 0
      ? e.weight
      : SEVERITY_WEIGHTS[e.severity.trim().toLowerCase()] ?? 10.0;
    if (w > (bestByCategory[e.category] ?? 0)) {
      bestByCategory[e.category] = w;
    }
  }
  return Math.min(100.0, Object.values(bestByCategory).reduce((a, b) => a + b, 0));
}

function _build_reason(
  yaraScore: number,
  pgScore: number,
  evidence: readonly YaraEvidence[],
  yaraSignal: boolean,
  pgSignal: boolean,
): string {
  const parts: string[] = [`YARA=${yaraScore.toFixed(0)}, PG=${pgScore.toFixed(2)}`];
  if (evidence.length > 0) {
    const ruleIds = [...new Set(evidence.map((e) => e.rule_id))].sort();
    parts.push(`YARA hits: ${ruleIds.join(", ")}`);
  }
  if (yaraSignal) parts.push("YARA strong");
  if (pgSignal) parts.push("PG strong");
  return parts.join(" | ");
}

// Default thresholds
const YARA_REVIEW_THRESHOLD = 15.0;
const YARA_HOLD_THRESHOLD = 40.0;
const PG_REVIEW_THRESHOLD = 0.4;
const PG_HOLD_THRESHOLD = 0.75;
const YARA_WEIGHT = 0.5;
const PG_WEIGHT = 0.5;

export class CheapRouter {
  private readonly yara_review_threshold: number;
  private readonly yara_hold_threshold: number;
  private readonly pg_review_threshold: number;
  private readonly pg_hold_threshold: number;
  private readonly yara_weight: number;
  private readonly pg_weight: number;

  constructor({
    yaraReviewThreshold = YARA_REVIEW_THRESHOLD,
    yaraHoldThreshold = YARA_HOLD_THRESHOLD,
    pgReviewThreshold = PG_REVIEW_THRESHOLD,
    pgHoldThreshold = PG_HOLD_THRESHOLD,
    yaraWeight = YARA_WEIGHT,
    pgWeight = PG_WEIGHT,
  }: {
    yaraReviewThreshold?: number;
    yaraHoldThreshold?: number;
    pgReviewThreshold?: number;
    pgHoldThreshold?: number;
    yaraWeight?: number;
    pgWeight?: number;
  } = {}) {
    if (yaraWeight < 0 || yaraWeight > 1 || pgWeight < 0 || pgWeight > 1) {
      throw new Error("Weights must be between 0 and 1");
    }
    if (yaraWeight === 0 && pgWeight === 0) {
      throw new Error("At least one of yara_weight or pg_weight must be non-zero");
    }
    if (yaraReviewThreshold < 0 || yaraReviewThreshold > yaraHoldThreshold) {
      throw new Error("yara_review_threshold must be <= yara_hold_threshold");
    }
    if (pgReviewThreshold < 0 || pgReviewThreshold > pgHoldThreshold) {
      throw new Error("pg_review_threshold must be <= pg_hold_threshold");
    }
    if (yaraHoldThreshold > 100.0) {
      throw new Error("yara_hold_threshold cannot exceed 100.0 (YARA score max)");
    }
    if (pgHoldThreshold > 1.0) {
      throw new Error("pg_hold_threshold cannot exceed 1.0 (PG score max)");
    }

    this.yara_review_threshold = yaraReviewThreshold;
    this.yara_hold_threshold = yaraHoldThreshold;
    this.pg_review_threshold = pgReviewThreshold;
    this.pg_hold_threshold = pgHoldThreshold;
    this.yara_weight = yaraWeight;
    this.pg_weight = pgWeight;
  }

  route(
    yaraFindings: readonly DetectionFinding[],
    pgScore: number,
  ): CheapChunkDecision {
    const evidence: YaraEvidence[] = yaraFindings.map((f) =>
      yaraEvidenceFromFinding(f),
    );
    const pgScoreClamped = Math.max(0.0, Math.min(1.0, pgScore));

    // Category-combination rules override numeric scoring
    const comboDecision = _check_category_combination_rules(evidence, pgScoreClamped);
    if (comboDecision !== null) {
      const yaraScore = _compute_yara_score(evidence);
      const riskScore = yaraScore * this.yara_weight + pgScoreClamped * 100 * this.pg_weight;
      const reason = _build_reason(yaraScore, pgScoreClamped, evidence, false, false);
      return Object.freeze({
        decision: comboDecision,
        risk_score: riskScore,
        pg_score: pgScoreClamped,
        yara_score: yaraScore,
        findings: evidence,
        reason,
      });
    }

    const hasHoldHint = evidence.some((e) => e.route_hint === "hold");
    const hasReviewHint = evidence.some((e) => e.route_hint === "review");
    const yaraScore = _compute_yara_score(evidence);

    const yaraStrong = this.yara_weight > 0 && yaraScore >= this.yara_hold_threshold;
    const yaraModerate = this.yara_weight > 0 && yaraScore >= this.yara_review_threshold;
    const pgStrong = this.pg_weight > 0 && pgScoreClamped >= this.pg_hold_threshold;
    const pgModerate = this.pg_weight > 0 && pgScoreClamped >= this.pg_review_threshold;

    const riskScore = yaraScore * this.yara_weight + pgScoreClamped * 100 * this.pg_weight;

    let decision: Decision;
    let reason: string;

    if (yaraStrong || pgStrong) {
      decision = DECISION_HOLD;
      reason = _build_reason(yaraScore, pgScoreClamped, evidence, yaraStrong, pgStrong);
    } else if (
      yaraModerate ||
      pgModerate ||
      riskScore >= 20.0 ||
      hasHoldHint ||
      hasReviewHint
    ) {
      decision = DECISION_REVIEW;
      reason = _build_reason(yaraScore, pgScoreClamped, evidence, yaraModerate, pgModerate);
    } else if (riskScore >= 10.0) {
      decision = DECISION_REVIEW;
      reason = _build_reason(yaraScore, pgScoreClamped, evidence, false, false);
    } else {
      decision = DECISION_SAFE;
      reason = `YARA=${yaraScore.toFixed(0)}, PG=${pgScoreClamped.toFixed(2)} — both signals weak.`;
    }

    if (!CHUNK_DECISIONS.has(decision)) {
      decision = DECISION_REVIEW;
      reason = `Unknown decision '${decision}' — routing to review. ${reason}`;
    }

    return Object.freeze({
      decision,
      risk_score: riskScore,
      pg_score: pgScoreClamped,
      yara_score: yaraScore,
      findings: evidence,
      reason,
    });
  }
}
