/**
 * Detection data models.
 * Ported from doc_analyse/detection/models.py
 */

export interface DetectionFinding {
  readonly span: string;
  readonly category: string;
  readonly severity: string;
  readonly reason: string;
  readonly start_char: number;
  readonly end_char: number;
  readonly source: string;
  readonly rule_id: string;
  readonly requires_llm_validation: boolean;
  readonly score: number | null;
  readonly metadata: Readonly<Record<string, unknown>>;
  readonly length: number;
}

export function createDetectionFinding(props: {
  span: string;
  category: string;
  severity: string;
  reason: string;
  start_char: number;
  end_char: number;
  source: string;
  rule_id: string;
  requires_llm_validation?: boolean;
  score?: number | null;
  metadata?: Readonly<Record<string, unknown>>;
}): DetectionFinding {
  return Object.freeze({
    span: props.span,
    category: props.category,
    severity: props.severity,
    reason: props.reason,
    start_char: props.start_char,
    end_char: props.end_char,
    source: props.source,
    rule_id: props.rule_id,
    requires_llm_validation: props.requires_llm_validation ?? false,
    score: props.score ?? null,
    metadata: props.metadata ?? Object.freeze({}),
    get length(): number {
      return props.end_char - props.start_char;
    },
  });
}
