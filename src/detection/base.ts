/**
 * Base detector abstract class and parallel detector.
 * Ported from doc_analyse/detection/base.py
 */

import pLimit from "p-limit";
import { DetectionFinding, createDetectionFinding } from "./models.js";
import { TextChunk } from "../ingestion/models.js";

export abstract class BaseDetector {
  abstract detect(chunk: TextChunk): readonly DetectionFinding[];

  protected _finalize_findings(
    findings: DetectionFinding[],
  ): readonly DetectionFinding[] {
    const deduped: DetectionFinding[] = [];
    const seen = new Set<string>();

    for (const finding of findings) {
      const key = BaseDetector._finding_key(finding);
      if (seen.has(key)) continue;
      seen.add(key);
      deduped.push(finding);
    }

    return Object.freeze(
      deduped.sort((a, b) => {
        const sc = a.start_char - b.start_char;
        if (sc !== 0) return sc;
        const ec = a.end_char - b.end_char;
        if (ec !== 0) return ec;
        return a.rule_id.localeCompare(b.rule_id);
      }),
    );
  }

  static _finding_key(finding: DetectionFinding): string {
    return `${finding.rule_id}|${finding.source}|${finding.start_char}|${finding.end_char}`;
  }

  static _build_finding(props: {
    chunk: TextChunk;
    span: string;
    category: string;
    severity: string;
    reason: string;
    rule_id: string;
    start_char: number;
    end_char: number;
    score?: number | null;
    requires_llm_validation?: boolean;
    metadata?: Readonly<Record<string, unknown>>;
  }): DetectionFinding {
    const resolvedMetadata: Record<string, unknown> = { ...props.chunk.metadata };
    if (props.metadata) {
      Object.assign(resolvedMetadata, props.metadata);
    }

    return createDetectionFinding({
      span: props.span,
      category: props.category,
      severity: props.severity,
      reason: props.reason,
      rule_id: props.rule_id,
      start_char: props.start_char,
      end_char: props.end_char,
      source: props.chunk.source,
      requires_llm_validation: props.requires_llm_validation ?? false,
      score: props.score ?? null,
      metadata: Object.freeze(resolvedMetadata),
    });
  }
}

export class ParallelDetector extends BaseDetector {
  private readonly detectors: readonly BaseDetector[];
  private readonly max_workers: number;

  constructor(detectors: BaseDetector[], maxWorkers?: number) {
    super();
    this.detectors = Object.freeze([...detectors]);
    this.max_workers = maxWorkers ?? this.detectors.length;
  }

  detect(chunk: TextChunk): readonly DetectionFinding[] {
    if (this.detectors.length === 0) return [];
    const findings: DetectionFinding[] = [];
    for (const detector of this.detectors) {
      try {
        findings.push(...detector.detect(chunk));
      } catch (err) {
        findings.push(_detector_error_finding(chunk, detector, err));
      }
    }
    return this._finalize_findings(findings);
  }

  async detect_many(chunks: readonly TextChunk[]): Promise<readonly DetectionFinding[]> {
    if (this.detectors.length === 0 || chunks.length === 0) return [];

    const findings: DetectionFinding[] = [];
    const limit = pLimit(this.max_workers);

    const tasks: Array<() => Promise<readonly DetectionFinding[]>> = [];

    for (const chunk of chunks) {
      for (const detector of this.detectors) {
        tasks.push(() =>
          limit(async () => {
            try {
              return detector.detect(chunk);
            } catch (err) {
              return [_detector_error_finding(chunk, detector, err)];
            }
          }),
        );
      }
    }

    const results = await Promise.all(tasks.map((t) => t()));
    for (const r of results) {
      findings.push(...r);
    }

    return this._finalize_findings(findings);
  }
}

function _detector_error_finding(
  chunk: TextChunk,
  detector: BaseDetector,
  exc: unknown,
): DetectionFinding {
  const detectorName = detector.constructor.name;
  return BaseDetector._build_finding({
    chunk,
    span: chunk.text,
    category: "detector_error",
    severity: "medium",
    reason: `${detectorName} failed; send this chunk to LLM validation.`,
    rule_id: `${detectorName.toLowerCase()}_error`,
    start_char: chunk.start_char,
    end_char: chunk.end_char,
    requires_llm_validation: true,
    metadata: Object.freeze({
      detector: detectorName,
      error: String(exc),
    }),
  });
}
