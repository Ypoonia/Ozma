/**
 * YARA detector with native RegExp reimplementation.
 * Ported from doc_analyse/detection/yara.py — replaces yara-python with JS RegExp.
 *
 * The YARA-to-RegExp transpiler parses the .yara source and converts each rule's
 * string patterns to JavaScript RegExp objects. The detector then runs all
 * compiled RegExp patterns against chunk text, producing DetectionFinding objects.
 */

import { readFileSync } from "fs";
import { DetectionFinding, createDetectionFinding } from "./models.js";
import { TextChunk } from "../ingestion/models.js";
import { _build_byte_to_char } from "../ingestion/chunking.js";
import { BaseDetector } from "./base.js";

// ---------------------------------------------------------------------------
// YARA rule types (internal)
// ---------------------------------------------------------------------------

interface YaraRuleMeta {
  rule_id: string;
  category: string;
  severity: string;
  weight: number;
  route_hint: string;
  requires_llm_validation: boolean;
  reason: string;
}

interface CompiledPattern {
  regex: RegExp;
  pattern: string;
}

interface CompiledRule {
  name: string;
  meta: YaraRuleMeta;
  patterns: CompiledPattern[];
  condition: "any" | "all";
}

export class YaraGlossaryError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "YaraGlossaryError";
  }
}

// ---------------------------------------------------------------------------
// YARA source parser and RegExp transpiler
// ---------------------------------------------------------------------------

function _read_yara_source(filePath: string): string {
  try {
    return readFileSync(filePath, "utf-8");
  } catch (err) {
    throw new YaraGlossaryError(
      `Could not read YARA rules file '${filePath}': ${err}`,
    );
  }
}

function _read_packaged_yara_source(filename: string): string {
  const fileUrl = new URL(`./${filename}`, import.meta.url);
  return readFileSync(fileUrl, "utf-8");
}

export function compile_yara_rules(source: string): CompiledRule[] {
  const rules: CompiledRule[] = [];
  const ruleBlocks = _split_rule_blocks(source);

  for (const block of ruleBlocks) {
    const rule = _parse_rule_block(block);
    if (rule) rules.push(rule);
  }

  return rules;
}

function _split_rule_blocks(source: string): string[] {
  const blocks: string[] = [];
  const regex = /\brule\s+\w+\s*\{/g;
  const matches: { offset: number; line: number }[] = [];

  let match: RegExpExecArray | null;
  const lines = source.split("\n");
  let charOffset = 0;
  const lineToChar: number[] = [0];
  for (const line of lines) {
    charOffset += line.length + 1;
    lineToChar.push(charOffset);
  }

  while ((match = regex.exec(source)) !== null) {
    matches.push({
      offset: match.index,
      line: source.substring(0, match.index).split("\n").length,
    });
  }

  for (let i = 0; i < matches.length; i++) {
    const start = matches[i]!.offset;
    const end = i < matches.length - 1 ? matches[i + 1]!.offset : source.length;
    blocks.push(source.slice(start, end));
  }

  return blocks;
}

function _parse_rule_block(block: string): CompiledRule | null {
  const nameMatch = block.match(/\brule\s+(\w+)\s*\{/);
  if (!nameMatch) return null;
  const name = nameMatch[1]!;

  const metaMatch = block.match(/meta:\s*([\s\S]*?)(?=strings:)/);
  const stringsMatch = block.match(/strings:\s*([\s\S]*?)(?=condition:)/);
  const conditionMatch = block.match(/condition:\s*([\s\S]*?)\}/);

  const meta = _parse_meta(metaMatch?.[1] ?? "");
  const patterns = _parse_strings(stringsMatch?.[1] ?? "");
  const condition = _parse_condition(conditionMatch?.[1] ?? "");

  return { name, meta, patterns, condition };
}

function _parse_meta(metaSection: string): YaraRuleMeta {
  const values = new Map<string, string>();
  const metaLineRegex = /^\s*([a-zA-Z_]\w*)\s*=\s*(.+?)\s*$/gm;

  let match: RegExpExecArray | null;
  while ((match = metaLineRegex.exec(metaSection)) !== null) {
    values.set(match[1]!.toLowerCase(), match[2]!.trim());
  }

  const get = (key: string, fallback: string | number | boolean): string | number | boolean => {
    const rawValue = values.get(key.toLowerCase());
    if (rawValue === undefined) {
      return fallback;
    }

    if (rawValue.startsWith("\"") && rawValue.endsWith("\"")) {
      return rawValue.slice(1, -1);
    }

    if (/^(true|yes|1|on)$/i.test(rawValue)) {
      return true;
    }

    if (/^(false|no|0|off)$/i.test(rawValue)) {
      return false;
    }

    if (/^\d+\.?\d*$/.test(rawValue)) {
      return parseFloat(rawValue);
    }

    return rawValue;
  };

  return {
    rule_id: String(get("rule_id", nameFromMeta(metaSection) || "unknown")),
    category: String(get("category", "unknown")),
    severity: String(get("severity", "medium")),
    weight: Number(get("weight", 0)),
    route_hint: String(get("route_hint", "evidence")),
    requires_llm_validation: Boolean(get("requires_llm_validation", false)),
    reason: String(get("reason", "YARA rule matched.")),
  };
}

function nameFromMeta(_metaSection: string): string | null {
  return null;
}

function _parse_strings(stringsSection: string): CompiledPattern[] {
  const patterns: CompiledPattern[] = [];
  // Pattern: $var = /regex/flags  OR  $var = "literal"
  // Pattern is followed by optional nocase modifier.
  // Group 1: var name, Group 2: /regex body/flags OR "literal body"
  const stringRegex = /\$([a-zA-Z_]\w*)\s*=\s*(\/(?:[^\/\\]|\\.)*\/[gimsuvy]*|"(?:[^"\\]|\\.)*")(?:\s+nocase)?/gi;

  let match: RegExpExecArray | null;
  while ((match = stringRegex.exec(stringsSection)) !== null) {
    const rawPattern = match[2]!;
    const isRegex = rawPattern.startsWith("/");
    const hasNocase = rawPattern.toLowerCase().includes("nocase") ||
      match[0]!.toLowerCase().includes("nocase");

    let regex: RegExp;
    try {
      if (isRegex) {
        const lastSlash = rawPattern.lastIndexOf("/");
        const body = rawPattern.substring(1, lastSlash);
        const flags = new Set(rawPattern.substring(lastSlash + 1).split(""));
        flags.add("g");
        if (hasNocase) flags.add("i");
        regex = new RegExp(body, [...flags].join(""));
      } else {
        const escaped = rawPattern.slice(1, -1).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        regex = new RegExp(escaped, hasNocase ? "gi" : "g");
      }
    } catch {
      continue;
    }

    patterns.push({ regex, pattern: rawPattern });
  }

  return patterns;
}

function _parse_condition(condition: string): "any" | "all" {
  const trimmed = condition.trim();
  if (trimmed.includes("any of them")) return "any";
  if (trimmed.includes("all of them")) return "all";
  if (trimmed.includes(" or ")) return "any";
  if (trimmed.includes(" and ")) return "all";
  return "any";
}

// ---------------------------------------------------------------------------
// YaraDetector
// ---------------------------------------------------------------------------

export class YaraDetector extends BaseDetector {
  private readonly compiled: CompiledRule[];

  constructor(compiled?: CompiledRule[] | null) {
    super();
    this.compiled = compiled ?? _DEFAULT_COMPILED_RULES ?? [];
  }

  static from_file(filePath: string): YaraDetector {
    const source = _read_yara_source(filePath);
    const compiled = compile_yara_rules(source);
    return new YaraDetector(compiled);
  }

  detect(chunk: TextChunk): readonly DetectionFinding[] {
    if (!chunk.text || !chunk.text.trim()) return [];

    if (!this.compiled || this.compiled.length === 0) {
      throw new YaraGlossaryError(
        "Default YARA rules failed to load. Provide rules via from_file().",
      );
    }

    const byteToChar = (chunk.metadata["byte_to_char"] as readonly number[] | null)
      ?? _build_byte_to_char(chunk.text);

    const findings: DetectionFinding[] = [];

    for (const rule of this.compiled) {
      const ruleMatches: Array<{
        spanText: string;
        charOffset: number;
        spanCharLen: number;
      }> = [];

      for (const { regex } of rule.patterns) {
        regex.lastIndex = 0;
        let m: RegExpExecArray | null;
        while ((m = regex.exec(chunk.text)) !== null) {
          const spanText = m[0]!;
          if (!spanText.trim()) continue;

          const byteOffset = m.index;
          const charOffset = byteToChar[byteOffset] ?? byteOffset;
          const spanCharLen = spanText.length;

          ruleMatches.push({ spanText, charOffset, spanCharLen });

          if (!regex.global) break;
        }
      }

      if (rule.condition === "any") {
        if (ruleMatches.length === 0) continue;
      } else {
        if (ruleMatches.length < rule.patterns.length) continue;
      }

      for (const { spanText, charOffset, spanCharLen } of ruleMatches) {
        const normalizedScore =
          rule.meta.weight > 0 ? Math.min(1.0, rule.meta.weight / 100.0) : null;

        findings.push(
          createDetectionFinding({
            span: spanText,
            category: rule.meta.category,
            severity: rule.meta.severity,
            reason: rule.meta.reason,
            rule_id: rule.meta.rule_id,
            start_char: chunk.start_char + charOffset,
            end_char: chunk.start_char + charOffset + spanCharLen,
            source: chunk.source,
            requires_llm_validation: rule.meta.requires_llm_validation,
            score: normalizedScore,
            metadata: Object.freeze({
              detector: "YaraDetector",
              yara_rule: rule.name,
              yara_weight: rule.meta.weight,
              route_hint: rule.meta.route_hint,
            }),
          }),
        );
      }
    }

    return this._finalize_findings(findings);
  }
}

// ---------------------------------------------------------------------------
// Module-level default compiled rules
// ---------------------------------------------------------------------------

let _DEFAULT_COMPILED_RULES: CompiledRule[] | null = null;

export function _load_default_rules(): CompiledRule[] | null {
  try {
    const source = _read_packaged_yara_source("default.yara");
    return compile_yara_rules(source);
  } catch {
    return null;
  }
}

export function _init_default_rules(): void {
  _DEFAULT_COMPILED_RULES = _load_default_rules();
}

_init_default_rules();
