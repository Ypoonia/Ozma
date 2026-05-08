/**
 * Ingestion data models.
 * Ported from doc_analyse/ingestion/models.py
 */

// Object.freeze is a built-in global — no import needed

export interface DocumentSegment {
  readonly text: string;
  readonly start_char: number;
  readonly end_char: number;
  readonly metadata: Readonly<Record<string, unknown>>;
}

export interface ExtractedDocument {
  readonly text: string;
  readonly source: string;
  readonly mime_type: string | null;
  readonly metadata: Readonly<Record<string, unknown>>;
  readonly segments: readonly DocumentSegment[];
}

export interface TextChunk {
  readonly text: string;
  readonly source: string;
  readonly start_char: number;
  readonly end_char: number;
  readonly metadata: Readonly<Record<string, unknown>>;
  readonly length: number;
}

export interface IngestedDocument {
  readonly document: ExtractedDocument;
  readonly chunks: readonly TextChunk[];
  readonly text: string;
  readonly source: string;
}

// ---------------------------------------------------------------------------
// Frozen object factories (replacing @dataclass(frozen=True))
// ---------------------------------------------------------------------------

export function createDocumentSegment(
  text: string,
  startChar: number,
  endChar: number,
  metadata?: Readonly<Record<string, unknown>>,
): DocumentSegment {
  return Object.freeze({
    text,
    start_char: startChar,
    end_char: endChar,
    metadata: metadata ?? Object.freeze({}),
  });
}

export function createExtractedDocument(
  text: string,
  source: string,
  mimeType: string | null = null,
  metadata?: Readonly<Record<string, unknown>>,
  segments?: readonly DocumentSegment[],
): ExtractedDocument {
  return Object.freeze({
    text,
    source,
    mime_type: mimeType,
    metadata: metadata ?? Object.freeze({}),
    segments: segments ?? [],
  });
}

export function createTextChunk(
  text: string,
  source: string,
  startChar: number,
  endChar: number,
  metadata?: Readonly<Record<string, unknown>>,
): TextChunk {
  return Object.freeze({
    text,
    source,
    start_char: startChar,
    end_char: endChar,
    metadata: metadata ?? Object.freeze({}),
    get length(): number {
      return endChar - startChar;
    },
  });
}

export function createIngestedDocument(
  document: ExtractedDocument,
  chunks: readonly TextChunk[],
): IngestedDocument {
  return Object.freeze({
    document,
    chunks,
    get text(): string {
      return document.text;
    },
    get source(): string {
      return document.source;
    },
  });
}
