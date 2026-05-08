/**
 * Text chunking for documents.
 * Ported from doc_analyse/ingestion/chunking.py
 */

import {
  createTextChunk,
  ExtractedDocument,
  TextChunk,
} from "./models.js";

export const DEFAULT_CHUNK_SIZE = 1500;
export const DEFAULT_CHUNK_OVERLAP = 200;

export class TextChunker {
  private readonly chunk_size: number;
  private readonly chunk_overlap: number;

  constructor({
    chunkSize = DEFAULT_CHUNK_SIZE,
    chunkOverlap = DEFAULT_CHUNK_OVERLAP,
  }: { chunkSize?: number; chunkOverlap?: number } = {}) {
    if (chunkSize <= 0) {
      throw new Error("chunk_size must be greater than 0.");
    }
    if (chunkOverlap < 0) {
      throw new Error("chunk_overlap must be greater than or equal to 0.");
    }
    if (chunkOverlap >= chunkSize) {
      throw new Error("chunk_overlap must be smaller than chunk_size.");
    }
    this.chunk_size = chunkSize;
    this.chunk_overlap = chunkOverlap;
  }

  chunk(document: ExtractedDocument): readonly TextChunk[] {
    const text = document.text;
    if (!text || !text.trim()) {
      throw new Error("document text must be a non-empty string.");
    }

    const chunks: TextChunk[] = [];
    let start = 0;
    const textLength = text.length;

    while (start < textLength) {
      let end = Math.min(start + this.chunk_size, textLength);
      end = _move_end_to_boundary(text, start, end);
      const chunkText = text.substring(start, end);

      if (chunkText.trim()) {
        chunks.push(
          createTextChunk(
            chunkText,
            document.source,
            start,
            end,
            _chunk_metadata(document.metadata, chunks.length, chunkText),
          ),
        );
      }

      if (end >= textLength) break;

      start = Math.max(end - this.chunk_overlap, start + 1);
      start = _move_start_to_boundary(text, start);
    }

    return Object.freeze(chunks);
  }
}

export function chunk_document(
  document: ExtractedDocument,
  chunkSize = DEFAULT_CHUNK_SIZE,
  chunkOverlap = DEFAULT_CHUNK_OVERLAP,
): readonly TextChunk[] {
  return new TextChunker({ chunkSize, chunkOverlap }).chunk(document);
}

function _move_end_to_boundary(text: string, start: number, end: number): number {
  if (end >= text.length) return end;

  const newlineIdx = text.lastIndexOf("\n", end - 1);
  const spaceIdx = text.lastIndexOf(" ", end - 1);
  const boundary = Math.max(newlineIdx, spaceIdx);

  if (boundary <= start) return end;
  return boundary;
}

function _move_start_to_boundary(text: string, start: number): number {
  while (start < text.length && text.charAt(start).trim() === "") {
    start++;
  }
  return start;
}

function _chunk_metadata(
  documentMetadata: Readonly<Record<string, unknown>>,
  index: number,
  text: string,
): Readonly<Record<string, unknown>> {
  const metadata: Record<string, unknown> = { ...documentMetadata };
  metadata["chunk_index"] = index;
  metadata["byte_to_char"] = _build_byte_to_char(text);
  return Object.freeze(metadata);
}

export function _build_byte_to_char(text: string): readonly number[] {
  const encoded = new TextEncoder().encode(text);
  const mapping = new Array<number>(encoded.length + 1).fill(0);
  let charIdx = 0;
  let byteIdx = 0;

  while (byteIdx < encoded.length) {
    const b = encoded[byteIdx]!;
    let seqLen: number;
    if (b < 0x80) {
      seqLen = 1;
    } else if (b < 0xe0) {
      seqLen = 2;
    } else if (b < 0xf0) {
      seqLen = 3;
    } else {
      seqLen = 4;
    }

    for (let i = 0; i < seqLen; i++) {
      mapping[byteIdx + i] = charIdx;
    }
    byteIdx += seqLen;
    charIdx += 1;
  }
  mapping[encoded.length] = charIdx;

  return Object.freeze(mapping);
}
