/**
 * Ingestion pipeline.
 * Ported from doc_analyse/ingestion/pipeline.py
 */

import { resolve } from "path";
import { createIngestedDocument, IngestedDocument } from "./models.js";
import { TextChunker, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP } from "./chunking.js";
import { ConverterRegistry, convert_document } from "./converters.js";

export function ingest_document(
  path: string,
  {
    registry,
    chunker,
    chunkSize = DEFAULT_CHUNK_SIZE,
    chunkOverlap = DEFAULT_CHUNK_OVERLAP,
  }: {
    registry?: ConverterRegistry;
    chunker?: TextChunker;
    chunkSize?: number;
    chunkOverlap?: number;
  } = {},
): IngestedDocument {
  const resolvedPath = resolve(path);
  const document = convert_document(resolvedPath, registry);

  const resolvedChunker =
    chunker ??
    new TextChunker({ chunkSize, chunkOverlap });

  return createIngestedDocument(document, resolvedChunker.chunk(document));
}
