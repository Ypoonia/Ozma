import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { describe, expect, it } from "vitest";
import * as ozma from "../src/index.js";
import * as ingestion from "../src/ingestion/index.js";
import {
  ConverterRegistry,
  createExtractedDocument,
  DocumentConversionError,
  MarkItDownDocumentConverter,
  TextChunker,
  UnsupportedDocumentError,
  chunk_document,
  convert_document,
  ingest_document,
} from "../src/ingestion/index.js";

class FakeMarkItDown {
  constructor(private readonly text: string) {}

  convert(path: string) {
    return { text_content: this.text, source: path };
  }
}

function withTempDir(run: (dir: string) => void): void {
  const dir = mkdtempSync(join(tmpdir(), "ozma-ingestion-"));
  try {
    run(dir);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

describe("ingestion", () => {
  it("reads text files", () => {
    withTempDir((dir) => {
      const path = join(dir, "sample.txt");
      writeFileSync(path, "First line\nSecond line", "utf-8");

      const document = convert_document(path);

      expect(document.text).toBe("First line\nSecond line");
      expect(document.source).toBe(path);
      expect(document.mime_type).toBe("text/plain");
      expect(document.metadata.converter).toBe("text");
      expect(document.metadata.extension).toBe(".txt");
      expect(document.segments).toHaveLength(1);
      expect(document.segments[0]!.start_char).toBe(0);
      expect(document.segments[0]!.end_char).toBe(document.text.length);
    });
  });

  it("keeps BaseDocumentConverter public", () => {
    expect("BaseDocumentConverter" in ingestion).toBe(true);
    expect(ingestion.BaseDocumentConverter).toBe(ozma.BaseDocumentConverter);
  });

  it("rejects missing, unsupported, and empty documents", () => {
    withTempDir((dir) => {
      expect(() => convert_document(join(dir, "missing.txt"))).toThrow(
        DocumentConversionError,
      );

      const unsupported = join(dir, "sample.bin");
      writeFileSync(unsupported, "binary", "utf-8");
      expect(() => new ConverterRegistry().convert(unsupported)).toThrow(
        UnsupportedDocumentError,
      );

      const empty = join(dir, "empty.md");
      writeFileSync(empty, "   ", "utf-8");
      expect(() => convert_document(empty)).toThrow(DocumentConversionError);
    });
  });

  it("normalizes MarkItDown conversions into an extracted document", () => {
    withTempDir((dir) => {
      const path = join(dir, "sample.docx");
      writeFileSync(path, "fake", "utf-8");
      const converter = new MarkItDownDocumentConverter();
      (converter as unknown as { _markitdown: FakeMarkItDown })._markitdown =
        new FakeMarkItDown("# Title\n\nBody");
      const registry = new ConverterRegistry([converter]);

      const document = convert_document(path, registry);

      expect(document.text).toBe("# Title\n\nBody");
      expect(document.metadata.converter).toBe("markitdown");
      expect(document.metadata.normalized_format).toBe("markdown");
      expect(document.segments[0]!.end_char).toBe(document.text.length);
    });
  });

  it("preserves chunk offsets and metadata", () => {
    const document = createExtractedDocument(
      "alpha beta gamma delta epsilon",
      "memory.txt",
      null,
      Object.freeze({ extension: ".txt" }),
    );

    const chunks = chunk_document(document, 12, 2);

    expect(chunks.map((chunk) => chunk.text)).toEqual([
      "alpha beta",
      "ta gamma",
      "ma delta",
      "ta epsilon",
    ]);
    expect(chunks.map((chunk) => [chunk.start_char, chunk.end_char])).toEqual([
      [0, 10],
      [8, 16],
      [14, 22],
      [20, 30],
    ]);
    expect(chunks.map((chunk) => chunk.metadata.chunk_index)).toEqual([0, 1, 2, 3]);
    expect(chunks.every((chunk) => chunk.metadata.extension === ".txt")).toBe(true);
    expect(chunks.every((chunk) => "byte_to_char" in chunk.metadata)).toBe(true);
  });

  it("rejects invalid chunker settings and ingests documents", () => {
    expect(() => new TextChunker({ chunkSize: 0 })).toThrow();
    expect(() => new TextChunker({ chunkSize: 10, chunkOverlap: 10 })).toThrow();
    expect(() => new TextChunker({ chunkSize: 10, chunkOverlap: -1 })).toThrow();

    withTempDir((dir) => {
      const path = join(dir, "sample.md");
      writeFileSync(path, "alpha beta gamma delta epsilon", "utf-8");

      const result = ingest_document(path, { chunkSize: 12, chunkOverlap: 2 });

      expect(result.document.source).toBe(path);
      expect(result.text).toBe("alpha beta gamma delta epsilon");
      expect(result.chunks.map((chunk) => chunk.metadata.chunk_index)).toEqual([
        0,
        1,
        2,
        3,
      ]);
    });
  });
});
