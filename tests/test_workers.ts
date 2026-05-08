import { describe, expect, it, vi } from "vitest";
import { createClassificationResult } from "../src/classifiers/base.js";
import * as factoryModule from "../src/classifiers/factory.js";
import { createTextChunk } from "../src/ingestion/models.js";
import {
  build_stateless_classifier_factory,
  ClassifierWorkerPool,
  StatelessClassifierWorker,
  WorkerPoolError,
} from "../src/workers/pool.js";

class FakeWorkerClassifier {
  readonly calls: string[] = [];

  classify(text: string, metadata?: Record<string, unknown>) {
    this.calls.push(text);
    return createClassificationResult({
      verdict: text.includes("RISKY-INSTRUCTION") ? "unsafe" : "suspicious",
      confidence: 0.5,
      reasons: ["worker"],
      findings: [],
      metadata,
    });
  }
}

describe("ClassifierWorkerPool", () => {
  it("classifies chunks in input order", async () => {
    const worker = new StatelessClassifierWorker(() => new FakeWorkerClassifier() as never);
    const chunks = [
      createTextChunk("chunk-0", "doc.txt", 0, 7),
      createTextChunk("chunk-1", "doc.txt", 8, 15),
      createTextChunk("chunk-2", "doc.txt", 16, 23),
    ];

    const pool = new ClassifierWorkerPool(worker, 3);
    const results = await pool.classify_chunks(chunks);

    expect(results.map((result) => result.chunk.text)).toEqual([
      "chunk-0",
      "chunk-1",
      "chunk-2",
    ]);
  });

  it("enriches metadata with chunk coordinates", () => {
    const seen: Record<string, unknown> = {};
    const worker = new StatelessClassifierWorker(
      () =>
        ({
          classify(_text: string, metadata?: Record<string, unknown>) {
            Object.assign(seen, metadata);
            return createClassificationResult({
              verdict: "safe",
              confidence: 0.9,
              reasons: ["ok"],
              findings: [],
            });
          },
        }) as never,
    );

    const result = worker.classify_chunk(
      createTextChunk("hello", "doc.txt", 10, 15, Object.freeze({ chunk_index: 3 })),
    );

    expect(result.classification.verdict).toBe("safe");
    expect(seen.chunk_index).toBe(3);
    expect(seen.source).toBe("doc.txt");
    expect(seen.start_char).toBe(10);
    expect(seen.end_char).toBe(15);
  });

  it("reuses the worker classifier instance", async () => {
    let factoryCalls = 0;
    const worker = new StatelessClassifierWorker(() => {
      factoryCalls += 1;
      return new FakeWorkerClassifier() as never;
    });

    const pool = new ClassifierWorkerPool(worker, 2);
    await pool.classify_chunks([
      createTextChunk("busy", "doc.txt", 0, 4),
      createTextChunk("busy", "doc.txt", 5, 9),
    ]);
    await pool.classify_chunks([createTextChunk("again", "doc.txt", 10, 15)]);

    expect(factoryCalls).toBe(1);
  });

  it("wraps worker errors with chunk index", async () => {
    const worker = new StatelessClassifierWorker(() => new FakeWorkerClassifier() as never);
    const pool = new ClassifierWorkerPool(worker, 1);

    await expect(
      pool.classify_chunks([
        createTextChunk("good", "doc.txt", 0, 4),
        createTextChunk("   ", "doc.txt", 5, 8),
      ]),
    ).rejects.toThrow(WorkerPoolError);
  });

  it("returns empty for no chunks and rejects reuse after close", async () => {
    const worker = new StatelessClassifierWorker(() => new FakeWorkerClassifier() as never);
    const pool = new ClassifierWorkerPool(worker);

    await expect(pool.classify_chunks([])).resolves.toEqual([]);
    pool.close();
    await expect(
      pool.classify_chunks([createTextChunk("hello", "doc.txt", 0, 5)]),
    ).rejects.toThrow(WorkerPoolError);
  });
});

describe("build_stateless_classifier_factory", () => {
  it("uses the classifier agent prompt for direct providers", () => {
    const spy = vi
      .spyOn(factoryModule, "build_classifier")
      .mockImplementation((_provider, _kwargs) => new FakeWorkerClassifier() as never);

    const factory = build_stateless_classifier_factory({
      provider: "openai",
      model: "gpt-test",
      api_key: "key",
    });

    factory();

    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy.mock.calls[0]![0]).toBe("openai");
    const kwargs = spy.mock.calls[0]![1]!;
    expect(kwargs.model).toBe("gpt-test");
    expect(String(kwargs.system_prompt).toLowerCase()).toContain(
      "document safety analyst",
    );
  });

  it("can use the env builder", () => {
    const spy = vi
      .spyOn(factoryModule, "classifier_from_env")
      .mockImplementation((_prefix, _kwargs) => new FakeWorkerClassifier() as never);

    const factory = build_stateless_classifier_factory({ prefix: "CUSTOM_PREFIX" });
    factory();

    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy.mock.calls[0]![0]).toBe("CUSTOM_PREFIX");
    expect(String(spy.mock.calls[0]![1]!.system_prompt).toLowerCase()).toContain(
      "document safety analyst",
    );
  });
});
