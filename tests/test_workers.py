import threading
import time

import pytest

from doc_analyse.classifiers import ClassificationResult
from doc_analyse.ingestion.models import TextChunk
from doc_analyse.workers import (
    ClassifierWorkerPool,
    StatelessClassifierWorker,
    WorkerPoolError,
    build_stateless_classifier_factory,
)


class FakeWorkerClassifier:
    def __init__(self, delay_by_text=None, thread_hits=None):
        self.delay_by_text = delay_by_text or {}
        self.thread_hits = thread_hits

    def classify(self, text, metadata=None):
        if self.thread_hits is not None:
            self.thread_hits.add(threading.current_thread().name)

        time.sleep(self.delay_by_text.get(text, 0.0))
        return ClassificationResult(
            verdict="suspicious",
            confidence=0.5,
            reasons=("worker",),
        )


def test_worker_pool_classifies_chunks_in_input_order():
    delay_by_text = {
        "chunk-0": 0.02,
        "chunk-1": 0.0,
        "chunk-2": 0.01,
    }
    worker = StatelessClassifierWorker(
        classifier_factory=lambda: FakeWorkerClassifier(delay_by_text=delay_by_text)
    )
    chunks = (
        TextChunk(text="chunk-0", source="doc.txt", start_char=0, end_char=7),
        TextChunk(text="chunk-1", source="doc.txt", start_char=8, end_char=15),
        TextChunk(text="chunk-2", source="doc.txt", start_char=16, end_char=23),
    )

    with ClassifierWorkerPool(worker=worker, max_workers=3) as pool:
        results = pool.classify_chunks(chunks)

    assert [result.chunk.text for result in results] == ["chunk-0", "chunk-1", "chunk-2"]
    assert all(result.classification.verdict == "suspicious" for result in results)


def test_worker_enriches_classification_metadata_with_chunk_coordinates():
    seen = {}

    class MetadataClassifier:
        def classify(self, text, metadata=None):
            seen["metadata"] = dict(metadata or {})
            return ClassificationResult(
                verdict="safe",
                confidence=0.9,
                reasons=("ok",),
            )

    worker = StatelessClassifierWorker(classifier_factory=lambda: MetadataClassifier())
    chunk = TextChunk(
        text="hello",
        source="doc.txt",
        start_char=10,
        end_char=15,
        metadata={"chunk_index": 3},
    )

    result = worker.classify_chunk(chunk)

    assert result.classification.verdict == "safe"
    assert seen["metadata"]["chunk_index"] == 3
    assert seen["metadata"]["source"] == "doc.txt"
    assert seen["metadata"]["start_char"] == 10
    assert seen["metadata"]["end_char"] == 15


def test_worker_uses_thread_local_classifier_cache():
    factory_calls = []
    thread_hits = set()

    def factory():
        factory_calls.append(1)
        return FakeWorkerClassifier(delay_by_text={"busy": 0.01}, thread_hits=thread_hits)

    worker = StatelessClassifierWorker(classifier_factory=factory)
    chunks = tuple(
        TextChunk(text="busy", source="doc.txt", start_char=index, end_char=index + 4)
        for index in range(12)
    )

    with ClassifierWorkerPool(worker=worker, max_workers=3) as pool:
        pool.classify_chunks(chunks)

    assert 1 <= len(factory_calls) <= 3
    assert len(factory_calls) == len(thread_hits)


def test_worker_pool_wraps_worker_errors_with_chunk_index():
    worker = StatelessClassifierWorker(classifier_factory=lambda: FakeWorkerClassifier())
    chunks = (
        TextChunk(text="good", source="doc.txt", start_char=0, end_char=4),
        TextChunk(text="   ", source="doc.txt", start_char=5, end_char=8),
    )

    with (
        ClassifierWorkerPool(worker=worker, max_workers=1) as pool,
        pytest.raises(WorkerPoolError, match="chunk index 1"),
    ):
        pool.classify_chunks(chunks)


def test_classifier_worker_pool_returns_empty_for_no_chunks():
    worker = StatelessClassifierWorker(classifier_factory=lambda: FakeWorkerClassifier())
    with ClassifierWorkerPool(worker=worker) as pool:
        assert pool.classify_chunks([]) == ()


def test_classifier_worker_pool_context_manager_shuts_down_executor():
    worker = StatelessClassifierWorker(classifier_factory=lambda: FakeWorkerClassifier())
    pool = ClassifierWorkerPool(worker=worker)
    chunk = TextChunk(text="hello", source="doc.txt", start_char=0, end_char=5)

    with pool:
        pool.classify_chunks((chunk,))

    # After close(), the executor rejects new submissions.
    with pytest.raises(RuntimeError):
        pool.classify_chunks((chunk,))


def test_build_stateless_classifier_factory_uses_classifier_agent_prompt(monkeypatch):
    captured = {}

    def fake_build_classifier(provider, **kwargs):
        captured["provider"] = provider
        captured["kwargs"] = kwargs
        return FakeWorkerClassifier()

    monkeypatch.setattr("doc_analyse.workers.pool.build_classifier", fake_build_classifier)
    factory = build_stateless_classifier_factory(provider="openai", model="gpt-test", api_key="key")

    classifier = factory()

    assert isinstance(classifier, FakeWorkerClassifier)
    assert captured["provider"] == "openai"
    assert captured["kwargs"]["model"] == "gpt-test"
    assert (
        "document safety analyst" in captured["kwargs"]["system_prompt"].lower()
    )


def test_build_stateless_classifier_factory_can_use_env_builder(monkeypatch):
    captured = {}

    def fake_classifier_from_env(prefix="DOC_ANALYSE_LLM", **kwargs):
        captured["prefix"] = prefix
        captured["kwargs"] = kwargs
        return FakeWorkerClassifier()

    monkeypatch.setattr("doc_analyse.workers.pool.classifier_from_env", fake_classifier_from_env)
    factory = build_stateless_classifier_factory(prefix="CUSTOM_PREFIX")

    classifier = factory()

    assert isinstance(classifier, FakeWorkerClassifier)
    assert captured["prefix"] == "CUSTOM_PREFIX"
    assert "system_prompt" in captured["kwargs"]
