from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import local
from time import sleep
from typing import Any, Callable, Iterable, Mapping, Optional

from doc_analyse.classifiers import (
    BaseClassifier,
    ClassificationResult,
    build_classifier,
    classifier_from_env,
)
from doc_analyse.ingestion.models import TextChunk
from doc_analyse.prompt.loader import load_classifier_agent_prompt


class WorkerPoolError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkerResult:
    chunk: TextChunk
    classification: ClassificationResult


RETRY_DELAYS = (1, 2, 4)  # exponential backoff in seconds
CHUNK_TIMEOUT = 120  # seconds per chunk


class StatelessClassifierWorker:
    """Thread-safe stateless worker.

    A worker call never keeps request memory. It only classifies the current
    chunk and metadata. The classifier instance is cached per-thread to avoid
    sharing non-thread-safe clients across worker threads.
    """

    def __init__(self, classifier_factory: Callable[[], BaseClassifier]) -> None:
        self._classifier_factory = classifier_factory
        self._thread_local = local()

    def classify_chunk(self, chunk: TextChunk) -> WorkerResult:
        if not isinstance(chunk.text, str) or not chunk.text.strip():
            raise ValueError("Worker chunk text must be a non-empty string.")

        classification = self._classifier().classify(
            text=chunk.text,
            metadata=_chunk_classification_metadata(chunk),
        )
        return WorkerResult(
            chunk=chunk,
            classification=classification,
        )

    def _classifier(self) -> BaseClassifier:
        classifier = getattr(self._thread_local, "classifier", None)
        if classifier is not None:
            return classifier

        classifier = self._classifier_factory()
        self._thread_local.classifier = classifier
        return classifier


def _classify_with_retry(worker: StatelessClassifierWorker, chunk: TextChunk) -> WorkerResult:
    """Call classify_chunk with exponential backoff retry on failure."""
    last_exc: Exception | None = None
    for delay in RETRY_DELAYS:
        try:
            return worker.classify_chunk(chunk)
        except Exception as exc:
            last_exc = exc
            sleep(delay)
    # Final attempt — let it propagate
    raise last_exc from None


def _cancel_pending(futures: dict, futures_to_cancel: set) -> None:
        for future in futures_to_cancel:
            future.cancel()


class ClassifierWorkerPool:
    """Thread pool that fans out chunk classification across stateless workers.

    The executor is created once at construction time and reused across all
    ``classify_chunks`` calls. Use the pool as a context manager — or call
    ``close()`` explicitly — to shut down the executor when done.
    """

    def __init__(
        self,
        worker: StatelessClassifierWorker,
        max_workers: Optional[int] = None,
    ) -> None:
        self.worker = worker
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def __enter__(self) -> ClassifierWorkerPool:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Shutdown the executor, waiting for in-flight tasks to finish."""
        self._executor.shutdown(wait=True)

    def classify_chunks(self, chunks: Iterable[TextChunk]) -> tuple[WorkerResult, ...]:
        indexed_chunks = tuple(enumerate(chunks))
        if not indexed_chunks:
            return ()

        results_by_index: dict[int, WorkerResult] = {}
        futures = {
            self._executor.submit(_classify_with_retry, self.worker, chunk): index
            for index, chunk in indexed_chunks
        }
        pending = set(futures.keys())

        try:
            for future in as_completed(futures):
                pending.discard(future)
                index = futures[future]
                try:
                    results_by_index[index] = future.result(timeout=CHUNK_TIMEOUT)
                except TimeoutError as exc:
                    raise WorkerPoolError(
                        f"Worker timed out after {CHUNK_TIMEOUT}s for chunk index {index}."
                    ) from exc
                except Exception as exc:
                    raise WorkerPoolError(
                        f"Worker failed for chunk index {index} after {len(RETRY_DELAYS)} retries: {exc}"
                    ) from exc
        finally:
            _cancel_pending(futures, pending)

        if len(results_by_index) != len(indexed_chunks):
            raise WorkerPoolError(
                f"Expected {len(indexed_chunks)} results but got {len(results_by_index)}."
            )

        return tuple(results_by_index[index] for index, _ in indexed_chunks)


def build_stateless_classifier_factory(
    *,
    provider: Optional[str] = None,
    prefix: str = "DOC_ANALYSE_LLM",
    system_prompt: Optional[str] = None,
    **classifier_kwargs: Any,
) -> Callable[[], BaseClassifier]:
    worker_system_prompt = load_classifier_agent_prompt(system_prompt)
    base_kwargs = dict(classifier_kwargs)
    base_kwargs["system_prompt"] = worker_system_prompt

    if provider is not None:
        normalized_provider = provider.strip().lower()
        return lambda: build_classifier(normalized_provider, **base_kwargs)

    return lambda: classifier_from_env(prefix=prefix, **base_kwargs)


def build_classifier_worker_pool(
    *,
    provider: Optional[str] = None,
    prefix: str = "DOC_ANALYSE_LLM",
    system_prompt: Optional[str] = None,
    max_workers: Optional[int] = None,
    **classifier_kwargs: Any,
) -> ClassifierWorkerPool:
    classifier_factory = build_stateless_classifier_factory(
        provider=provider,
        prefix=prefix,
        system_prompt=system_prompt,
        **classifier_kwargs,
    )
    worker = StatelessClassifierWorker(classifier_factory=classifier_factory)
    return ClassifierWorkerPool(worker=worker, max_workers=max_workers)


def _chunk_classification_metadata(chunk: TextChunk) -> Mapping[str, Any]:
    metadata = dict(chunk.metadata)
    metadata.setdefault("source", chunk.source)
    metadata.setdefault("start_char", chunk.start_char)
    metadata.setdefault("end_char", chunk.end_char)
    return metadata
