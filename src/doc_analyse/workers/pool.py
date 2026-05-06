from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from threading import local
from time import monotonic, sleep
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple

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


RETRY_DELAYS = (1, 2, 4)  # exponential backoff in seconds between retries
CHUNK_TIMEOUT = 120  # seconds per chunk — enforced from submission to completion

# Backpressure: max futures in flight at any time
_MAX_CONCURRENT = 16


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


def _is_retryable(exc: Exception) -> bool:
    """Return True for transient errors that may succeed on retry."""
    if isinstance(exc, (ValueError, TypeError, KeyError, IndexError, AttributeError)):
        return False
    return True


def _classify_with_retry(worker: StatelessClassifierWorker, chunk: TextChunk) -> WorkerResult:
    """Call classify_chunk with exponential backoff retry on transient failures.

    Attempts classify_chunk once, then retries up to len(RETRY_DELAYS) times
    with intervening sleeps. Does NOT sleep after the final failure before
    propagating.
    """
    last_exc: Exception | None = None
    try:
        return worker.classify_chunk(chunk)
    except Exception as exc:
        last_exc = exc

    for delay in RETRY_DELAYS:
        if not _is_retryable(last_exc):
            raise last_exc from None
        sleep(delay)
        try:
            return worker.classify_chunk(chunk)
        except Exception as exc:
            last_exc = exc

    raise last_exc from None


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

    def classify_chunks(self, chunks: Iterable[TextChunk]) -> Tuple[WorkerResult, ...]:
        indexed_chunks = tuple(enumerate(chunks))
        if not indexed_chunks:
            return ()

        results_by_index: dict[int, WorkerResult] = {}

        # P3 fix: bounded submission with batching for backpressure
        futures: dict[Any, int] = {}
        batch_size = _MAX_CONCURRENT

        for i in range(0, len(indexed_chunks), batch_size):
            batch = indexed_chunks[i : i + batch_size]
            batch_futures = {
                self._executor.submit(_classify_with_retry, self.worker, chunk): index
                for index, chunk in batch
            }
            futures.update(batch_futures)

        # P1 fix: track per-future deadline; enforce it when future completes
        futures_with_deadline = {
            future: (index, monotonic() + CHUNK_TIMEOUT)
            for future, index in futures.items()
        }

        try:
            while futures_with_deadline:
                wait_result = wait(
                    list(futures_with_deadline.keys()),
                    return_when=FIRST_COMPLETED,
                    timeout=0.5,
                )

                done = wait_result.done
                now = monotonic()

                for future in list(done):
                    if future not in futures_with_deadline:
                        continue
                    index, deadline = futures_with_deadline.pop(future)
                    elapsed = now - (deadline - CHUNK_TIMEOUT)

                    if elapsed > CHUNK_TIMEOUT:
                        for f in futures_with_deadline:
                            f.cancel()
                        raise WorkerPoolError(
                            f"Worker timed out after {CHUNK_TIMEOUT:.1f}s "
                            f"(elapsed={elapsed:.3f}s) for chunk index {index}."
                        )
                    try:
                        results_by_index[index] = future.result()
                    except Exception as exc:
                        raise WorkerPoolError(
                            f"Worker failed for chunk index {index} "
                            f"after initial attempt + {len(RETRY_DELAYS)} retries: {exc}"
                        ) from exc

                if not done:
                    # No futures completed this poll — check for deadline expiry.
                    # A hung future that never finishes would otherwise spin forever.
                    for f, (idx, dl) in list(futures_with_deadline.items()):
                        if now > dl:
                            elapsed = now - (dl - CHUNK_TIMEOUT)
                            for pending_f in futures_with_deadline:
                                pending_f.cancel()
                            raise WorkerPoolError(
                                f"Worker timed out after {CHUNK_TIMEOUT:.1f}s "
                                f"(elapsed={elapsed:.3f}s) for chunk index {idx}."
                            )

        finally:
            # Cancel any futures that did not complete (e.g. on error paths).
            # The executor itself is kept alive for subsequent classify_chunks calls;
            # it is only shut down by close() / the context-manager __exit__.
            for f in futures_with_deadline:
                f.cancel()

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