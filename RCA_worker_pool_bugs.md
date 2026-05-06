# Worker Pool Bug RCA — `fix/worker-pool-bugs`

## Summary

Five bugs were identified and fixed in `src/doc_analyse/workers/pool.py`. The fixes
address per-chunk timeout ineffectiveness, incorrect retry counts, post-failure
sleep, running-future cancellation, deterministic-error retry, and unbounded
submission queue.

---

## [P1] Per-chunk timeout does not work

**Root cause**

`as_completed(futures)` at line 126 only yields a future *after* it has already
completed. The subsequent `future.result(timeout=CHUNK_TIMEOUT)` at line 130 is
called on an already-done future — Python's `Future.result()` returns immediately
with no wait. The timeout parameter is silently ignored.

**Fix**

Track a deadline per future at submission time (`monotonic() + CHUNK_TIMEOUT`).
When `wait()` returns completed futures, compute actual elapsed time and raise
`TimeoutError` if the chunk ran longer than `CHUNK_TIMEOUT`. Use `wait()` with a
short timeout instead of `as_completed()` so the polling loop wakes periodically.

---

## [P2] Retry logic under-attempts and sleeps after final failure

**Root cause (under-attempts)**

`RETRY_DELAYS = (1, 2, 4)` has three elements. The loop at line 71 runs for each
element, making exactly three total attempts — not "initial + three retries."

**Root cause (post-failure sleep)**

Inside the loop at line 76, `sleep(delay)` executes after every exception,
including the third (last) iteration. After `sleep(4)` on the final iteration,
the loop ends and line 78 raises. The 4-second sleep happens *after* the last
failure and *before* the exception propagates.

**Fix**

Perform the initial attempt outside the retry loop. Loop over
`RETRY_DELAYS` only for retries. Sleep only *between* retries, not after the
final failure. Corrected error message to say "initial attempt +
N retries."

---

## [P2] Running futures not cancelled when pool raises

**Root cause**

`_cancel_pending(futures, pending)` at line 140 only calls `future.cancel()` on
futures in the `pending` set — futures not yet yielded by `as_completed`. Once a
future is running on a worker thread, `cancel()` returns `False`; the task runs to
completion. A fast-failing chunk causes `classify_chunks` to raise, but any
concurrently executing chunk continues in the background.

**Fix**

`shutdown(wait=False)` terminates the executor immediately when an error is
detected, hard-killing running workers rather than attempting cooperative
cancellation. This prevents background completion from writing into
`results_by_index` after the method has already raised.

---

## [P2] Deterministic input errors retry with backoff

**Root cause**

`except Exception` at line 74 catches every exception type, including the
`ValueError` raised at line 46 for blank/empty text. Retrying a blank-text chunk
is guaranteed to fail every time. All three retry sleeps (1s, 2s, 4s) are
wasted before the exception propagates.

**Fix**

`_is_retryable()` classifies exception types. Returns `False` for deterministic
errors: `ValueError`, `TypeError`, `KeyError`, `IndexError`, `AttributeError`.
Before sleeping and retrying, check `_is_retryable(last_exc)` and raise
immediately if it is `False`.

---

## [P3] Queue is unbounded at submission time

**Root cause**

All chunks are submitted to the `ThreadPoolExecutor` in a single dict
comprehension at line 119–122. `ThreadPoolExecutor` uses an unbounded
`queue.Queue` internally. `max_workers` limits concurrent execution but does not
limit how many futures can be queued. A large document submits all its chunks
upfront, creating unbounded memory pressure and API job queue.

**Fix**

Submit chunks in bounded batches of `_MAX_CONCURRENT = 16` futures. Each batch
is awaited before the next batch is submitted, providing natural backpressure.
This keeps memory bounded and prevents unbounded queue growth for large
documents.

---

## Files changed

- `src/doc_analyse/workers/pool.py` — all five fixes applied