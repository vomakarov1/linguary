# Final review: translation reliability

Branch `feature/translation-reliability`, base `36b1e8fa75dc9973d6c929cfb5e8edc11aa271d6`, head `de37c0c79c25ef051d510379eb830e8d1304237e`. Read-only review of product code and tests against the brief, the SDD, and the plan's Global Constraints. The reported suite (`uv run pytest -m "not slow"`: 4484 passed, 7 deselected, 0 failed) was not re-run. No focused test was required: the failures below are visible on the success path.

`progress.md` marks Tasks 1–21 complete (through `de37c0c`). The trailing "Task 2–21: pending" list is a stale ledger. Of the deferred notes, only Task 10's "cancel after `mark_attempt_sent` cannot undo a charged sent row" must be fixed before merge. It is the Critical issue. The other deferred notes are triaged under Declined or Minor.

### Declined to judge

- HTTP 500 stays `known_transient` — SDD P1.T2 classifies it that way; 401 and 402 stay known and can be continued while scope budget remains.
- An idle `PROCESSING` book with no failures stops auto-refresh — AC-10 requires a stalled book to stop refreshing.
- `invalid_limit` is HTTP 409 — the SDD maps `ExtendRecoveryLimitError.reason_code` to 409.
- Migration backfill skips chapter 0 — counters require `chapter_number >= 1`, and the note-id pattern is chapter-based; no collision with live ids.
- An existing counter row does not merge note ids written outside `allocate_note_ids` — the counter is authoritative inside the caller's transaction and gaps are allowed; production save paths allocate inside that transaction.
- Callers still default `validation_policy` to `v1`, and `_TRANSLATION_CONTEXT_V2` still says "Политика translation-context-v2" — required so legacy operations and the v2 prompt text are not rewritten to `translation-reliability-v3`.
- Strict line preservation applies only when `validation_policy == "v2"` — AC-5; v1 books keep the old check.
- `stop_root` without `expected_generation` still refuses complete and superseded roots — a protective fence, not a hole.
- A discarded live incomplete root fails the dispatcher after siblings may have saved (`process_chapter.py`) — the SDD says not to skip the completeness check; siblings are committed and isolated.
- `scope_limit` is restart-only and is not accepted by `ExtendRecoveryLimit` — SDD Q2. Lifecycle tests that expect `calls_used == 3` and `scope_limit` match that contract.
- Book-level `recovery.reason` prefers the literal `unknown` — per-task retry buttons are gated separately, and the export page still shows the provider-failure retry.
- A live `sent` attempt does not by itself clear `can_retry` — `mark_attempt_sent` leaves the scope claimed, and `can_retry` is already false while a scope is claimed.
- A lifespan task that is already cancelled can skip `worker.cancel()` because of `asyncio.shield` — normal ASGI shutdown calls lifespan `__aexit__`, which the grace test drives; the 5 second grace is the spec.
- A falsy embedded error object is skipped, and the raise after a gateway status is unreachable — the body then fails as a format error.
- Regeneration links an already-stopped same-block root with a raw `superseded_by` update — `restart_root` would create another successor; this path is not an unknown-outcome restart and writes no billing audit.
- An unclassified local exception after a seen response is stored as `unknown` — a second model call is a new charge, so requiring acknowledgement is the conservative side; `ResultValidationError` stays known.
- Rooted format, capacity, configuration, and provider errors stay `sent` and then become known failures — they are not relabeled `unknown`.
- A `ConnectError` on plain HTTP through a non-direct proxy stays `unknown` — the proxy may already have forwarded the request.
- A crash after `mark_attempt_sent` and before any exception is observed stays `unknown` on reconcile — that send is ambiguous and must not be rolled back.
- Historical accepted results are not rewritten — `_validate_historical` uses `validation_policy='v1'`, export still accepts validators `(1, 3, 4)`, and continuing an old `translation-context-v2` operation stays on v1.
- Suite-fix `de37c0c` — Alice now emits one line per non-empty source line, which is the new v2 contract; the quality-review test asserts both correction and review after its own raw replace; the export page keeps exactly one provider-failure retry form; the analysis fake grew `begin_local_acceptance` because `READ_AS_IS` now calls it; the prompt test asserts the v2 policy sentence. Assertions were not loosened to hide a product bug.
- Domain and application forbidden-import scan (SQLAlchemy, httpx, pydantic, FastAPI, Alembic, and the banned stdlib set, including `TYPE_CHECKING`) found no matches. The architecture suite was not re-run.
- Task 3 diagnostics-lock behavior was not re-read far enough to judge a learner-facing failure.
- An unchanged segment inside a changed correction join, and `CORRECTION_VERSIONS_V3` bumping validation to 3 — snapshot equality does distinguish v3 from v2; no concrete false accept or false reject of a reader field was shown.
- `progress.md` still lists Tasks 2–21 as pending — the complete bullets and the commit chain are the ledger.

### Strengths

- A proven `unknown` translation or regeneration outcome is not auto-retried (`retry_policy.py:321`), ordinary `RetryTask` refuses it (`retry_task.py:86`), restart without `acknowledge_unknown_outcome` fails (`restart_translation.py:145`), and the audit is stored on `successor.compatibility` including `restarted_at` (`restart_translation.py:149`). `restart_root`'s signature is unchanged (`persistence.py:429`). Late writes are generation-fenced. The AC-3 test uses real SQLite.
- `BookConditions` validators stay `(1, 3, 4)` (`block_processing.py:283`, `export_book.py:521`). Current policy is `translation-reliability-v3` with validator string `v2`; legacy policies stay `v1` (`translation_policy.py:10`, `translation_policy.py:33`). The v2 prompt text is not rewritten (`context_builder.py:140`).
- Note ids are allocated inside the caller's save transaction. Migration `0018` backfills with an idempotent `MAX` upsert, skips revisions that are not on a live run, and rejects chapter 0. The AC-4/AC-11 test uses real Alembic, `foreign_keys=ON`, rollback, and concurrent threads; a rolled-back save does not burn `ln-ch03-001`.
- The FIFO limiter records `last_admitted` at admit time, so a cooldown does not burst (`rate_limit.py:186`). `hold` ignores non-finite and non-positive delays and caps at `max_hold_seconds` (`rate_limit.py:266`). Cancelling a waiter removes the ticket and wakes the next head (`rate_limit.py:204`). AC-8 drives a real limiter and a controllable clock.
- Processing marks the attempt sent in a short transaction and only then awaits HTTP (`book_runtime.py:289`). Correction HTTP is also outside its transaction. `READ_AS_IS` uses `begin_local_acceptance` and save in one transaction, with no send gap.
- Known 401/402/429/503 on a rooted processing call stay known and, while scope budget remains, the same root can be continued. AC-2 covers that path.
- Restart of one root in a multi-root dispatcher puts a failed sibling task back to pending or succeeded (`work_state.py:1217`). AC-1 isolates a parallel restart.
- The audit file (`tests/integration/processing/test_translation_reliability_audit.py`) uses real SQLite, Alembic, `ExecutorLock`, and a controllable clock. The paths it covers are not mocked green. The hole is the path it never drives, below.
- The web billing checkbox is separate from the generic `confirmed` checkbox and is shown only when `has_unknown_outcome`. The translation and regeneration restart use case still checks the acknowledgement on the server.
- Worker startup recovery honors a stop that arrived before `run_forever` (`worker.py:114`). That change in `de37c0c` is a product fix.

### Issues

#### Critical (Must Fix)

1. A proven not-sent connection failure after `mark_attempt_sent` still spends the scope budget and, at the defaults, stops the chapter so the same operation cannot be continued.

`src/linguary/composition/book_runtime.py:248`, `src/linguary/adapters/db/work_state.py:1426`, `src/linguary/adapters/db/work_state.py:1483`, `src/linguary/application/block_processing.py:914`, `src/linguary/adapters/llm/openai_compatible.py:240`, `src/linguary/application/retry_policy.py:321`, `src/linguary/application/block_processing.py:993`, `src/linguary/adapters/db/work_state.py:1375`, `src/linguary/adapters/db/work_state.py:1821`.

Processing order is limiter, then `mark_attempt_sent` (`calls_used += 1`, state `created` → `sent`), then HTTP. `ConnectError`, `ConnectTimeout`, and `PoolTimeout` are `not_sent`, except plain HTTP through a non-direct proxy, which correctly stays `unknown`. `_settle_transport` calls `cancel_unsent_attempt`, which raises unless the row is still `created`, and the error is swallowed. The comment at `book_runtime.py:251` says a charged send stays sent. The attempt remains `sent` with the charge. `scope_attempt_count` counts every non-cancelled row.

`_failure_outcome` still returns `not_sent`, but `_record_recovery_failure` rewrites that to `known` once the state is `sent` (`block_processing.py:918`). The adapter then stores `failed` and releases the scope to pending without decrementing `calls_used` (`work_state.py:1531`). Retry policy treats `not_sent` as transient, so it auto-retries up to `DEFAULT_MAX_ATTEMPTS` (3). Each retry charges again. After the third failure `_terminal_reason` sees `scope_attempt_count >= max_scope_calls` (3) while `calls_used` is still under `max_calls` (12) and stops the root with `scope_limit`. `ExtendRecoveryLimit` accepts only reason `limit` (`work_state.py:1251`). The operator must restart a chapter the provider never received.

Even with `max_attempts = 1`, each such failure permanently consumes one of the three scope slots. A crash after the failed cancel and before `record_recovery_failure` leaves state `sent`; reconcile then marks it `unknown` and demands a billing acknowledgement for a request that never connected (`work_state.py:1794`).

AC-2 and the audit test never drive this path. `_pause_then_continue` injects 503/401/402/429 and asserts `calls_used == attempts` (`test_translation_reliability_audit.py:596`), which is right for a request that was sent. The limiter-queue cancel stays `created` and is tested. Nothing asserts that a completed `ConnectError` leaves `calls_used` and the scope count unchanged.

How to fix: when the transport outcome is a proven `not_sent` after `mark_attempt_sent`, roll the charge back in `_settle_transport` before the exception reaches the retry loop. Decrement `calls_used` only when it is positive and the generation still matches, move `sent` → `cancelled`, and release the claimed scope to pending the way `cancel_unsent_attempt` does for `created`. Do not store it as a known `failed` attempt. `record_recovery_failure` already returns without charging when the row is `cancelled`. Leave proxy-HTTP connect errors, write and read timeouts, and a crash before any exception on `unknown`. Also stop labeling scope exhaustion as `limit` inside `begin_recovery_attempt` (`work_state.py:1377`); that reason would offer an extension that does not raise `max_scope_calls`. Add a regression that a post-boundary `ConnectError` does not increase `calls_used` or `scope_attempt_count`, leaves the same root continuable, and is not reconciled as `unknown_outcome` after the rollback commits.

#### Important (Should Fix)

1. A direct restart can supersede an in-flight sent call without a billing acknowledgement.

`src/linguary/application/use_cases/restart_translation.py:141`, `src/linguary/adapters/db/work_state.py:1194`, `src/linguary/application/book_overview.py:476`, `src/linguary/web/routes/processing.py:321`, `src/linguary/cli/app.py:409`.

Acknowledgement is required only when the root reason is `unknown_outcome` or `interrupted`, or some attempt is already `unknown`. A root with a claimed scope and an attempt still `sent` (the provider has the request, the response has not come back) is accepted. `restart_root` cancels only `created` and `reserved` rows and leaves the `sent` row in place, then starts a successor that will send again. The late response is generation-fenced, so it is not saved, but the provider can bill both calls. The overview hides the button while any scope is claimed (`can_restart` requires `not busy`). The HTTP route and the CLI call `RestartTranslation` directly and do not repeat that check. AC-3 only restarts while the attempt is still `created` inside the limiter, where `calls_used` is still 0.

How to fix: refuse restart while any scope is `claimed` or any attempt is `sent`, the same way the projection already does. If a restart during that window must remain possible, require `acknowledge_unknown_outcome` and keep the audit. Cover it with a server-side test, not only an overview assertion.

2. A timed-out chapter correction is still an ordinary retry, and every failure after the correction send mark is stored as unknown.

`src/linguary/composition/correction_runtime.py:419`, `src/linguary/composition/correction_runtime.py:453`, `src/linguary/application/use_cases/retry_task.py:103`, `src/linguary/web/routes/quality.py:149`.

`CorrectionProvenanceClient` moves the attempt to `sent` before `await self.inner.generate`. The rate limiter sits inside that client, so a cancel while the correction is only queued is already `sent`. The `except BaseException` handler then stores every still-`sent` failure as `unknown`: connect errors, HTTP 429, format errors, and that queued cancel. `CHAPTER_CORRECTION` is outside `_RECOVERY_TASKS` and outside `_RECOVERY_RETRY_TYPES`, so `is_task_retryable_for_recovery` returns true and the quality page keeps the ordinary retry button. A `ReadTimeout` or gateway 502/504 is a real ambiguous send. The operator can run it again from the normal button and pay twice, with no billing checkbox. The handler itself predates this branch's correction diff; the branch left it in place while adding the billing rule for translation and regeneration. The brief forbids ordinary manual retry of an unknown paid call, not only those two task types.

How to fix: classify the correction failure with the same transport outcome as processing. Roll back a proven `not_sent`, keep a known transient as a known failure, and mark `unknown` only for an ambiguous send. Put `chapter_correction` under the same ordinary-retry refusal as translation and regeneration when an attempt is `unknown`, and require the explicit acknowledgement before another send.

#### Minor (Nice to Have)

1. `acquire` drops its FIFO ticket only on `CancelledError` (`rate_limit.py:204`). Any other exception leaves a dead head, and `_signal` on a closed loop returns without waking the waiter (`rate_limit.py:263`). Later calls in that process wait forever. The default sleep does not raise, and a closed loop is a shutdown, so this is not the operator's steady state. The class docstring still says "reservations" (`rate_limit.py:54`). Remove the ticket in a `finally`, and wake the next head when the signal cannot be delivered. The closed-loop test should not set private `Task._log_destroy_pending`.

2. Non-string `message.content` with `finish_reason=length` is a format error with `truncated=False` (`openai_compatible.py:165`, `openai_compatible.py:499`). Empty content and `content is None` are handled. This shape can be retried as the same request instead of stopping as `output_truncated`. Copy the local truncated flag onto that error.

3. `RegenerateBlock` still wraps `save` with a fallback that allocates note ids in its own transaction and then calls `save` (`regenerate_block.py:169`). `chapter_num` exists on `TranslationCall.save`, so the wrapper delegates and the fallback is dead. The comment describes the opposite. Delete the fallback so a future signature change cannot allocate outside the save transaction.

### Assessment

**Ready to merge?** No

**Reasoning:** A connection failure the provider never received still consumes the scope budget and stops the chapter as `scope_limit`, so the operator cannot continue that operation. A direct restart of an in-flight send and an ordinary retry of a timed-out chapter correction can each start a second paid call without the billing acknowledgement.

### Fix report

The new regressions were run red first (14 failed). Proven `not_sent` after the send mark now rolls `calls_used` back, cancels the attempt, and releases the claimed scope; it is not stored as `failed` or reconciled as `unknown`. Scope-only exhaustion stops as `scope_limit`; a spent call budget, including when both budgets are exhausted together, stays `limit`.

An in-flight restart (claimed scope or `sent` attempt) is refused unless `acknowledge_unknown_outcome` is set, and that acknowledgement is what the existing fencing tests pass. Chapter correction uses the same transport outcomes, rolls back a proven `not_sent`, and an ordinary retry of an `unknown` correction requires the same acknowledgement. The limiter drops a FIFO ticket in `finally` and skips a closed-loop head. Non-string truncated content keeps `truncated=True`. The dead `RegenerateBlock` save wrapper is gone.

`uv run pytest tests/architecture -q`: 22 passed. `uv run pytest -m "not slow" -q`: 4496 passed, 7 deselected, 0 failed, 0 errors.
