# План реализации: Надёжность основного перевода книги (`BRIEF-translation-reliability-2026-11-22`)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Устранить блокировки продолжения, гонки нумерации заметок, потерю `finish_reason=length`, пропуск потери строк и служебных сообщений, залповый выпуск ограничителя частоты и рассинхронизацию статусов и действий в основном переводе книги и общих механизмах записи.

**Architecture:** Сохраняются локальный монолит Linguary, SQLite (WAL, `foreign_keys=ON`, `BEGIN IMMEDIATE`) и `ExecutorLock`. Для клиента обработки и повторной генерации граница отправки (`created -> sent`, проверка поколения и списание попытки) переносится внутрь — после FIFO-ограничителя частоты и непосредственно перед HTTP-транспортом (не затрагивая клиент анализа `AnalyzeChapter`); транспорт разделяет `not_sent`, `known_transient` и `unknown`; координатор главы изолирует ошибки независимых корней; номера заметок выделяются через таблицу `note_number_counters(run_id, chapter_number)` в момент сохранения; строгие проверки привязываются к новым версиям политик операций (`translation-reliability-v3`, `chapter-correction-v3`) без изменения `BookConditions.versions.validators` и истории.

**Tech Stack:** Python 3.13+, SQLAlchemy 2.0, Alembic, SQLite, httpx, FastAPI, Jinja2, pytest, pytest-asyncio, uv.

**Spec:** `.ai/develop-context/BRIEF-translation-reliability-2026-11-22/BRIEF-translation-reliability-2026-11-22.md`. Архитектура — `.ai/develop-context/BRIEF-translation-reliability-2026-11-22/ADR-translation-reliability-2026-11-22.md`. Фазы и контракты — `.ai/develop-context/BRIEF-translation-reliability-2026-11-22/SDD-backend-translation-reliability-2026-11-22.md`. Аудит — `.ai/develop-context/BRIEF-translation-reliability-2026-11-22/AUDIT-2026-11-22.md`.

## Global Constraints

- Соблюдать архитектурные границы `tests/architecture`: в `domain` и `application` запрещены сторонние пакеты (`sqlalchemy`, `httpx`, `pydantic`, `fastapi`, `alembic`) и инфраструктурная стандартная библиотека (`os`, `io`, `pathlib`, `sqlite3`, `http`, `urllib`, `socket`, `ssl`, `subprocess`), включая блоки `if TYPE_CHECKING:` и вызовы `open()`.
- Никаких HTTP/LLM-вызовов внутри открытой транзакции БД.
- Не менять `BookConditions.versions.validators` (остаётся `(1, 3, 4)`) и текст промпта `_TRANSLATION_CONTEXT_V2` в `context_builder.py`: новые строгие правила приёмки закрепляются на `TranslationPolicy` и политике исправлений.
- Не менять сигнатуру `WorkRepository.restart_root(root_id, *, expected_generation, successor, scopes, task)` (закреплена тестом `test_translation_policy.py:44,50`): аудит перезапуска передаётся через `successor.compatibility`.
- Исторические принятые результаты, номера заметок, свидетельства покрытия и экспорт уже обработанных книг не переписываются и не отвергаются задним числом.
- Неизвестный исход отправленного запроса никогда не повторяется автоматически и не снимается обычным `RetryTask`; явный перезапуск требует отдельного подтверждения `acknowledge_unknown_outcome=True` и сохраняет аудит.
- Не включать в коммиты фичи существующие незакоммиченные правки разработчика; в каждом коммите указывать `Assisted-by: <роль> (<инструмент>, <модель>)`.
- Не начинать реализацию до утверждения спецификаций и выбора режима выполнения.

## Review Focus

- Отмена задачи во время ожидания `Retry-After` в `LLMRateLimiter` (AC-8 + AC-3): попытка остаётся `cancelled`, `calls_used` и `max_scope_calls` не расходуются, `UniqueConstraint('root_id', 'ordinal')` не нарушается при следующей попытке, слот ограничителя не утекает, а если за время паузы корень был перезапущен, устаревшая корутина не отправляет HTTP-запрос.
- Известные ошибки ответа провайдера (`LLMResponseFormatError`, `finish_reason=length`, `LLMContextCapacityError`, HTTP `401`/`402`/`429`/`503`) в `ProvenanceClient.generate()` не переводят попытку в `'unknown'`, поэтому сохраняются разбиение по обрыву (AC-6), повтор без соседей и ручное продолжение с остатком бюджета (AC-2).
- Перезапуск одного корня на упавшей (`FAILED`) мультикорневой задаче-диспетчере, где второй корень уже `complete`: диспетчер переходит в `succeeded`, книга не зависает в `PROCESSING`, а Web UI не уходит в бесконечное автообновление.
- Аварийная остановка между `record_rejection` (`attempt='rejected'`) и `record_recovery_failure` (`scope='claimed'`): `reconcile_recovery_attempts()` находит и разблокирует `claimed`-область при старте под `ExecutorLock` (включая `linguary worker --once`).
- Миграция `0018_note_number_counters` (`revision='0018'`, `down_revision='0017'`) на БД с удалёнными через `ResetProcessing` запусками в `artifact_revisions.run_id` и ожидающими `correction_candidates`: при `PRAGMA foreign_keys=ON` миграция не падает, номера кандидатов учтены, повторное первичное заполнение идемпотентно.

---

### Task 1: FIFO-ограничитель частоты, пауза без залпа и безопасный `Retry-After`

Соответствует SDD `[P1.T1]`.

**Files:**
- Modify: `src/linguary/adapters/llm/rate_limit.py:44-228`
- Test: `tests/unit/llm/test_rate_limited_client.py`

**Interfaces:**
- Consumes: `LLMTransportError.retry_after_seconds`
- Produces: `LLMRateLimiter(requests_per_minute: int | None = None, *, max_concurrency: int | None = None, max_hold_seconds: float = 120.0, clock: Clock | None = None, sleep: Sleep | None = None)` с очередью FIFO в `acquire()`, безопасным `hold()` и `set_limit()`

- [ ] **Step 1: Написать падающие тесты на FIFO-паузу, отмену, бесконечный `Retry-After` и `set_limit`**

```python
@pytest.mark.asyncio
async def test_hold_spaces_three_waiters_at_20_21_22_and_fourth_at_23():
    # 60 RPM (interval=1.0), hold(20.0) при t=0; 3 ожидающих входят в t=0, 4-й в t=20.01
    # Ожидаемые моменты допуска: [20.0, 21.0, 22.0, 23.0]
```
Добавить проверки: `test_cancelled_waiter_does_not_leak_slot_or_block_queue`, `test_hold_caps_huge_and_ignores_infinite_retry_after`, `test_set_limit_recalculates_waiting_head_immediately`, `test_negative_monotonic_clock_and_closed_loop_wakeup`.

- [ ] **Step 2: Проверить падение тестов**

Run: `uv run pytest tests/unit/llm/test_rate_limited_client.py -v`
Expected: новые тесты падают (три ожидающих выходят одновременно в `20.0`, `inf` замораживает паузу).

- [ ] **Step 3: Переработать `LLMRateLimiter` в `src/linguary/adapters/llm/rate_limit.py`**

Заменить `_last_numeric_slot` на `_last_admitted: float | None = None`, `_provider_blocked_until: float | None = None` и очередь FIFO `_pacing_waiters: deque`. Только голова очереди вычисляет `deadline = max(now, (_last_admitted + _interval) if _interval and _last_admitted is not None else now, _provider_blocked_until or now)` и после пробуждения повторно проверяет условия под `_lock`, фиксируя `_last_admitted = now` в момент фактического допуска и будя следующего. При `CancelledError` удалять запись из очереди и будить нового голову, если отменён первый. В `hold(seconds)` игнорировать нечисловые, `not math.isfinite(seconds)` и `seconds <= 0`, ограничивать сверху `min(seconds, self._max_hold_seconds)`. В `set_limit()` будить голову очереди. В `_wake_waiters_locked()` при `RuntimeError` на `call_soon_threadsafe` откатывать `_in_flight -= 1` и продолжать цикл.

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest tests/unit/llm/test_rate_limited_client.py -v`
Expected: все тесты зелёные.

- [ ] **Step 5: Коммит**

```bash
git add src/linguary/adapters/llm/rate_limit.py tests/unit/llm/test_rate_limited_client.py
git commit -m "fix(llm): enforce fifo pacing after provider cooldown and cap retry-after" -m "Assisted-by: implementer (<tool>, <model>)"
```

---

### Task 2: Типизация исходов HTTP-транспорта, `finish_reason=length` и отказ провайдера

Соответствует SDD `[P1.T2]`.

**Files:**
- Modify: `src/linguary/application/ports/llm.py:60-72`
- Modify: `src/linguary/adapters/llm/openai_compatible.py:115-430`
- Modify: `src/linguary/adapters/llm/payloads.py:200-295`
- Test: `tests/unit/llm/test_openai_compatible_client.py`

**Interfaces:**
- Consumes: `Violation`, `LLMResponseKind`
- Produces: `TransportOutcome = Literal["not_sent", "known_transient", "unknown"]`, `LLMTransportError(..., outcome: TransportOutcome = "known_transient")`, гарантированные `Violation("provider_truncated", ...)`, `Violation("provider_refusal", ...)`, `Violation("provider_content_filter", ...)`

- [ ] **Step 1: Написать падающие тесты**

В `tests/unit/llm/test_openai_compatible_client.py` добавить:
- `test_transport_outcome_classification_not_sent_known_unknown` (`ConnectError`/`ConnectTimeout`/`PoolTimeout` -> `"not_sent"`, незашифрованный HTTP через прокси -> `"unknown"`; HTTP `429`/`500`/`503` -> `"known_transient"`; `ReadTimeout`/`WriteTimeout`/`RemoteProtocolError`/HTTP `502`/`504`/`524` -> `"unknown"`);
- `test_length_truncation_preserved_on_null_content_and_dto_errors` (`finish_reason="length"` + `content=None`, а также ошибка `_to_dto`/`HelpEvidenceSchema` с `truncated=True` -> `error.truncated is True` и `error.violation.rule_code == "provider_truncated"`);
- `test_message_refusal_and_content_filter_detected_for_translation`;
- `test_openrouter_200_error_and_coded_context_overflow`;
- `test_retry_after_rejects_infinity`.

- [ ] **Step 2: Проверить падение тестов**

Run: `uv run pytest tests/unit/llm/test_openai_compatible_client.py -k "transport_outcome or null_content or refusal_and_content_filter or openrouter_200 or rejects_infinity" -v`
Expected: FAIL.

- [ ] **Step 3: Реализовать изменения в `ports/llm.py`, `openai_compatible.py`, `payloads.py`**

1. В `ports/llm.py`: добавить `type TransportOutcome = Literal["not_sent", "known_transient", "unknown"]`, параметр `outcome: TransportOutcome = "known_transient"` в `LLMTransportError.__init__` и атрибут `self.outcome`.
2. В `openai_compatible.py`:
   - В `_post()`: разделить `httpx.ConnectError`/`ConnectTimeout`/`PoolTimeout` (если `urlsplit(self._settings.base_url).scheme == "http"` и `self._settings.proxy_mode != "direct"` -> `outcome="unknown"`, иначе `"not_sent"`) и `httpx.WriteTimeout`/`WriteError`/`ReadTimeout`/`ReadError`/`RemoteProtocolError`/`CloseError`/`DecodingError`/`TimeoutException`/`HTTPError` (`outcome="unknown"`).
   - В `_raise_for_status()`: HTTP `429`/`500`/`503` -> `outcome="known_transient"`, HTTP `502`/`504`/`524` (и прочие `>= 500`, кроме `500`/`503`) -> `outcome="unknown"`.
   - В `_is_context_capacity_error()`: проверять `_CONTEXT_OVERFLOW_PHRASES` в `error.message` независимо от наличия `code`.
   - В `_retry_after()`: добавить проверку `not math.isfinite(seconds)`.
   - В `generate()`: при отсутствии `choices` и наличии `document.get("error")` обработать как ошибку шлюза; проверять `message.get("refusal")` для всех `LLMResponseKind` (`truncated=False`, `Violation("provider_refusal", ...)`); проверять `finish_reason == "content_filter"` (`truncated=False`, `Violation("provider_content_filter", ...)`); при `_content(choice)` с `content is None` и `truncated=True` поднимать `LLMResponseFormatError(..., truncated=True, violation=Violation("provider_truncated", request.kind.value, ...))`.
3. В `payloads.py:257-288`: в ветках `_to_dto`, `byr_evidence_items`, `content_evidence_items` при `truncated=True` задавать `rule_code="provider_truncated"`.

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest tests/unit/llm/ -v`
Expected: все тесты зелёные.

- [ ] **Step 5: Коммит**

```bash
git add src/linguary/application/ports/llm.py src/linguary/adapters/llm/openai_compatible.py src/linguary/adapters/llm/payloads.py tests/unit/llm/test_openai_compatible_client.py
git commit -m "fix(llm): classify transport outcomes and preserve truncation and refusals" -m "Assisted-by: implementer (<tool>, <model>)"
```

---

### Task 3: Политика повторов для `unknown`-исходов и устойчивость `LoggingLLMClient` / `Diagnostics`

Соответствует SDD `[P1.T3]`.

**Files:**
- Modify: `src/linguary/application/retry_policy.py:222-333, 510-524`
- Modify: `src/linguary/adapters/llm/logging.py:52-224`
- Modify: `src/linguary/application/diagnostics.py:65-95`
- Test: `tests/unit/application/test_retry_policy.py`
- Create: `tests/unit/llm/test_logging_cancellation.py`

**Interfaces:**
- Consumes: `LLMTransportError.outcome`, `Violation.rule_code`, `PersistenceError`
- Produces: `RetryPolicy.decide()` -> `GIVE_UP` для `unknown`/`provider_refusal`/`provider_content_filter` и неклассифицированных ошибок; `LoggingLLMClient` записывает `CancelledError` и не блокирует процесс необратимо при временной занятости SQLite

- [ ] **Step 1: Написать падающие тесты**

В `test_retry_policy.py` добавить `test_retry_policy_never_retries_unknown_transport_or_provider_refusal` (включая проверку неклассифицированного исключения без автоповтора при сохранении существующих тестов `classify_error`). В `tests/unit/llm/test_logging_cancellation.py` добавить `test_logging_client_records_cancelled_call_and_reraises` и `test_transient_sqlite_busy_does_not_permanently_poison_diagnostics`.

- [ ] **Step 2: Проверить падение тестов**

Run: `uv run pytest tests/unit/application/test_retry_policy.py tests/unit/llm/test_logging_cancellation.py -v`
Expected: новые тесты падают.

- [ ] **Step 3: Реализовать изменения**

1. В `retry_policy.py`: в `RetryPolicy.decide()` перед веткой `is_retried_automatically` возвращать `RetryDecision(action=RetryAction.GIVE_UP, error=task_error)` для `isinstance(error, LLMTransportError) and error.outcome == "unknown"`, для `violation.rule_code in {"provider_refusal", "provider_content_filter"}` и для неклассифицированных исключений, не входящих в `_ERROR_TYPES` (проверить совместимость с существующими тестами `test_retry_policy.py`); в `violation_of()` для `LLMTransportError` сопоставлять `"not_sent" -> "transport_not_sent"`, `"known_transient" -> "transport_transient"`, `"unknown" -> "transport_unknown"`.
2. В `logging.py`: в `generate()` добавить `except asyncio.CancelledError as error:` — записать лог (`success=False`, `decision='unknown'`) и повторно пробросить `raise`; в `_log()` при временной ошибке блокировки `PersistenceError` пробросить ошибку без установки `self._observation_failed = True` и без `diagnostics.fail()`.
3. В `diagnostics.py`: в `finalize_attempt()` и `purge_expired()` не выставлять необратимый `self._failed = True` на временной ошибке блокировки `PersistenceError` (пробрасывать её наверх).

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest tests/unit/application/test_retry_policy.py tests/unit/llm/test_logging_cancellation.py -v`
Expected: все тесты зелёные.

- [ ] **Step 5: Коммит**

```bash
git add src/linguary/application/retry_policy.py src/linguary/adapters/llm/logging.py src/linguary/application/diagnostics.py tests/unit/application/test_retry_policy.py tests/unit/llm/test_logging_cancellation.py
git commit -m "fix(llm): stop auto-retry on unknown outcomes and harden call logging" -m "Assisted-by: implementer (<tool>, <model>)"
```

---

### Task 4: Модель `NoteNumberCounterRow` и миграция `0018_note_number_counters.py`

Соответствует SDD `[P2.T1]`.

**Files:**
- Modify: `src/linguary/adapters/db/models.py`
- Create: `alembic/versions/0018_note_number_counters.py`
- Create: `tests/integration/db/test_note_number_counters_migration.py`

**Interfaces:**
- Consumes: `Base`, `NAMING_CONVENTION`, `processing_runs`, `language_notes`, `correction_candidates`, `artifact_revisions`
- Produces: `NoteNumberCounterRow` (`note_number_counters(run_id, chapter_number, last_ordinal)`), `backfill_note_number_counters(connection)` в `0018_note_number_counters.py` (`revision = '0018'`, `down_revision = '0017'`)

- [ ] **Step 1: Написать падающие тесты миграции**

В `tests/integration/db/test_note_number_counters_migration.py` создать тесты на реальной SQLite с миграцией `0017 -> 0018`:
- `test_migration_0018_backfills_notes_candidates_and_live_revisions`: заполнить `language_notes` (с пропусками `ln-ch01-001`, `ln-ch01-007`), `correction_candidates.payload_json` (`ln-ch01-009` в `result` и `ln-ch02-004` в `current_result`), `artifact_revisions.body_json` (`ln-ch01-012` для существующего `run_id`, нетиповой `custom-note` игнорируется) → `last_ordinal` равен `12` для главы 1 и `4` для главы 2.
- `test_migration_0018_ignores_orphaned_revision_runs_under_fk_on`: вставить `artifact_revisions` с несуществующим `run_id` при `PRAGMA foreign_keys=ON` → миграция проходит без `IntegrityError`.
- `test_migration_0018_backfill_is_idempotent_and_matches_models_and_downgrades`: повторный вызов `backfill_note_number_counters` не уменьшает повышенный счётчик; `downgrade("0017")` чисто удаляет таблицу; `upgrade("head")` снова поднимает и проходит `test_alembic_migrations.py`.

- [ ] **Step 2: Проверить падение тестов**

Run: `uv run pytest tests/integration/db/test_note_number_counters_migration.py -v`
Expected: FAIL (нет миграции `0018` и модели).

- [ ] **Step 3: Добавить `NoteNumberCounterRow` и миграцию `0018_note_number_counters.py`**

1. В `models.py`: добавить `NoteNumberCounterRow(__tablename__ = "note_number_counters")` с `run_id` (`ForeignKey("processing_runs.id", ondelete="CASCADE")`, PK), `chapter_number` (`Integer`, PK), `last_ordinal` (`Integer`, `nullable=False`), `CheckConstraint("chapter_number >= 1", name="chapter_number_positive")`, `CheckConstraint("last_ordinal >= 0", name="last_ordinal_nonnegative")`. Экспортировать в `__all__`.
2. В `alembic/versions/0018_note_number_counters.py` строго указать `revision = "0018"` и `down_revision = "0017"` (как в `0017_learner_layout_segments.py:7-8`): создать таблицу, реализовать `backfill_note_number_counters(connection)` с regex `^ln-ch(\d+)-(\d+)$`, фильтром `artifact_revisions.run_id IN (SELECT id FROM processing_runs)` и вставкой с обновлением `INSERT INTO note_number_counters ... ON CONFLICT (run_id, chapter_number) DO UPDATE SET last_ordinal = MAX(last_ordinal, excluded.last_ordinal)`. В `downgrade()` удалить таблицу.

- [ ] **Step 4: Прогнать тесты миграций**

Run: `uv run pytest tests/integration/db/test_note_number_counters_migration.py tests/integration/db/test_alembic_migrations.py -v`
Expected: все тесты зелёные (отсутствие расхождений со схемой моделей).

- [ ] **Step 5: Коммит**

```bash
git add src/linguary/adapters/db/models.py alembic/versions/0018_note_number_counters.py tests/integration/db/test_note_number_counters_migration.py
git commit -m "feat(db): add note_number_counters table and idempotent backfill migration" -m "Assisted-by: implementer (<tool>, <model>)"
```

---

### Task 5: Атомарное выделение номеров заметок (`allocate_note_ids`)

Соответствует SDD `[P2.T2]`.

**Files:**
- Modify: `src/linguary/application/ports/persistence.py`
- Modify: `src/linguary/adapters/db/repositories.py`
- Create: `tests/integration/db/test_note_id_allocation.py`

**Interfaces:**
- Consumes: `NoteNumberCounterRow`, `build_language_note_id`
- Produces: `BlockResultRepository.allocate_note_ids(self, run_id: str, chapter_number: int, count: int) -> tuple[str, ...]`

- [ ] **Step 1: Написать падающие тесты**

В `tests/integration/db/test_note_id_allocation.py` добавить `test_allocate_note_ids_monotonic_unique_across_concurrent_transactions` (несколько параллельных потоков с `SqlAlchemyUnitOfWork.transaction()` на реальной SQLite -> 0 дубликатов), `test_allocate_note_ids_lazy_seeds_from_existing_notes_and_candidates`, `test_allocate_note_ids_zero_count_and_cascade_on_run_delete`.

- [ ] **Step 2: Проверить падение тестов**

Run: `uv run pytest tests/integration/db/test_note_id_allocation.py -v`
Expected: FAIL (`AttributeError: allocate_note_ids`).

- [ ] **Step 3: Реализовать `allocate_note_ids` в `ports/persistence.py` и `adapters/db/repositories.py`**

При `count == 0` вернуть `()`. При `count < 0` или `chapter_number < 1` бросить `ValueError`. Внутри текущей транзакции `BEGIN IMMEDIATE`: прочитать `NoteNumberCounterRow(run_id, chapter_number)`; если строки нет, вычислить `base` как максимум из `language_notes`, `correction_candidates.payload_json` и `artifact_revisions.body_json` для `(run_id, chapter_number)` (regex `^ln-ch(\d+)-(\d+)$`), вставить строку с `last_ordinal = base + count`; если строка есть, обновить `last_ordinal = row.last_ordinal + count` и `flush()`. Вернуть кортеж `build_language_note_id(chapter_number=chapter_number, ordinal=base + i)` для `i in range(1, count + 1)`.

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest tests/integration/db/test_note_id_allocation.py -v`
Expected: все тесты зелёные.

- [ ] **Step 5: Коммит**

```bash
git add src/linguary/application/ports/persistence.py src/linguary/adapters/db/repositories.py tests/integration/db/test_note_id_allocation.py
git commit -m "feat(db): implement atomic per-run per-chapter note id allocation" -m "Assisted-by: implementer (<tool>, <model>)"
```

---

### Task 6: Первичный обрыв `truncated=True`, `TranslationPolicy` v3 и политика исправлений v3

Соответствует SDD `[P3.T1]`.

**Files:**
- Modify: `src/linguary/application/translation_policy.py`
- Modify: `src/linguary/domain/correction.py`
- Modify: `src/linguary/application/validators/result.py:117-235`
- Test: `tests/unit/application/test_result_validators.py`
- Test: `tests/unit/application/test_translation_policy.py`
- Test: `tests/integration/processing/test_translation_context_accuracy.py`

**Interfaces:**
- Consumes: `truncation_diagnostic`
- Produces: `TranslationPolicy.validation_policy`, `CURRENT_TRANSLATION_POLICY = "translation-reliability-v3"`, `CORRECTION_POLICY_VERSION_V3 = "chapter-correction-v3"`, `is_candidate_correction_policy(version: str | None) -> bool`, `TranslationResultValidator.validate(..., validation_policy: str = "v1", protected_quotes: Sequence[str] = (), excluded_service_texts: Mapping[str, frozenset[str]] | None = None)`

- [ ] **Step 1: Написать падающие тесты и обновить ожидания по текущей версии политики**

1. В `tests/unit/application/test_result_validators.py` добавить:
   - `test_truncated_true_precedes_coverage_and_structure_violations` (валидный JSON с `truncated=True` и пропущенным блоком / неверным режимом / пустым текстом -> `ResultValidationError` с `violation.rule_code == "provider_truncated"`);
   - `test_translation_policy_v3_and_v2_legacy_coexist`;
   - `test_candidate_correction_policy_helper_covers_v2_and_v3`.
2. Обновить `tests/unit/application/test_translation_policy.py:25-27` (добавить `"translation-reliability-v3"` в ожидаемое множество `TRANSLATION_POLICIES`) и `tests/integration/processing/test_translation_context_accuracy.py:115` (ожидать `CURRENT_TRANSLATION_POLICY`; строка L121 с текстом промпта `Политика translation-context-v2.` остаётся неизменной).

- [ ] **Step 2: Проверить падение тестов**

Run: `uv run pytest tests/unit/application/test_result_validators.py tests/unit/application/test_translation_policy.py tests/integration/processing/test_translation_context_accuracy.py -k "truncated_true_precedes or policy_v3 or candidate_correction_policy or test_translation_policy or test_translation_context_accuracy" -v`
Expected: FAIL.

- [ ] **Step 3: Реализовать изменения**

1. В `translation_policy.py`: добавить `validation_policy: str = "v1"` в `TranslationPolicy`, сохранить `"translation-context-legacy-v1"` и `"translation-context-v2"` с `validation_policy="v1"`, добавить `CURRENT_TRANSLATION_POLICY = "translation-reliability-v3"` с `validation_policy="v2"`.
2. В `domain/correction.py`: добавить `CORRECTION_POLICY_VERSION_V3 = "chapter-correction-v3"` и `is_candidate_correction_policy(version: str | None) -> bool`, расширить допустимое множество в `correction_policy()`.
3. В `validators/result.py`: переместить проверку `if truncated:` (`truncation_diagnostic(..., truncated=True)`) в начало `validate()` (сразу после расчёта `expected`, **до** `validate_assistance_body`, `coverage_violation`, `treatment_violation`, `content_violation`, `paragraph_violation`, `_foreign_marker_violation`). Принять необязательные параметры `validation_policy: str = "v1"`, `protected_quotes: Sequence[str] = ()`, `excluded_service_texts: Mapping[str, frozenset[str]] | None = None`; в поблочном цикле передавать `include_assistances=False`, а `payload.fragment_assistances` при `validation_policy == "v2"` проверять **отдельным шагом вне поблочного цикла** (с `Violation(..., scope="unit")` или по владельцу сложности), чтобы ошибка помощи не приписывалась чужому блоку.

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest tests/unit/application/test_result_validators.py tests/unit/application/test_translation_policy.py tests/integration/processing/test_translation_context_accuracy.py -v`
Expected: все тесты зелёные.

- [ ] **Step 5: Коммит**

```bash
git add src/linguary/application/translation_policy.py src/linguary/domain/correction.py src/linguary/application/validators/result.py tests/unit/application/test_result_validators.py tests/unit/application/test_translation_policy.py tests/integration/processing/test_translation_context_accuracy.py
git commit -m "feat(validation): prioritize truncation check and version acceptance policies" -m "Assisted-by: implementer (<tool>, <model>)"
```

---

### Task 7: Запрет потери строк, защита стихов и хвостовые маркеры сносок

Соответствует SDD `[P3.T2]`.

**Files:**
- Modify: `src/linguary/application/validators/paragraphs.py`
- Modify: `src/linguary/application/validators/truncation.py:111-115`
- Test: `tests/unit/application/test_result_validators.py`

**Interfaces:**
- Consumes: `TranslationResultValidator.validate(..., validation_policy=...)`, `without_footnote_markers`
- Produces: `paragraph_violation(..., strict_line_preservation: bool = False)`

- [ ] **Step 1: Написать падающие тесты**

В `tests/unit/application/test_result_validators.py` добавить:
- `test_strict_paragraphs_rejects_first_line_only_three_line_quote`;
- `test_strict_paragraphs_preserves_dialogue_split_and_partial_segments`;
- `test_strict_poem_rejects_blank_line_substitution` (`"L1\nL2\nL3"` -> `"С1\n\nС3"` отклоняется при `validation_policy="v2"`);
- `test_legacy_v1_policy_allows_historical_line_count_decrease`;
- `test_truncation_ignores_trailing_footnote_markers` (`"He left.[^1]"`).

- [ ] **Step 2: Проверить падение тестов**

Run: `uv run pytest tests/unit/application/test_result_validators.py -k "strict_paragraphs or strict_poem or legacy_v1_policy or trailing_footnote_markers" -v`
Expected: FAIL.

- [ ] **Step 3: Реализовать изменения в `paragraphs.py` и `truncation.py`**

1. В `paragraphs.py`: добавить `strict_line_preservation: bool = False` в `_check_paragraphs`, `_poem_line_violation`, `_block_paragraph_violation`, `paragraph_violation`. При `strict_line_preservation=True` отклонять `translated_count < source_count` в `_check_paragraphs`, а в `_poem_line_violation` дополнительно проверять сохранение числа непустых строк.
2. В `truncation.py:_ends_with_punctuation`: предварительно снимать маркеры сносок через `without_footnote_markers(text)` из `linguary.domain`.

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest tests/unit/application/test_result_validators.py -v`
Expected: все тесты зелёные.

- [ ] **Step 5: Коммит**

```bash
git add src/linguary/application/validators/paragraphs.py src/linguary/application/validators/truncation.py tests/unit/application/test_result_validators.py
git commit -m "fix(validation): reject dropped lines under strict policy and fix footnote tail check" -m "Assisted-by: implementer (<tool>, <model>)"
```

---

### Task 8: Служебные сообщения во всех читательских полях и включение снимка `chapter-correction-v3`

Соответствует SDD `[P3.T3]`.

**Files:**
- Modify: `src/linguary/application/validators/service_messages.py`
- Modify: `src/linguary/application/correction_dto.py:15-30, 200-240`
- Modify: `src/linguary/composition/correction_runtime.py:85-105`
- Create: `tests/unit/application/test_service_message_validation.py`

**Interfaces:**
- Consumes: `source_quote_fragments`, `CORRECTION_POLICY_VERSION_V3`, `is_candidate_correction_policy`
- Produces: `service_message_violation(payload, *, strict_fields: bool = False, protected_quotes: Sequence[str] = (), excluded_texts: Mapping[str, frozenset[str]] | None = None, include_assistances: bool = True) -> str | None`, `unchanged_reader_texts(previous: BlockProcessingResult) -> frozenset[str]`, `CORRECTION_VERSIONS_V3` и поддержка V2/V3 в `CorrectionSnapshotV2`, `parse_correction_snapshot()`, `correction_snapshot()`

- [ ] **Step 1: Написать падающие тесты**

В `tests/unit/application/test_service_message_validation.py` добавить:
- `test_service_messages_rejects_refusal_in_english_help_and_translated_excerpt`;
- `test_service_messages_normalizes_curly_apostrophes` (`"I’m sorry, but I cannot..."`, `"I can’t fulfill..."`);
- `test_service_messages_allows_verbatim_source_quotes`;
- `test_correction_checks_only_changed_texts_and_parses_v2_and_v3_snapshots` (проверяет, что неизменённые перенесённые заметки/сегменты не отвергаются при добавлении новой заметки в тот же блок, а `correction_snapshot()` создаёт `CORRECTION_POLICY_VERSION_V3` и `parse_correction_snapshot()` читает оба варианта V2 и V3).

- [ ] **Step 2: Проверить падение тестов**

Run: `uv run pytest tests/unit/application/test_service_message_validation.py -v`
Expected: FAIL.

- [ ] **Step 3: Реализовать изменения в `service_messages.py`, `correction_dto.py` и `correction_runtime.py`**

1. В `service_messages.py`: нормализовать типографские апострофы (`’`, `‘`, `ʼ`, `‛` -> `'`) перед поиском; маскировать подстроки из `protected_quotes` (от длинных к коротким) до `_marker_in`; при `strict_fields=True` проверять `translated_text`, `block.segments[].translated_text`, `glossary_hints`, `language_notes.note`, `language_notes.translated_excerpt` и (при `include_assistances=True`) `payload.fragment_assistances[].text`; пропускать точные неизменённые текстовые значения, присутствующие в `excluded_texts.get(block_id, frozenset())`.
2. В `correction_dto.py`: добавить `unchanged_reader_texts(previous: BlockProcessingResult) -> frozenset[str]` (собирает все существующие читательские строки из `previous`); добавить `CORRECTION_VERSIONS_V3` и разрешить `is_candidate_correction_policy(policy_version)` (с допустимыми версиями V2/V3) в `CorrectionSnapshotV2.__post_init__` и `parse_correction_snapshot()`.
3. В `correction_runtime.py:92-98`: создавать для новых запусков снимок с `policy_version=CORRECTION_POLICY_VERSION_V3` и `versions=CORRECTION_VERSIONS_V3`.

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest tests/unit/application/test_service_message_validation.py tests/unit/application/test_result_validators.py -v`
Expected: все тесты зелёные.

- [ ] **Step 5: Коммит**

```bash
git add src/linguary/application/validators/service_messages.py src/linguary/application/correction_dto.py src/linguary/composition/correction_runtime.py tests/unit/application/test_service_message_validation.py
git commit -m "fix(validation): cover all reader fields for service messages and enable correction v3 snapshots" -m "Assisted-by: implementer (<tool>, <model>)"
```

---

### Task 9: Порт и реализация `work_state.py`: ограждение поколений, честный `unknown` и `reconcile`

Соответствует SDD `[P4.T1]`.

**Files:**
- Modify: `src/linguary/application/ports/persistence.py:410-450`
- Modify: `src/linguary/adapters/db/work_state.py:201, 339, 372, 935, 1058, 1132-1626`
- Test: `tests/integration/db/test_translation_operation_ownership.py`
- Create: `tests/integration/db/test_recovery_work_state_reliability.py`

**Interfaces:**
- Consumes: `is_candidate_correction_policy`
- Produces: в `WorkRepository` и `SqlAlchemyWorkRepository`: `mark_attempt_sent(attempt_id, *, expected_root_generation)`, `begin_local_acceptance(...)`, `cancel_unsent_attempt(attempt_id)`, `stop_root(root_id, reason, *, expected_generation=None)`, `pause_root(root_id, reason, *, expected_generation)`, `record_recovery_failure(attempt_id, violation, *, outcome="known")`, `split_scope(..., expected_root_generation=None)`, неизменная сигнатура `restart_root(root_id, *, expected_generation, successor, scopes, task)` с сохранением `successor.compatibility` и очисткой `FAILED`-диспетчера, исправленный `reconcile_recovery_attempts()`

- [ ] **Step 1: Написать падающие тесты**

В `tests/integration/db/test_recovery_work_state_reliability.py` добавить:
- `test_stop_root_and_split_scope_reject_stale_generation`;
- `test_record_recovery_failure_never_downgrades_unknown_to_failed`;
- `test_cancel_unsent_attempt_keeps_unique_ordinal_and_budget` (отмена попытки `ordinal=1` -> следующая получает `ordinal=2` без `IntegrityError`, `calls_used` не сгорает, `scope_attempt_count` и выборка двух подряд `rejected` не считают `'cancelled'`);
- `test_restart_root_cleans_up_failed_multiroot_dispatcher_and_persists_audit`;
- `test_reconcile_recovery_attempts_recovers_claimed_scope_with_rejected_attempt`.

- [ ] **Step 2: Проверить падение тестов**

Run: `uv run pytest tests/integration/db/test_recovery_work_state_reliability.py -v`
Expected: FAIL.

- [ ] **Step 3: Реализовать изменения в `ports/persistence.py` и `adapters/db/work_state.py`**

1. Заменить 5 жёстких сравнений `== CORRECTION_POLICY_VERSION_V2` (`L201, 339, 372, 935, 1058`) на `is_candidate_correction_policy(...)`.
2. В `begin_recovery_attempt`: вычислять `ordinal = SELECT COALESCE(MAX(ordinal), 0) + 1 FROM llm_attempts WHERE root_id = :root_id`, считать `scope_attempt_count` по `LLMAttemptRow.state != 'cancelled'`, проверять лимиты, но `calls_used` ещё не увеличивать.
3. Добавить `mark_attempt_sent(attempt_id, *, expected_root_generation: int)`: атомарно проверить владение, `superseded_by IS NULL`, `root.generation == expected_root_generation`, лимиты (`calls_used < max_calls + extra_calls` и `scope_attempt_count < max_scope_calls` без `'cancelled'`), увеличить `calls_used += 1`, выставить `root.status='running', root.reason=None` и перевести `attempt` `created -> sent` (НЕ меняя `ordinal`).
4. Добавить `begin_local_acceptance(root_id, scope_id, *, task_id, run_id, stage, assigned_model, prompt_hash, policy_version)`: в текущей транзакции проверить владение и поколение, занять `scope`, создать `CallAttempt` со `state='validated'` и монотонным `ordinal = COALESCE(MAX(ordinal), 0) + 1` **без** увеличения `calls_used`, вернуть `CallContext`.
5. Добавить `cancel_unsent_attempt(attempt_id)`: перевести `created -> cancelled`, вернуть `scope` `claimed -> pending` (с увеличением `scope.generation`), не трогая `calls_used`.
6. В `stop_root` и `pause_root`: проверять `generation == expected_generation` (если задан), `superseded_by IS NULL`, `status != 'complete'`, и увеличивать `generation += 1`. В `split_scope`: проверять `root.generation`/`superseded_by`/`status` и увеличивать `scope.generation += 1`.
7. В `record_recovery_failure(..., outcome="known")`: при `outcome == "unknown"` или `attempt.state == "unknown"` перевести `attempt -> 'unknown'`, `scope -> 'stopped'`, `root -> ('stopped', 'unknown_outcome')` с увеличением поколений и вернуть `False`; при `outcome == "not_sent"` делегировать в `cancel_unsent_attempt`; в выборке двух последних попыток для разбиения (`L1379`) добавить `.where(LLMAttemptRow.state != 'cancelled')`.
8. В `restart_root` (сохранив точную сигнатуру без новых параметров): `self.add_root(successor)` уже сохраняет `successor.compatibility`; после удаления `root_id` из `operation_refs` диспетчера `T1`, если `T1` был `failed` и все оставшиеся корни уже `complete` — перевести `T1` в `succeeded`; если среди оставшихся есть повторяемые незавершённые — вернуть `T1` в `pending` с очисткой ошибки.
9. В `reconcile_recovery_attempts()`: обрабатывать также все `RecoveryScopeRow.status == 'claimed'` (даже если их последняя попытка `'rejected'` или `'failed'`), группировать обновление `root.generation` один раз на корень, пропускать уже финализированные `decision in ('interrupted', 'stale')` строки `unknown`.
10. Обновить сигнатуры протокола `WorkRepository` в `src/linguary/application/ports/persistence.py`.

- [ ] **Step 4: Прогнать DB-тесты и тест контракта политик**

Run: `uv run pytest tests/integration/db/test_recovery_work_state_reliability.py tests/integration/db/test_translation_operation_ownership.py tests/unit/application/test_translation_policy.py -v`
Expected: все тесты зелёные.

- [ ] **Step 5: Коммит**

```bash
git add src/linguary/application/ports/persistence.py src/linguary/adapters/db/work_state.py tests/integration/db/test_translation_operation_ownership.py tests/integration/db/test_recovery_work_state_reliability.py
git commit -m "fix(work-state): fence root mutations, preserve unknown outcomes, and fix reconcile" -m "Assisted-by: implementer (<tool>, <model>)"
```

---

### Task 10: Перестановка декораторов для `processing_client` и восстановление в `drain_tasks`

Соответствует SDD `[P4.T2]`.

**Files:**
- Modify: `src/linguary/composition/book_runtime.py:215-285`
- Modify: `src/linguary/composition/container.py:385-425, 619-640, 875-888`
- Test: `tests/integration/processing/test_recovery_lifecycle.py`
- Create: `tests/integration/processing/test_send_boundary_and_rate_limit.py`

**Interfaces:**
- Consumes: `work.mark_attempt_sent`, `work.cancel_unsent_attempt`, `LLMTransportError.outcome`
- Produces: для `processing_client` порядок `RateLimitedLLMClient -> ProvenanceClient -> LoggingLLMClient -> OpenAICompatibleLLMClient` (не ломая `analysis_client` и `StageRoutedLLMClient`); восстановление в `drain_tasks(application)` под `ExecutorLock`

- [ ] **Step 1: Написать падающие тесты и обновить `test_recovery_lifecycle.py:124-147`**

В `tests/integration/processing/test_send_boundary_and_rate_limit.py` добавить:
- `test_cancel_while_waiting_in_rate_limiter_stays_not_sent_without_budget_burn`;
- `test_restarted_root_waiting_in_rate_limiter_never_sends_http`;
- `test_known_format_and_429_errors_do_not_turn_attempt_into_unknown` (проверяет, что `LLMResponseFormatError`, `LLMContextCapacityError`, `LLMConfigurationError`, `LLMProviderError` и HTTP `429`/`503` оставляют попытку в `'sent'` для дальнейшей обработки, а НЕ переводят в `'unknown'`);
- `test_drain_tasks_recovers_interrupted_when_alone_under_executor_lock`.
В `tests/integration/processing/test_recovery_lifecycle.py:124-147` (`test_crash_boundaries_do_not_reset_counter_or_repeat_unknown`): перевести переход `created -> sent` на `mark_attempt_sent` и ожидать `calls_used == 1` при отменённой `created`-попытке.

- [ ] **Step 2: Проверить падение тестов**

Run: `uv run pytest tests/integration/processing/test_send_boundary_and_rate_limit.py -v`
Expected: FAIL.

- [ ] **Step 3: Реализовать изменения в `book_runtime.py` и `container.py`**

1. В `container.py`: вынести вспомогательный `_logged_client(...) -> LoggingLLMClient` (без `RateLimitedLLMClient`). Для `processing_client` (используемого в `ProcessChapter` и `RegenerateBlock`) собирать `RateLimitedLLMClient(ProvenanceClient(_logged_client(...), ...), limiter=rate_limiter)`. Клиент анализа (`analysis_client` с `StageRoutedLLMClient` и прямым доступом к `attempts`/`reject_last`) не перестраивать. Проверить, что `_collect_context_clients` находит `ProvenanceClient` внутри `RateLimitedLLMClient._inner`.
2. В `book_runtime.py:ProvenanceClient.generate()`: вызывать `transaction.work.mark_attempt_sent(attempt.id, expected_root_generation=...)` (после выхода из `RateLimitedLLMClient`, непосредственно перед `self.inner.generate(request)`). В обработке исключений после `mark_attempt_sent`:
   - `except LLMTransportError as error:` при `error.outcome == "not_sent"` вызывать `cancel_unsent_attempt(attempt.id)` (для корневых попыток); при `error.outcome == "known_transient"` оставить `'sent'` (не переводить в `'unknown'`); при `error.outcome == "unknown"` перевести `sent -> unknown`; затем `raise` (чтобы внешний `RateLimitedLLMClient` получил `retry_after_seconds` и вызвал `hold()`).
   - `except (LLMResponseFormatError, LLMContextCapacityError, LLMConfigurationError, LLMProviderError):` оставить попытку в `'sent'` (НЕ переводить в `'unknown'`) и `raise`.
   - `except BaseException:` (включая `asyncio.CancelledError` и неклассифицированные исключения) перевести `sent -> unknown` и `raise`.
3. В `container.py:drain_tasks(application)`: если `lock.join()` вернул `True` (`alone`), вызвать `await application.runner.recover_interrupted()` до `lock.share()` (аналогично `run_worker` в `container.py:868`). Проверить все существующие интеграционные тесты, вызывающие `drain_tasks()`.

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest tests/integration/processing/test_send_boundary_and_rate_limit.py tests/integration/processing/test_recovery_commit.py tests/integration/processing/test_recovery_lifecycle.py -v`
Expected: все тесты зелёные.

- [ ] **Step 5: Коммит**

```bash
git add src/linguary/composition/book_runtime.py src/linguary/composition/container.py tests/integration/processing/test_send_boundary_and_rate_limit.py tests/integration/processing/test_recovery_lifecycle.py
git commit -m "fix(runtime): move send boundary after rate limiter for processing and recover in drain_tasks" -m "Assisted-by: implementer (<tool>, <model>)"
```

---

### Task 11: Неделимые типы блоков, историческая валидация и устойчивость `SqliteTaskStore`/`Worker`

Соответствует SDD `[P4.T3]`.

**Files:**
- Modify: `src/linguary/application/recovery.py:165-178, 271-365`
- Modify: `src/linguary/adapters/runtime/task_store.py:110-140`
- Modify: `src/linguary/adapters/runtime/worker.py:83-93`
- Test: `tests/unit/application/test_recovery_partition.py`

**Interfaces:**
- Consumes: `BlockType`, `TranslationResultValidator.validate(..., validation_policy="v1")`
- Produces: исправленный `Recovery.prepare_roots`, `Recovery._validate_historical`, `Recovery.split`, устойчивый `SqliteTaskStore.claim_next_pending`

- [ ] **Step 1: Написать падающие тесты**

В `tests/unit/application/test_recovery_partition.py` добавить:
- `test_cohesive_poem_quote_list_table_blocks_are_indivisible`;
- `test_validate_historical_uses_v1_validation_policy`;
- `test_task_store_claim_retries_transient_sqlite_busy`.

- [ ] **Step 2: Проверить падение тестов**

Run: `uv run pytest tests/unit/application/test_recovery_partition.py -k "cohesive_poem or historical_uses_v1 or claim_retries" -v`
Expected: FAIL.

- [ ] **Step 3: Реализовать изменения**

1. В `recovery.py:173`: заменить `first.type.value in {'list_item', 'poem', 'quote', 'table'}` на `first.type in {BlockType.LIST_ITEM, BlockType.POEM, BlockType.QUOTE, BlockType.TABLE}`. В `_validate_historical`: передавать `validation_policy="v1"` в `validator.validate(...)`. В `split()`: проверять `root.status in {'pending', 'running'}` и передавать `expected_generation=root.generation` в `stop_root`/`split_scope`.
2. В `task_store.py:claim_next_pending()`: добавить цикл до 10 повторов при `OperationalError` (`locked`/`busy`) с явным `session.rollback()` и короткой паузой (по образцу `unit_of_work.py:84-93`).
3. В `worker.py:drain()`: при `BaseException` отменять незавершённые `_drain_worker` и ожидать их перед повторным пробросом.

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest tests/unit/application/test_recovery_partition.py -v`
Expected: все тесты зелёные.

- [ ] **Step 5: Коммит**

```bash
git add src/linguary/application/recovery.py src/linguary/adapters/runtime/task_store.py src/linguary/adapters/runtime/worker.py tests/unit/application/test_recovery_partition.py
git commit -m "fix(recovery): fix cohesive block types, historical policy, and task store busy retry" -m "Assisted-by: implementer (<tool>, <model>)"
```

---

### Task 12: Бюджет попыток, `unknown_outcome`, `output_truncated` и выделение ID заметок в `block_processing.py`

Соответствует SDD `[P5.T1]`.

**Files:**
- Modify: `src/linguary/application/block_processing.py:105-580`
- Test: `tests/integration/processing/test_recovery_lifecycle.py`
- Test: `tests/integration/processing/test_restart_translation.py`
- Create: `tests/integration/processing/test_budget_and_truncation_reliability.py`

**Interfaces:**
- Consumes: `work.pause_root`, `work.stop_root(..., expected_generation=...)`, `work.cancel_unsent_attempt`, `results.allocate_note_ids`, `TranslationPolicy.validation_policy`, `source_quote_fragments`
- Produces: обновлённый `TranslationCall.request()` и `TranslationCall.save(..., chapter_num: int | None = None, transaction=None)`

- [ ] **Step 1: Написать падающие тесты и синхронизировать ожидания по `scope_limit`**

1. В `tests/integration/processing/test_budget_and_truncation_reliability.py` добавить:
   - `test_max_attempts_1_transient_error_preserves_budget_for_manual_retry`;
   - `test_unknown_transport_outcome_stops_without_auto_retry`;
   - `test_indivisible_truncation_stops_with_output_truncated_without_repeat`;
   - `test_save_allocates_db_note_ids_atomically_and_skips_stale`.
2. При принятом умолчании Q2 (выделение `scope_limit`) обновить ожидания `root.reason` в `test_recovery_lifecycle.py:25` и `test_restart_translation.py:45`.

- [ ] **Step 2: Проверить падение тестов**

Run: `uv run pytest tests/integration/processing/test_budget_and_truncation_reliability.py -v`
Expected: FAIL.

- [ ] **Step 3: Реализовать изменения в `src/linguary/application/block_processing.py`**

1. В `TranslationCall.request()`:
   - Передавать `validation_policy=policy.validation_policy` и `protected_quotes=source_quote_fragments(...)` в `self._validator.validate(...)`, использовать временную фабрику ID заметок (окончательные ID выдаст `save()`).
   - Обработать `except asyncio.CancelledError:`: если `physical_context` создан, финализировать попытку (если ещё `created` -> `cancel_unsent_attempt`, если `sent`/`unknown` -> `record_recovery_failure(..., outcome="unknown")`), затем `raise`.
   - В `except Exception as error:`: определить `outcome` (`"not_sent"` для `LLMTransportError(outcome="not_sent")`; `"unknown"` для `LLMTransportError(outcome="unknown")` или неклассифицированного исключения после отправки; `"known"` для известных ответов и ошибок валидации). Передать `outcome` в `record_recovery_failure`.
   - Если `outcome == "unknown"`: сразу поднять `RecoveryError("unknown_outcome", recovery[0], original_error=error)` без вызова `RetryPolicy`.
   - При `not decision.should_retry`:
     - `ConditionsError` -> `stop_root(..., "conditions_unavailable", expected_generation=...)`;
     - неделимый `provider_truncated` -> `stop_root(..., "output_truncated", expected_generation=...)`;
     - `provider_refusal` / `provider_content_filter` -> `stop_root(..., "provider_refusal", expected_generation=...)`;
     - исчерпан `max_calls + extra_calls` -> `stop_root(..., "limit", expected_generation=...)`;
     - исчерпан `max_scope_calls` -> `stop_root(..., "scope_limit", expected_generation=...)`;
     - иначе (долговременный бюджет ещё есть — например `max_attempts=1` на временной ошибке или `401/402/403`) -> `pause_root(root.id, reason, expected_generation=...)` (оставляя `status='pending'` с причиной `'transient_error'`/`'configuration_error'`/`'provider_error'`/`'validation_error'`), не записывать `unsafe_partition`.
   - Для `LLMContextCapacityError`: разрешать 1 повтор без соседей в рамках `policy.capacity_retries_without_neighbours` и долговременного бюджета даже при `max_attempts=1`; иначе `stop_root(..., "context_capacity", expected_generation=...)`.
2. В `TranslationCall.save(..., chapter_num: int | None = None, transaction=None)`: поддержать передачу открытой `transaction` через `nullcontext(transaction)` (для единой транзакции `READ_AS_IS`); когда все проверки владения/поколения пройдены и `results` содержат заметки, при наличии `chapter_num` (и `allocate_note_ids` у `transaction.results`) заменить временные `note.id` на атомарно выделенные БД-номера непосредственно перед `replace_results()` и `archive_artifacts()`.

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest tests/integration/processing/test_budget_and_truncation_reliability.py tests/integration/processing/test_recovery_commit.py tests/integration/processing/test_recovery_lifecycle.py -v`
Expected: все тесты зелёные.

- [ ] **Step 5: Коммит**

```bash
git add src/linguary/application/block_processing.py tests/integration/processing/test_recovery_lifecycle.py tests/integration/processing/test_restart_translation.py tests/integration/processing/test_budget_and_truncation_reliability.py
git commit -m "fix(processing): preserve retry budget, isolate unknown outcomes, and allocate note ids on save" -m "Assisted-by: implementer (<tool>, <model>)"
```

---

### Task 13: Изоляция корней в `ProcessChapter` и единая транзакция `READ_AS_IS`

Соответствует SDD `[P5.T2]`.

**Files:**
- Modify: `src/linguary/application/use_cases/process_chapter.py:178-491, 620-651`
- Test: `tests/integration/processing/test_restart_translation.py`

**Interfaces:**
- Consumes: `work.begin_local_acceptance(...)`, `TranslationCall.save(..., chapter_num=work.chapter_number, transaction=...)`
- Produces: изолированный `ProcessChapter._process_roots` при параллелизме 1 и 2; атомарный локальный путь `READ_AS_IS` в одной транзакции

- [ ] **Step 1: Написать падающие тесты**

В `tests/integration/processing/test_restart_translation.py` добавить:
- `test_restart_one_root_at_concurrency_2_keeps_sibling_running_and_saved`;
- `test_restart_waiting_root_at_concurrency_1_does_not_fail_dispatcher`;
- `test_local_read_as_is_commits_in_single_transaction` (проверяет, что локальный блок создаёт принятую попытку для `root_coverage_complete`, не тратит `calls_used` и не открывает две раздельные транзакции);
- `test_discarded_result_never_marks_incomplete_chapter_succeeded`.

- [ ] **Step 2: Проверить падение тестов**

Run: `uv run pytest tests/integration/processing/test_restart_translation.py -k "concurrency_2 or waiting_root or read_as_is or discarded_result" -v`
Expected: FAIL.

- [ ] **Step 3: Реализовать изменения в `src/linguary/application/use_cases/process_chapter.py`**

1. В `_process_roots`: обернуть тело `_process_single_root(root_id)` в перехват ожидаемых `Exception` (возвращать `_RootOutcome(result, error, superseded=...)`), не позволяя исключению одного корня отменять соседние `asyncio.Task` в `gather()`. Внешний `asyncio.CancelledError` по-прежнему отменяет все незавершённые задачи и дожидается их завершения.
2. При `RecoveryError` с `reason in {'interrupted', 'superseded', 'conditions_unavailable'}` проверять в БД, был ли `root_id` вытеснен (`superseded_by is not None`) или удалён из `operation_refs` задачи: если да — считать корень корректно выбывшим, не роняя диспетчер и соседей.
3. При `root_result.discarded`: проверить, вытеснен ли корень или закрыт запуск; если корень остался живым и неполным — не завершать его молча успехом.
4. Передавать `chapter_num=work.chapter_number` в `self._call.save(...)`.
5. Свернуть локальный путь `READ_AS_IS` (`L288-367`) в одну короткую транзакцию `with self._unit_of_work.transaction() as tx:`: вызвать `tx.work.begin_local_acceptance(...)` (создаёт локальный `CallAttempt` со `state='validated'` без расхода `calls_used`), затем в той же транзакции вызвать `self._call.save(..., transaction=tx)` — без разрыва на 2 транзакции и при полном соблюдении `root_coverage_complete`.

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest tests/integration/processing/test_restart_translation.py -v`
Expected: все тесты зелёные.

- [ ] **Step 5: Коммит**

```bash
git add src/linguary/application/use_cases/process_chapter.py tests/integration/processing/test_restart_translation.py
git commit -m "fix(processing): isolate chapter roots across concurrency 1 and 2" -m "Assisted-by: implementer (<tool>, <model>)"
```

---

### Task 14: Единая нумерация и версии политик в `RegenerateBlock` и `CorrectChapter`

Соответствует SDD `[P5.T3]`.

**Files:**
- Modify: `src/linguary/application/use_cases/regenerate_block.py:220-235, 341-391`
- Modify: `src/linguary/application/use_cases/correct_chapter.py:875-1110`
- Modify: `src/linguary/application/correction.py:860-875`
- Create: `tests/integration/processing/test_concurrent_note_ids_and_policies.py`

**Interfaces:**
- Consumes: `TRANSLATION_POLICIES`, `is_candidate_correction_policy`, `unchanged_reader_texts`, `results.allocate_note_ids`
- Produces: совместимый `BlockRegenerationSequence.start()`, `CorrectChapter._candidates()`, `prepare_correction_candidates()`, `invalidate_correction()`

- [ ] **Step 1: Написать падающие тесты**

В `tests/integration/processing/test_concurrent_note_ids_and_policies.py` добавить:
- `test_concurrent_chapter_regeneration_and_correction_never_collide_on_note_ids`;
- `test_regenerate_block_does_not_duplicate_live_v2_operation_under_v3`;
- `test_new_regeneration_supersedes_stopped_regeneration_root`;
- `test_correction_preserves_old_ids_and_candidate_recheck_burns_zero_counter` (включая проверку, что отказ валидации в конце `_candidates()` не тратит счётчик `note_number_counters`).

- [ ] **Step 2: Проверить падение тестов**

Run: `uv run pytest tests/integration/processing/test_concurrent_note_ids_and_policies.py -v`
Expected: FAIL.

- [ ] **Step 3: Реализовать изменения**

1. В `regenerate_block.py`:
   - Передать `chapter_num=work.chapter_number` в `self._call.save(...)`.
   - В `BlockRegenerationSequence.start()` (L358): проверять `existing.policy_version not in TRANSLATION_POLICIES` вместо `!= CURRENT_TRANSLATION_POLICY`.
   - Перед созданием нового корня регенерации найти и вытеснить (`superseded_by = root.id`) любой предыдущий остановленный корень `operation_kind == 'regeneration'` для того же `(run_id, block_id)`.
2. В `correct_chapter.py`:
   - В `_candidates()`: валидировать с временной фабрикой ID, передавать `validation_policy` от `inputs.snapshot.policy_version` (`"v2"` для `CORRECTION_POLICY_VERSION_V3`, `"v1"` для более старых), `protected_quotes` и `excluded_service_texts={block.block_id: unchanged_reader_texts(previous)}`; сопоставлять `old.id` для сохранившихся заметок и выделять новые БД-номера через `tx.results.allocate_note_ids(...)` **в самом конце `_candidates()`, только после успешного завершения всех проверок** (`content_plan`, защищённая помощь и т. д.) и только когда `allocate_from_db=True`.
   - В `prepare_correction_candidates()`: вызывать `_candidates(..., allocate_from_db=False)`.
3. В `correction.py:866`: заменить `== CORRECTION_POLICY_VERSION_V2` на `is_candidate_correction_policy(...)`.

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest tests/integration/processing/test_concurrent_note_ids_and_policies.py tests/integration/processing/test_translation_policy_chapter.py -v`
Expected: все тесты зелёные.

- [ ] **Step 5: Коммит**

```bash
git add src/linguary/application/use_cases/regenerate_block.py src/linguary/application/use_cases/correct_chapter.py src/linguary/application/correction.py tests/integration/processing/test_concurrent_note_ids_and_policies.py
git commit -m "fix(processing): unify note allocation and policy compatibility across regeneration and correction" -m "Assisted-by: implementer (<tool>, <model>)"
```

---

### Task 15: Явный `RestartTranslation` с аудитом и сценарий `ExtendRecoveryLimit`

Соответствует SDD `[P6.T1]`.

**Files:**
- Modify: `src/linguary/application/use_cases/restart_translation.py`
- Create: `src/linguary/application/use_cases/extend_recovery_limit.py`
- Modify: `src/linguary/composition/container.py:575-605`
- Test: `tests/integration/processing/test_recovery_lifecycle.py`

**Interfaces:**
- Consumes: `work.restart_root(root_id, expected_generation=..., successor=..., scopes=..., task=...)`, `work.extend_recovery_limit(...)`
- Produces: `RestartTranslationCommand(..., acknowledge_unknown_outcome: bool = False)`, `ExtendRecoveryLimit(unit_of_work)` с `ExtendRecoveryLimitCommand`, `ExtendRecoveryLimitNotFoundError` (404), `ExtendRecoveryLimitError(message, reason_code=...)`, `ExtendRecoveryLimitConflict`, полем `Application.extend_recovery_limit`

- [ ] **Step 1: Написать падающие тесты**

В `tests/integration/processing/test_recovery_lifecycle.py`:
- добавить `test_restart_translation_allows_unknown_outcome_with_acknowledgement_and_audit`;
- добавить `test_restart_translation_rejects_unknown_without_acknowledgement`;
- добавить `test_restart_translation_allows_unsafe_partition_and_output_truncated`;
- добавить `test_extend_recovery_limit_use_case_enforces_ownership_and_reason_codes`.

- [ ] **Step 2: Проверить падение тестов**

Run: `uv run pytest tests/integration/processing/test_recovery_lifecycle.py -k "allows_unknown or rejects_unknown or unsafe_partition_and_output or extend_recovery_limit_use_case" -v`
Expected: FAIL.

- [ ] **Step 3: Реализовать изменения**

1. В `restart_translation.py`: разрешить перезапуск для `root.status == "stopped"` с причинами `{"limit", "scope_limit", "unknown_outcome", "interrupted", "unsafe_partition", "output_truncated", "context_capacity", "provider_refusal"}`. Определять `had_unknown = root.reason in {"unknown_outcome", "interrupted"} or any(a.state == "unknown" for a in attempts)`. Если `had_unknown` и `not command.acknowledge_unknown_outcome` -> `RestartTranslationError("требуется явное подтверждение перезапуска после неизвестного исхода (возможна повторная оплата)")`. Сформировать аудит `{"schema_version": 1, "predecessor_root_id": root.id, "observed_generation": command.expected_generation, "previous_reason": root.reason, "had_unknown_outcome": had_unknown, "acknowledged_duplicate_billing_risk": command.acknowledge_unknown_outcome, "restarted_at": datetime.now(UTC).isoformat()}` и передать его в `RecoveryRoot(..., compatibility=audit)` (не меняя сигнатуру `work.restart_root`).
2. Создать `src/linguary/application/use_cases/extend_recovery_limit.py`: проверить существование книги и корня (`ExtendRecoveryLimitNotFoundError` для 404), делегировать в `tx.work.extend_recovery_limit(...)`, преобразовывать `WorkStateError` (`generation_conflict`, `unknown`, `state_conflict`, `invalid_limit`, `conditions_unavailable`) в `ExtendRecoveryLimitConflict` / `ExtendRecoveryLimitError` с атрибутом `reason_code`.
3. В `src/linguary/composition/container.py`: зарегистрировать `extend_recovery_limit=ExtendRecoveryLimit(unit_of_work=unit_of_work)` на `Application`.

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest tests/integration/processing/test_recovery_lifecycle.py -v`
Expected: все тесты зелёные.

- [ ] **Step 5: Коммит**

```bash
git add src/linguary/application/use_cases/restart_translation.py src/linguary/application/use_cases/extend_recovery_limit.py src/linguary/composition/container.py tests/integration/processing/test_recovery_lifecycle.py
git commit -m "feat(recovery): allow explicit restart of stopped/unknown roots and add extend-limit use case" -m "Assisted-by: implementer (<tool>, <model>)"
```

---

### Task 16: Серверные проверки восстановления в `RetryTask`

Соответствует SDD `[P6.T2]`.

**Files:**
- Modify: `src/linguary/application/use_cases/retry_task.py`
- Create: `tests/integration/processing/test_retry_task_recovery_guards.py`
- Test: `tests/integration/web/test_quality_pages.py`

**Interfaces:**
- Consumes: `transaction.runs.active_for_book`, `transaction.work.latest_conditions`, `transaction.work.get_binding`, `transaction.work.get_root`, `transaction.work.list_root_attempts`
- Produces: `is_task_retryable_for_recovery(transaction, task) -> tuple[bool, str | None]` и защищённый `RetryTask.__call__`

- [ ] **Step 1: Написать падающие тесты и синхронизировать `test_quality_pages.py:124`**

1. В `tests/integration/processing/test_retry_task_recovery_guards.py` добавить:
   - `test_retry_task_allows_paused_transient_root_with_remaining_budget`;
   - `test_retry_task_rejects_unknown_outcome_and_limit_stopped_roots`;
   - `test_retry_task_allows_partial_chapter_when_one_root_retryable`.
2. Обновить `tests/integration/web/test_quality_pages.py:124` (где `retry_task` вызывался на корне `(stopped, 'limit')`): ожидать `RetryTaskError` (или проверять повтор после продления лимита / перезапуска).

- [ ] **Step 2: Проверить падение тестов**

Run: `uv run pytest tests/integration/processing/test_retry_task_recovery_guards.py -v`
Expected: FAIL.

- [ ] **Step 3: Реализовать проверки в `src/linguary/application/use_cases/retry_task.py`**

Выделить `is_task_retryable_for_recovery(transaction, task) -> tuple[bool, str | None]`: для `task.task_type in {TaskType.PROCESS_TRANSLATION_UNIT, TaskType.REGENERATE_BLOCK}` (если у `transaction` есть `work` и привязанные корни) проверять активный запуск, актуальность `conditions` и наличие хотя бы одного не вытесненного незавершённого корня в `status in ('pending', 'running')` без `attempt.state == 'unknown'` и с остатком бюджета. В `RetryTask.__call__` вызывать эту проверку и при отказе поднимать `RetryTaskError` с понятным указанием действия (явный перезапуск или продление лимита).

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest tests/integration/processing/test_retry_task_recovery_guards.py tests/integration/web/test_quality_pages.py -v`
Expected: все тесты зелёные.

- [ ] **Step 5: Коммит**

```bash
git add src/linguary/application/use_cases/retry_task.py tests/integration/processing/test_retry_task_recovery_guards.py tests/integration/web/test_quality_pages.py
git commit -m "fix(recovery): enforce server-side recovery guards in RetryTask" -m "Assisted-by: implementer (<tool>, <model>)"
```

---

### Task 17: Проекция `book_overview.py` — живой `sent` vs `unknown`, по-задачный повтор и метаданные корней

Соответствует SDD `[P6.T3]`.

**Files:**
- Modify: `src/linguary/application/book_overview.py:350-460, 1260-1275, 1600-1630`
- Test: `tests/integration/web/test_quality_projections.py`

**Interfaces:**
- Consumes: `is_candidate_correction_policy`, `is_task_retryable_for_recovery`
- Produces: обновлённый словарь `recovery.roots[*]` (`section_id`, `chapter_title`, `status`, `has_unknown_outcome`, `in_flight`), исправленный `unknown`, по-задачный `TaskFailure.is_retryable`

- [ ] **Step 1: Написать падающие тесты**

В `tests/integration/web/test_quality_projections.py` добавить:
- `test_in_flight_sent_attempt_not_shown_as_unknown`;
- `test_unknown_and_stopped_roots_expose_can_restart_and_chapter_metadata`;
- `test_one_unknown_root_does_not_disable_can_retry_for_other_chapters`.

- [ ] **Step 2: Проверить падение тестов**

Run: `uv run pytest tests/integration/web/test_quality_projections.py -v`
Expected: новые тесты падают.

- [ ] **Step 3: Реализовать изменения в `src/linguary/application/book_overview.py`**

1. В `_quality_projection`: вычислять `has_unknown_outcome = root.reason in ('unknown_outcome', 'interrupted') or any(a.state == 'unknown' for a in attempts)` и отдельно `in_flight = any(a.state == 'sent' for a in attempts)` (не приравнивать живой `'sent'` к `unknown`).
2. Считать `scope_counts` без учёта `attempt.state == 'cancelled'`.
3. Открыть `can_restart` для всех допустимых остановленных корней (`limit`, `scope_limit`, `unknown_outcome`, `interrupted`, `unsafe_partition`, `output_truncated`, `context_capacity`, `provider_refusal`) и незавершённых невыполняемых корней (включая `has_unknown_outcome=True`).
4. Добавить в каждый элемент `root_summaries`: `section_id`, `chapter_title`, `status`, `has_unknown_outcome`, `in_flight`.
5. В построении `TaskFailure` (`L1607-1621`): для `PROCESS_TRANSLATION_UNIT` / `REGENERATE_BLOCK` вычислять `is_retryable` через `task.is_retryable and is_task_retryable_for_recovery(transaction, task)[0]`.
6. Заменить `== CORRECTION_POLICY_VERSION_V2` (`L1266`) на `is_candidate_correction_policy(...)`.

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest tests/integration/web/test_quality_projections.py -v`
Expected: все тесты зелёные.

- [ ] **Step 5: Коммит**

```bash
git add src/linguary/application/book_overview.py tests/integration/web/test_quality_projections.py
git commit -m "fix(overview): separate live sent calls from unknown outcomes and compute per-task retry" -m "Assisted-by: implementer (<tool>, <model>)"
```

---

### Task 18: Роуты Web UI, отдельное подтверждение `unknown` и защита от бесконечного обновления

Соответствует SDD `[P7.T1]`.

**Files:**
- Modify: `src/linguary/web/routes/books.py:195-225`
- Modify: `src/linguary/web/routes/processing.py:50-65, 295-355`
- Modify: `src/linguary/web/routes/quality.py:120-140, 210-230`
- Test: `tests/integration/web/test_quality_api.py`

**Interfaces:**
- Consumes: `RestartTranslationCommand`, `Application.extend_recovery_limit`, `ExtendRecoveryLimitNotFoundError`, `ExtendRecoveryLimitError`, `is_task_retryable_for_recovery`
- Produces: исправленный `_is_stalled`, отдельное поле `_RestartRecoveryInput.acknowledge_unknown_outcome`, роут продления лимита через `application.extend_recovery_limit`, по-задачный фильтр `quality_ui.retry_tasks`

- [ ] **Step 1: Написать падающие тесты**

В `tests/integration/web/test_quality_api.py` добавить:
- `test_stalled_processing_without_failed_tasks_stops_auto_refresh`;
- `test_web_restart_route_requires_separate_unknown_acknowledgement` (`confirmed=True, acknowledge_unknown_outcome=False` на `unknown_outcome`-корне -> 409; с `acknowledge_unknown_outcome=True` -> 202/303);
- `test_web_extend_limit_uses_application_use_case_and_preserves_reason_codes` (404 для чужой/несуществующей книги, 409 с `detail.reason_code` для `generation_conflict` и `unknown`).

- [ ] **Step 2: Проверить падение тестов**

Run: `uv run pytest tests/integration/web/test_quality_api.py -k "stalled_processing or separate_unknown_acknowledgement or preserves_reason_codes" -v`
Expected: FAIL.

- [ ] **Step 3: Реализовать изменения**

1. В `web/routes/books.py`: определить `_is_stalled` как `not _has_active_tasks(response) and (bool(response.progress.failures) or response.book.status == "PROCESSING")` (сохраняя прежнюю остановку обновления для `IMPORTING`/`ANALYZING` при наличии `failures`).
2. В `web/routes/processing.py`: добавить в `_RestartRecoveryInput` отдельное поле `acknowledge_unknown_outcome: bool = False`, принимать его в JSON и в HTML-форме и передавать в `RestartTranslationCommand(..., acknowledge_unknown_outcome=payload.acknowledge_unknown_outcome)`.
3. В `web/routes/quality.py`: вызывать `application.extend_recovery_limit(ExtendRecoveryLimitCommand(...))`, преобразовывать `ExtendRecoveryLimitNotFoundError` в 404 и `ExtendRecoveryLimitError` / `ExtendRecoveryLimitConflict` в 409 с `reason_code` через `input_error_response`; в построении `quality_ui.retry_tasks` (`L216-224`) фильтровать задачи через по-задачную проверку `is_task_retryable_for_recovery`.

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest tests/integration/web/test_quality_api.py tests/integration/web/test_failure_recovery.py -v`
Expected: все тесты зелёные.

- [ ] **Step 5: Коммит**

```bash
git add src/linguary/web/routes/books.py src/linguary/web/routes/processing.py src/linguary/web/routes/quality.py tests/integration/web/test_quality_api.py
git commit -m "fix(web): require unknown-outcome acknowledgement, wire extend-limit, and stop refresh loop" -m "Assisted-by: implementer (<tool>, <model>)"
```

---

### Task 19: Метки статусов и шаблоны карточки книги

Соответствует SDD `[P7.T2]`.

**Files:**
- Modify: `src/linguary/web/labels.py:310-345`
- Modify: `src/linguary/web/templates/books/detail.html:130-165`
- Modify: `src/linguary/web/templates/partials/quality_review.html:15-115`
- Test: `tests/integration/web/test_quality_pages.py`

**Interfaces:**
- Consumes: `root.chapter_title`, `root.has_unknown_outcome`, `root.can_restart`, `root.reason`, по-задачный `failure.is_retryable`, отфильтрованный `quality_ui.retry_tasks`
- Produces: русские метки в `QUALITY_REASON_LABELS` и `QUALITY_OPERATION_ERRORS['unknown']`, формы с подписью главы и отдельным чекбоксом `acknowledge_unknown_outcome`

- [ ] **Step 1: Написать падающие тесты**

В `tests/integration/web/test_quality_pages.py` добавить:
- `test_detail_page_keeps_retry_button_for_other_chapter_when_one_root_unknown`;
- `test_quality_partial_shows_chapter_label_and_billing_warning_checkbox_for_unknown_restart`.

- [ ] **Step 2: Проверить падение тестов**

Run: `uv run pytest tests/integration/web/test_quality_pages.py -k "keeps_retry_button or billing_warning_checkbox" -v`
Expected: FAIL.

- [ ] **Step 3: Реализовать изменения**

1. В `web/labels.py`: добавить в `QUALITY_REASON_LABELS` русские метки для `unknown_outcome`, `output_truncated`, `scope_limit`, `context_capacity`, `provider_refusal`, `transient_error`, `configuration_error`, `provider_error`, `validation_error`; обновить `QUALITY_OPERATION_ERRORS['unknown']` (убрать несуществующее «сначала нужно подтвердить исход», пояснить явный перезапуск).
2. В `detail.html:141` и `quality_review.html:107`: убрать глобальную проверку `data.recovery.reason not in ('unknown', 'limit')` (опираться на по-задачный `failure.is_retryable` и отфильтрованный `quality_ui.retry_tasks`).
3. В `quality_review.html`: согласовать баннер на `data.recovery.reason in ('unknown', 'unknown_outcome')`, подписывать каждый корень названием главы (`chapter_title`) и причиной; при `root.has_unknown_outcome` выводить предупреждение о возможной повторной оплате запроса у провайдера и отдельный чекбокс `name="acknowledge_unknown_outcome" value="true"`.

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest tests/integration/web/test_quality_pages.py -v`
Expected: все тесты зелёные.

- [ ] **Step 5: Коммит**

```bash
git add src/linguary/web/labels.py src/linguary/web/templates/books/detail.html src/linguary/web/templates/partials/quality_review.html tests/integration/web/test_quality_pages.py
git commit -m "fix(web): clarify recovery labels, per-task retry visibility, and unknown billing confirmation" -m "Assisted-by: implementer (<tool>, <model>)"
```

---

### Task 20: Паритет CLI и штатная остановка воркера Web UI

Соответствует SDD `[P7.T3]`.

**Files:**
- Modify: `src/linguary/cli/app.py:140-165, 400-420`
- Modify: `src/linguary/web/app.py:90-115, 260-280`
- Test: `tests/integration/cli/test_cli.py`

**Interfaces:**
- Consumes: `RestartTranslationCommand(..., acknowledge_unknown_outcome=...)`, `RestartTranslationError`, `RestartTranslationConflict`, `ExtendRecoveryLimitError`, `ExtendRecoveryLimitConflict`
- Produces: CLI-флаг `--acknowledge-unknown-outcome`, константа `DEFAULT_SHUTDOWN_GRACE_SECONDS = 5.0` и сопоставление ошибок в `_ERROR_STATUSES`

- [ ] **Step 1: Написать падающие тесты**

В `tests/integration/cli/test_cli.py` добавить `test_cli_restart_translation_requires_acknowledge_flag_for_unknown_outcome` и `test_web_lifespan_waits_gracefully_before_cancelling_worker` (с подменой `DEFAULT_SHUTDOWN_GRACE_SECONDS` на малое значение для быстрого прогона).

- [ ] **Step 2: Проверить падение тестов**

Run: `uv run pytest tests/integration/cli/test_cli.py -k "acknowledge_flag or waits_gracefully" -v`
Expected: FAIL.

- [ ] **Step 3: Реализовать изменения**

1. В `cli/app.py`: добавить `--acknowledge-unknown-outcome` (`action="store_true"`) в подкоманду `restart-translation` и передавать в `RestartTranslationCommand`.
2. В `web/app.py`: добавить `DEFAULT_SHUTDOWN_GRACE_SECONDS = 5.0`; добавить `RestartTranslationConflict -> 409`, `RestartTranslationError -> 409`, `ExtendRecoveryLimitConflict -> 409`, `ExtendRecoveryLimitError -> 409`, `ExtendRecoveryLimitNotFoundError -> 404` в `_ERROR_STATUSES`; в `lifespan` после `built.begin_close()` ждать `asyncio.wait_for(asyncio.shield(worker), timeout=DEFAULT_SHUTDOWN_GRACE_SECONDS)` с перехватом `TimeoutError` перед `worker.cancel()`.

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest tests/integration/cli/test_cli.py -v`
Expected: все тесты зелёные.

- [ ] **Step 5: Коммит**

```bash
git add src/linguary/cli/app.py src/linguary/web/app.py tests/integration/cli/test_cli.py
git commit -m "fix(cli,web): add unknown-outcome confirmation flag and graceful worker shutdown" -m "Assisted-by: implementer (<tool>, <model>)"
```

---

### Task 21: Сквозное закрытие критериев AC-1..AC-12 и полная верификация

Соответствует SDD `[P8.T1]`.

**Files:**
- Create: `tests/integration/processing/test_translation_reliability_audit.py`

**Interfaces:**
- Consumes: реальная SQLite, миграции Alembic, `ProcessChapter`, `RegenerateBlock`, `CorrectChapter`, `RestartTranslation`, `RetryTask`, `LLMRateLimiter`, `book_overview`
- Produces: воспроизводящий сквозной регрессионный набор по всем AC-1..AC-11

- [ ] **Step 1: Написать сквозные регрессионные тесты по AC-1..AC-11**

Добавить 7 сквозных тестов (`test_ac1_parallel_restart_isolation`, `test_ac2_max_attempts_1_manual_continue`, `test_ac3_unknown_outcome_explicit_restart_and_fencing`, `test_ac4_ac11_concurrent_note_ids_and_migration`, `test_ac5_ac6_ac7_ac9_validation_and_policy_compat`, `test_ac8_rate_limit_pacing_after_pause`, `test_ac10_crash_recovery_and_overview_projection`) с реальной SQLite и управляемыми асинхронными часами без внешних LLM.

- [ ] **Step 2: Запустить сквозной набор, архитектурные тесты и полный прогон**

Run:
```bash
uv run pytest tests/integration/processing/test_translation_reliability_audit.py -v && uv run pytest tests/architecture -q && uv run pytest -m "not slow" -q
```
Expected: 0 failures, все архитектурные (22+) и профильные тесты (4117+ исходных и все новые) зелёные.

- [ ] **Step 3: Коммит**

```bash
git add tests/integration/processing/test_translation_reliability_audit.py
git commit -m "test(processing): add end-to-end reliability regression suite for AC-1..AC-12" -m "Assisted-by: implementer (<tool>, <model>)"
```
