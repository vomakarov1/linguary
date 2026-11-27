# SDD (Backend): Контракты, алгоритмы и тестовая матрица (Harry Potter pipeline scalability & quality hardening)

**Дата:** 2026-11-27  
**Статус:** Approved (пройдены 3 независимых ревью: Spec Architect Reviewer, Regression & Affected Code Auditor, Final Independent Spec Auditor; все замечания устранены)  
**Связанные документы:** `BRIEF-hp-pipeline-scalability-and-quality-hardening-2026-11-27.md`, `ADR-hp-pipeline-scalability-and-quality-hardening-2026-11-27.md`

---

## Глобальные ограничения

- **Границы слоёв:** `domain ← application ← adapters / web / cli` (`uv run pytest tests/architecture`). В `domain` и `application` нет сторонних библиотек и прямого I/O.
- **Схема БД:** 0 миграций Alembic. Новые поля в JSON (`payload_json`, `summary_json`) строго обратно совместимы со старыми записями SQLite.
- **Замороженные SHA-256:** `_TRANSLATION_INSTRUCTIONS`, `snapshot_prompt_templates(source_language="en", target_language="ru", reader_level="b2")` и вариант с `pronunciation_help_enabled=True` (`profile="legacy-v1"`) остаются побайтово неизменными (`test_baseline_b2_templates_sha256_unchanged`).
- **Универсальность:** никаких хардкодов под конкретные книги, персонажей, локации или модели LLM.
- **Неизменяемые контракты:** OpenAI strict JSON Schema (без `minItems`, без `default_factory` на обязательном `used_fact_ids`), `test_ambiguous_identity_remains_separate`, `test_transitive_identity_conflict_triangle_order_independent_deterministic`, `test_dialogue_line_breaks_rejected_if_extra_line_lacks_dash`, `test_dialogue_validator.py`, `test_normalize_a2_dialogue_response_clears_before_you_read`, `test_translation_context.py` (`[bN] режим GLOSSARY_ONLY\n`), `test_book_layout.py:2047-2128`, `test_analysis_merge_scenes.py:1216-1248`, `test_work_state_indexes.py`, `test_recovery_partition.py`, `test_illustration_chapter.py:790-890`, `registry_sha256` через `query_registry_items(..., kind=None)`.

---

## WP1 — Анализ глав: `lowercase_non_glossary`, `_opening_remark_names_speaker` и восстановление после merge

### 1.1 `src/linguary/application/analysis_merge.py`

1. **`_first_name_block_index(name, block_tokens, glossary_candidates)` (`L2776-2865`):**
   - На `L2854-2855` заменить ветку `if lowercase_non_glossary:` на:
     ```python
     if lowercase_non_glossary:
         cap_pos = scan(require_capitalized=True)
         if cap_pos is None:
             return None
         any_pos = scan(require_capitalized=False)
         if any_pos is not None and any_pos < cap_pos:
             return None
         return cap_pos
     ```
   - *Проверка контракта:*
     - `first-years` (строчные вхождения в блоках 19/21/28/32, заглавное `First-years` в блоке 145): `any_pos = 19 < cap_pos = 145` → `None`!
     - `'voldemort'`, `'tom riddle'`, `'harry potter'`, `'professor mcgonagall'` без глоссария (`test_analysis_merge_scenes.py:1216-1248`): строчных вхождений до заглавного нет (`any_pos == cap_pos`) → возвращает `cap_pos` (все 5 ассертов сохраняются!).

2. **`_opening_remark_names_speaker(text: str, value: str) -> bool` (`L7600-7644`):**
   - Извлечь `remark = _opening_remark(text)` и построить токены с регистром, используя ту же нормализацию, что `expression_words`:
     ```python
     raw_tokens = re.findall(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё'’-]*", remark)
     remark_pairs: list[tuple[str, bool]] = []
     for raw in raw_tokens:
         is_cap = raw[:1].isupper()
         for norm in expression_words(raw):
             remark_pairs.append((norm, is_cap))
     ```
   - Полное фразовое совпадение `value_words and _contains_speaker_phrase([w for w, _ in remark_pairs], value_words)` сохранить без изменений.
   - В пословном цикле `for name in _speaker_name_words(value):` по `index, (word, is_cap)` **требовать `is_cap` (`if not is_cap: continue`)**, сохраняя существующие проверки `prev in _ADDRESSEE_PREPOSITIONS`, `prev in conflicting_titles` и `prev in _SPEAKER_TITLES and prev2 in _ADDRESSEE_PREPOSITIONS`.
   - *Эффект:* строчное `"fat"` в `"a fat little monk"` больше не совпадает с `"Fat Friar"`, а `"Harry said"`, `"said Hagrid"`, `"asked Professor McGonagall"` работают без изменений.

3. **`_explicit_speaker_if_remark_names(...)` (`L7199-7230`):**
   - Добавить опциональные keyword-only аргументы:
     ```python
     block_tokens: Sequence[tuple[tuple[str, bool, str | None], ...]] | None = None,
     block_order: Mapping[str, int] | None = None,
     ```
   - Вынести локальную проверку `would_leak`:
     ```python
     def _would_leak_spoiler() -> bool:
         if block_tokens is None or block_order is None:
             return False
         fact_pos = block_order.get(fact.subject_anchor.block_id)
         if fact_pos is None:
             return False
         first_pos = _first_name_block_index(fact.value, block_tokens, glossary_candidates)
         return first_pos is not None and first_pos > fact_pos
     ```
   - Проверять `if _would_leak_spoiler(): return fact` **до** возврата `replace(fact, certainty="explicit")` для **обеих** веток (и `_opening_remark_names_speaker`, и `_opening_remark_pronoun_resolves_speaker` на `L7218`).
   - Вычисление и передача:
     - В `validate_narrative_response()` (`L3563-3595`): вычислить `block_tokens = _extract_block_spoiler_tokens(ordered_block_ids, texts)` один раз до `_collect_spoiler_name_leaks` (передав/переиспользовав) и передать `block_tokens=block_tokens, block_order=block_order` в `_explicit_speaker_if_remark_names` (`L3584`).
     - В `content_plan_from_response()` (`L4024-4037`): передавать `block_tokens` и `block_order`, только когда `texts is not None and ordered_block_ids` (защита от `texts=None`).
     - В `combine_staged_analysis_responses()` (`L3767-3792`): вычислять `block_tokens` и `block_order` по `resolved_blocks` (если переданы), проверяя `fact.subject_anchor.block_id` напрямую по `block_order`.

### 1.2 `src/linguary/application/use_cases/analyze_chapter.py`

- В `AnalyzeChapter._run_staged_narrative_pass()` (`L576-615`), после `narrative = merge_narrative_fragments(...)`:
  - Обернуть `scenes, facts = validate_narrative_response(narrative, ...)` в обработку `AnalysisMergeError as error`:
    - Если `error.violation is not None and error.violation.rule_code == "spoiler_name_leak"`:
      - Вычислить `block_tokens = _extract_block_spoiler_tokens(ordered_block_ids, texts)` и `block_order = {bid: i for i, bid in enumerate(ordered_block_ids)}`.
      - Найти в `narrative.context_facts` только те `explicit`-факты (`kind in {"speaker", "pronoun_referent"}`, `fact.value` непусто), для которых реально выполняется `first_pos > fact_pos` (`_first_name_block_index(fact.value, block_tokens, narrative.glossary_candidates) > block_order[resolved_block_id]`).
      - Если такие факты есть — создать `downgraded` с заменой только этих фактов на `certainty="inferred"`, повторно вызвать `scenes, facts = validate_narrative_response(downgraded, ...)`. При успехе переприсвоить и `scenes, facts`, и `narrative = downgraded` (чтобы именно `downgraded` сохранился через `_save_pass`, без очистки фрагментного кэша!).
    - Если таких фактов нет или повторная валидация снова упала (например, спойлер в `scenes[].participants`) — выполнить `self._clear_fragment_passes(command.book_id, command.section_id, AnalysisPassKind.NARRATIVE)` и пробросить исключение.

### 1.3 Тестовая матрица WP1
- `tests/unit/application/test_analysis_merge_scenes.py`:
  - `_first_name_block_index("first-years", ...)` со строчными вхождениями в блоках 19/21 и заглавным `First-years` в блоке 145 → `None`;
  - все существующие тесты `L1216-1248` (`'voldemort'`, `'tom riddle'`, `'harry potter'`, `'professor mcgonagall'`) и `L1943-2021` (`'the dark voldemort'`, `'Chocolate Boy'`, `'younger boy'`) остаются зелёными;
  - `_opening_remark_names_speaker("What looked like a fat little monk was saying: 'Forgive...'", "Fat Friar")` → `False`, `"Hagrid said: 'Hello'"` → `True`, `"Harry said: 'Hi'"` для `"Harry Potter"` → `True`;
  - `_explicit_speaker_if_remark_names` не повышает ни именную, ни местоименную ветку при `first_pos > fact_pos`;
  - идемпотентность `validate_narrative_response` (фрагмент → `merge_narrative_fragments` → повторная валидация как в `illustration_source.py:930`) для `Fat Friar` в блоке 19 (`inferred`) и блоке 23 (`explicit`).
- `tests/unit/application/test_staged_analyze_chapter.py`:
  - межфрагментный `spoiler_name_leak` на `explicit` speaker → точечное понижение только спойлерного факта в `inferred`, сохранение `downgraded` в `_save_pass` без сброса фрагментного кэша;
  - межфрагментный `spoiler_name_leak` в `scenes[].participants` → по-прежнему ошибка и очистка кэша.

---

## WP2 — SQLite `database is locked`, пул воркеров и $O(N^2)$ `dispatch_scope_task`

### 2.1 `src/linguary/adapters/db/engine.py`
- На `L28`: увеличить `DEFAULT_BUSY_TIMEOUT_MS = 15_000`.
- В `create_database_engine(...)` (`L41-48`): передавать `connect_args={"timeout": busy_timeout_ms / 1000.0}` в `create_engine(...)` (сохраняя `PRAGMA busy_timeout = ...` в `connect` listener).

### 2.2 `src/linguary/adapters/runtime/task_store.py`
- Вынести приватный метод `_run_immediate_write(self, action: Callable[[Session], _T]) -> _T`:
  - Явное управление сессией (без `with session.begin():`, как уже сделано в `claim_next_pending` `L117-150`):
    - цикл `for attempt in range(1, 11):` (сохраняет точные 10 попыток и `sleep(min(0.005 * attempt, 0.02))`, закреплённые в `test_recovery_partition.py:2290-2304` и `test_translation_reliability_audit.py:1786`!);
    - `session.connection().exec_driver_sql(_BEGIN_IMMEDIATE)`;
    - `result = action(session)`;
    - `session.commit()`, возврат `result`;
    - при `OperationalError` с `"locked"` или `"busy"` в `str(error).lower()`: `session.rollback()`; на `attempt == 10` кидает `PersistenceError("база данных занята после ограниченного повтора") from None`, иначе `sleep(min(0.005 * attempt, 0.02))`;
    - при прочих исключениях (`IntegrityError`, `PersistenceError` и др.): `session.rollback()` и проброс;
    - в `finally`: `session.close()`.
- Перевести на `_run_immediate_write`:
  1. `claim_next_pending()` (оставить быструю предпроверку `read_session` без write-лока);
  2. `save(task)` (сохранить обработку `PersistenceError` для удалённой/superseded строки внутри `action(session)`);
  3. `enqueue(identity)` (внутри `_run_immediate_write` проверить существование и вставить задачу, сохранив `except IntegrityError:` fallback и `self.notify_work_available()` после коммита).

### 2.3 `src/linguary/adapters/db/unit_of_work.py`
- **НЕ** добавлять `session.rollback()` внутри `with session.begin():` (`L83-96` — в SQLAlchemy 2.0 это вызывает `InvalidRequestError`, а `_FakeSession` в `test_work_state_indexes.py:125-135` не имеет `.rollback()`) и **сохранить** 10 попыток и `sleep(min(0.005 * attempt, 0.02))` (закреплены `test_work_state_indexes.py:139-166` и `test_translation_reliability_audit.py:1737-1750`). Реальная защита обеспечивается увеличенным `DEFAULT_BUSY_TIMEOUT_MS = 15_000` + `connect_args["timeout"]`, устранением DEFERRED-транзакций в `SqliteTaskStore` и устранением $O(N^2)$ в `dispatch_scope_task`.

### 2.4 `src/linguary/adapters/runtime/worker.py`
- Добавить `from time import monotonic` и в `_drain_worker()` (`L167-171`) добавить перехват `is_transient_lock_error(error)` с двойным предохранителем (не более `_MAX_CONSECUTIVE_DRAIN_LOCK_ERRORS = 10` подряд и не дольше `_MAX_DRAIN_LOCK_WAIT_SECONDS = 30.0` по `monotonic()`):
  ```python
  consecutive_locks = 0
  lock_wait_started: float | None = None
  while True:
      try:
          task = await self._runner.run_next()
          consecutive_locks = 0
          lock_wait_started = None
      except Exception as error:
          if not is_transient_lock_error(error):
              raise
          now = monotonic()
          if lock_wait_started is None:
              lock_wait_started = now
          consecutive_locks += 1
          if (
              consecutive_locks >= _MAX_CONSECUTIVE_DRAIN_LOCK_ERRORS
              or (now - lock_wait_started) >= _MAX_DRAIN_LOCK_WAIT_SECONDS
          ):
              raise
          await asyncio.sleep(min(0.05 * consecutive_locks, 0.5))
          continue
      if task is None:
          return
      finished.append(task)
  ```

### 2.5 Точечный `list_for_scope` в `TaskRepository` и `dispatch_scope_task()`
- **`src/linguary/application/ports/persistence.py`:** импортировать `TaskType` из `linguary.domain` и добавить в протокол `TaskRepository`:
  ```python
  def list_for_scope(
      self, book_id: str, task_type: TaskType, scope_id: str
  ) -> tuple[ProcessingTask, ...]: ...
  ```
- **`src/linguary/adapters/db/repositories.py`:** реализовать `SqlAlchemyTaskRepository.list_for_scope(...)` через SQL `WHERE book_id = ? AND task_type = ? AND scope_id = ? ORDER BY created_at, id` (как в `list_for_book`).
- **`src/linguary/application/illustrations.py:102-106` (`dispatch_scope_task`):** использовать `list_for_scope` через `getattr(tx.tasks, "list_for_scope", None)` с fallback на `tx.tasks.list_for_book(...)` для fake-репозиториев в unit-тестах.

### 2.6 Тестовая матрица WP2
- `tests/integration/tasks/test_concurrent_claim.py` и `tests/integration/tasks/test_worker_pool.py`:
  - Конкурентные `SqliteTaskStore.save()` и `enqueue()` при удержании write-лока (краткосрочный лок → успех; постоянный лок → `PersistenceError`);
  - `_drain_worker()` переживает транзиентный `PersistenceError("база данных занята")` и дорабатывает очередь, но падает при превышении лимита/дедлайна;
  - `SqlAlchemyTaskRepository.list_for_scope` возвращает только задачи нужного `(book_id, task_type, scope_id)`, а `dispatch_scope_task` вызывает `list_for_scope` вместо `list_for_book`.
- **Обновление 1 строки теста:** в `tests/unit/application/test_recovery_partition.py:2306` (где на `L2298` задаётся `script["remaining"] = 20`, из которых 10 съедает `claim_next_pending()` на `L2301`, и оставшиеся 10 ранее не мешали DEFERRED `store.enqueue()` на `L2306`) явно сбросить `script["remaining"] = 0` перед `store.enqueue(...)` на `L2306`.
- Прогон `tests/unit/adapters/test_work_state_indexes.py`, `tests/integration/processing/test_translation_reliability_audit.py`, `tests/unit/application/test_recovery_partition.py`.

---

## WP3 — Подсистема иллюстраций: линейный `alias`, чистый `get_registry`, полный глоссарий и гигиена промптов

### 3.1 `звезда + цепочка` + представительский `alias` (`src/linguary/application/illustration_reference.py`)
- Константы:
  - `_EXACT_GROUP_CLIQUE_MAX = 4`
  - `_CROSS_GROUP_REPS_MAX = 2`
- В `build_alias_candidate_pairs(mentions, *, glossary_links=None, fact_counts=None)` (`L328-396`):
  1. Добавить опциональный `fact_counts: Mapping[str, int] | None = None` (в `illustration_reference.py:2360-2405` предварительно посчитать `fact_counts = Counter(str(f["subject_mention_key"]) for f in facts if f.get("subject_mention_key"))` до вызова `build_alias_candidate_pairs`).
  2. Ранг упоминания (100% порядок-независим):
     ```python
     def _rep_rank(m: Mapping[str, Any]) -> tuple[int, str, str]:
         mk = str(m["mention_key"])
         cnt = fact_counts.get(mk, 0) if fact_counts else 0
         return (-cnt, str(m.get("position_key") or ""), mk)
     ```
  3. Сгруппировать упоминания по `group_key = (entity_kind, normalize_name_token(name))` и отсортировать каждую группу по `_rep_rank`:
     - **Внутри группы размером `k`:**
       - `k <= _EXACT_GROUP_CLIQUE_MAX`: все $\binom{k}{2}$ пар (`<= 6`).
       - `k > _EXACT_GROUP_CLIQUE_MAX`: **звезда + цепочка** (`2k - 3` пар):
         - звезда: `(anchor, m_i)` для `i = 1..k-1`, где `anchor = group[0]["mention_key"]`;
         - цепочка: `(group[i]["mention_key"], group[i + 1]["mention_key"])` для `i = 1..k-2` (страхует от распада группы, если `anchor` оказался редким омонимом!).
     - **Представители группы для межгрупповых бакетов:**
       - `base_reps = [m["mention_key"] for m in group[:_CROSS_GROUP_REPS_MAX]]`
       - `hint_reps = [m["mention_key"] for m in group if (m.get("alias_of_hint") or "").strip()]`
       - `reps[group_key] = list(dict.fromkeys([*base_reps, *hint_reps]))` (гарантирует сохранение всех `alias_of_hint`!).
  4. **Межгрупповые бакеты (`exact:*`, `tok:*`, `stripped:*`, `gloss:*`):**
     - В каждом бакете собирать уникальные `group_key` (по `entity_kind`).
     - Для каждой пары **различных** `group_key` `(g1, g2)` одного `entity_kind` добавлять пары `reps[g1] × reps[g2]`.
  5. **Контекстный блокинг анонимных дескрипторов (`_is_anonymous_person_descriptor`, `L360-395`):**
     - После сортировки соседей в секции по `(tier_prio, abs(o_blk - blk_ord), okey)` отбирать не более `MAX_CONTEXT_ALIAS_NEIGHBORS = 8` соседей с **уникальным `normalize_name_token(o["name"])`**.
  6. **Обновление 2 ассертов в существующем тесте:** в `tests/unit/application/test_illustration_reference.py:917-985` (`test_alias_blocking_batched_and_same_name_not_identity`, `L969-970`):
     - обновить `assert sum(alias_batch_sizes) == 21` на `11` **и** `assert len(alias_batch_sizes) >= 2` на `assert len(alias_batch_sizes) == 1` (`2*7 - 3 = 11` пар для 7 `"John"` укладываются в 1 батч; все 7 остаются отдельными сущностями `len(entities) == 7`);
     - добавить отдельный тест батчинга `> 16` на 12 `"John"` (`2*12 - 3 = 21` пара → 2 батча `[16, 5]`, все 12 остаются отдельными сущностями).

### 3.2 Фильтрация `alias_decision` в `get_registry` (`ports/illustrations.py`, `illustration_repository.py`, `illustration_view.py`)
- В `IllustrationRepository.query_registry_items(...)` (уже возвращает `(rows, total)`, отдельного `count_registry_items` нет!): добавить опциональный аргумент `kinds: Sequence[IllustrationRegistryKind] | None = None`:
  - `kind is not None`: прежний фильтр `kind == kind.value`;
  - `kind is None and kinds is not None`: `kind IN (...)`;
  - `kind is None and kinds is None`: все виды (100% сохраняет `registry_sha256` на `illustration_reference.py:2835`!).
- В `build_illustration_registry_view()` (`illustration_view.py:1063-1145`): при `kind is None` передавать `kinds=(IllustrationRegistryKind.ENTITY, IllustrationRegistryKind.FACT, IllustrationRegistryKind.DESIGN)` (с защитным fallback-перехватом `TypeError` для старых fake-репозиториев в unit-тестах). При явном `kind=IllustrationRegistryKind.ALIAS_DECISION` возвращать решения алиасов.
- Обновить `tests/integration/web/test_illustrations_pages.py:1144-1157`: передать `registry_kind=alias_decision` в запрос, проверяющий отображение `"один и тот же образ"` / `"разные образы"`, и добавить проверку, что запрос без `registry_kind` скрывает `alias_decision`.

### 3.3 Лимит `used_fact_ids` (`src/linguary/adapters/llm/illustration_payloads.py`)
- Добавить `_MAX_PROMPT_USED_FACT_IDS = 128` рядом с `_MAX_COVERED_IDS = 256` (`L35-36`).
- В `IllustrationPromptSchema` (`L370`): `used_fact_ids: list[str] = Field(max_length=_MAX_PROMPT_USED_FACT_IDS)` (**без `default_factory=list`** — поле остаётся строго обязательным!).

### 3.4 Полный словарь `locked > approved > suggested` без взрыва `DESIGN` (`illustration_context.py`, `illustration_reference.py`)
- В `src/linguary/application/illustration_context.py` выделить чистый хелпер:
  ```python
  def collect_book_glossary_entries(
      tx: Any,
      book_id: str,
      *,
      max_section_order: int | None = None,
      generation_id: str | None = None,
  ) -> dict[str, tuple[str, str]]:
      """{glossary_key(source): (canonical_source, translation_ru)} с приоритетом locked > approved > suggested."""
  ```
  - Убрать ранний `if not concepts: return {}` (`illustration_context.py:635`).
  - Единая шкала приоритетов для записей из `tx.glossary.list_for_book(book_id)`: `max(concept_rank, ru_variant_rank)` где `LOCKED=3, APPROVED=2, SUGGESTED=1` (согласовано с `illustration_reference.py:2389` и `illustration_context.py:766`); при `max_section_order is not None` сохраняется действующий антиспойлерный фильтр `seen_keys` по секциям `<= max_section_order` **и `ENTITY`-реестру `generation_id`** (`test_illustration_selection.py:1508-1509`!).
  - Источник 2 (приоритет `suggested=1`): собрать `GlossaryCandidate` по секциям (дедуплицируя по `section_id`: если у секции есть принятый `ChapterAnalysis` из `tx.analyses.list_for_book(book_id)`, брать его; иначе брать `NARRATIVE`-пасс из `tx.analyses.list_passes_for_book(...)` — исключает двойной подсчёт глав!), фильтровать `sec_ord <= max_section_order` (если задан), и для ключей, не занятых `locked`/`approved`, выбирать перевод по наибольшей частоте (тай-брейк — минимальный `(sec_ord, suggested_translation)`).
- В `illustration_reference.py:2374-2395`:
  - `glossary_links` и `approved_glossary_keys` (множество `glossary_key(concept.canonical_source_text)` для записей из `tx.glossary` со статусом `APPROVED`/`LOCKED`) строить **строго по `tx.glossary`** (не подмешивать `suggested` в `glossary_links`!).
  - `glossary_ru_by_key` строить через `collect_book_glossary_entries(tx, generation.book_id)` для локализации `ENTITY.glossary_translation_ru` и `ENTITY.alias_translations_ru`.
  - В отборе `DESIGN` (`illustration_reference.py:2690-2703`): заменить `or bool(e.get("glossary_translation_ru"))` на **`or glossary_key(str(e.get("canonical_name") or "")) in approved_glossary_keys`** (не меняет схему `entity_payload` и `payload_sha256` и не даёт 177 `suggested`-терминам взрывать число вызовов `DESIGN`!).
- В `_collect_book_glossary_terms_ru(...)` (`illustration_context.py:616-775`): использовать `collect_book_glossary_entries(..., max_section_order=max_section_order, generation_id=generation_id)` и добавить `priority_keys: frozenset[str] = frozenset()` (ключи сущностей текущей сцены/момента), чтобы их русские имена гарантированно входили в `limit=40`.

### 3.5 Устранение 5 дефектов промптов и сокращение дробности `illustration_chapter`
1. **Изоляция `episodic`-фактов (`illustration_chapter.py:1047-1074`):**
   - Вычислить `scene_start_pos: str`: построить карту `{b.id: idx for idx, b in enumerate(textual_blocks(section))}` (**0-based индекс `enumerate`**, как в `illustration_reference.py:1698-1700` и `illustration_selection.py:526`, и `entry.order`!). Если `card.scene_start_block_id` непуст и найден в карте как `start_idx`, взять `format_position_key(entry.order, start_idx, 0)`; иначе fallback — `format_position_key(entry.order, 0, 0)`.
   - Сразу после выборки фактов сущности (до построения `all_registry_facts` и `entity_facts`):
     - оставлять все `temporal_scope in {"permanent", "state_change"}`;
     - для `temporal_scope == "episodic"` оставлять **только факты с `scene_start_pos <= fact_pos <= event_pos`** (убирает и носки из чулана гл. 2 в зоопарке, и `baggy old clothes` из гл. 2 в гл. 8).
2. **Инструкции `EXTRACT`, `FACT_SCOPE`, `CANDIDATES`, `PROMPT` (`illustration_context.py`):**
   - `EXTRACT`: смена повседневной одежды/формы/обстановки → `temporal_scope="state_change"`, разовая деталь эпизода → `"episodic"`; `back_applicable=true` запрещён для возрастных/этапных черт и предметов, появившихся после временного скачка.
   - `FACT_SCOPE`: более поздний `state_change` или факт текущей сцены перезаписывает устаревшую одежду/статус в `running_summary_en`; при возрастном несоответствии (младенец vs старший возраст) исключать несовместимые `is_back_applicable`-факты.
   - `PROMPT`: добавить запрет комиксных речевых облачков (`speech balloon`, `speech bubble`, `thought bubble`, `comic bubble`, `caption`) и читаемых текстовых надписей/табличек.
   - `CANDIDATES`: добавить в `_STAGE_INSTRUCTIONS[CANDIDATES]` правило не использовать в `title_ru` и `event_ru` имя персонажа, если оно появляется в тексте только после якоря события.
3. **Валидатор запрета speech bubbles (`illustration_validation.py`, `illustration_chapter.py:2394-2540`):**
   - В `illustration_validation.py` добавить чистую функцию `find_speech_bubble_term(text: str) -> str | None` на базе `_SPEECH_BUBBLE_RE = re.compile(r"\b(?:speech|thought|text|comic)\s+(?:balloon|bubble|box|caption)s?\b", re.IGNORECASE)`.
   - В `IllustrationChapter._validate_render_response()` (`illustration_chapter.py:2394-2540`) проверять `find_speech_bubble_term(payload.scene_composition_en)` — при совпадении отклонять ответ с явным сообщением для компактного repair.
4. **`CANDIDATES` и `chunk_end_pos` (`illustration_selection.py:84, 103-153, 964-1003, 2417-2452`):**
   - В `_known_entities_context_for_chapter(...)` (`L103-153`):
     - добавить опциональные `max_position_key: str | None = None` (по умолчанию `chapter_end_pos`, сохраняет `test_illustration_selection.py:1493`) и `preloaded_entities` (кэшировать `_load_entity_payloads` один раз на главу в `plan_scopes_in_tx`);
     - **вынести проверку `if first_pos and first_pos > effective_max_pos: continue` наружу из `if not in_current:`** (`L125-136`), и передавать `effective_max_pos` в `_display_name_at_position(ent, effective_max_pos)` и фильтр `alias_first_positions` (`pos <= effective_max_pos`) — иначе персонажи из поздних чанков текущей главы всё равно просочились бы в ранний чанк!
   - В `plan_scopes_in_tx` (`L964-1003`, а не в чистом конструкторе `_build_candidate_scope_spec` `L596-612`): вычислять `chunk_end_pos` по последнему блоку каждого чанка и передавать почанковые `known_characters` / `known_locations` / `known_entity_aliases` в `_build_candidate_scope_spec`; в `_split_children_for_scope` (`L2417-2452`) сохранить наследование списков от родителя (сохраняет `test_illustration_selection.py:1315-1321, 1504-1509`).
5. **Снижение дробности `illustration_chapter`:**
   - В `illustration_selection.py:84`: `_MAX_CANDIDATE_BLOCKS_PER_CHUNK = 64` (wire-budget продолжает ограничивать реальный размер чанка).
   - В `illustration_chapter.py:1126-1191`:
     - Для `len(entity_facts) == 1`: сохранить текущее правило без изменений (вкл. `back_applicable=True`, сохраняет `test_illustration_chapter.py:790-890`!).
     - Для `2 <= len(entity_facts) <= 4`: включить fast-path, только если **все** факты имеют `temporal_scope == "permanent"`, все локальные (`_parse_section_order(fact_pos) <= entry.order`), все `description_en` и `description_ru` непусты, `description_en` без кириллицы, без цветовых слов в `monochrome_ink`, а дедуплицированные через `dict.fromkeys(...)` и склеенные через `"; "` строки укладываются в `len(joined_en) <= 400` и `len(joined_ru) <= 600` (уважая `_MAX_COMPACT_DESC = 600`). Обновить `ent-eva` в `test_trivial_permanent_fact_scope_fast_path_and_guards` (`test_illustration_chapter.py:2742-2901`).

### 3.6 Тестовая матрица WP3
- `tests/unit/application/test_illustration_reference.py`: `звезда + цепочка` ($k=68 \to 133$ пар, омоним на `anchor` не раскалывает `group[1:]`), `alias_of_hint` не теряется в кросс-группах, `fact_counts` влияет на выбор `anchor`, анонимная дедупликация соседей, `same name != identity` на 7 и 12 `"John"`, `suggested` глоссарий локализует `name_ru` без роста `DESIGN`-скоупов.
- `tests/unit/application/test_illustration_chapter.py` и `tests/integration/web/test_illustrations_pages.py`: `get_registry(kind=None)` скрывает `alias_decision`, `registry_kind=alias_decision` показывает их, `registry_sha256` неизменен; `used_fact_ids` = 48 проходит `IllustrationPromptSchema`; `episodic` из предыдущей сцены/главы отсекается; `speech balloon` отклоняется в `_validate_render_response`; `CANDIDATES` использует `chunk_end_pos`; fast-path `FACT_SCOPE` работает на 1 (вкл. `back_applicable`) и на 2–4 локальных `permanent`-фактах с лимитом длины 400/600.

---

## WP4 — Коррекция глав (`chapter_correction`): полнобатчевый retry и снятие Catch-22 диалогов

### 4.1 `src/linguary/application/use_cases/correct_chapter.py`
- Добавить хелпер `_batch_expectation(...) -> ResultExpectation` по **всем** блокам батча коррекции в исходном порядке: при наличии `inputs.analysis` — через `expectation_for_unit(TranslationUnit.of_blocks(...), ...)` (сохраняя `segment_treatments`), при `inputs.analysis is None` — напрямую через `ResultExpectation(..., blocks=tuple(ExpectedBlock(...) for ...))`.
- Передавать `batch_expectation` аргументом в `_candidates(..., batch_expectation=batch_expectation)` (`L933-956`) и напрямую в `self.validator.validate(..., expectation=batch_expectation)` (при этом `violation.block_ids` остаётся локальным `(block.block_id,)`).
- В `_group()` (`L773, 864-882`) и `_propose_group()` (`L1320, 1407-1425`): при перехвате **`LLMPayloadError`** (от `validate_for()` при неполном покрытии батча) или `LLMResponseFormatError` **до** вызова `_alias_error_for_retry` (который на `L404-405` делает ранний возврат при пустых `replacements` и не создаёт `expectation`) прикрепить `batch_expectation` к копии ошибки (`err = copy.copy(error); err.expectation = batch_expectation`) и передать в `_alias_error_for_retry` → `self.retry.decide(...)`.

### 4.2 `src/linguary/application/validators/paragraphs.py`
- Добавить `_count_post_speech_narrative_sentences(source_text: str) -> int`:
  - Находит закрывающие кавычки прямой речи после терминальной пунктуации (`[.!?…]["”’']`), за которыми вне кавычек следует самостоятельное авторское предложение с заглавной буквы (исключая соседнюю открывающую кавычку вроде `‘Yes.’ ‘No.’` и апострофы внутри слов).
- В `_check_paragraphs()` (`L34-38`), в ветке `if translated_count > source_count and has_speech:`:
  - Прежняя ветка `all(line.lstrip().startswith(_DIALOGUE_DASHES) for line in extra_lines)` остаётся без изменений.
  - Новая ветка: если `post_count = _count_post_speech_narrative_sentences(source_text) > 0`:
    - не-тире строк среди `extra_lines` не больше `min(post_count, 2)` и общее число `extra_lines <= post_count + 1`;
    - каждая не-тире строка из `extra_lines` начинается с заглавной буквы;
    - первая строка `translated_lines[0]` не содержит уже встроенного самостоятельного авторского предложения после `— ` с заглавной буквы (защищает от склейки чужого соседнего абзаца!).
  - *Проверка `test_dialogue_line_breaks_rejected_if_extra_line_lacks_dash` (`test_result_validators.py:784`):* для `'"Hello," he said. "How are you?"'` `post_count == 0` → по-прежнему отклоняется!

### 4.3 `src/linguary/application/reader_text.py` и `src/linguary/composition/correction_runtime.py`
- В `_A2_DIALOGUE_AND_CALQUE_RULES` (`reader_text.py:516-529`, не входит в замороженный SHA-256!) и `CORRECTION_PROMPT_V2` (`correction_runtime.py`):
  - Добавить явное правило для авторского предложения после реплики в одном абзаце: оформлять в 1 строку через длинное тире с прописной буквы (`— Реплика? — Авторское предложение.`) либо с переносом строк (если после авторского предложения речь продолжается — 3 строки: `— Реплика 1.\nАвторское предложение.\n— Реплика 2.`, **не** 2 строки вида `Авторское предложение. — Реплика 2.` и не кавычки-ёлочки с начала абзаца).

### 4.4 Тестовая матрица WP4
- `tests/unit/application/test_correct_chapter_v2.py` и `tests/unit/composition/test_correction_runtime.py`:
  - ошибка валидации на 1 из 10 блоков батча → `repair_instruction` требует вернуть все 10 блоков;
  - `LLMPayloadError` (неполное покрытие батча из `validate_for`) → `repair_instruction` получает полный `batch_expectation`;
  - ошибка на 1 из 3 блоков → перечислены все 3 блока.
- `tests/unit/application/test_result_validators.py` и `test_dialogue_validator.py`:
  - абзац Олливандера проходит и в 1 строку через тире, и в 3 строки; поглощённый соседний абзац отклоняется; `test_dialogue_line_breaks_rejected_if_extra_line_lacks_dash` зелёный.

---

## WP5 — `book_conditions`: отделение транспортно-операционных настроек и прокси

### 5.1 `src/linguary/composition/book_runtime.py` и `correction_runtime.py`
1. **Операционные поля (`_OPERATIONAL_LLM_FIELDS`):**
   `("base_url", "api_key", "proxy_mode", "proxy_url", "timeout_seconds", "requests_per_minute", "max_attempts", "debug_llm_payloads", "input_price_per_mtok", "output_price_per_mtok")`.
   - В `BookRuntimeResolver.settings_for()` (`book_runtime.py:478-488`) и `CorrectionRuntimeResolver.settings_for()` (`correction_runtime.py:210-219`): формировать единый словарь `llm_kwargs = {**frozen_llm_params, **operational_overrides}` перед `LLMSettings(**llm_kwargs)` (исключает дублирование kwargs и `TypeError` в `CorrectionRuntimeResolver`!).
   - Обновить `_runtime_fingerprint()` обоих резолверов:
     - `BookRuntimeResolver` (`L431`): добавить `timeout_seconds`, `requests_per_minute`, `max_attempts`, `proxy_mode`, `input_price_per_mtok`, `output_price_per_mtok`;
     - `CorrectionRuntimeResolver` (`L151`): добавить `timeout_seconds`, `max_attempts`, `proxy_mode`, `input_price_per_mtok`, `output_price_per_mtok`.
   - Ключи в `conditions_payload()` и `CORRECTION_PARAMETERS` (`correction_dto.py:26-29`) **не удалять** (100% обратная совместимость старых записей SQLite и `CorrectionSnapshot.__post_init__`).
2. **Маршрут и прокси (`route_snapshot()` и `routes_compatible()` в `book_runtime.py:64-73`):**
   - В `route_snapshot(llm)`: **сохранить `public_url(llm.base_url)`** и убрать чтение `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY`/`NO_PROXY` из `os.environ`:
     ```python
     safe_base_url = public_url(llm.base_url)
     identity = {"provider": llm.provider, "base_url": safe_base_url}
     return {
         "provider": llm.provider,
         "base_url": safe_base_url,
         "proxy": {"mode": llm.proxy_mode},
         "route_id": payload_digest(identity),
     }
     ```
   - Добавить `routes_compatible(current_route: Mapping[str, Any], pinned_route: Mapping[str, Any]) -> bool`:
     `return (current_route.get("provider"), current_route.get("base_url")) == (pinned_route.get("provider"), pinned_route.get("base_url"))`.
   - Применить `routes_compatible(...)` в `BookRuntimeResolver.settings_for()` (`book_runtime.py:474`) и `CorrectionRuntimeResolver.settings_for()` (`correction_runtime.py:208`). (`illustration_runtime.py:185` не содержит проверки маршрута и не требует правки.)

### 5.2 Тестовая матрица WP5
- `tests/integration/processing/test_pinned_runtime.py`:
  - Обновить `test_changed_proxy_route_rejected_without_request` (`L38-50`) → смена прокси не блокирует, смена `provider`/`base_url` даёт `ConditionsError('route_unavailable')`;
  - Обновить `test_auto_proxy_route_fingerprint_excludes_credentials` (`L67-77`) → `route_snapshot` использует `public_url`, не содержит env-прокси и стабилен при смене `HTTPS_PROXY`;
  - Добавить тесты на переопределение `timeout_seconds`/`requests_per_minute`/`max_attempts` в `BookRuntimeResolver` и `CorrectionRuntimeResolver` (вкл. старые БД с env-прокси и без `TypeError`) и обновление `_runtime_fingerprint`.

---

## WP6 — Вёрстка сносок в EPUB и точность глоссарного валидатора

### 6.1 `src/linguary/adapters/epub/layout_render.py` (`L288-357`)
- В `_insert_markers(source, unit, notes)` группировать маркеры по `offset` (убывание `offset`, внутри группы — возрастание `index`) и вызывать `_insert_marker_group(source, offset - unit.start, group)`:
  - `links = [_marker_link(number, note_id) for number, note_id in group]`;
  - `position >= length`: `for link in links: source.append(link)`;
  - внутри `_enclosing_anchor`: цепочка `prev.addnext(link)` с переносом `anchor.tail` (как `anchor.tail or None`) на `links[-1].tail`;
  - `_split_insert_group(node, is_tail, offset, links)`:
    - `is_tail=False`: `node.text = before or None`, `for i, link in enumerate(links): node.insert(i, link)`, `links[-1].tail = rest or None`;
    - `is_tail=True`: `node.tail = before or None`, цепочка `prev.addnext(link)` от `node`, `links[-1].tail = rest or None`.

### 6.2 CSS-разделитель смежных сносок (`writer.py`, `linguary.css`)
- Добавить отдельным правилом в `src/linguary/adapters/epub/writer.py` (`L181-191`) и `src/linguary/web/static/linguary.css` (`L555-565`):
  ```css
  a.lg-ref + a.lg-ref {
    margin-left: 0.18em;
  }
  ```

### 6.3 `src/linguary/application/validators/language.py` (`L261-333`)
- В `_glossary_mismatch_warnings(...)`:
  1. Сортировать термины по `(-len(entry.source_text.strip()), not entry.status.is_user_decision, entry.source_text)`.
  2. Добавить `_source_term_spans(source_text: str, term: str, *, entry_type: ConceptType | None) -> list[tuple[int, int]]`:
     - Границы: `left = r"(?<![\w-])"` если `not term.startswith("-")` else `r"(?<!\w)"`; `right = r"(?![\w-])"` если `not term.endswith("-")` else `r"(?!\w)"` (сохраняет `You-Know-Who`, исключает `half-moon`).
     - Определение регистрозависимости:
       - `case_sensitive = (entry_type in {ConceptType.PERSON, ConceptType.PLACE, ConceptType.ORGANIZATION}) or (bool(term) and term[0].isupper())`.
       - Если `case_sensitive`: если термин многословный и начинается с артикля `article` (`the`, `a`, `an`, без учёта регистра), строить паттерн с точным регистронезависимым артиклем термина `(?i:{re.escape(article)})\s+` + регистрозависимый хвост (так `"the Sorting Hat"` матчит `"The Sorting Hat"`, а `"The Great Hall"` матчит `"the Great Hall"`!). Допускать также ALL-CAPS вариант `term.upper()`.
       - Если `not case_sensitive`: `re.IGNORECASE`.
  3. Маскировать спаны `covered_spans` по принципу longest-match-first: более короткий термин (`Sorting`) проверяется, только если у него есть хотя бы один спан, не вложенный в `covered_spans`.

### 6.4 Тестовая матрица WP6
- `tests/unit/adapters/epub/test_layout_render.py` и `tests/integration/layout/test_a2_harry_potter_regressions.py`: парные сноски при `< length`, `>= length`, внутри `<em>`/`<a>` и при пустом `before`/`rest` (`None`); наличие `a.lg-ref + a.lg-ref` в CSS EPUB.
- `tests/unit/application/test_result_validators.py`: `Moon`/`half-moon`/`full moon`/`MOON`, `Bill`/`a bill`/`Bill`, `The Sorting Hat` + `"the Sorting Hat"` маскируют `Sorting`, `You-Know-Who` сохраняется.

---

## WP7 — Педагогика A2: стихи, строфы/двустишия, `GLOSSARY_ONLY`, `fragment_assistance`, `speech_style`

### 7.1 Политика подстрочного перевода стихов и `GLOSSARY_ONLY` (`context_builder.py`, `adapters/llm/schemas.py`)
- В `_NARRATIVE_ANALYSIS_INSTRUCTIONS` (`context_builder.py:207`, не входит в замороженный SHA-256): добавить правило не требовать рифмованный/стихотворный перевод стихов, песен и гимнов (только точный смысловой построчный подстрочник без рифмы).
- В `ContextBuilder._translation_context()` / `_shape_instructions()` (`context_builder.py:1402, 2988-3004`, динамическая часть, не входит в замороженный SHA-256):
  - Вычислять `verse_block_ids` по **всей главе** (`detect_verse_stanza_runs(textual_blocks(section))` или из переданных блоков секции) и при `not self._english_help`:
    - для `BlockType.POEM` сохранять `_POEM_TRANSLATION_RULE`;
    - для `PARAGRAPH`-блоков из `verse_block_ids` добавлять правило построчного подстрочника без попыток рифмовать в ущерб смыслу и с единым обращением `ты`/`вы`.
  - **НЕ менять** заголовки `[bN] режим GLOSSARY_ONLY\n` в `user_prompt` (сохраняет `test_translation_context.py:2402-2731`!). Вместо этого в `_shape_instructions(unit, ...)` добавлять напоминание (*«Для каждого блока в режиме GLOSSARY_ONLY обязательно верни не менее 1 словарной подсказки в `glossary_hints`.»*) **только при `segment_lines and not english_help`** (когда есть `BlockTreatment.GLOSSARY_ONLY` в `unit`) и вставлять его в `parts` **ПЕРЕД `_SEGMENT_TRANSLATION_RULE` и `_POEM_TRANSLATION_RULE`** (сохраняет оба ассерта `test_translation_context.py:2449, 2451`!).
- В `adapters/llm/schemas.py:218`: усилить `Field(description=...)` у `BlockResultSchema.glossary_hints` (без `minItems`!).

### 7.2 Детектор строф и двустиший (`src/linguary/application/verse.py`)
- В `_extract_implicit_verse_subruns()` (`L136-158`):
  - Допускать до 2 промежуточных end-stopped строк между enjambed-парами (`idx <= cur_end + 3`), если все промежуточные блоки проходят `_is_verse_paragraph_candidate` и не содержат глаголов речи (`said`/`asked`/…).
  - Допускать **расширение хвоста** валидного прогона до закрытия незакрытой открывающей кавычки (`‘`/`“`, открытой внутри прогона и не закрытой к `cur_end`), присоединяя до 4 следующих подряд коротких `candidate`-блоков (≤ 80 символов, без глаголов речи) вплоть до блока с закрывающей кавычкой (захватывает хвостовую строфу `#62–#65` песни Шляпы); при проверке порога `enjambed_caps * 2 >= non_final` (`L120`) оценивать enjambment по основной части прогона без хвостового закрывающего куска.
- В `_detect_implicit_couplets(blocks)` (вызывается на блоках вне длинных строф):
  - **Только строгие 2-строчные закавыченные двустишия** (без ветки курсива и без обрыва на тире): два смежных `PARAGRAPH`-блока `(b1, b2)`, оба проходят `_is_verse_paragraph_candidate`, длина каждого ≤ 100 символов, нет глаголов речи (`said`, `asked`, `replied`, `whispered`, `shouted`, `muttered`), `b1` открывает кавычку (`‘`/`“`) без закрытия и заканчивается `,`/`;`/без терминального знака (`.!?`), а `b2` начинается с заглавной и закрывает ту же кавычку с терминальной пунктуацией (`.!?`).

### 7.3 Починка двух причин `unresolved` в `fragment_assistance` (`layout/notes.py`, `a2_core_vocab.py`)
1. **`src/linguary/application/layout/notes.py:65-84, 118-120`:** добавить `_RESULTATIVE_PARTICLES = frozenset({"open", "shut", "closed", "free", "loose", "clear", "clean", "dry", "flat", "short", "still", "tight", "wide"})` в `_PHRASE_PREPOSITIONS` (чинит `"flicked it open"` ↔ `"to flick open"` и в `validators/content_plan.py`, и в `find_hint_marker_end`).
2. **`src/linguary/application/a2_core_vocab.py:1012-1032`:** добавить `"going"` в `A2_CORE_VOCABULARY` и разрешить 2-буквенные основы `{"go", "do", "be"}` в `_matches_ing` (`"doing"`, `"being"`, `"does"`, `"goes"` уже покрыты существующими таблицами).

### 7.4 Дедупликация `speech_style` (`src/linguary/application/layout/book.py:1730-1839`)
1. В `_repeated_speech_style_ids()` (`L1794-1818`):
   - Фильтровать `fact.kind == "speaker"` и `isinstance(fact.value, str)`.
   - Добавлять в `anchors_by_speaker[key]` **и `fact.subject_anchor`, и `fact.anchors`**.
   - Строить `block_speakers: dict[str, set[str]]` **строго по `fact.subject_anchor.block_id`**: если в блоке `block_id` все `speaker`-факты принадлежат ровно одному спикеру `only_key`, привязывать `speech_style`-трудность этого блока к `only_key` при отсутствии пересечения якорей.
2. В `_speech_style_lexemes()` (`L1730-1759`):
   - Токены, оканчивающиеся на `"in'"` (длина ≥ 4, напр. `flyin'`, `payin'`, `deliverin'`), заменять на общий маркер `"-in'"` (при этом прочие лексемы той же реплики вроде `"yer"` сохраняются в множестве `lexemes` и не подавляются!).
3. В цикле отбора (`L1820-1835`):
   - Ввести `_SPEECH_STYLE_BOOK_CAP = 4` (сохраняет `test_book_layout.py:2047-2128`, который проверяет 4-ю уникальную заметку `d6`, и обрезает 7–9 повторов Хагрида до ≤ 4 на книгу!).

### 7.5 Тестовая матрица WP7
- `tests/unit/application/test_context_builder.py` и `tests/unit/application/test_translation_context.py`: `test_baseline_b2_templates_sha256_unchanged` и ассерты `[bN] режим GLOSSARY_ONLY\n` зелёные; `_NARRATIVE_ANALYSIS_INSTRUCTIONS` запрещает рифмованный перевод стихов; `_shape_instructions` подключает подстрочник для `PARAGRAPH`-строф (и НЕ подключает при `english_help=True`) и напоминание `GLOSSARY_ONLY`.
- `tests/unit/application/test_analysis_merge_scenes.py` (или новый `tests/unit/application/test_verse.py`): хвостовая строфа `#61–#65` и 2-строчное закавыченное заклинание `#215–#216` распознаются, а письма/курсив/диалоги с тире НЕ дают ложных срабатываний.
- `tests/unit/application/layout/test_notes.py` и `tests/unit/application/test_a2_core_vocab.py`: `"flicked it open"` ↔ `"to flick open"`, `"going"` в core A2.
- `tests/unit/application/layout/test_book_layout.py`: `test_book_layout_keeps_first_two_speech_style_and_unique_idioms` (`L2047-2128`) зелёный; новый тест на привязку через `fact.subject_anchor`, block-level fallback, нормализацию `-in'` и `_SPEECH_STYLE_BOOK_CAP = 4`.
