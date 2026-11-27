# Harry Potter Pipeline Scalability & Quality Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Устранить все верифицированные системные баги, блокеры и узкие места масштабируемости из прогона *Harry Potter* (`hp-ch1-ch9p2-gpt61terra-20261127`): блокер анализа главы 7, падения SQLite `database is locked`, $O(M^2)$ взрыв `alias` в иллюстрациях, 39.2% сбоев `chapter_correction`, заморозку операционных параметров в `book_conditions`, инверсию сносок в EPUB, ложные глоссарные предупреждения и дефекты стихов/педагогики A2 — без миграций БД и без регрессий.

**Architecture:** Изменения строго соблюдают Clean Architecture (`domain ← application ← adapters / web / cli`) и разбиты на 7 слабосвязанных рабочих пакетов (WP1–WP7), каждый со своим TDD-циклом, тестовым гейтом и отдельным коммитом.

**Tech Stack:** Python 3.15+, Pydantic v2, SQLite (без изменения схемы), pytest, Ruff, ty.

**Spec (одобрены по итогам 3 независимых ревью; все замечания устранены):**
- `docs/superpowers/specs/2026-11-27-hp-pipeline-scalability-and-quality-hardening/BRIEF-hp-pipeline-scalability-and-quality-hardening-2026-11-27.md`
- `docs/superpowers/specs/2026-11-27-hp-pipeline-scalability-and-quality-hardening/ADR-hp-pipeline-scalability-and-quality-hardening-2026-11-27.md`
- `docs/superpowers/specs/2026-11-27-hp-pipeline-scalability-and-quality-hardening/SDD-backend-hp-pipeline-scalability-and-quality-hardening-2026-11-27.md`

## Global Constraints

- **0 миграций Alembic:** схема SQLite не меняется; новые поля в JSON обратно совместимы со старыми записями БД.
- **Границы слоёв:** `uv run pytest tests/architecture` зелёный после каждого WP.
- **Замороженные SHA-256:** `_TRANSLATION_INSTRUCTIONS` и `snapshot_prompt_templates(..., reader_level="b2", profile="legacy-v1")` не изменяются (`test_baseline_b2_templates_sha256_unchanged`).
- **Неизменяемые контракты:** отсутствие `minItems` и `default_factory` на обязательном `used_fact_ids`; `public_url(llm.base_url)` в `route_snapshot`; 10 попыток и backoff `[0.005..0.02]` в `UnitOfWork` и `SqliteTaskStore`; сохранение заголовков `[bN] режим GLOSSARY_ONLY\n`; `test_ambiguous_identity_remains_separate`, `test_transitive_identity_conflict_triangle_order_independent_deterministic`, `test_dialogue_line_breaks_rejected_if_extra_line_lacks_dash`, `test_dialogue_validator.py`, `test_analysis_merge_scenes.py:1216-1248`, `test_book_layout.py:2047-2128`, `test_illustration_chapter.py:790-890`, `registry_sha256`.
- **Универсальность:** никаких хардкодов под книги, персонажей, локации или модели LLM.
- **Подпись коммитов:** каждый коммит содержит `Assisted-by: <роль> (<инструмент>, <модель>)` согласно `AGENTS.md`.

---

### Task 1 (WP1): Анализ глав — `lowercase_non_glossary`, регистр в `_opening_remark_names_speaker` и мягкое восстановление merge

**Files:**
- Modify: `src/linguary/application/analysis_merge.py:2776-2865, 3563-3596, 3767-3800, 4024-4037, 7199-7230, 7600-7644`
- Modify: `src/linguary/application/use_cases/analyze_chapter.py:576-615` (`_run_staged_narrative_pass`)
- Test: `tests/unit/application/test_analysis_merge_scenes.py`, `tests/unit/application/test_staged_analyze_chapter.py`

- [ ] **Step 1:** Написать падающие тесты на `first-years` + `First-years`, `"a fat little monk"` vs `"Fat Friar"`, антиспойлерный предохранитель на обе ветки `_explicit_speaker_if_remark_names`, идемпотентность `validate_narrative_response` и точечное понижение `explicit → inferred` через `error.violation` в `_run_staged_narrative_pass`.
- [ ] **Step 2:** Реализовать правки в `analysis_merge.py` и `analyze_chapter.py` по SDD WP1.
- [ ] **Step 3:** Прогнать `uv run pytest tests/unit/application/test_analysis_merge_scenes.py tests/unit/application/test_staged_analyze_chapter.py tests/architecture`.

---

### Task 2 (WP2): SQLite и пул воркеров — `BEGIN IMMEDIATE`, `_drain_worker`, `busy_timeout=15s` и `list_for_scope`

**Files:**
- Modify: `src/linguary/adapters/db/engine.py:28, 41-48` (`create_database_engine`)
- Modify: `src/linguary/adapters/runtime/task_store.py:82-171`
- Modify: `src/linguary/adapters/runtime/worker.py:167-171`
- Modify: `src/linguary/application/ports/persistence.py`
- Modify: `src/linguary/adapters/db/repositories.py`
- Modify: `src/linguary/application/illustrations.py:102-106`
- Test: `tests/integration/tasks/test_concurrent_claim.py`, `tests/integration/tasks/test_worker_pool.py`, `tests/unit/adapters/test_work_state_indexes.py`, `tests/integration/processing/test_translation_reliability_audit.py`, `tests/unit/application/test_recovery_partition.py`

- [ ] **Step 1:** Обновить `test_recovery_partition.py:2306` (сбросить `script["remaining"] = 0` перед `store.enqueue(...)`), и написать падающие тесты на конкурентные `SqliteTaskStore.save()`/`enqueue()` под lock, восстановление `_drain_worker()` при транзиентном lock (и падение по лимиту/дедлайну), а также `list_for_scope` в `dispatch_scope_task`.
- [ ] **Step 2:** Реализовать `_run_immediate_write` (сохраняя 10 попыток / `[0.005..0.02]`), `DEFAULT_BUSY_TIMEOUT_MS = 15_000` + `connect_args["timeout"]`, retry + deadline в `_drain_worker` (`from time import monotonic`), `list_for_scope` (`ORDER BY created_at, id`, импорт `TaskType` в `ports/persistence.py`) по SDD WP2 (не трогая `with session.begin():` в `unit_of_work.py`).
- [ ] **Step 3:** Прогнать `uv run pytest tests/integration/tasks tests/unit/adapters/test_work_state_indexes.py tests/integration/processing/test_translation_reliability_audit.py tests/unit/application/test_recovery_partition.py tests/architecture`.

---

### Task 3 (WP3): Иллюстрации — линейный `alias`, фильтрация `get_registry`, `used_fact_ids`, полный глоссарий и гигиена промптов

**Files:**
- Modify: `src/linguary/application/illustration_reference.py:328-396, 2360-2410, 2690-2703`
- Modify: `src/linguary/application/ports/illustrations.py`
- Modify: `src/linguary/adapters/db/illustration_repository.py:1603-1650`
- Modify: `src/linguary/application/illustration_view.py:1063-1145`
- Modify: `src/linguary/adapters/llm/illustration_payloads.py:35, 370`
- Modify: `src/linguary/application/illustration_context.py:490-594, 616-775`
- Modify: `src/linguary/application/illustration_selection.py:84, 103-153, 890, 2417-2452`
- Modify: `src/linguary/application/illustration_chapter.py:1047-1191, 2394-2540`
- Modify: `src/linguary/application/illustration_validation.py`
- Test: `tests/unit/application/test_illustration_reference.py`, `tests/unit/application/test_illustration_chapter.py`, `tests/unit/application/test_illustration_selection.py`, `tests/integration/web/test_illustrations_pages.py`

- [ ] **Step 1:** Обновить 3 существующих теста (`test_alias_blocking_batched_and_same_name_not_identity`: `sum == 11` и `len == 1` + новый тест батчинга на 12 `"John"`, `test_illustrations_pages.py:1144-1157`, `ent-eva` в `test_illustration_chapter.py:2742-2901`) и добавить новые тесты по SDD WP3.
- [ ] **Step 2:** Реализовать `звезда + цепочка` с `fact_counts` (по `f["subject_mention_key"]`) и сохранением `alias_of_hint`, `kinds` в `query_registry_items` / `build_illustration_registry_view`, `_MAX_PROMPT_USED_FACT_IDS = 128` (без `default_factory`), `collect_book_glossary_entries` + `approved_glossary_keys` в `DESIGN`, изоляцию `episodic`-фактов по сцене (`enumerate(textual_blocks(section))`), `find_speech_bubble_term` в `_validate_render_response`, `chunk_end_pos` в `plan_scopes_in_tx` / `_known_entities_context_for_chapter` (с вынесенной наружу проверкой `first_pos > effective_max_pos`), `_MAX_CANDIDATE_BLOCKS_PER_CHUNK = 64` и расширение fast-path `FACT_SCOPE` на 2–4 локальных `permanent`-фактах с лимитом 400/600.
- [ ] **Step 3:** Прогнать `uv run pytest tests/unit/application/test_illustration* tests/integration/web/test_illustrations_pages.py tests/integration/processing/test_illustration_reliability_integration.py tests/architecture`.

---

### Task 4 (WP4): Коррекция глав — полнобатчевый `batch_expectation` и снятие Catch-22 диалогов

**Files:**
- Modify: `src/linguary/application/use_cases/correct_chapter.py:773, 864-882, 933-956, 1320, 1407-1425`
- Modify: `src/linguary/application/validators/paragraphs.py:23-55`
- Modify: `src/linguary/application/reader_text.py:516-529`
- Modify: `src/linguary/composition/correction_runtime.py:1-90`
- Test: `tests/unit/application/test_correct_chapter_v2.py`, `tests/unit/composition/test_correction_runtime.py`, `tests/unit/application/test_result_validators.py`, `tests/unit/application/test_dialogue_validator.py`

- [ ] **Step 1:** Написать падающие тесты на 1-блочный repair-баг и `LLMPayloadError` в `CorrectChapterV2`, а также на 1-строчный / 3-строчный абзац с репликой и авторским предложением (и защиту от поглощённого соседнего абзаца).
- [ ] **Step 2:** Реализовать `batch_expectation` (через `expectation_for_unit` для V1 и V2 + `LLMPayloadError`), `_count_post_speech_narrative_sentences` в `paragraphs.py` и синхронизировать правила в `reader_text.py` и `correction_runtime.py`.
- [ ] **Step 3:** Прогнать `uv run pytest tests/unit/application/test_correct_chapter* tests/unit/composition/test_correction_runtime.py tests/unit/application/test_result_validators.py tests/unit/application/test_dialogue_validator.py tests/unit/application/test_context_builder.py tests/architecture`.

---

### Task 5 (WP5): `book_conditions` — развод контентных и транспортно-операционных настроек

**Files:**
- Modify: `src/linguary/composition/book_runtime.py:64-100, 431-578`
- Modify: `src/linguary/composition/correction_runtime.py:104-112, 151-169, 206-219`
- Test: `tests/integration/processing/test_pinned_runtime.py`, `tests/unit/composition/test_correction_runtime.py`

- [ ] **Step 1:** Обновить 2 прокси-теста в `test_pinned_runtime.py:38-50, 67-77` под новый контракт и добавить тесты на переопределение операционных параметров (без `TypeError` в `CorrectionRuntimeResolver`), сохранность `public_url`, совместимость со старыми записями БД и `_runtime_fingerprint`.
- [ ] **Step 2:** Реализовать `_OPERATIONAL_LLM_FIELDS` (мерж в единый `llm_kwargs`), `routes_compatible`, `route_snapshot` с `public_url` без env-прокси и обновление `_runtime_fingerprint`.
- [ ] **Step 3:** Прогнать `uv run pytest tests/integration/processing/test_pinned_runtime.py tests/unit/composition tests/architecture`.

---

### Task 6 (WP6): EPUB-вёрстка сносок и глоссарный валидатор

**Files:**
- Modify: `src/linguary/adapters/epub/layout_render.py:288-357`
- Modify: `src/linguary/adapters/epub/writer.py:181-191`
- Modify: `src/linguary/web/static/linguary.css:555-565`
- Modify: `src/linguary/application/validators/language.py:261-333`
- Test: `tests/unit/adapters/epub/test_layout_render.py`, `tests/unit/application/test_result_validators.py`, `tests/integration/layout/test_a2_harry_potter_regressions.py`

- [ ] **Step 1:** Написать падающие тесты на парные сноски в середине (`< length`) и в конце (`>= length`) абзаца, внутри `<em>`/`<a>` и при `None` text/tail, а также на `Moon`/`half-moon`, `Bill`/`a bill`, `The Sorting Hat`/`the Sorting Hat` + `Sorting`, `You-Know-Who`.
- [ ] **Step 2:** Реализовать `_insert_marker_group` (с `before or None` / `rest or None`), CSS `a.lg-ref + a.lg-ref { margin-left: 0.18em; }` и longest-match + регистро-/артикле-/дефисозависимый `_source_term_spans` в `validators/language.py`.
- [ ] **Step 3:** Прогнать `uv run pytest tests/unit/adapters/epub tests/unit/application/test_result_validators.py tests/integration/layout/test_a2_harry_potter_regressions.py tests/architecture`.

---

### Task 7 (WP7): Педагогика A2 — стихи, строфы/двустишия, `GLOSSARY_ONLY`, `fragment_assistance`, `speech_style`

**Files:**
- Modify: `src/linguary/application/context_builder.py:207, 1402, 2977-3004`
- Modify: `src/linguary/application/verse.py:87-198`
- Modify: `src/linguary/adapters/llm/schemas.py:218`
- Modify: `src/linguary/application/layout/notes.py:65-84, 118-120`
- Modify: `src/linguary/application/a2_core_vocab.py:1012-1032`
- Modify: `src/linguary/application/layout/book.py:1730-1839`
- Test: `tests/unit/application/test_context_builder.py`, `tests/unit/application/test_translation_context.py`, `tests/unit/application/test_analysis_merge_scenes.py`, `tests/unit/application/layout/test_notes.py`, `tests/unit/application/test_a2_core_vocab.py`, `tests/unit/application/layout/test_book_layout.py`

- [ ] **Step 1:** Написать падающие тесты на подстрочник стихов (с проверкой `not english_help` и `verse_block_ids` по всей главе), хвостовую строфу `#62–#65` и 2-строчное закавыченное заклинание `#215–#216`, динамическое напоминание `GLOSSARY_ONLY` в `_shape_instructions` (только `segment_lines and not english_help`, перед `_SEGMENT_TRANSLATION_RULE`), `"flicked it open"` ↔ `"to flick open"`, `"going"` в core A2 и `_repeated_speech_style_ids` через `subject_anchor` + `-in'` + `_SPEECH_STYLE_BOOK_CAP = 4`.
- [ ] **Step 2:** Реализовать правки по SDD WP7, проверяя сохранность `test_baseline_b2_templates_sha256_unchanged`, `test_translation_context.py` и `test_book_layout.py:2047-2128`.
- [ ] **Step 3:** Прогнать полный локальный гейт: `make check && uv run pytest -m "not slow"`.
