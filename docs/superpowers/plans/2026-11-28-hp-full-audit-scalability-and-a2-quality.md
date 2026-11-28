# HP Full Audit: Scalability, Pipeline Resilience & A2 Quality Implementation Plan

> **For agentic workers:** Реализуйте план пошагово по рабочим пакетам (WP), соблюдая TDD (сначала падающие тесты, затем минимальная чистая реализация, затем целевой прогон + `tests/architecture` + `ruff`/`ty`). Используйте чекбоксы (`- [ ]`) для отслеживания прогресса.

**Goal:** Устранить P0-блокеры обработки полной 18-главной книги (хоровая речь в гл. 6/10, взрывной рост и усечение промптов > 120 КБ в `narrative` и `correction`), повысить надёжность анализа (пакетный валидатор сегментов, анти-спойлер `known_names`, устойчивый кэш чанков при `retry`, батчинг ревью корректуры), защитить ключевые имена глоссария без дедлока `suggested_note`, закрыть методические и лингвистические пробелы профиля A2, отполировать экспорт EPUB 3.3 и защитить локальный мост субагентов от явного и тихого усечения UTF-8.

**Architecture:** Clean Architecture (`domain ← application ← adapters / web / cli`, 0 миграций Alembic, сохранение замороженных SHA-256 и `A2_CONTENT_PLAN_V8_VERSIONS`). План разделён на **Трек A (git репозитория: WP1–WP7)** и **Трек B (локальный скилл `.agents/`: WP8-local)**.

**Tech Stack:** Python 3.15+, Pydantic v2, lxml, SQLite (0 миграций), pytest, Ruff, ty.

**Specs (доработаны по итогам двух независимых ревью спецификаций и задетого кода `HEAD d9ff0f9`):**
- `docs/superpowers/specs/2026-11-28-hp-full-audit-scalability-and-a2-quality/BRIEF-hp-full-audit-scalability-and-a2-quality-2026-11-28.md`
- `docs/superpowers/specs/2026-11-28-hp-full-audit-scalability-and-a2-quality/ADR-hp-full-audit-scalability-and-a2-quality-2026-11-28.md`
- `docs/superpowers/specs/2026-11-28-hp-full-audit-scalability-and-a2-quality/SDD-backend-hp-full-audit-scalability-and-a2-quality-2026-11-28.md`

---

## Global Constraints & Exact Wave / File-Overlap Plan

- **0 миграций Alembic:** схема SQLite не меняется; все расширения обратно совместимы с существующими записями `version=3`.
- **Границы слоёв и статанализ:** `uv run pytest tests/architecture`, `uv run ruff check src tests` и `uv run ty check` зелёные после каждого WP. Чтение ZIP/OPF исходного EPUB (`clean_dc_text`, `read_source_language_tag`) размещается строго в `src/linguary/adapters/epub/inline.py`.
- **Замороженные SHA-256 и версии:** `test_baseline_b2_templates_sha256_unchanged` и `A2_CONTENT_PLAN_V8_VERSIONS` остаются зелёными и неизменными.
- **Покрытие тестами:** итоговый прогон `uv run pytest --cov=linguary --cov-fail-under=90` зелёный.
- **Разделение git и `.agents/`:** WP1–WP7 коммитятся в git (с обязательной подписью `Assisted-by: <роль> (<инструмент>, <модель>)` по `AGENTS.md`); WP8-local правит только `.agents/skills/linguary-subagent-run/` и никогда не добавляется в `git add`.

**Матрица пересечения файлов и безопасные волны параллелизации:**
- Общие файлы: `context_builder.py` (`WP1`, `WP2`, `WP4`, `WP5`, `WP6`), `analysis_merge.py` (`WP1`, `WP3`, `WP5`), `analyze_chapter.py` (`WP2`, `WP3`, `WP4`), `layout/book.py` (`WP1`, `WP5`), `adapters/epub/inline.py` (`WP5`, `WP7`), `adapters/epub/writer.py` (`WP5`, `WP7`), `cli/app.py` (`WP3`, `WP4`, `WP5`).
- **Волна 1 (полностью без пересечения файлов, параллельно):**
  - `WP1` (хоровая речь + `visual_facts`)
  - `WP7` (экспорт EPUB/CSS + `clean_dc_text` в `inline.py` + `writer.py` + `layout_render.py`)
  - `WP8-local` (локальный мост `.agents/skills/linguary-subagent-run/`)
- **Волна 2 (после `WP1`, последовательно из-за `analyze_chapter.py`):**
  - `WP2` (бюджеты промптов + батчинг ревью корректуры) → `WP3` (валидатор сегментов + анти-спойлер + устойчивый кэш `retry`)
- **Волна 3 (после `WP2`, `WP3`, `WP7`, последовательно из-за `context_builder.py` / `cli/app.py` / `reader_text.py`):**
  - `WP4` (глоссарий `exempt_type_review` + склонение титулов) → `WP5` (педагогика A2 + произношение + дубли глосс + `<sup>` + «Говорит X» + `legend.xhtml`) → `WP6` (лингвистика промптов + корректор)

---

## Трек A — Репозиторий Linguary (`src/linguary/`, `tests/`)

### Task 1 (WP1): Совместная (хоровая) речь и устранение коллизии `visual_facts.subject_anchor` (P0)

**Files:**
- Modify: `src/linguary/domain/content_plan.py:110-149, 220-234` (`ContextFact.joint`, `ContentPlan.__post_init__`)
- Modify: `src/linguary/application/llm_dto.py:430` (`QuotedContextFactResponse.joint`)
- Modify: `src/linguary/adapters/llm/schemas.py:455-473` (`QuotedContextFactSchema.joint`)
- Modify: `src/linguary/adapters/llm/payloads.py:844-852` (`_quoted_context_fact`)
- Modify: `src/linguary/application/analysis_pass_cache.py:125-135, 224-230` (`encode_pass`, `decode_pass`)
- Modify: `src/linguary/adapters/db/mapping.py:917-928` (`_boolean`, `_context_fact`)
- Modify: `src/linguary/application/analysis_merge.py:3272-3284, 3563-3625, 3666-3670, 4013-4024, 4055-4069` (`validate_narrative_response`, `_normalize_joint_speaker_facts`, `_quoted_fact`, `content_plan_from_response`, `_conflicting_subject_block_ids`)
- Modify: `src/linguary/application/context_builder.py:206-216` (`_NARRATIVE_ANALYSIS_INSTRUCTIONS`)
- Modify: `src/linguary/application/layout/book.py:957-1030` (`_speakers_for_block`)
- Modify: `src/linguary/application/layout/units.py:48-149, 252-270, 429-443` (`block_units`, `_speakers_by_part`, `_format_speaker_names`, `_speaker_note`; сохранить `LayoutUnit.speaker: str | None`)
- Test: `tests/unit/application/test_analysis_merge.py`, `tests/unit/llm/test_llm_response_schemas.py`, `tests/unit/llm/test_analysis_segment_contract.py`, `tests/unit/application/layout/test_book_layout.py`, `tests/unit/application/test_context_builder.py`, `tests/integration/web/test_result_and_export.py`

- [ ] **Step 1:** Написать падающие тесты:
  - `ContextFact(..., joint=True)` валиден для `speaker` (`explicit`/`inferred`), отклоняется для `gender`/`pronoun_referent`/`unknown`; `joint=False` не попадает в `to_dict()` и сохраняет прежние `canonical_id` и `ContentPlan.sha256`;
  - `ContentPlan` разрешает несколько `speaker` с `joint=True` на одном `subject_anchor`, но отклоняет конфликт `joint=False`, смесь `joint=True`/`False` или расхождение `certainty` для одного говорящего;
  - полный сквозной поток `validate_narrative_response` → `merge_narrative_fragments` → `combine_staged_analysis_responses` → `content_plan_from_response` принимает `‘Hermione!’ they said together.` и `‘Harry Potter,’ chorused the twins.` (с `joint=True` и через защитную нормализацию);
  - негативные тесты `_JOINT_SPEECH_RE`: слово `together` внутри кавычек (`‘Let’s go together,’ said Harry.`) и соседнее предложение (`‘Hermione!’ they said together. ‘Run!’ said Harry.`) не скрывают реальный конфликт атрибуции;
  - атомарность `_explicit_speaker_if_remark_names` для `joint=True`-группы (если в ремарке назван только один из участников, вся группа остаётся `inferred`);
  - round-trip `encode_pass`/`decode_pass` и `_encode_content_plan`/`_decode_content_plan` (старые записи без `'joint'`, новые с `joint=True`, отказ `_boolean` на `0`/`1`);
  - `layout/book.py` + `layout/units.py`: детерминированный порядок «Говорят Гарри Поттер и Рон», отсутствие склейки двух разных `joint=False` реплик в одном юните, сохранение `LayoutUnit.speaker: str | None` и валидная сериализация `ResultUnitResponse.of_unit(unit)`.
- [ ] **Step 2:** Реализовать WP1 строго по SDD WP1 (синхронизировать ассерты `QuotedContextFactSchema` в `test_llm_response_schemas.py` / `test_analysis_segment_contract.py` при необходимости).
- [ ] **Step 3:** Прогнать `uv run pytest tests/unit/application/test_analysis_merge.py tests/unit/llm/test_llm_response_schemas.py tests/unit/llm/test_analysis_segment_contract.py tests/unit/application/layout/test_book_layout.py tests/unit/application/test_context_builder.py tests/architecture && uv run ruff check src tests && uv run ty check`.

---

### Task 2 (WP2): Байтовые бюджеты промптов `narrative` / `correction` и батчинг `chapter_correction_review` (P0 + P1)

**Files:**
- Modify: `src/linguary/application/context_builder.py:115-120, 1940-1985, 2027-2067, 3208-3232, 3341-3405` (обогащение совпавших `TermConcept(SUGGESTED)` + компактный бэкфилл, байтовые бюджеты, дедупликация `findings.candidates`)
- Modify: `src/linguary/application/use_cases/analyze_chapter.py:100, 965-995, 1140-1160, 1230-1243` (импорт и сплит `LLMContextCapacityError` в `_run_staged_fragments` / `_request_staged`, компактные `findings`, сброс `assistant_response` при > 65 КБ / `raw_bytes > 20 КБ`)
- Modify: `src/linguary/application/use_cases/correct_chapter.py:200-277, 303-335, 461-576, 1480-1496, 1750-1855` (безусловная очистка и фильтрация `scenes` через `section.ordered_blocks`, поблочный байтовый чанкинг `propose`/`review` в обоих вызовах, батчинг ревью по главе с проверкой `group['candidate_ids']`)
- Test: `tests/unit/application/test_context_builder.py`, `tests/unit/application/test_staged_analyze_chapter.py`, `tests/unit/application/test_correction_inputs.py`, `tests/integration/processing/test_segment_correction_lifecycle.py`, `tests/e2e/test_correction_v2_pipeline.py`

- [ ] **Step 1:** Обновить `tests/unit/application/test_context_builder.py:3119` (компактный бэкфилл для несовпавшей `Alexandra`) и написать падающие тесты:
  - совпавшие `TermConcept(SUGGESTED)` (`test_context_builder.py:3110-3116`) по-прежнему обогащаются полными `note`/`description`, а 200 несовпавших прошлых кандидатов укладываются в `DEFAULT_MAX_BACKFILL_BYTES = 12_000` (итоговый промпт < 45 КБ UTF-8);
  - `LLMContextCapacityError` (без `.usage`) в `_run_staged_fragments` делит фрагмент пополам без `AttributeError` и сохраняет накопленный `usage` (при 1 блоке ошибка пробрасывается);
  - при `LLMResponseFormatError` с 40 КБ `error.raw_content` (и при сумме с `repair_turns` > 65 КБ) `assistant_response` сбрасывается в `None` и 40 КБ JSON не попадает в `repair_turns`;
  - `_llm_chapter_analysis` удаляет `visual_facts`/`visual_assumptions`/UUID и при `allowed_block_ids=None`, и при оконной фильтрации по `section.ordered_blocks` (сравнение до замены на алиасы);
  - `_chunk_correction_blocks` с `max_request_bytes=72_000` не деградирует до 1 блока при большом `chapter_analysis` главы и корректно учитывает двойной вес кандидатов в режиме `review` (включая вызов внутри `_review_group:1763`);
  - `CorrectChapterV2` объединяет 3 кандидатов разных `group_id` в 1 вызов `chapter_correction_review` и фиксирует `unchanged_block_ids` через `group['candidate_ids']` (в т. ч. при разбиении 1 группы на 2 чанка ревью).
- [ ] **Step 2:** Реализовать WP2 по SDD WP2, не затрагивая `_ANALYSIS_INSTRUCTIONS` и `test_baseline_b2_templates_sha256_unchanged`.
- [ ] **Step 3:** Прогнать `uv run pytest tests/unit/application/test_context_builder.py tests/unit/application/test_staged_analyze_chapter.py tests/unit/application/test_correction_inputs.py tests/integration/processing/test_segment_correction_lifecycle.py tests/e2e/test_correction_v2_pipeline.py tests/architecture && uv run ruff check src tests && uv run ty check`.

---

### Task 3 (WP3): Пакетный валидатор сегментов, анти-спойлер и устойчивый кэш фрагментов при `retry` (P1)

**Files:**
- Modify: `src/linguary/application/analysis_merge.py:262-350, 2873-2933` (`plan_items_from_response`, `_plan_item_from_response`, `_collect_spoiler_name_leaks`)
- Modify: `src/linguary/application/use_cases/analyze_chapter.py:605-610, 1025-1065, 1085-1099, 1118-1255, 3126-3160` (`fragment_known_names`, кэшируемый пофрагментный даунгрейд, `known_names` в вызове уровня главы, авто-сброс битого кэша и при сбое инкрементального `merge_chapter_analysis`)
- Modify: `src/linguary/application/use_cases/retry_task.py:40-86` (`RetryTaskCommand.clear_cache`, сохранение кэша по умолчанию)
- Modify: `src/linguary/cli/app.py` (флаг `retry --clear-cache`)
- Test: `tests/unit/application/test_analysis_merge.py`, `tests/unit/application/test_staged_analyze_chapter.py`, `tests/integration/processing/test_retry_task_recovery_guards.py`, `tests/integration/cli/test_cli.py`

- [ ] **Step 1:** Обновить `tests/integration/processing/test_retry_task_recovery_guards.py:299-455` и написать падающие тесты:
  - односегментный блок с согласованным `segments` нормализуется без ошибки; 3 битых блока (`b16`, `b28`, `b51`) возвращают единую ошибку со всеми 3 алиасами `bN` (включая `TreatmentPlanError` и сохранение `require_all=True`);
  - `_collect_spoiler_name_leaks` пропускает `mapped_facts` для `known_names` (`Dudley`, `Sorting Hat`, `Professor McGonagall`);
  - заголовок секции и предыдущие заголовки входят в `fragment_known_names`; пофрагментный даунгрейд `explicit → inferred` сохраняет пониженный `payload` в `analysis_pass_results` и `parts`, а вызов уровня главы (`L605`) не понижает имена из `known_names`;
  - повреждённый/невалидный кэш фрагмента автоматически сбрасывается (cache-miss), сбой инкрементального `merge_chapter_analysis` очищает фрагментный кэш;
  - `RetryTask` сохраняет фрагментные кэши по умолчанию (включая `VALIDATION`), а `clear_cache=True` (`cli retry --clear-cache`) очищает фрагментные кэши, сохраняя целый `NARRATIVE`.
- [ ] **Step 2:** Реализовать WP3 по SDD WP3.
- [ ] **Step 3:** Прогнать `uv run pytest tests/unit/application/test_analysis_merge.py tests/unit/application/test_staged_analyze_chapter.py tests/integration/processing/test_retry_task_recovery_guards.py tests/integration/cli/test_cli.py tests/architecture && uv run ruff check src tests && uv run ty check`.

---

### Task 4 (WP4): Защита ключевых `exempt_type_review` имён в `glossary_snapshot` и грамматические правила анализа (P0/P1)

**Files:**
- Modify: `src/linguary/application/glossary_review.py:75-155, 255-270, 435-485, 550-572` (`GlossaryReviewItem.exempt_type_review`, `is_auto_mandatory`, `glossary_snapshot` с `note=""` для автопродвинутых записей)
- Modify: `src/linguary/application/context_builder.py:198, 207, 447-469` (правило склонения титулов перед женской фамилией, второстепенные персонажи/клички, документация `scaled_chapter_note_limit`)
- Modify: `src/linguary/adapters/config/settings.py:329` (описание масштабируемого `language_notes_per_chapter`)
- Modify: `src/linguary/application/use_cases/analyze_chapter.py` (логирование эффективного лимита заметок главы)
- Modify: `src/linguary/cli/app.py` (отображение `auto_mandatory` в `glossary list`)
- Test: `tests/unit/domain/test_glossary_review.py`, `tests/unit/application/test_book_glossary.py`, `tests/unit/application/test_approve_glossary.py`, `tests/unit/application/test_name_translation_analysis.py`, `tests/unit/application/test_name_translation_evaluation.py`, `tests/unit/application/test_context_builder.py`, `tests/integration/cli/test_cli.py`

- [ ] **Step 1:** Написать падающие тесты:
  - ключевые `exempt_type_review` сущности (`PERSON`/`PLACE`/`ORGANIZATION`/`FICTIONAL_CONCEPT`, `is_frequent` или `>= 2` упоминаний/глав, непустой `suggested_translation`) не попадают в ручную очередь `awaiting_decision`, но получают `TermStatus.APPROVED` и `note=""` в `glossary_snapshot(...).mandatory_entries` (не вызывая `glossary_fact_conflict` в `validate_fact_glossary` при противоречивой `suggested_note`!);
  - `mention_count == 0` при `is_frequent=False`, пустой `suggested_translation`, разовый `TERM` и пользовательский `SKIP` (`TermConcept(status=SUGGESTED)`) остаются `TermStatus.SUGGESTED`;
  - `_NARRATIVE_ANALYSIS_INSTRUCTIONS` содержит правило склонения титулов мужского рода (`профессор`, `доктор`, `директор`) перед женской фамилией;
  - `cli glossary list --json` выводит `auto_mandatory`.
- [ ] **Step 2:** Реализовать WP4 по SDD WP4 (синхронизировать тесты `test_name_translation_analysis.py` / `test_name_translation_evaluation.py` / `test_book_glossary.py`, не ломая `test_baseline_b2_templates_sha256_unchanged`).
- [ ] **Step 3:** Прогнать `uv run pytest tests/unit/domain/test_glossary_review.py tests/unit/application/test_book_glossary.py tests/unit/application/test_approve_glossary.py tests/unit/application/test_name_translation_analysis.py tests/unit/application/test_name_translation_evaluation.py tests/unit/application/test_context_builder.py tests/integration/cli/test_cli.py tests/architecture && uv run ruff check src tests && uv run ty check`.

---

### Task 5 (WP5): Педагогика A2 — произношение, порог `GLOSSARY_ONLY`, анти-дубли глосс, `<sup>`, «Говорит X» и `legend.xhtml` (P0/P1)

**Files:**
- Modify: `src/linguary/domain/document.py` (`detect_english_variant`)
- Modify: `src/linguary/adapters/epub/inline.py` (`read_source_language_tag`)
- Modify: `src/linguary/application/analysis_merge.py:4832-4840, 6012-6036` (`_promote_overloaded_glossary_only` для блоков и сегментов после `_promote_a2_stoplist_item`)
- Modify: `src/linguary/application/context_builder.py:231, 2166-2192` (ограничение «пинг-понга» + запрет разрыва реплики пополам, синхронизация коротких реплик с 1 глаголом речи, скрытые ловушки B1)
- Modify: `src/linguary/application/layout/notes.py:544-630, 1000-1018` (`_suppress_hints_covered_by_notes` с сохранением IPA и сегментным инвариантом, `_find_span` + `_trim_terminal_punctuation` с защитой `flyin’`/`Mr.`)
- Modify: `src/linguary/application/a2_core_vocab.py:766-960, 1138-1170` (профильные наборы + 2-словные A1-существительные вроде `school uniform`)
- Modify: `src/linguary/application/layout/book.py:542, 556, 923-1030` (`_strip_quoted_speech`, `_part_explicitly_names_speaker` с защитой однофамильцев/притяжательных/предлогов, комментарий о статусе `who_is_who`/`review`)
- Modify: `src/linguary/application/layout/legend.py:24-46, 95-108` («Пометка в начале примечания», разведение глосс/заметок, описание `Previously in…`)
- Modify: `src/linguary/adapters/epub/book_pages.py:52-68` (`legend_document(..., source_language=None, include_example=False)`)
- Modify: `src/linguary/adapters/epub/writer.py:499-505` (передача `source_language` и `include_example` в `legend_document`)
- Modify: `src/linguary/web/routes/books.py:112-143`, `src/linguary/web/templates/books/detail.html:34-55`, `src/linguary/cli/app.py:144-149, 480-508` (автоопределение `uk`/`us`, предвыбор A2 только в `detail.html` при `IMPORTED` HTML, `--pronunciation-variant auto` в CLI)
- Test: `tests/unit/domain/test_canonical_document.py`, `tests/unit/application/test_analysis_merge.py`, `tests/unit/application/layout/test_notes.py`, `tests/unit/application/test_a2_core_vocab.py`, `tests/unit/application/layout/test_book_layout.py`, `tests/unit/application/layout/test_legend.py`, `tests/integration/epub/test_learner_book_pages.py`, `tests/integration/layout/test_a2_harry_potter_regressions.py`, `tests/integration/web/test_conditions_choices.py`, `tests/integration/cli/test_cli.py`

- [ ] **Step 1:** Обновить ассерты формулировки легенды в `tests/unit/application/layout/test_legend.py` и написать падающие тесты:
  - `detect_english_variant("en-GB") == "uk"`, `detect_english_variant("en-US") == "us"`, `detect_english_variant("en") is None`;
  - `GLOSSARY_ONLY` блок и `GLOSSARY_ONLY` сегмент с ≥ 5 не-A2 словами (в т. ч. после `_promote_a2_stoplist_item`) повышаются до `TRANSLATE`, без регрессий в `test_a2_harry_potter_regressions.py`;
  - `_suppress_hints_covered_by_notes` подавляет дубли (`Fancy`/`Fancy seeing`, `wrong`/`How very wrong he was.`, `to get`/`Make Harry get it.`), НЕ подавляет глоссы под длинной несвязанной заметкой (> 6 слов), сохраняет `ipa` и соблюдает сегментный инвариант `GLOSSARY_ONLY`;
  - `_trim_terminal_punctuation`: `How very wrong he was.` → до `.`; `as we was flyin’` → `’` сохраняется; `Mr.` → `.` сохраняется; `‘No.’` → до `.`;
  - `is_trivial_a1_hint` для `bacon`, `perfect`, `mistake`, `wrong`, `to enter`, `to celebrate`, `a mistake`, `violet`, `unusual`, `darkness`, `angrily`, `school uniform`;
  - `_part_explicitly_names_speaker`: `‘Oh.’ Dudley sat down heavily.` не выводит «Говорит Дадли», но контрпримеры (`Mrs Dursley` vs `Mr Dursley`, `Harry’s aunt`, `looking at Dudley`, `‘The Dursleys’ car…’`) сохраняют плашку;
  - `legend_document`: `test_learner_book_pages.py:63-65` остаётся зелёным (без мини-примера при `_ORIGINAL`+`_IPA`), а при `include_example=True, source_language="en"` рендерится нейтральный мини-пример;
  - `detail.html` предвыбирает `pronunciation_help` для A2 (не трогая `partials/quality_review.html` и не читая ZIP вне `IMPORTED` HTML); `cli analyze --pronunciation-help --pronunciation-variant auto` корректно разрешает `uk`/`us`.
- [ ] **Step 2:** Реализовать WP5 по SDD WP5 (сохранив `BookConditionsService.create(pronunciation_help_enabled=False)` и `test_baseline_b2_templates_sha256_unchanged`).
- [ ] **Step 3:** Прогнать `uv run pytest tests/unit/domain/test_canonical_document.py tests/unit/application/test_analysis_merge.py tests/unit/application/layout tests/unit/application/test_a2_core_vocab.py tests/integration/epub/test_learner_book_pages.py tests/integration/layout/test_a2_harry_potter_regressions.py tests/integration/web/test_conditions_choices.py tests/integration/cli/test_cli.py tests/architecture && uv run ruff check src tests && uv run ty check`.

---

### Task 6 (WP6): Лингвистическое усиление промптов перевода, заметок, пересказа и балансировка корректора (P1)

**Files:**
- Modify: `src/linguary/application/reader_text.py:508-560` (`_A2_NOTE_DISCIPLINE`, `_A2_DIALOGUE_AND_CALQUE_RULES`, `learner_instructions`)
- Modify: `src/linguary/application/context_builder.py:237-250` (`_PEDAGOGICAL_ANALYSIS_INSTRUCTIONS`)
- Modify: `src/linguary/composition/correction_runtime.py:85-116` (`CORRECTION_PROMPT_V2`, `CORRECTION_REVIEW_PROMPT`)
- Test: `tests/unit/application/test_context_builder.py`, `tests/unit/composition/test_correction_runtime.py`

- [ ] **Step 1:** Написать падающие тесты на присутствие новых правил в A2 / `content-plan-v3` промптах (`speech_style` без анализа русского перевода + корректный якорь, лаконичность заметок с пометкой «для уровня A2», географическая точность заметок, сквозной регистр персонажей, единство рода животных, отказ от избыточных притяжательных местоимений, согласование времён в `Previously in…`, защита корректора от гиперкоррекции несобственно-прямой речи) при неизменном `test_baseline_b2_templates_sha256_unchanged`.
- [ ] **Step 2:** Внести точечные правки в `reader_text.py`, `context_builder.py` и `correction_runtime.py` по SDD WP6.
- [ ] **Step 3:** Прогнать `uv run pytest tests/unit/application/test_context_builder.py tests/unit/composition/test_correction_runtime.py tests/integration/layout/test_a2_harry_potter_regressions.py tests/architecture && uv run ruff check src tests && uv run ty check`.

---

### Task 7 (WP7): Полировка экспорта EPUB, метаданных, CSS и политики front matter (P1/P2)

**Files:**
- Modify: `src/linguary/adapters/epub/inline.py` (`clean_dc_text`)
- Modify: `src/linguary/adapters/epub/reader.py:547-566` (`clean_dc_text` при импорте)
- Modify: `src/linguary/adapters/epub/writer.py:113-308, 539-582, 705-734, 992-1135, 1462-1536` (`clean_dc_text` в `_read_source_dc_metadata` и `_append_metadata`, `_exported_sections` / `_is_dropped_front_matter`, `.lg-cover` включая opaque SVG, ведущий `pagebreak` до `_reserve_opaque_ids`, `.lg-unit-long`, `_cover_image_href` с `sections_by_id`, `properties="cover-image"`)
- Modify: `src/linguary/adapters/epub/layout_render.py:59-85` (рендер `.lg-cover` и модификатора `.lg-unit-long`)
- Modify: `src/linguary/web/static/linguary.css` (синхронизация `.lg-cover` и `.lg-unit.lg-unit-long`)
- Test: `tests/unit/test_epub_writer.py`, `tests/unit/adapters/epub/test_layout_render.py`, `tests/integration/epub/test_epub_export_content.py`, `tests/integration/epub/test_learner_book_pages.py`

- [ ] **Step 1:** Синхронизировать существующие тесты front matter (`test_epub_export_content.py:769`, `test_learner_book_pages.py:59,76,181`, `test_epub_writer.py:627-642`) и написать падающие тесты:
  - `clean_dc_text` очищает `<p>Text <em>here</em>` при импорте, fallback-чтении и повторном экспорте из БД (`_append_metadata`);
  - `properties="cover-image"` проставляется в `content.opf`, а `_cover_image_href` не падает на синтетических главах (`linguary-legend` и др.);
  - обложка рендерится в `.lg-cover` без `.lg-unit` и `<p class="lg-src">` (как для `BlockType.IMAGE`, так и для opaque SVG);
  - ведущий пустой `pagebreak` (и `BlockType.PAGE_BREAK`, и opaque `p.pagebreak`) удаляется до `_reserve_opaque_ids`, переносит `id` на `<section epub:type="chapter">` и остаётся валидной целью ссылок;
  - длинный юнит (оригинал + перевод + заметки > 450 симв. или ≥ 4 предложений) получает `lg-unit-long`;
  - multi-book EPUB с `title`/`dedication` между книгами сохраняет нумерацию глав и `previous` для `Previously in…`.
- [ ] **Step 2:** Реализовать WP7 по SDD WP7, проверив все инварианты CSS в `test_layout_render.py`.
- [ ] **Step 3:** Прогнать полный финальный гейт Трека A:
  - `uv run pytest tests/unit/test_epub_writer.py tests/unit/adapters/epub tests/integration/epub tests/architecture`
  - `uv run ruff check src tests`
  - `uv run ty check`
  - `uv run pytest -m "not slow"`
  - `uv run pytest --cov=linguary --cov-fail-under=90`

---

## Трек B — Локальный скилл `linguary-subagent-run` (вне git, `.agents/`)

### Task 8 (WP8-local): Превентивный потолок UTF-8, компактизация схемы, единичная запись метрик и прогресс-монитор (P0/P1)

**Files (не коммитить в git репозитория Linguary!):**
- Modify: `.agents/skills/linguary-subagent-run/scripts/bridge_core.py`
- Modify: `.agents/skills/linguary-subagent-run/SKILL.md`
- Test: `.agents/skills/linguary-subagent-run/tests/test_subagent_bridge.py`

- [ ] **Step 1:** Написать падающие unit-тесты в `test_subagent_bridge.py`:
  - `compact_schema_for_prompt` удаляет метаданные `"title"`/`"description"`, сохраняя поля `"title"`/`"description"` внутри `"properties"`;
  - превентивный потолок `prompt_bytes > PROMPT_HARD_LIMIT_BYTES` возвращает HTTP `400` `context_length_exceeded` без вызова `worker` и пишет ровно **одну** запись в `bridge_metrics.json`;
  - `_is_truncated_prompt_refusal` срабатывает на первом ответе, в ветке `expect_json=False` и на ремонтном ходе, возвращая HTTP `400` `context_length_exceeded` и записывая ровно **одну** запись в `bridge_metrics.json`;
  - `bridge_metrics.json` фиксирует `prompt_bytes`, `truncation_risk` и `repair_errors`;
  - фоновый read-only монитор `run_cmd(..., db_path=..., progress_interval_seconds=...)` безопасно читает SQLite (`?mode=ro`), не падает при отсутствии/занятости БД и чисто завершается при отмене.
- [ ] **Step 2:** Реализовать WP8-local в `bridge_core.py` и обновить `SKILL.md` по SDD WP8-local.
- [ ] **Step 3:** Прогнать `uv run pytest .agents/skills/linguary-subagent-run/tests/test_subagent_bridge.py` и убедиться через `git status`, что `.agents/` не отслеживается git.
