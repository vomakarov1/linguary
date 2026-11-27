# Subagent Audit Fixes & A2 Prompt Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Устранить 6 групп верифицированных дефектов приложения, экспорта EPUB, CLI и промптов/валидаторов A2, выявленных сквозным прогоном `hp-ch1-3-a2` и двумя независимыми ревью: сделать `StartProcessing` атомарным относительно маршрута `book_conditions` во всех точках входа, исправить fallback `is_cover_section` и перенос `dc:rights`/`dc:description` / очистку ресурсов / CSS в EPUB и Web UI, довести CLI до операционной полноты для автономных прогонов, удалить ошибочный хардкод «Прайвет-драйв» и усилить фонетику `sound`, исправить кореференцию сегментов / калибровку A2 / снятие артиклей в дедупликации `glossary_hints`, а также закрыть лишнее тире перед авторским текстом после реплики — без миграций БД и без регрессий.

**Architecture:** Изменения строго соблюдают Clean Architecture (`domain ← application ← adapters / web / cli`) и разбиты на 6 слабосвязанных рабочих пакетов (WP1–WP6), каждый со своим TDD-циклом, тестовым гейтом и отдельным коммитом.

**Tech Stack:** Python 3.15+, Pydantic v2, SQLite (без изменения схемы), pytest, Ruff, ty.

**Spec (одобрены по итогам 3 независимых ревью кода и архитектуры; все замечания устранены):**
- `docs/superpowers/specs/2026-11-28-subagent-audit-fixes-and-prompt-hardening/BRIEF-subagent-audit-fixes-and-prompt-hardening-2026-11-28.md`
- `docs/superpowers/specs/2026-11-28-subagent-audit-fixes-and-prompt-hardening/ADR-subagent-audit-fixes-and-prompt-hardening-2026-11-28.md`
- `docs/superpowers/specs/2026-11-28-subagent-audit-fixes-and-prompt-hardening/SDD-backend-subagent-audit-fixes-and-prompt-hardening-2026-11-28.md`

## Global Constraints

- **0 миграций Alembic:** схема SQLite (`books` в `adapters/db/models.py` и `mapping.py`) не меняется; `dc:rights`/`dc:description` при экспорте из БД добираются резервным чтением из `request.source_epub_path`.
- **Границы слоёв:** `uv run pytest tests/architecture` зелёный после каждого WP. Никаких импортов `composition`/`adapters` из `application`/`domain`.
- **Замороженные SHA-256 и версии:** `_TRANSLATION_INSTRUCTIONS`, `_ANALYSIS_INSTRUCTIONS` (`L2427-2430`) и `snapshot_prompt_templates(source_language="en", target_language="ru", reader_level="b2", profile="legacy-v1")` (с/без `pronunciation_help_enabled`, `name_translation_policy=None`) остаются побайтово неизменными (`test_baseline_b2_templates_sha256_unchanged`). `A2_CONTENT_PLAN_V8_VERSIONS` (`prompts: 8`, `context_builder: 6`) не меняется.
- **Неизменяемые контракты:** отсутствие `minItems` и `default_factory` на обязательных полях OpenAI strict JSON Schema; сохранение `_NARRATIVE_GLOSSARY_WORD_LIMIT = 45`, `_HINT_LOCAL_WINDOW_BLOCKS = 20`, `glossary_only_rescue_by_bucket`, `test_dialogue_line_breaks_rejected_if_extra_line_lacks_dash`, `test_pinned_runtime.py`, `test_layout_render.py`, `test_book_layout.py`, `test_epub_export_content.py`.
- **Универсальность:** 0 новых хардкодов под конкретные книги, персонажей, локации или модели LLM (ошибочный хардкод «Прайвет-драйв» удаляется в WP4 из обеих точек `context_builder.py`).
- **Подпись коммитов:** каждый коммит содержит `Assisted-by: <роль> (<инструмент>, <модель>)` согласно `AGENTS.md`.

---

### Task 1 (WP1): Атомарность `StartProcessing` и единый `route` во всех точках старта

**Files:**
- Modify: `src/linguary/composition/book_runtime.py:454-540` (`BookRuntimeResolver.validate_route`)
- Modify: `src/linguary/application/use_cases/approve_glossary.py:1-25, 455-548` (`StartProcessing`)
- Modify: `src/linguary/composition/container.py:211-230, 466-478` (`Application.start_processing`, `_start_book_processing`)
- Modify: `src/linguary/web/routes/processing.py:172-186` (`start_processing`)
- Test: `tests/unit/application/test_approve_glossary.py`, `tests/integration/processing/test_pinned_runtime.py`, `tests/integration/web/test_glossary_and_processing.py`

- [ ] **Step 1:** Обновить `tests/integration/web/test_glossary_and_processing.py:406` под единый структурированный контракт `{"reason_code": "conditions_unavailable"}` и написать падающие unit- и integration-тесты:
  - при ошибке `validate_route` (`ConditionsError("route_unavailable")`) `StartProcessing` откатывает транзакцию целиком: статус книги остаётся `GLOSSARY_REVIEW`, `runs.active_for_book()` возвращает `None`, `work.get_binding("run", ...)` не создаётся;
  - при полном совместимом `pinned_route` (`{"provider", "base_url", "proxy", "route_id"}`) `work_bindings["run"].payload["correction"]["route"]` совпадает с `conditions.payload["route"]`, а минимальные моки `{"provider": ...}` в существующих unit-тестах не ломаются;
  - сценарий смены порта (`127.0.0.1:51247` → `127.0.0.1:51594`) перед `process` (статус `GLOSSARY_REVIEW`) и при `auto_start_processing` (статус `READY_TO_PROCESS`, без `ProcessingRun`): отказ с `route_unavailable` без перехода в `PROCESSING`, а после возврата `51247` — успешный перевод и `chapter_correction`;
  - `POST /books/{id}/process` при `route_unavailable` возвращает `409` со структурированным `{"reason_code": "route_unavailable"}`.
- [ ] **Step 2:** Добавить `BookRuntimeResolver.validate_route`, keyword-only `validate_route: Callable[[Mapping[str, Any] | None], None] | None = None` и безопасное копирование 4-ключевого `route` в `StartProcessing`, подключить его в **обеих** точках `container.py` (с живым `runtime_settings.current.settings` в `_start_book_processing`) и добавить `except ConditionsError` (вокруг `start_processing()` и `processing.start()`) в `web/routes/processing.py:176-181`.
- [ ] **Step 3:** Прогнать `uv run pytest tests/unit/application/test_approve_glossary.py tests/unit/application/test_block_processing.py tests/integration/processing/test_pinned_runtime.py tests/integration/web tests/architecture`.

---

### Task 2 (WP2): Экспорт EPUB — fallback `is_cover_section`, `dc:rights`/`dc:description`, очистка ресурсов и CSS

**Files:**
- Modify: `src/linguary/domain/document.py:358-370, 727-734` (`BookMetadata`, fallback-ветка `is_cover_section`)
- Modify: `src/linguary/adapters/epub/reader.py` (чтение `dc:rights`, `dc:description`)
- Modify: `src/linguary/adapters/epub/writer.py:111-285, 299-341, 586-625, 1315-1330` (`_STYLESHEET`, конвейер `write`, `_resource_payloads`, `_read_source_dc_metadata`, `_append_metadata`)
- Modify: `src/linguary/adapters/epub/layout_render.py:242-254` (`_append_notes` — класс `lg-note-plain`)
- Modify: `src/linguary/web/static/linguary.css` (синхронизация `.center`, `.pagebreak`, `p.pagebreak`, `.lg-notes > li.lg-note-plain`)
- Modify: `src/linguary/web/templates/partials/result_block.html:70` (класс `lg-note-plain` при `item.note.number is none`)
- Test: `tests/unit/domain/test_canonical_document.py`, `tests/unit/test_epub_writer.py`, `tests/unit/adapters/epub/test_layout_render.py`, `tests/integration/epub/test_epub_export_content.py`, `tests/integration/web/test_templates.py`

- [ ] **Step 1:** Написать падающие тесты:
  - `is_cover_section` возвращает `False` для титульной секции без `cover*`-имени с `BlockType.HEADING` + `BlockType.IMAGE` и сохраняет `True` для всех явных обложек (`cover.html`, заголовок `"Cover"`, чисто графическая обложка);
  - `dc:rights` и `dc:description` переносятся в экспортированный `content.opf` как из `document.metadata`, так и через fallback-чтение `source_epub_path` после загрузки книги из SQLite (0 миграций);
  - при `request.layout is not None` неиспользуемые CSS, `.xpgt` и картинки отброшенных секций не попадают в архив, OPF-манифест и `ExportWarnings` (`MISSING_RESOURCE`), а используемые `<img src>` (в т.ч. `../Images/fig%201.jpg`), SVG `<image xlink:href>`, `<object data>` и CSS `url(...)` сохраняются;
  - ненумерованная заметка в `.lg-notes` получает `class="lg-note-plain"` и в EPUB (`layout_render.py`), и в Web UI (`result_block.html`), нумерованная — нет; `.center`, `.pagebreak`, `p.pagebreak`, `.lg-notes > li.lg-note-plain` есть в `_STYLESHEET` и `web/static/linguary.css`.
- [ ] **Step 2:** Реализовать правки по SDD WP2, не трогая ранние ветки `is_cover_section` (`L715-725`), таблицу `books` и `?v=5` в `base.html` (или синхронно обновляя `test_templates.py:965` при смене версии).
- [ ] **Step 3:** Прогнать `uv run pytest tests/unit/domain/test_canonical_document.py tests/unit/test_epub_writer.py tests/unit/adapters/epub tests/integration/epub/test_epub_export_content.py tests/integration/web/test_templates.py tests/architecture`.

---

### Task 3 (WP3): Операционная полнота CLI (`glossary`, `corrections`, `export --strict-corrections`)

**Files:**
- Modify: `src/linguary/cli/app.py` (подкоманды `glossary`, `corrections` с `dest="json_output"`, флаг `--strict-corrections` до `export_book`)
- Test: `tests/integration/cli/test_cli.py`

- [ ] **Step 1:** Написать падающие CLI-тесты на:
  - `glossary list <book_id> [--pending-only] [--json]` и `glossary accept-all <book_id> [--json]` (разблокировка `cli process` при `model_flagged_review = true`);
  - `corrections list <book_id> [--awaiting-only] [--json]` (включая `section_id`, `proposal_sha256`, `input_sha256`) и `corrections resolve <book_id> <candidate_id> --action {accept|keep} [--reason ...] [--proposal-sha256 ...] [--input-sha256 ...] [--json]` (включая отказ при устаревшем CAS-хэше);
  - `export <book_id> --output ... --strict-corrections` (код возврата `1` **без записи файла** при наличии `awaiting_human`, код `0` после `corrections resolve` или без флага с предупреждением в `stderr`).
- [ ] **Step 2:** Реализовать парсеры и обработчики `glossary`, `corrections` (с проверкой `application.resolve_correction_candidate is not None` и `dest="json_output"`) и проверку `--strict-corrections` до `export_book` напрямую через `tx.work` без подавления `WorkStateError` по SDD WP3.
- [ ] **Step 3:** Прогнать `uv run pytest tests/integration/cli tests/architecture`.

---

### Task 4 (WP4): Промпты анализа и глоссария — удаление «Прайвет-драйв», фонетика `sound` и обязательность вложенных полей

**Files:**
- Modify: `src/linguary/application/context_builder.py:207, 3247-3262` (`_NARRATIVE_ANALYSIS_INSTRUCTIONS`, `_with_name_policy`)
- Test: `tests/unit/application/test_context_builder.py`, `tests/unit/application/test_illustration_reliability.py`

- [ ] **Step 1:** Обновить 2 существующих теста (`test_illustration_reliability.py:938` и `test_context_builder.py:78`) под нейтральные примеры и добавить падающие тесты:
  - во всех шаблонах `snapshot_prompt_templates(..., profile="content-plan-v3", name_translation_policy=NameTranslationPolicy(strategy=SOUND))` отсутствует `"Прайвет-драйв"`;
  - присутствуют правило фонетической сверки (и для анализа, и для перевода) и явное требование полноты `name_translation_assessment.rationale` и `scenes[].visual_facts.{anchor, subject, detail}`.
- [ ] **Step 2:** Заменить «Прайвет-драйв» в `_NARRATIVE_ANALYSIS_INSTRUCTIONS:207` и `_with_name_policy:3262`, добавить напоминание о вложенных полях и фонетическую сверку в ветке `NameTranslationStrategy.SOUND`, не трогая `_ANALYSIS_INSTRUCTIONS:2427-2430`, `_TRANSLATION_INSTRUCTIONS` и строки `L3250-3259`.
- [ ] **Step 3:** Прогнать `uv run pytest tests/unit/application/test_context_builder.py tests/unit/application/test_illustration_reliability.py tests/integration/layout/test_a2_harry_potter_regressions.py tests/architecture` (убедиться, что `test_baseline_b2_templates_sha256_unchanged` зелёный).

---

### Task 5 (WP5): Педагогика A2 — кореференция сегментов, калибровка режимов и снятие артиклей в `vocabulary_dedup_key` / `is_trivial_a1_hint`

**Files:**
- Modify: `src/linguary/application/context_builder.py:2150-2199, 3686` (`_PEDAGOGICAL_ANALYSIS_INSTRUCTIONS`, `_CONTENT_PLAN_PROCESSING_V3`, `_reader_level_treatment_guidance`, `_a2_progression_guidance`)
- Modify: `src/linguary/application/reader_text.py:508-514` (`_A2_NOTE_DISCIPLINE`)
- Modify: `src/linguary/application/layout/notes.py:609-619` (`vocabulary_dedup_key`)
- Modify: `src/linguary/application/a2_core_vocab.py:1124-1145` (`is_trivial_a1_hint`)
- Test: `tests/unit/application/test_context_builder.py`, `tests/unit/application/layout/test_notes.py`, `tests/unit/application/test_a2_core_vocab.py`, `tests/unit/application/layout/test_book_layout.py`

- [ ] **Step 1:** Обновить 5 существующих проверок в `tests/unit/application/test_context_builder.py` (`L1437`, `L2393`, `L3815`: `"45 слов"` → `"30 слов"`; `L398`, `L1422`: `"4–8 слов"` → `"1–8 слов"`) и написать падающие тесты:
  - промпт-инструкции A2 содержат порог `30 слов` (в т.ч. в `_a2_progression_guidance`), окно `1–8 слов`, *Complex Object* / разрыв глагола-дополнения / эллипсис в стоп-листе `GLOSSARY_ONLY`, запрет «висячих» местоимений на границе сегментов и правило оформления цитат в `_A2_NOTE_DISCIPLINE`;
  - `vocabulary_dedup_key("the reptile house") == vocabulary_dedup_key("reptile house")`;
  - `is_trivial_a1_hint("a glass") is True`, `is_trivial_a1_hint("the paper") is True`, `is_trivial_a1_hint("a wand") is False`;
  - в `build_book_layout` повтор `"the reptile house"` через 3 блока после `"reptile house"` подавляется, а `_HINT_LOCAL_WINDOW_BLOCKS = 20` и `glossary_only_rescue_by_bucket` сохраняют прежний контракт.
- [ ] **Step 2:** Внести правки в `context_builder.py`, `reader_text.py`, `layout/notes.py` и `a2_core_vocab.py` по SDD WP5 (не трогая `_NARRATIVE_GLOSSARY_WORD_LIMIT = 45` в `analysis_merge.py` и не дублируя `_deduplicate_glossary_hints` в `layout/book.py`).
- [ ] **Step 3:** Прогнать `uv run pytest tests/unit/application/test_context_builder.py tests/unit/application/layout/test_notes.py tests/unit/application/test_a2_core_vocab.py tests/unit/application/layout/test_book_layout.py tests/unit/application/test_staged_analyze_chapter.py tests/architecture`.

---

### Task 6 (WP6): Пунктуация прямой речи — автомат в `dialogue.py`, поддержка британских реплик в `paragraphs.py` и единая норма в промптах

**Files:**
- Modify: `src/linguary/application/validators/paragraphs.py:23, 48-58` (`_count_post_speech_narrative_sentences`)
- Modify: `src/linguary/application/validators/dialogue.py:153-198, 350-390` (`_has_block_post_quote_narrative_dash`, `_has_dash_before_post_quote_narration`, `_text_dialogue_violation`)
- Modify: `src/linguary/application/reader_text.py:524` (`_A2_DIALOGUE_AND_CALQUE_RULES`)
- Modify: `src/linguary/composition/correction_runtime.py:91` (`CORRECTION_PROMPT_V2`)
- Test: `tests/unit/application/test_dialogue_validator.py`, `tests/unit/application/test_result_validators.py`, `tests/unit/composition/test_correction_runtime.py`

- [ ] **Step 1:** Обновить 3 существующих теста (`test_dialogue_validator.py:657-681`, `test_result_validators.py:3787-3836`, `test_correction_runtime.py:342-350`) под отказ от однострочного `«— Реплика? — Авторское предложение.»` и написать падающие тесты:
  - все 5 реальных негативных кейсов прогона + британский случай (`‘Hello,’ he said. Then he ran.` → `— Привет, — сказал он. — Потом он убежал.`) отклоняются `dialogue_violation(..., strict_v3=True)`;
  - перенос на новую строку без тире (`— Реплика!\nОн взъерошил волосы.` и `— Привет, — сказал он.\nПотом он убежал.`) принимается и `dialogue_violation(..., strict_v3=True)`, и `paragraph_violation(...)`;
  - легитимные 2-репликовые конструкции (`— Посмотри на это! — крикнул он. — Посмотри! — ахнула тётя Петуния.`, `— Первая, — сказал он. — Вторая.`), сокращения (`Mr.`, `г.`) и сегментированные блоки не дают ложных срабатываний;
  - `'"Hello," he said. "How are you?"'` и `"‘Hello,’ said Mr. Ollivander. ‘Wait.’"` дают `_count_post_speech_narrative_sentences == 0`; `"‘Hello!’ he cried. Then he ran."` даёт `1`; встроенные цитаты `b35_single_leading_dash` (`test_dialogue_validator.py:576-596`, `test_result_validators.py:3734-3744`) остаются зелёными.
- [ ] **Step 2:** Реализовать расширение `_count_post_speech_narrative_sentences` в `paragraphs.py`, автоматную проверку `_has_block_post_quote_narrative_dash` / `_has_dash_before_post_quote_narration` в `dialogue.py` (`strict_v3=True`) и синхронно обновить `_A2_DIALOGUE_AND_CALQUE_RULES` и `CORRECTION_PROMPT_V2` по SDD WP6.
- [ ] **Step 3:** Прогнать полный локальный гейт:
  - `uv run pytest tests/unit/application/test_dialogue_validator.py tests/unit/application/test_result_validators.py tests/unit/composition/test_correction_runtime.py tests/architecture`
  - `uv run pytest -m "not slow"`
  - `uv run pytest --cov=linguary --cov-fail-under=90`
