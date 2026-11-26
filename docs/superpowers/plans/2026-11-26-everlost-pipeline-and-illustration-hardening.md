# Everlost Pipeline & Illustration Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Устранить все верифицированные системные баги кода и промптов/схем из прогона *Everlost* (кэш-дедлок анализа, распад реестра и перерасход вызовов иллюстраций, рассинхрон диалогового валидатора и баги раскладки A2) без миграций БД и без регрессий.

**Architecture:** Изменения строго соблюдают Clean Architecture (`domain ← application ← adapters / web / cli`) и разбиты на 5 слабосвязанных рабочих пакетов (WP1–WP5), каждый со своим независимым TDD-циклом, тестовым гейтом и коммитом.

**Tech Stack:** Python 3.15+, Pydantic v2, SQLite (без изменения схемы), pytest, Ruff, ty.

**Spec (одобрены по итогам 3 независимых ревью):**
- `docs/superpowers/specs/2026-11-26-everlost-pipeline-and-illustration-hardening/BRIEF-everlost-pipeline-and-illustration-hardening-2026-11-26.md`
- `docs/superpowers/specs/2026-11-26-everlost-pipeline-and-illustration-hardening/ADR-everlost-pipeline-and-illustration-hardening-2026-11-26.md`
- `docs/superpowers/specs/2026-11-26-everlost-pipeline-and-illustration-hardening/SDD-backend-everlost-pipeline-and-illustration-hardening-2026-11-26.md`

## Global Constraints

- **0 миграций Alembic:** схема SQLite не меняется; все новые поля живут в `payload_json` и обратно совместимы со старыми записями.
- **Границы слоёв:** `uv run pytest tests/architecture` зелёный после каждого WP; в `domain` и `application` нет сторонних библиотек и прямого I/O.
- **Замороженные SHA-256:** `_TRANSLATION_INSTRUCTIONS`, `_SEGMENT_TRANSLATION_RULE` и `snapshot_prompt_templates(..., reader_level="b2")` не изменяются (`test_baseline_b2_templates_sha256_unchanged`).
- **Неизменяемые контракты:** `temperature=policy.temperature` в `_clean_llm_settings()`, `_exported_sections()` (`title.xhtml`/`dedication.xhtml`), `.lg-notes p { display: inline; margin: 0; }`, покрытие `ChapterCorrectionV2Schema`, `test_ambiguous_identity_remains_separate` и `test_transitive_identity_conflict_triangle_order_independent_deterministic` сохраняются.
- **Универсальность:** никаких хардкодов под названия книг, персонажей, локаций или моделей LLM.
- **Подпись коммитов:** каждый коммит содержит `Assisted-by: <роль> (<инструмент>, <модель>)` согласно `AGENTS.md`.

## Review Focus

1. **Чтение старых `ENTITY`-записей из БД без новых полей (`glossary_translation_ru`, `alias_translations_ru`) и старых скоупов `EXTRACT`/`CANDIDATES` с сырыми UUID:** код обязан работать без `KeyError`, а wire-алиасинг должен применяться только на границе `LLMRequest` при `len(raw_id) > 16`, не ломая сплиты и `find_reusable_scope` (проверяется в WP2 и WP3).
2. **Порядок-независимость `resolve_verdict_graph_clusters()` и сохранность обоих тестов на мосты:** любая перестановка `same_edges` и `diff_edges` обязана давать тождественное разбиение; `test_ambiguous_identity_remains_separate` и `test_transitive_identity_conflict_triangle_order_independent_deterministic` остаются зелёными (проверяется в WP2).
3. **Раздельный и бережный кэш анализа:** очистка через `tx.analyses.delete_fragment_passes_for_section(book_id, section_id, pass_kind)` при `AnalysisMergeError` на `PEDAGOGICAL` (вкл. `_assemble_staged`) не сносит общеглавный `NARRATIVE`, а `RetryTask` не сносит фрагментный кэш при `ErrorType.TRANSIENT` (проверяется в WP1).
4. **Согласованность `max_completion_tokens` и `reasoning_max_tokens` на OpenRouter:** при `base_llm.reasoning_max_tokens >= 6144` и `stage=DESIGN` (`max_completion_tokens=6144`) `serialize_request` не падает с `LLMConfigurationError` (проверяется в WP3).
5. **Отсутствие спойлеров и сохранение границ валидатора диалогов:** `glossary_terms_ru`, `known_entity_aliases` и обиходные имена говорящих фильтруются строго по `<= section.order` / `chapter_end_pos`; `_is_allowed_embedded_quote_span` пропускает однострочный и двухстрочный B35 со `strict_v3=True`, но по-прежнему отклоняет `«Привет, — сказал он»` с начала строки (проверяется в WP2, WP4, WP5).

---

### Task 1 (WP1): Анализ глав — монотонность `spoiler_name_leak`, дескрипторы, инвалидация кэша (`tx.analyses`) и автосинтез имён

**Files:**
- Modify: `src/linguary/application/context_builder.py:197-205`
- Modify: `src/linguary/application/analysis_merge.py:2221-2249, 2738-2816`
- Modify: `src/linguary/application/use_cases/analyze_chapter.py:575-590, 688-706, 2537-2580`
- Modify: `src/linguary/application/use_cases/retry_task.py:54-75`
- Test: `tests/unit/application/test_context_builder.py`
- Test: `tests/unit/application/test_analysis_merge_scenes.py`
- Test: `tests/unit/application/test_staged_analyze_chapter.py`
- Test: `tests/integration/processing/test_retry_task_recovery_guards.py`
- Test: `tests/integration/processing/test_analysis_everlost_regressions.py`

**Interfaces:**
- Produces:
  - `_is_generic_person_candidate(source_text: str) -> bool` (сохраняя шаблоны `[0..1]` + строчные 2–4 слова с головой из `_GENERIC_PERSON_HEADS`)
  - монотонный `_first_name_block_index(...) -> int | None` (с `matches_capitalized_glossary`)
  - `_clear_fragment_passes(self, book_id: str, section_id: str, pass_kind: AnalysisPassKind) -> None` через `tx.analyses.delete_fragment_passes_for_section`
  - очистка фрагментов в `RetryTask.__call__()` для `ANALYZE_CHAPTER` при `last_error.type is not ErrorType.TRANSIENT`
  - расширенный `_a2_vocab_known_names_from_glossary` / `_a2_vocab_known_names_with_narrative` (`analyze_chapter.py:2537-2580`).

- [ ] **Step 1: Написать падающие тесты и обновить 3 старых теста примера `'the freckled boy'`**
  - Обновить `test_context_builder.py:3737` (пример `"Iron Hand"`), `test_analysis_merge_scenes.py:1465-1484, 1778-1827` (`'the freckled boy'` в отклоняемых, `'Iron Hand'` в разрешённых).
  - Добавить тесты: `_is_generic_person_candidate` на `"younger boy"`, `"the freckled kid"`, `"the baker"` (`True`) и `"Chocolate Boy"`, `"Iron Hand"` (`False`); монотонность `_first_name_block_index` при `"Chocolate Boy"` в блоке 38 с сохранением спойлера `'the dark voldemort'` (`L1214`); очистка фрагментов через `tx.analyses` при `AnalysisMergeError` на `NARRATIVE` и `PEDAGOGICAL` (вкл. `_assemble_staged`); `RetryTask` очищает фрагменты при `ErrorType.VALIDATION` и сохраняет при `ErrorType.TRANSIENT`; `_a2_vocab_known_names_*` распознаёт `Allie` из `note` `Alexandra` без поломки `test_staged_analyze_chapter.py:4714-4720`.

- [ ] **Step 2: Прогнать новые тесты и убедиться, что они падают**
  Run: `uv run pytest tests/unit/application/test_analysis_merge_scenes.py tests/unit/application/test_staged_analyze_chapter.py tests/integration/processing/test_retry_task_recovery_guards.py -k "generic_person or spoiler_monotonic or fragment_cache or allie_alias" -v`
  Expected: FAIL на новых тестах.

- [ ] **Step 3: Реализовать WP1 по SDD § 1.1–1.3**
  - Обновить `_NARRATIVE_ANALYSIS_INSTRUCTIONS:200` в `context_builder.py`.
  - Обновить `_is_generic_person_candidate()` и `_first_name_block_index()` в `analysis_merge.py`.
  - Добавить `_clear_fragment_passes` (через `tx.analyses`), обёртки на `L575-590` и `L688-706` (вкл. `_assemble_staged`), расширение `_a2_vocab_known_names_*` (`L2537-2580`) в `analyze_chapter.py`, и точечную очистку в `retry_task.py:54-75`.

- [ ] **Step 4: Прогнать тесты WP1 и архитектуру**
  Run: `uv run pytest tests/architecture tests/unit/application/test_context_builder.py tests/unit/application/test_analysis_merge_scenes.py tests/unit/application/test_staged_analyze_chapter.py tests/integration/processing/test_retry_task_recovery_guards.py tests/integration/processing/test_analysis_everlost_regressions.py -q`
  Expected: PASS.

- [ ] **Step 5: Коммит WP1**
  ```bash
  git add src/linguary/application/context_builder.py src/linguary/application/analysis_merge.py src/linguary/application/use_cases/analyze_chapter.py src/linguary/application/use_cases/retry_task.py tests/
  git commit -m "fix(analysis): make spoiler check monotonic, broaden generic person filter, and clear fragment cache on merge failure/retry"
  ```

---

### Task 2 (WP2): Реестр иллюстраций — рёберное правило мостов, контекстный блокинг, `_choose_canonical_name`, `_display_name_at_position`, алиасы и глоссарий

**Files:**
- Modify: `src/linguary/application/illustration_reference.py:264-425, 2400-2445, 2781-2849`
- Modify: `src/linguary/application/illustration_selection.py:103-155, 670-705, 2085-2120`
- Modify: `src/linguary/application/illustration_chapter.py:299-335, 995-1050, 1380-1415`
- Modify: `src/linguary/application/illustration_context.py:490-556`
- Test: `tests/unit/application/test_illustration_reference.py`
- Test: `tests/unit/application/test_illustration_selection.py`
- Test: `tests/unit/application/test_illustration_chapter.py`
- Test: `tests/integration/processing/test_illustration_reliability_integration.py`

**Interfaces:**
- Consumes: `getattr(tx, "glossary", None)` (защитное чтение `TermConcept`/`TermVariant`).
- Produces:
  - рёберное правило мостов в `resolve_verdict_graph_clusters(node_ids, *, same_edges, diff_edges)`
  - `build_alias_candidate_pairs(mentions, *, glossary_links=None)` с контекстным блокингом `tier == 0` при валидном `position_key` (`<= 8` соседей)
  - `_choose_canonical_name(members, *, glossary_source_keys=frozenset()) -> str` и приоритет `canonical_name` в `_display_name_at_position()`
  - опциональные поля `ENTITY`: `glossary_translation_ru`, `alias_translations_ru`
  - `_known_entities_context_for_chapter(...)` → чистые `known_characters`/`known_locations` + отдельное поле `known_entity_aliases` (сохраняя `_known_entity_names_for_chapter`)
  - `_display_name_ru_at_position(...)` (для `name_ru` и fallback `description_ru`) и неспойлерный `_collect_book_glossary_terms_ru(...)` в пределах wire budget.

- [ ] **Step 1: Написать падающие тесты WP2**
  - Тест на `resolve_verdict_graph_clusters`: 9-клик (35 `same` + 1 `diff`) сохраняет 7 вершин (0 `diff` в кластере, 20 случайных перестановок рёбер), `A-B-C, A-D, diff(A,C)` → `{("A","D"), ("B",), ("C",)}`, `{A1..A3}-X-{B1,B2}` с `diff(A1,B1)` → 3 кластера.
  - Тест на `build_alias_candidate_pairs`: `the boy` (с валидным `position_key`) получает `<= 8` ближайших `person`-соседей в главе, а упоминания без `position_key` или с собственным именем (`the Smiths’ son` vs `the Bakers’ son`) не связываются.
  - Тест на `_choose_canonical_name` и `_display_name_at_position`: `Nick` (`count=8, in_glossary=True`) > `Chocolate Boy` (`count=1`), `Professor McGonagall` > `McGonagall` (`count=1`), и `_display_name_at_position` в гл. 4 возвращает `"Nick"` (не `"Chocolate Boy"`).
  - Тест на `known_entity_aliases`, `_display_name_ru_at_position` и `glossary_terms_ru`: `known_characters` остаётся чистым списком, алиасы и термины будущих глав не спойлерятся, старые `ENTITY` без новых полей работают без `KeyError`, wire budget усекает `glossary_terms_ru` при дефиците байт.

- [ ] **Step 2: Прогнать новые тесты и убедиться, что они падают**
  Run: `uv run pytest tests/unit/application/test_illustration_reference.py tests/unit/application/test_illustration_selection.py tests/unit/application/test_illustration_chapter.py -k "clique_single_diff or anonymous_context_blocking or canonical_name_glossary or display_name_preserves_canonical or known_entity_aliases or glossary_terms_ru" -v`
  Expected: FAIL на новых тестах.

- [ ] **Step 3: Реализовать WP2 по SDD § 2.1–2.3**
  - Обновить `resolve_verdict_graph_clusters()`, `build_alias_candidate_pairs()`, `_choose_canonical_name()`, `_display_name_at_position()` и формирование `ENTITY` (`L2400-2445`) в `illustration_reference.py`.
  - Добавить `known_entity_aliases` в `illustration_selection.py` (`plan_scopes_in_tx` / `_build_req` / `_split_children_for_scope`).
  - Реализовать `_display_name_ru_at_position()`, `_collect_book_glossary_terms_ru()`, защитную чистку в `_resolve_entity_name()` (`L299`) и проброс `glossary_terms_ru` в wire-ограниченные запросы в `illustration_chapter.py` / `illustration_context.py`.

- [ ] **Step 4: Прогнать тесты WP2 (вкл. `test_ambiguous_identity_remains_separate` и `test_illustration_reliability_integration.py`)**
  Run: `uv run pytest tests/architecture tests/unit/application/test_illustration_reference.py tests/unit/application/test_illustration_selection.py tests/unit/application/test_illustration_chapter.py tests/integration/processing/test_illustration_reliability_integration.py -q`
  Expected: PASS.

- [ ] **Step 5: Коммит WP2**
  ```bash
  git add src/linguary/application/illustration_reference.py src/linguary/application/illustration_selection.py src/linguary/application/illustration_chapter.py src/linguary/application/illustration_context.py tests/
  git commit -m "fix(illustrations): use edge-level bridge pruning, add anonymous descriptor blocking, prioritize glossary names, and propagate book glossary"
  ```

---

### Task 3 (WP3): Иллюстрации — потолки токенов, авторетрай `provider_truncated`, wire-алиасинг длинных ID, экономия вызовов, плотность и гигиена

**Files:**
- Modify: `src/linguary/composition/illustration_runtime.py:43-85`
- Modify: `src/linguary/application/illustration_context.py:465-645`
- Modify: `src/linguary/adapters/llm/illustration_payloads.py:225-275`
- Modify: `src/linguary/application/illustration_reference.py:640-690, 1225-1260, 2485-2515`
- Modify: `src/linguary/application/illustration_selection.py:260-320, 580-610, 1140-1180, 1720-1750, 1960-1980, 2210-2240`
- Modify: `src/linguary/application/illustration_chapter.py:470-560, 1030-1165, 1900-1960, 2070-2170`
- Modify: `src/linguary/application/illustration_validation.py:115-145`
- Test: `tests/unit/composition/test_illustration_runtime.py`
- Test: `tests/unit/application/test_illustration_reference.py`
- Test: `tests/unit/application/test_illustration_selection.py`
- Test: `tests/unit/application/test_illustration_chapter.py`
- Test: `tests/unit/application/test_illustration_evaluation.py`
- Test: `tests/unit/llm/test_illustration_payloads.py`
- Test: `tests/integration/processing/test_illustration_pipeline.py`

**Interfaces:**
- Consumes: `IllustrationPolicy`, `ENTITY.glossary_translation_ru` из WP2.
- Produces:
  - `MIN_ILLUSTRATION_STAGE_COMPLETION_TOKENS = 6144` и `_stage_max_completion_tokens(stage, policy) -> int`
  - `encode_wire_id_aliases` / `decode_wire_id_aliases` / `rewrite_repair_ids_to_wire` (только `len(raw_id) > 16`)
  - `effective_chapter_illustration_cap(max_per_chapter: int, content_block_count: int, scene_count: int) -> int`
  - `moments_conflict_diversity(a, b, *, block_order_by_id=None, min_block_gap=0) -> bool`
  - fast-path `FACT_SCOPE` (строго для 1 `permanent`-факта) и отбор `DESIGN`.

- [ ] **Step 1: Написать падающие тесты WP3 и обновить 3 теста `FACT_SCOPE` на 1 `permanent`-факте**
  - Тест в `test_illustration_runtime.py`: `build_for_kind(..., stage=DESIGN)` ставит `max_completion_tokens <= 6144`, `EXTRACT` `<= 8192`; при `base_llm.reasoning_max_tokens=8192` `_clean_llm_settings` обнуляет `reasoning_max_tokens` и `serialize_request` проходит без `LLMConfigurationError` (сохраняя `L136, L260, L267, L316-321`).
  - Тесты на авторетрай `provider_truncated` для не-composite областей во всех 3 модулях (`DESIGN`, `PROMPT`, 1-групповой `SELECTION`).
  - Тесты на wire-алиасинг: `len > 16` (36-char UUID, 64-hex `scene_id`) кодируются в `b1/u1/s1` и разворачиваются обратно + переписываются в `repair_summary`, при этом короткие ID (`b-0`, `b-atomic`, `sec0-b0`) не меняются, а `input_sha256` и сплиты остаются на канонических ID.
  - Тесты на `quote_anchors` `description`, отбор `DESIGN` (сохраняя `test_illustration_reference.py:1066`), fast-path `FACT_SCOPE` (строго для 1 `permanent`-факта; обновить `test_illustration_chapter.py:634, 638-700, 886`), `effective_chapter_illustration_cap(3, 2, 1) == 1`, `min_block_gap=2`, `episodic_context_only` при `len(permanent)==1` (сохраняя `L1158-1169`), `anchor_quotes` и детерминированную фильтрацию технических ключей в `assumptions_ru`.

- [ ] **Step 2: Прогнать новые тесты и убедиться, что они падают**
  Run: `uv run pytest tests/unit/composition/test_illustration_runtime.py tests/unit/application/test_illustration_reference.py tests/unit/application/test_illustration_selection.py tests/unit/application/test_illustration_chapter.py tests/unit/llm/test_illustration_payloads.py -k "stage_max_completion_tokens or reasoning_max_tokens_stage_cap or truncated_retry or wire_id_aliases or trivial_permanent_fact_scope or chapter_illustration_cap or technical_assumptions" -v`
  Expected: FAIL на новых тестах.

- [ ] **Step 3: Реализовать WP3 по SDD § 3.1–3.5**
  - Внедрить `_stage_max_completion_tokens`, защиту `reasoning_max_tokens` в `_clean_llm_settings` и авторетрай `provider_truncated` в `illustration_reference.py:1232`, `illustration_chapter.py:1906`, `illustration_selection.py:1148`.
  - Внедрить wire-алиасинг `len > 16`, `Field(description=...)` для `quote_anchors`, отбор `DESIGN`, fast-path `FACT_SCOPE`, `effective_chapter_illustration_cap` (с сохранением `"effective_max_per_chapter"` в `scope.summary` для `_split_children_for_scope`), `min_block_gap`, `episodic_context_only`, `anchor_quotes` и фильтр `assumptions_ru` в `illustration_validation.py:119` (сохраняя `characters[].description_en` в `_STAGE_INSTRUCTIONS[PROMPT]` для `test_illustration_reliability.py:1478`).

- [ ] **Step 4: Прогнать все тесты WP3 и интеграцию иллюстраций**
  Run: `uv run pytest tests/architecture tests/unit/composition/test_illustration_runtime.py tests/unit/llm/test_illustration_payloads.py tests/unit/application/test_illustration_reference.py tests/unit/application/test_illustration_selection.py tests/unit/application/test_illustration_chapter.py tests/unit/application/test_illustration_evaluation.py tests/unit/application/test_illustration_reliability.py tests/integration/processing/test_illustration_reliability_integration.py tests/integration/processing/test_illustration_pipeline.py -q`
  Expected: PASS.

- [ ] **Step 5: Коммит WP3**
  ```bash
  git add src/linguary/composition/illustration_runtime.py src/linguary/application/illustration_context.py src/linguary/adapters/llm/illustration_payloads.py src/linguary/application/illustration_reference.py src/linguary/application/illustration_selection.py src/linguary/application/illustration_chapter.py src/linguary/application/illustration_validation.py tests/
  git commit -m "fix(illustrations): enforce per-stage token caps and truncation retries, wire-alias long IDs, cut redundant scopes, and harden prompt hygiene"
  ```

---

### Task 4 (WP4): Перевод — синхронизация правил в `reader_text.py` и вложенные цитаты в `validators/dialogue.py`

**Files:**
- Modify: `src/linguary/application/reader_text.py:508-529`
- Modify: `src/linguary/application/validators/dialogue.py:155-205, 310-335`
- Test: `tests/unit/application/test_context_builder.py`
- Test: `tests/unit/application/test_dialogue_validator.py`
- Test: `tests/unit/application/test_result_validators.py`

**Interfaces:**
- Produces: обновлённые `_A2_DIALOGUE_AND_CALQUE_RULES` и `_A2_NOTE_DISCIPLINE` (без изменения замороженных SHA-256) и `_is_allowed_embedded_quote_span(...)` в `validators/dialogue.py` (используемый и в `_has_author_speech_inside_guillemets`, и в `_has_inline_dialogue_dash`).

- [ ] **Step 1: Написать падающие тесты WP4**
  - Тест в `test_context_builder.py`: в A2/V3 промпте присутствуют правила переноса строки/двоеточия после точки, мн. ч. «вы» к группе и запрета `to` для существительных; `test_baseline_b2_templates_sha256_unchanged` зелёный.
  - Тесты в `test_dialogue_validator.py` и `test_result_validators.py`: смешанный абзац `“We were on a mountain road…” Only this wasn’t... “A dead forest,” Dad had said, pointing. “It happens.”` принимается со `strict_v3=True` как в однострочном, так и в двухстрочном переводе (`рукой. — Так` внутри встроенных `«…»`), а обычная реплика `«Привет, — сказал он.»` с начала строки по-прежнему отклоняется.

- [ ] **Step 2: Прогнать новые тесты и убедиться, что они падают**
  Run: `uv run pytest tests/unit/application/test_context_builder.py tests/unit/application/test_dialogue_validator.py tests/unit/application/test_result_validators.py -k "embedded_quote_in_narrative or dialogue_dash_prompt_rule or baseline_b2_templates" -v`
  Expected: FAIL на новых тестах (при зелёном `baseline_b2_templates`).

- [ ] **Step 3: Реализовать WP4 по SDD § 4.1–4.2**
  - Обновить `_A2_DIALOGUE_AND_CALQUE_RULES` и `_A2_NOTE_DISCIPLINE` в `reader_text.py`.
  - Реализовать `_is_allowed_embedded_quote_span` и применить его в `_has_author_speech_inside_guillemets` (L173) и `_has_inline_dialogue_dash` (L187, L314-329) в `validators/dialogue.py`.

- [ ] **Step 4: Прогнать тесты WP4**
  Run: `uv run pytest tests/architecture tests/unit/application/test_context_builder.py tests/unit/application/test_dialogue_validator.py tests/unit/application/test_result_validators.py -q`
  Expected: PASS.

- [ ] **Step 5: Коммит WP4**
  ```bash
  git add src/linguary/application/reader_text.py src/linguary/application/validators/dialogue.py tests/
  git commit -m "fix(translation): sync dialogue dash/plural/noun prompt rules and allow embedded narrative quotes in dialogue validator"
  ```

---

### Task 5 (WP5): Раскладка учебного слоя A2 — единая сегментная привязка подсказок, «Говорит …» и тавтологии

**Files:**
- Modify: `src/linguary/application/layout/notes.py:720-795`
- Modify: `src/linguary/application/layout/book.py:370-390, 555-705, 1035-1165`
- Test: `tests/unit/application/layout/test_notes.py`
- Test: `tests/unit/application/layout/test_book_layout.py`
- Test: `tests/unit/application/layout/test_units.py`
- Test: `tests/integration/layout/test_a2_harry_potter_regressions.py`

**Interfaces:**
- Produces:
  - `find_hint_marker_end(text, original, *, anchor_spans=(), preferred_spans=()) -> int | None` в `layout/notes.py`
  - `_is_tautological_gloss(translation: str, note: str) -> bool` (вызывается после L779)
  - `_extract_short_name_pair(concept, analyses, language) -> tuple[str, str] | None` (накопительный подсчёт `<= section.order`, без сокращения титульных форм)
  - `_part_self_identifies_speaker(part_text: str, speaker_tokens: Sequence[str]) -> bool` (с исключением служебных/родовых токенов).

- [ ] **Step 1: Написать падающие тесты WP5**
  - Тест в `layout/test_notes.py` и `layout/test_book_layout.py`: на смешанном блоке `[TRANSLATE("...base of the cliff..."), GLOSSARY_ONLY("Now they lay at the base of the cliff.")]` подсказка `cliff — обрыв` получает `marker_end` внутри сегмента 1 (`GLOSSARY_ONLY`), а при явной опоре `Difficulty` в сегменте 0 (`TRANSLATE`) остаётся на слове внутри сегмента 0; `_hint_segment_index()` полностью согласован с `_hint_draft()`.
  - Тест в `layout/test_book_layout.py`: при `Alexandra -> Александра` и доминирующем `Allie -> Элли` в разделах `<= section.order` выводится «Говорит Элли» (без спойлера в ранней главе, где `Allie` ещё не появлялось; титульные имена вроде `дядя Вернон` не сокращаются); на `“My name’s Nick.”` сноска «Говорит Ник» подавляется, а на `“I'm the boy who...”` — нет.
  - Тест в `layout/test_notes.py`: `grown-ups` (`translation="взрослые"`, `note="взрослые люди"`) очищает `note`, `translation="взрослого"`, `note="взрослого человека"` очищает `note`, а реальное пояснение сохраняется.

- [ ] **Step 2: Прогнать новые тесты и убедиться, что они падают**
  Run: `uv run pytest tests/unit/application/layout/test_notes.py tests/unit/application/layout/test_book_layout.py -k "mixed_segment_hint_anchor or self_identifies_speaker or common_short_speaker_name or tautological_gloss" -v`
  Expected: FAIL на новых тестах.

- [ ] **Step 3: Реализовать WP5 по SDD § 5.1–5.3**
  - Реализовать `find_hint_marker_end()` и `_is_tautological_gloss()` (после L779) в `layout/notes.py`.
  - Прокинуть `plan` (и `dropped_difficulty_ids`) из `_deduplicate_glossary_hints()` (`L1154`) через `_hint_is_glossary_only()` и `_glossary_only_bucket()` в `_hint_segment_index()`, предвычислить накопительные частоты `<= section.order` в `_SpeakerIndex`, реализовать `_extract_short_name_pair()` и `_part_self_identifies_speaker()` (по `fact.value`, `canonical_source_text` и `short_en`) в `layout/book.py`.

- [ ] **Step 4: Прогнать тесты раскладки, EPUB и архитектуру**
  Run: `uv run pytest tests/architecture tests/unit/application/layout/test_notes.py tests/unit/application/layout/test_book_layout.py tests/unit/application/layout/test_units.py tests/unit/adapters/epub/ tests/integration/layout/ tests/integration/epub/ -q`
  Expected: PASS.

- [ ] **Step 5: Коммит WP5**
  ```bash
  git add src/linguary/application/layout/notes.py src/linguary/application/layout/book.py tests/
  git commit -m "fix(layout): unify mixed-block hint anchoring, use non-spoiler short speaker names, and filter tautological gloss notes"
  ```

---

### Task 6: Полная верификация и контрольный прогон

- [ ] Прогнать полный `make check` и `uv run pytest -m "not slow"`.
- [ ] Проверить `git status` и полный `git diff main...HEAD` независимым ревьюером.
- [ ] Провести контрольный прогон `everlost-ch1-ch4p2.epub` в новом `book_id` (анализ → глоссарий → перевод → иллюстрации → экспорт `EPUBCheck: passed`) и сравнить метрики и качество с базовым прогоном `everlost-ch1-ch4p2-20261126`.
