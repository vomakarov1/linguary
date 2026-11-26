# План реализации: надёжность коррекции, адаптивный бюджет рассуждений и связность промптов иллюстраций

**Goal:** Устранить все 9 верифицированных системных дефектов Round 3 (+ 2 детерминированных бага форматирования/типографики) без заплаток, без миграций БД и без регрессий.

**Architecture:** Изменения строго следуют Clean Architecture (`domain ← application ← adapters / web / cli`, проверяется `tests/architecture/test_layer_dependencies.py`), разбиты на 4 слабосвязанных рабочих пакета (WP1–WP4) с независимой тестовой проверкой каждого пакета.

**Specs (актуализированы после глубокого ревью):**
- `docs/superpowers/specs/2026-11-26-correction-budget-and-illustration-coherence/BRIEF-correction-budget-and-illustration-coherence-2026-11-26.md`
- `docs/superpowers/specs/2026-11-26-correction-budget-and-illustration-coherence/ADR-correction-budget-and-illustration-coherence-2026-11-26.md`
- `docs/superpowers/specs/2026-11-26-correction-budget-and-illustration-coherence/SDD-backend-correction-budget-and-illustration-coherence-2026-11-26.md`

## Global Constraints

- **0 миграций Alembic:** схема SQLite не меняется; новые ключи в JSON-полезных нагрузках обратно совместимы со старыми записями.
- **Границы слоёв:** `uv run pytest tests/architecture` зелёный после каждого WP.
- **Замороженные SHA-256:** `_TRANSLATION_INSTRUCTIONS`, `_SEGMENT_TRANSLATION_RULE`, `snapshot_prompt_templates(..., reader_level="b2", profile="legacy-v1")` не трогать (`test_baseline_b2_templates_sha256_unchanged`).
- **Приватность ошибок:** `parse_payload()` использует статический белый список `_STATIC_CONTRACT_ERROR_HINTS` (никогда не выводит сырой `str(error)`).
- **Универсальность:** никаких хардкодов под конкретную книгу, персонажей или модели LLM.
- **Подпись коммитов:** каждый коммит в ветке содержит строку `Assisted-by: <роль> (<инструмент>, <модель>)` согласно `AGENTS.md`.

---

### Task 1 (WP1): Коррекция сегментированных блоков + безопасная диагностика `LLMPayloadError`

**Files:**
- Modify: `src/linguary/application/correction_dto.py`
- Modify: `src/linguary/composition/correction_runtime.py`
- Modify: `src/linguary/adapters/llm/payloads.py`
- Modify: `tests/unit/application/test_correction_inputs.py`
- Modify: `tests/unit/llm/test_llm_payload_mapping.py`
- Modify: `tests/integration/processing/test_segment_correction_lifecycle.py`

**Steps:**
- [ ] Добавить `_resolve_segment_entries()` и поддержку `segments`, `segments.<N>`, `segments.<N>.translated_text` (строгий 1-based поиск по `entry['index'] == req_idx` + `req_idx == 0` fallback в `_evidence_value()`, толерантный quote fallback только в `CurrentQuoteEvidence.valid_for()`).
- [ ] Документировать `segments.<index>.translated_text` в `CORRECTION_PROMPT_V2` (`correction_runtime.py`).
- [ ] Добавить `_STATIC_CONTRACT_ERROR_HINTS` в `parse_payload()` (`payloads.py`), оставив `Violation.safe_message` статичным и защитив от утечки текста книги.
- [ ] Написать и прогнать unit- и integration-тесты WP1.
- [ ] Коммит WP1.

---

### Task 2 (WP2): Адаптивный бюджет рассуждений, контракт `segments`/`translated_text` и паритет многоточия

**Files:**
- Modify: `src/linguary/application/token_budget.py`
- Modify: `src/linguary/application/block_processing.py`
- Modify: `src/linguary/adapters/llm/schemas.py`
- Modify: `src/linguary/adapters/llm/payloads.py`
- Modify: `src/linguary/application/validators/treatment.py`
- Modify: `src/linguary/application/context_builder.py`
- Modify: `src/linguary/application/analysis_merge.py`
- Modify: `tests/unit/application/test_token_budget.py`
- Modify: `tests/unit/llm/test_segment_translation_contract.py`
- Modify: `tests/unit/llm/test_llm_payload_mapping.py`
- Modify: `tests/unit/application/test_result_validators.py`
- Modify: `tests/unit/application/test_analysis_merge_scenes.py`

**Steps:**
- [ ] Обновить `recommended_unit_max_completion_tokens()` в `token_budget.py` для совместного учёта активного `reasoning_effort` (`!= "none"`) и `implicit_reasoning_tokens`.
- [ ] Разрешить implicit-эскалацию при `configured_max_output_tokens is None` для активного `reasoning_effort` и устранить недостижимый код на `can_escalate_recovery` в `block_processing.py`.
- [ ] Сделать двунаправленным сообщение ошибки `segments` + `translated_text` в `schemas.py` и `_STATIC_VALUE_ERROR_HINTS` в `payloads.py` (обновив `test_llm_payload_mapping.py`); уточнить `_block_content_violation()` в `validators/treatment.py`; добавить правило про несегментированные блоки в `_TRANSLATION_CONTEXT_V2` и `_CONTENT_PLAN_PROCESSING_V3` (`context_builder.py`), не затрагивая замороженные шаблоны.
- [ ] Добавить типографскую эквивалентность `(?P<ellipsis>…|\.{3})` в `_lenient_quote_pattern()` (`analysis_merge.py`).
- [ ] Написать и прогнать unit-тесты WP2 + `test_baseline_b2_templates_sha256_unchanged`.
- [ ] Коммит WP2.

---

### Task 3 (WP3): Надёжность `SELECTION` и ремонта `EXTRACT`/`CANDIDATES` в иллюстрациях

**Files:**
- Modify: `src/linguary/domain/illustrations.py`
- Modify: `src/linguary/application/illustration_context.py`
- Modify: `src/linguary/adapters/llm/illustration_payloads.py`
- Modify: `tests/unit/application/test_illustration_selection.py`
- Modify: `tests/unit/application/test_illustration_reference.py`

**Steps:**
- [ ] Обновить дефолт `IllustrationPolicy.max_repair_turns = 2` (включая `from_payload()`).
- [ ] Уточнить `_STAGE_INSTRUCTIONS[SELECTION]` и `compact_repair_summary(..., stage=...)` (с сохранением сигнатуры, идемпотентности и лимита `MAX_REPAIR_SUMMARY_BYTES`, преамбула полного ответа для `EXTRACT`/`CANDIDATES`).
- [ ] Добавить `Field(description=...)` в `IllustrationSelectionSchema` (`illustration_payloads.py`), без избыточного `@model_validator`.
- [ ] Обновить/добавить unit-тесты WP3 (включая `test_hallucinated_and_duplicate_candidate_ids_rejected`).
- [ ] Коммит WP3.

---

### Task 4 (WP4): Идентичность сущностей, защита смены облика, `depicted_entity_ids`, привязка локаций и гигиена пунктуации

**Files:**
- Modify: `src/linguary/application/illustration_reference.py`
- Modify: `src/linguary/application/illustration_selection.py`
- Modify: `src/linguary/application/illustration_chapter.py`
- Modify: `src/linguary/application/illustration_context.py`
- Modify: `src/linguary/application/illustration_dto.py`
- Modify: `src/linguary/application/illustration_validation.py`
- Modify: `src/linguary/adapters/llm/illustration_payloads.py`
- Modify: `tests/unit/application/test_illustration_reference.py`
- Modify: `tests/unit/application/test_illustration_selection.py`
- Modify: `tests/unit/application/test_illustration_chapter.py`
- Modify: `tests/unit/application/test_illustration_reliability.py`
- Modify: `tests/unit/llm/test_illustration_payloads.py`
- Modify: `tests/unit/composition/test_illustration_runtime.py`
- Modify: `tests/integration/processing/test_illustration_reliability_integration.py`

**Steps:**
- [ ] Реализовать 3-уровневый `_choose_canonical_name()`, `alias_first_positions`, `section_orders`, передачу `aliases` в `_build_bounded_design_input()`, очистку эхо-префикса `^<trait>:` до проверок в `_validate_design()`, подавление анатомических черт `DESIGN` при `temporal_scope == "state_change"` в `(*explicit_facts, *excluded_facts)`, расширение `_TRAIT_PATTERNS["build"]` и 3-символьные токены в `_echo_content_tokens()` (`illustration_reference.py`).
- [ ] Реализовать `_display_name_at_position()` во всех 3 точках (`FACT_SCOPE` `entity_name`, `name_ru`, `name_en`), двухступенчатый `_resolve_entity_name()` (`"resolved"` / `normalize_name_token`, приоритет `canonical_name` > `aliases`), `depicted_entity_ids` с фильтрацией в `_validate_render_response()` (до `compose_monochrome_prompt()`) и `_advance_chapter_rendering()`, `scene_location_ru` в `_build_prompt_scope_spec()`, чистку пунктуации в `illustration_chapter.py` и `illustration_validation.py` (включая `_clause_covered_in_text()`).
- [ ] В `plan_scopes_in_tx()` (`illustration_selection.py`) загрузить `ENTITY` через `query_registry_items()`, отфильтровать `first_position_key <= chapter_end`, пробросить `known_characters` (`person`) и `known_locations` (`location`) в `_build_req` (`pack_items_by_wire_budget()`) и `_split_children_for_scope()`; обновить инструкции `EXTRACT`, `ALIAS`, `DESIGN`, `CANDIDATES`, `PROMPT` (`illustration_context.py`).
- [ ] Написать и прогнать unit- и integration-тесты WP4 (включая обновление 2 raw-JSON фикстур `PROMPT`).
- [ ] Коммит WP4.

---

### Task 5: Полная верификация, глубокое ревью, слияние и повторный прогон

- [ ] Запустить `uv run pytest tests/architecture` и `uv run pytest -m "not slow"`.
- [ ] Передать полный diff (`git diff main...HEAD`), продуктовый код и тесты новому субагенту на глубокое ревью.
- [ ] Устранить все валидные замечания, повторно прогнать тесты и при необходимости сделать контрольное ревью.
- [ ] Слить ветку в `main` и запушить в `origin/main`.
- [ ] Повторно прогнать `hp_ch1_ch2p5_round3.epub` с новым `book_id` (`r4`), выгрузить чистые сопоставимые дампы `r3` и `r4` и передать новому субагенту на сравнительный анализ.
