# План реализации: надёжное восстановление перевода, хронологическая точность и монохромная гигиена иллюстраций

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Устранить системные причины остановки перевода на `recovery:limit` (ложные отказы валидаторов кавычек и языкового просачивания, непрозрачные ошибки схемы сегментов, заниженный бюджет скрытых рассуждений, переразмеренные диалоговые юниты, голодание правой ветви при сплите) и хронологические/стилевые дефекты иллюстраций (перекрытие `FACT_SCOPE` статическим `DESIGN`, утечка цветов и дублирование `Visual details:`, заглушки `assumptions_ru`, пропуск описательных алиасов в `ALIAS`, противоречие лемматизации `glossary_hints` в A2).

**Architecture:** Изменения ложатся строго в существующие слои `domain ← application ← adapters`, без новых зависимостей и без миграций SQLite/Alembic. В переводе валидаторы (`truncation.py`, `mixed_script.py`) переходят к контекстно-корректной проверке (британские кавычки + агрегация по `TRANSLATE`-сегментам; дословные многословные цитаты оригинала), схема и ремонт (`schemas.py`, `payloads.py`) становятся самодокументируемыми, а планировщик (`translation_units.py`, `recovery.py`, `block_processing.py`) ограничивает число блоков на юнит, адаптируется к скрытым рассуждениям и обходит скоупы в ширину (BFS). В иллюстрациях `DESIGN` очищается от эпизодических состояний, `resolve_appearance_with_design_fallback()` учитывает `excluded_facts` и `compact_description_*`, финальные образы материализуются строго по `used_design_keys`, а монохромная проекция выполняется уже в `FACT_SCOPE` с защитой собственных имён и клаузной дедупликацией `Visual details:`.

**Tech Stack:** Python 3.15+, Pydantic v2, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-25-translation-recovery-and-illustration-fidelity/BRIEF-translation-recovery-and-illustration-fidelity-2026-11-25.md`. Архитектура — `ADR-translation-recovery-and-illustration-fidelity-2026-11-25.md`. Детальные контракты — `SDD-backend-translation-recovery-and-illustration-fidelity-2026-11-25.md`.

## Global Constraints

- **0 миграций Alembic:** схема SQLite не меняется; существующие книги и поколения в `~/.linguary/linguary.db` остаются полностью читаемыми.
- **Границы слоёв:** `domain ← application ← adapters / web / cli` проверяются через `uv run pytest tests/architecture`. Никаких сторонних библиотек и I/O-модулей в `application`.
- **Замороженные SHA-256 промптов:** `_TRANSLATION_INSTRUCTIONS` и `snapshot_prompt_templates(..., reader_level="b2")` должны сохранять точные дайджесты (`test_baseline_b2_templates_sha256_unchanged`). `_SEGMENT_TRANSLATION_RULE` не изменяется.
- **Универсальность:** никаких хардкодов под конкретную книгу, персонажей или названия моделей LLM.
- **Подпись коммитов:** каждый коммит содержит строку `Assisted-by: <роль> (<инструмент>, <модель>)` согласно `AGENTS.md`.

## Review Focus

- `truncation.py`: британская и американская многосегментная прямая речь проходит, но реальные незакрытые кавычки (`test_truncation_validator_checks_only_translated_segments`, `test_unbalanced_quote_is_rejected`, `test_truncation_allows_british_dialect_apostrophes_and_unclosed_quotes`) по-прежнему отклоняются.
- `mixed_script.py`: многословная дословная цитата из оригинала проходит, а одиночное немаркированное `very` между русскими словами и кириллица в английских кавычках отклоняются.
- `schemas.py` / `payloads.py`: `SegmentTranslationSchema.index` имеет `description` на wire-схеме без `minimum`; `_schema_hints()` безопасно поясняет `greater_than_equal`, `greater_than`, `value_error` без круглых скобок и сырого текста книги.
- `block_processing.py` / `translation_units.py` / `recovery.py`: адаптивный запас на скрытые рассуждения работает без хардкода имён моделей; `DEFAULT_MAX_UNIT_BLOCKS = 16` не ломает короткие фикстуры и связные списки; BFS в `plan_unresolved` сначала обрабатывает здоровую ветвь на меньшей глубине.
- `illustration_*`: `FACT_SCOPE` + `excluded_facts` блокируют воскрешение устаревших черт `DESIGN`; `used_design_keys` реально управляет итоговым `prompt_en` и `assumptions_ru`; собственные имена с «цветными» словами (*Mr Brown*, *Violet*, *Green Gables*) не отклоняются в `monochrome_ink`; ID-only промпты по-прежнему отклоняются `validate_moment()`.

---

### Task 1: Баланс кавычек в мультисегментной прямой речи (`WP-1.1`)

Соответствует SDD `WP-1` (§ 1.1).

**Files:**
- Modify: `src/linguary/application/validators/truncation.py`
- Modify: `tests/unit/application/test_result_validators.py`

**Interfaces:**
- Consumes: `british_quote_counts`, `has_unclosed_british_quote` из `validators/quotes.py`
- Produces: точный подсчёт британских кавычек в `_balances()` и двухуровневую проверку баланса кавычек в `truncation_violation` / `truncation_diagnostic`

- [ ] **Step 1: Написать падающие регрессионные тесты**

В `tests/unit/application/test_result_validators.py` добавить тесты:
- сегментированный блок с открытием британской кавычки `‘` в середине сегмента после `: ` и закрытием `’` в последующем сегменте (`«...` / `...»`) → принят;
- сегментированный блок с несколькими парами британских реплик через разные сегменты → принят;
- сегментированный блок с прямой `"` в сег. 1 и закрывающей `"` в сег. 2 → принят;
- сегментированный блок, где `TRANSLATE`-сегмент открыл `«`, но ни один сегмент не закрыл `»` при закрытом оригинале → отклонён (`quote_balance`).

Run: `uv run pytest tests/unit/application/test_result_validators.py -k "truncation" -q`
Expected: новые тесты на британскую/прямую многосегментную прямую речь падают.

- [ ] **Step 2: Реализовать фикс в `truncation.py`**

1. В `_balances(text)` добавить `brit_open, brit_close = british_quote_counts(text)` в `openings` и `closings`.
2. Выделить `_quote_balance_matches(tr_balance, *, source, source_balance)` с поддержкой `tr_balance == 0`, `tr_balance == source_balance`, нечётных `"` (`abs(tr_balance) == 1 and source.count('"') % 2 == 1`) и `has_unclosed_british_quote`.
3. Объединить проверку одного блока в общий хелпер `_block_truncation_issue(block, item)` для `truncation_violation` и `truncation_diagnostic`: для сегментированных блоков проверять пунктуацию и скобки посегментно, а кавычки — посегментно с агрегированной компенсацией **на склеенных строках** `joined_tr` / `joined_src` всех `TRANSLATE`-сегментов блока (а не через арифметическую сумму `_balances`).

- [ ] **Step 3: Прогнать тесты валидаторов**

Run: `uv run pytest tests/unit/application/test_result_validators.py -q --tb=line`
Expected: все тесты зелёные.

- [ ] **Step 4: Commit**

```bash
git add src/linguary/application/validators/truncation.py tests/unit/application/test_result_validators.py
git commit -m "fix(validators): support British and multi-segment quote balance in truncation checks"
```

---

### Task 2: Контекстное исключение дословных цитат оригинала в `note_language_leak_violation` (`WP-1.2`)

Соответствует SDD `WP-1` (§ 1.2).

**Files:**
- Modify: `src/linguary/application/validators/mixed_script.py`
- Modify: `src/linguary/application/validators/result.py`
- Modify: `tests/unit/application/test_result_validators.py`

**Interfaces:**
- Consumes: `unit.items` (исходные тексты блоков) и `original_excerpt` / `hint.original`
- Produces: опциональный `source_texts: Mapping[str, str] | None = None` в `note_language_leak_violation` и `_cited_source_spans()`

- [ ] **Step 1: Написать падающие регрессионные тесты**

В `tests/unit/application/test_result_validators.py` добавить тесты:
- русская заметка, цитирующая многословную фразу `How very wrong` из `original_excerpt` / текста блока → принята;
- одиночное немаркированное слово `very` между русскими словами при наличии `very` в тексте блока → отклонена (`note_language_leak`);
- многословная немаркированная английская вставка с `very`, отсутствующая в исходнике → отклонена (`note_language_leak`).

Run: `uv run pytest tests/unit/application/test_result_validators.py -k "language_leak" -q`
Expected: тест с дословной цитатой `How very wrong` падает.

- [ ] **Step 2: Реализовать `_cited_source_spans` в `mixed_script.py` и передать `source_texts` из `result.py`**

1. В `mixed_script.py` добавить опциональный `source_texts: Mapping[str, str] | None = None` в `note_language_leak_violation()` и пробросить его в обе ветки `_note_texts_of()`.
2. Реализовать `_cited_source_spans(text, allowed_sources)` без смещения индексов `(start, end)` исходного `text`, для непрерывных многословных (≥ 2 слова, допускается внутренняя запятая) латинских отрезков, дословно совпадающих по границам слов с нормализованными `allowed_sources`.
3. В `result.py` передать `source_texts={item.block.id: item.block.text for item in unit.items}` в оба вызова `note_language_leak_violation()`.

- [ ] **Step 3: Прогнать тесты валидаторов**

Run: `uv run pytest tests/unit/application/test_result_validators.py -q --tb=line`
Expected: все тесты зелёные.

- [ ] **Step 4: Commit**

```bash
git add src/linguary/application/validators/mixed_script.py src/linguary/application/validators/result.py tests/unit/application/test_result_validators.py
git commit -m "fix(validators): allow verbatim multi-word source quotes in Russian notes"
```

---

### Task 3: Контракт `segments[].index`, подсказки `_schema_hints()` и правила `source_block_id` / `translated_excerpt` (`WP-2`)

Соответствует SDD `WP-2`.

**Files:**
- Modify: `src/linguary/adapters/llm/schemas.py`
- Modify: `src/linguary/adapters/llm/payloads.py`
- Modify: `src/linguary/application/context_builder.py` (`_TRANSLATION_CONTEXT_V2`, `_CONTENT_PLAN_PROCESSING_V3`)
- Modify: `src/linguary/application/validators/content_plan.py`
- Modify: `tests/unit/llm/test_segment_translation_contract.py`
- Modify: `tests/unit/llm/test_llm_payload_mapping.py`
- Test: `tests/unit/llm/test_openai_compatible_client.py`
- Test: `tests/integration/processing/test_llm_logging.py`
- Test: `tests/unit/application/test_context_builder.py`

**Interfaces:**
- Consumes: `SegmentTranslationSchema`, `_schema_hints`, `_TRANSLATION_CONTEXT_V2`, `_CONTENT_PLAN_PROCESSING_V3`, `_translated_excerpt_matches`
- Produces: wire-`description` для `index`, безопасные статические подсказки ремонта для `greater_than_equal`/`greater_than`/`value_error` (без `source_block_id` и `treatment`), правила `bN` vs `bN.M` в незамороженных промптах

- [ ] **Step 1: Написать падающие тесты**

Добавить проверки:
- `strict_json_schema(schema_for(LLMResponseKind.TRANSLATION_RESULT, contract_version=4))` содержит `"description"` для `SegmentTranslationSchema.index` и по-прежнему не содержит `"minimum"`;
- `_schema_hints()` для `segments[0].index: greater_than_equal` и для `BlockResultSchema` `value_error` формирует точные статические подсказки без круглых скобок, **без** `source_block_id` и **без** значения `treatment`;
- `_TRANSLATION_CONTEXT_V2` и `_CONTENT_PLAN_PROCESSING_V3` содержат явное правило для `segments[].index` (1-based `M` из `[bN.M]`) и запрет `bN.M` в `source_block_id`.

- [ ] **Step 2: Внести изменения в `schemas.py`, `payloads.py`, `context_builder.py`, `content_plan.py`**

В `_schema_hints()` для `value_error` использовать строгий allowlist / маппинг суффиксов `BlockResultSchema._check_treatment_composition` (не копировать сырой `item["msg"]`!). Строго не трогать `_TRANSLATION_INSTRUCTIONS`, `_SEGMENT_TRANSLATION_RULE` и B2/`legacy-v1`.

- [ ] **Step 3: Прогнать тесты схемы, приватности логов, контекста и замороженных SHA-256**

Run: `uv run pytest tests/unit/llm/test_segment_translation_contract.py tests/unit/llm/test_llm_payload_mapping.py tests/unit/llm/test_openai_compatible_client.py tests/integration/processing/test_llm_logging.py tests/unit/application/test_context_builder.py tests/unit/application/test_translation_context.py -q --tb=line`
Expected: все тесты зелёные, включая `test_baseline_b2_templates_sha256_unchanged` и проверки приватности логов.

- [ ] **Step 4: Commit**

```bash
git add src/linguary/adapters/llm/schemas.py src/linguary/adapters/llm/payloads.py src/linguary/application/context_builder.py src/linguary/application/validators/content_plan.py tests/unit/llm/ tests/unit/application/
git commit -m "fix(llm): document segment index contract and expose safe static schema hints"
```

---

### Task 4: Адаптивный бюджет скрытых рассуждений, лимит блоков в `TranslationUnit` и BFS-обход скоупов `Recovery` (`WP-3`)

Соответствует SDD `WP-3`.

**Files:**
- Modify: `src/linguary/application/token_budget.py`
- Modify: `src/linguary/application/block_processing.py`
- Modify: `src/linguary/application/translation_units.py`
- Modify: `src/linguary/application/recovery.py`
- Modify: `tests/unit/application/test_translation_units.py`
- Modify: `tests/unit/application/test_recovery_partition.py`
- Modify: `tests/unit/application/test_token_budget.py`

**Interfaces:**
- Consumes: `TokenUsage.reasoning_tokens` (`domain/llm_log.py` через `response.usage` и `error.usage`), `plan_translation_units`, `Recovery.plan_unresolved`
- Produces: `IMPLICIT_REASONING_MIN_TOKENS`, `DEFAULT_MAX_UNIT_BLOCKS = 16`, потокобезопасная калибровка `TranslationCall._implicit_reasoning_floor`, BFS-сортировка скоупов по `(depth, attempts, first_block_order)`

- [ ] **Step 1: Написать падающие тесты**

1. В `test_translation_units.py`: 32 коротких блока по 100 символов при дефолтном `max_blocks=16` делятся на 2 юнита по 16; при `max_blocks=None` — 1 юнит; `max_blocks=0` бросает `TranslationUnitError`; связная серия > `max_blocks` разбивается.
2. В `test_recovery_partition.py`: после сплита корня на `(L, R)` и последующего сплита `L` на `(L1, L2)` вызов `plan_unresolved` возвращает `R` перед `L1`, `L2`.
3. В `test_token_budget.py`: `recommended_unit_max_completion_tokens(..., implicit_reasoning_tokens=5000)` добавляет запас; после первого ответа с `TokenUsage.reasoning_tokens > 0` или после `LLMResponseFormatError(..., truncated=True)` (включая случай `max_output_tokens=None` и `error.usage`) следующий вызов в `TranslationCall` сразу получает повышенный `max_completion_tokens`.

- [ ] **Step 2: Реализовать изменения в `token_budget.py`, `block_processing.py`, `translation_units.py`, `recovery.py`**

- [ ] **Step 3: Прогнать модульные и интеграционные тесты обработки и совместимости**

Run: `uv run pytest tests/unit/application/test_token_budget.py tests/unit/application/test_translation_units.py tests/unit/application/test_recovery_partition.py tests/integration/processing/ -q --tb=line`
Expected: все тесты зелёные (включая `test_translation_legacy_compatibility.py`).

- [ ] **Step 4: Commit**

```bash
git add src/linguary/application/token_budget.py src/linguary/application/block_processing.py src/linguary/application/translation_units.py src/linguary/application/recovery.py tests/
git commit -m "fix(processing): bound unit block count, schedule recovery BFS, and adapt implicit reasoning budget"
```

---

### Task 5: Хронологическая точность `DESIGN` / `FACT_SCOPE`, `used_design_keys` и чистка `assumptions_ru` (`WP-4`)

Соответствует SDD `WP-4`.

**Files:**
- Modify: `src/linguary/adapters/llm/illustration_payloads.py`
- Modify: `src/linguary/application/illustration_context.py`
- Modify: `src/linguary/application/illustration_source.py`
- Modify: `src/linguary/application/illustration_validation.py`
- Modify: `src/linguary/application/illustration_reference.py`
- Modify: `src/linguary/application/illustration_chapter.py`
- Modify: `tests/unit/llm/test_illustration_payloads.py`
- Modify: `tests/unit/application/test_illustration_reference.py`
- Modify: `tests/unit/application/test_illustration_chapter.py`
- Modify: `tests/unit/application/test_illustration_reliability.py`

**Interfaces:**
- Consumes: `IllustrationDesignTraitSchema`, `_build_bounded_design_input`, `resolve_appearance_with_design_fallback`, `compute_chapter_fingerprint`, `IllustrationChapter._advance_chapter_rendering`, `_validate_render_response`
- Produces: изоляция `episodic` в `DESIGN` (всегда с булевым `"non_permanent_context_only"`), пустой `assumption_ru=""`, учёт `excluded_facts` и `compact_description_*` в fallback (без `bundle` и со строгим 2-токенным эхом с `_ECHO_STOPWORDS`), унифицированная материализация обеих веток `illustration_chapter.py` строго по `used_design_keys` с `"base_fallback_entity_ids"`, фильтр заглушек `filter_non_assumption_placeholders`, `illustration_chapter_v2`

- [ ] **Step 1: Написать падающие регрессионные тесты**

1. `_build_bounded_design_input` при наличии `permanent`-фактов не кладёт `episodic`/`state_change` в `facts` и ставит `non_permanent_context_only: False`; при их отсутствии ставит `non_permanent_context_only: True`.
2. `IllustrationDesignTraitSchema` принимает `assumption_ru=""`.
3. `_validate_design` отклоняет хроматический цвет в `hair`/`attire_default` при `monochrome_ink` (сохраняя подстроку `"monochrome_ink"` в `repair_reason`), но не отклоняет имя персонажа (*Mr Brown* / *Violet*).
4. `resolve_appearance_with_design_fallback` с `excluded_facts` (устаревший возраст/одежда/сон/освещение) и актуальным `compact_description_en` подавляет устаревшие черты `DESIGN`, сохраняя `palette_value` и не подавляя нейтральную черту `"Short dark hair"` из-за `"short dark cloak"` в `excluded_facts`, а также сохраняя тест `bundle`/`managed`/`parties`/`big eyes`.
5. `IllustrationChapter` (обе ветки — с фактами и без фактов): неиспользованный `DESIGN`-ключ не попадает в `prompt_en`, `character_descriptions` и `assumptions_ru`; заглушки «Допущений нет.» отфильтровываются.

- [ ] **Step 2: Реализовать изменения WP-4**

1. Перенести `_MONOCHROME_COLOR_EN_RE`/`_MONOCHROME_COLOR_RU_RE` и реализовать `find_monochrome_color_word(..., strict_palette=...)` + `filter_non_assumption_placeholders()` в `illustration_validation.py`.
2. Обновить `IllustrationDesignTraitSchema`, `_STAGE_INSTRUCTIONS[DESIGN]`, `_build_bounded_design_input()`, `_build_design_scope_spec()` (`entity_names`), `_validate_design()`, `resolve_appearance_with_design_fallback()`, `compute_chapter_fingerprint()` (`illustration_chapter_v2`).
3. Унифицировать обе ветки (`838-870` и `1170-1207`) в `illustration_chapter.py`, сохраняя `design_traits_by_key`, `base_fallback_entity_ids` и `entity_names` в `summary` (с безопасным fallback для старых `scope.summary`).

- [ ] **Step 3: Прогнать тесты иллюстраций**

Run: `uv run pytest tests/unit/llm/test_illustration_payloads.py tests/unit/application/test_illustration_reference.py tests/unit/application/test_illustration_chapter.py tests/unit/application/test_illustration_reliability.py -q --tb=line`
Expected: все тесты зелёные.

- [ ] **Step 4: Commit**

```bash
git add src/linguary/adapters/llm/illustration_payloads.py src/linguary/application/illustration_context.py src/linguary/application/illustration_source.py src/linguary/application/illustration_validation.py src/linguary/application/illustration_reference.py src/linguary/application/illustration_chapter.py tests/
git commit -m "fix(illustrations): respect FACT_SCOPE exclusions and used_design_keys over static DESIGN"
```

---

### Task 6: Ахроматическая проекция `monochrome_ink` в `FACT_SCOPE`/`PROMPT` и дедупликация `Visual details:` (`WP-5`)

Соответствует SDD `WP-5`.

**Files:**
- Modify: `src/linguary/application/illustration_validation.py`
- Modify: `src/linguary/application/illustration_context.py`
- Modify: `src/linguary/application/illustration_chapter.py`
- Modify: `tests/unit/application/test_illustration_chapter.py`
- Modify: `tests/unit/application/test_illustration_reliability.py`

**Interfaces:**
- Consumes: `find_monochrome_color_word`, `compose_monochrome_prompt`, `validate_moment`, `_build_fact_scope_spec`, `_validate_render_response`
- Produces: передача `style` в `FACT_SCOPE`, проверка хроматических цветов в `FACT_SCOPE.compact_description_en` и `PROMPT.scene_composition_en` с безопасным маскированием имён и без ложных отказов на предметах (`copper kettle`, `ice cream`), клаузная дедупликация `Visual details:` (точное совпадение токенов + простое мн. ч. `-s`/`-es`) и клаузная проверка в `validate_moment()`

- [ ] **Step 1: Написать падающие тесты**

1. `find_monochrome_color_word()`: ловит `blue eyes`, `purple cloak`, `glow orange`, `изумрудный плащ`; пропускает *Mr Brown*, *Violet*, *Green Gables*, но отклоняет строчные `brown coat` / `green dress` при `allowed_names=("Mr Brown", "The Woman in Green")`; пропускает ахроматические слова и физические предметы (`copper kettle`, `bronze statue`, `ice cream`) при `strict_palette=False`.
2. `_validate_render_response(FACT_SCOPE)` (на `compact_description_en`) и `(PROMPT)` (на `scene_composition_en`) при `monochrome_ink` отклоняют хроматические цвета с понятным `repair_reason`, сохраняя `test_contradictory_facts_preserved_separately` зелёным.
3. `compose_monochrome_prompt()` не дублирует в `Visual details:` клаузы `description_en`, уже покрытые в `scene_composition_en` (без префиксных ложных совпадений `scar` ↔ `scarf`); `validate_moment()` принимает такой момент и по-прежнему отклоняет ID-only промпт.

- [ ] **Step 2: Реализовать изменения WP-5**

1. Передать `"style": revision.settings.style` в `_build_fact_scope_spec()`, `_build_fs_probe()` и `_build_consol_probe()`, сохранить `"entity_names"` в `summary`.
2. Обновить `_STAGE_INSTRUCTIONS[FACT_SCOPE]`, `_STAGE_INSTRUCTIONS[PROMPT]` и проверки в `_validate_render_response()` для `FACT_SCOPE` (`compact_description_en`) и `PROMPT` (`scene_composition_en`).
3. Реализовать `_clause_covered_in_text()` и `_appearance_reflected_in_prompt()` в `illustration_validation.py`, подключить их в `compose_monochrome_prompt()` и `validate_moment()`.

- [ ] **Step 3: Прогнать все тесты иллюстраций**

Run: `uv run pytest tests/unit/application/test_illustration_chapter.py tests/unit/application/test_illustration_reliability.py tests/unit/application/test_illustration_reference.py -q --tb=line`
Expected: все тесты зелёные (включая `test_contradictory_facts_preserved_separately`, `test_id_only_prompt_without_description_en_rejected` и `test_compose_monochrome_prompt_uses_name_en_and_avoids_cyrillic`).

- [ ] **Step 4: Commit**

```bash
git add src/linguary/application/illustration_validation.py src/linguary/application/illustration_context.py src/linguary/application/illustration_chapter.py tests/unit/application/
git commit -m "fix(illustrations): project monochrome_ink tones in FACT_SCOPE and deduplicate Visual details"
```

---

### Task 7: Blocking в `ALIAS`, `alias_of_hint` в `EXTRACT` и лемматизация `glossary_hints` в A2 (`WP-6`)

Соответствует SDD `WP-6`.

**Files:**
- Modify: `src/linguary/application/illustration_reference.py`
- Modify: `src/linguary/application/illustration_context.py`
- Modify: `src/linguary/application/reader_text.py`
- Modify: `tests/unit/application/test_illustration_reference.py`
- Modify: `tests/unit/application/test_context_builder.py`
- Test: `tests/unit/application/test_reader_policy.py`

**Interfaces:**
- Consumes: `_BLOCKING_STOPWORDS`, `_blocking_keys_for_mention`, `_STAGE_INSTRUCTIONS[EXTRACT]`, `learner_instructions`, `_A2_NOTE_DISCIPLINE`
- Produces: сингуляризованные ключи blocking, расширенные родовые стоп-слова, инструкцию `alias_of_hint`, непротиворечивые правила лемматизации A2

- [ ] **Step 1: Написать падающие тесты**

1. В `test_illustration_reference.py`: `build_alias_candidate_pairs` формирует пару для `("John Smith", "the Smiths’ son")`, НЕ формирует пару для `("the Smiths’ son", "the Bakers’ son")`, и не портит имена на `-ss`/`-us`/`-is`.
2. В `test_context_builder.py`: `learner_instructions("a2")` не требует сохранять `glossary_hints.original` дословно в противоречии с правилом инфинитива `to`, а `test_baseline_b2_templates_sha256_unchanged()` остаётся зелёным.

- [ ] **Step 2: Реализовать изменения WP-6**

- [ ] **Step 3: Целевая и финальная проверка всего репозитория**

Run: `uv run pytest tests/unit/application/test_illustration_reference.py tests/unit/application/test_context_builder.py tests/unit/application/test_reader_policy.py tests/architecture -q --tb=line && uv run pytest -m "not slow" -q --tb=line`
Expected: все архитектурные тесты и весь набор модульных/интеграционных тестов проходят без ошибок; `git status` показывает 0 изменений в `alembic/`.

- [ ] **Step 4: Commit**

```bash
git add src/linguary/application/illustration_reference.py src/linguary/application/illustration_context.py src/linguary/application/reader_text.py tests/unit/application/
git commit -m "fix(alias,prompts): improve plural surname blocking and align A2 glossary hint rules"
```
