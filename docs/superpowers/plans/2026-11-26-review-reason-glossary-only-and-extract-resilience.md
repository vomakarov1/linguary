# План реализации: `reason` в проверке коррекции, пустые `GLOSSARY_ONLY`-сегменты и устойчивость `EXTRACT`

**Дата:** 2026-11-26  
**Спецификации:** `docs/superpowers/specs/2026-11-26-review-reason-glossary-only-and-extract-resilience/`  
**Ветка реализации:** `fix/review-reason-glossary-only-and-extract-resilience` (от `main` `3496d8c`)  
**Миграции БД:** 0

---

## Задачи

### Шаг 1 — WP1: `CorrectionVerdict` и `CorrectionVerdictSchema`
- [ ] `src/linguary/adapters/llm/correction_payloads.py`: ослабить `CorrectionVerdictSchema.reason` до `str` (без `min_length=1`, остаётся обязательным ключом), добавить `@model_validator(mode="after")` с требованием непустого `reason` при `not confirmed` и нормализацией пробельной строки в `""` при `confirmed`.
- [ ] `src/linguary/adapters/llm/payloads.py`: добавить статическую безопасную подсказку `"для вердикта needs_human нужна непустая причина"` в `_STATIC_VALUE_ERROR_HINTS`.
- [ ] `src/linguary/application/correction_dto.py`: в `CorrectionVerdict.__post_init__` требовать `self.reason.strip()` только при `not self.confirmed` (после проверки всех типов) и нормализовать пробельный `reason` в `""` при `confirmed`.
- [ ] `tests/unit/application/test_correction_inputs.py`: добавить тесты для всех комбинаций `verdict` × булевые флаги × `reason`, `parse_payload` и 7-аргументного legacy-вызова.

### Шаг 2 — WP2: Пустые `GLOSSARY_ONLY`-сегменты и смешанные блоки
- [ ] `src/linguary/application/analysis_merge.py`: реализовать `_demote_vacuous_mixed_glossary_segments()` (с защитами `1 <= segment.index <= len(spans)` и `sentence_index in sentence_bounds`) и подключить в `normalize_pedagogical_response()` (после Шага 3, перед Шагом 4, при `not english_help`).
- [ ] `src/linguary/application/context_builder.py`:
  - уточнить `_reader_level_treatment_guidance("a2")` (не заморожено);
  - уточнить `_CONTENT_PLAN_PROCESSING_V3` и `_SEGMENT_TRANSLATION_RULE` для смешанных блоков с `GLOSSARY_ONLY` (не заморожено).
- [ ] `src/linguary/application/validators/treatment.py`: добавить индексы сегментов `GLOSSARY_ONLY` во все три сообщения `_segmented_content()`.
- [ ] Тесты:
  - `tests/unit/application/test_analysis_merge.py` — тесты понижения пустых `GLOSSARY_ONLY`-сегментов и сохранения непустых/сложных/нелатинских/несегментированных/стихотворных;
  - `tests/unit/application/test_result_validators.py` — проверка индексов сегментов в сообщениях;
  - `tests/unit/application/test_translation_context.py` — обновление зеркальной константы `_SEGMENT_TRANSLATION_RULE` + проверка промптов;
  - `tests/unit/application/test_context_builder.py` — проверка новых правил и прохождение `test_baseline_b2_templates_sha256_unchanged`.

### Шаг 3 — WP3: Компактная нарезка, автодробление и цитаты `EXTRACT`
- [ ] `src/linguary/application/illustration_reference.py`:
  - добавить `_MAX_EXTRACT_TARGET_BYTES = 12_288`, `_MAX_EXTRACT_BLOCKS_PER_PAGE = 16`, `_MAX_EXTRACT_SCENE_UNITS_PER_PAGE = 2`, `_MAX_EXTRACT_BLOCK_UNITS_PER_PAGE = 16` и `_pack_extract_section_units()`;
  - перевести `plan_scopes()` на посекционную упаковку (с пропуском пустых секций `if sec_units:`) и последовательными `extract:p{page_idx:04d}`;
  - добавить пост-ремонтный вызов `_handle_post_send_split()` для составных `EXTRACT`-скоупов в обеих ветках `run_task()` (`LLMResponseFormatError` и `not validation.ok`);
  - уточнить текст ремонтной ошибки в `_validate_extract()` для неизвестного `subject_mention_id`.
- [ ] `src/linguary/application/illustrations.py`: добавить `or s.stage in _REFERENCE_STAGE_SET` в проверку `has_failed` (строка 458), чтобы упавший посекционный `EXTRACT` блокировал `prepare(..., section_id=...)`.
- [ ] `src/linguary/application/illustration_context.py`:
  - дополнить `_STAGE_INSTRUCTIONS[IllustrationScopeStage.EXTRACT]` правилом `facts[].subject_mention_id ∈ mentions[].mention_id`;
  - добавить `_QUOTE_CHAR_MAP`, `_normalize_quote_with_spans()`, `_normalized_occurrences()` и типографско-пробельный запасной путь в `resolve_quote_anchor()`.
- [ ] Тесты:
  - `tests/unit/application/test_illustration_reference.py` — посекционная упаковка, лимиты сцен/блоков/байт, одиночная крупная сцена, автодробление `EXTRACT` после исчерпания `max_repair_turns` на схеме и валидации, лимит `max_split_depth`, атомарный `EXTRACT`;
  - `tests/unit/application/test_illustration_reliability.py` — инструкция `subject_mention_id`, запасной поиск `resolve_quote_anchor` (кавычки, тире, многоточие, пробелы, дословный `SourceAnchor.quote`, `ambiguous`, частичное совпадение многоточия);
  - `tests/integration/processing/test_illustration_reliability_integration.py` и `tests/integration/processing/test_illustration_pipeline.py` — обновить стабы 2-главных тестов для корректной обработки `ILLUSTRATION_ALIAS` при посекционном `EXTRACT` + тест блокировки `prepare(section_id=...)` при упавшем `EXTRACT` другой секции.

### Шаг 4 — Полная верификация и ревью
- [ ] Прогнать `uv run pytest tests/architecture`.
- [ ] Прогнать полный `uv run pytest -m "not slow"` (100% зелёный).
- [ ] Отдать полный `git diff main...HEAD` на глубокое ревью кода и тестов независимому субагенту.
- [ ] Исправить все подтверждённые замечания ревью, повторно прогнать тесты.
- [ ] Влить ветку в `main` (`--no-ff`) и отправить в `origin/main` с подписью `Assisted-by:` в коммитах.
