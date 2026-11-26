# ADR: Архитектурные решения по `reason` в проверке коррекции, пустым `GLOSSARY_ONLY`-сегментам и устойчивости `EXTRACT`

**Дата:** 2026-11-26  
**Статус:** Accepted (после глубокого ревью спецификаций)

---

## ADR-1: Разрешить пустой `reason` только при фактически подтверждённом вердикте `CorrectionVerdict`

### Контекст
`CORRECTION_REVIEW_PROMPT` (`composition/correction_runtime.py:96-98`) просит указывать краткую причину на русском только для `needs_human`. `strict_json_schema()` вырезает `minLength` из JSON-схемы провайдера, но оставляет поле `reason` в `required`. В результате модели возвращают `"verdict": "confirmed", ..., "reason": ""`, на чём падают `CorrectionVerdictSchema` (`min_length=1`) и `CorrectionVerdict.__post_init__`.

### Решение
- В `CorrectionVerdictSchema` (`adapters/llm/correction_payloads.py`) заменить `reason: str = Field(min_length=1)` на `reason: str` (поле остаётся обязательным, но допускает `""`) и добавить `@model_validator(mode="after")`, требующий `bool(self.reason.strip())` во всех случаях, когда не выполнено `verdict == "confirmed" and error_supported and evidence_valid and minimal and no_semantic_loss`, а для подтверждённого вердикта нормализующий пробельную строку `reason` в `""`.
- В `_STATIC_VALUE_ERROR_HINTS` (`adapters/llm/payloads.py`) добавить соответствующую статическую подсказку без интерполяции данных книги/модели.
- В `CorrectionVerdict.__post_init__` (`application/correction_dto.py`) проверять непустоту `self.reason.strip()` только при `not self.confirmed` (после проверки типов всех четырёх булевых полей и `isinstance(self.reason, str)`).
- `CORRECTION_REVIEW_PROMPT` не менять: он уже соответствует контракту и не ломает хеши `CorrectionSnapshotV2`.

### Альтернативы и причины отказа
1. **Разрешить пустой `reason` для всех вердиктов (включая `needs_human`):** отклонено — интерфейс ручного разбора (`correction_candidate.html:40`) показывает `review_reason` пользователю, чтобы объяснить, почему автоприменение отклонено.
2. **Требовать в промпте причину и для `confirmed`:** отклонено — тратит лишние выходные токены и всё равно остаётся хрупким, если модель вернёт пустую строку.

### Последствия
- Проволочная схема `strict_json_schema(ChapterCorrectionReviewSchema)` не меняется.
- `work_state.py` (`record_correction_verdict`, `_confirmed_candidate`), `book_overview.py` и `correction_candidate.html` уже безопасно работают с `reason == ""` при `confirmed`.
- Позиционный legacy-вызов `CorrectionVerdict` с 7 аргументами сохраняет работоспособность (`isinstance("", str) is True`).

---

## ADR-2: Понижать пустые элементарные `GLOSSARY_ONLY`-сегменты на этапе нормализации плана, не ослабляя доменный инвариант

### Контекст
В Round 4 на смешанном блоке `b#5` элементарное предложение (*«The Dursleys knew that the Potters had a small son, too, but they had never even seen him.»*) получило режим `GLOSSARY_ONLY` без единой трудности и без небазовой лексики A2. Поскольку валидатор перевода (`validators/treatment.py`) требует подсказки из сегмента `GLOSSARY_ONLY`, а промпт перевода не уточняет это для смешанных блоков и ошибка не называет индекс сегмента, перевод зациклился на 9 попыток и породил мусорные подсказки `even`/`too`.

### Решение
1. **Устранить первопричину в планировании (`application/analysis_merge.py`):**
   - Добавить чистую функцию `_demote_vacuous_mixed_glossary_segments(...)` и вызывать её в `normalize_pedagogical_response()` после Шага 3 (унификация стихов и списков) и до Шага 4/5 (при `not english_help`, для всех уровней читателя, кроме блоков `BlockType.POEM`).
   - Функция рассматривает только **смешанные сегментированные блоки** (`item.segments` содержит хотя бы один `TRANSLATE` и хотя бы один `GLOSSARY_ONLY`).
   - Внутри такого блока для каждого предложения (сгруппированного по `sentence_index`), чей сильнейший режим — `GLOSSARY_ONLY`, если:
     - `dominant_script(s_text) == "LATIN"`,
     - нет привязанных трудностей в диапазоне предложения (`not _has_difficulty_in_range(..., whole_block=False)`),
     - нет стоп-листа (`not _has_a2_nonlexical_stoplist(s_text)`),
     - нет слов из `_SHORT_B1_WORDS`, фраз из `_SHORT_B1_PHRASES` и небазовых глаголов речи (`_NONBASIC_REPORTING - _BASIC_REPORTING`),
     - нет фразовых глаголов `_A2_B1_PHRASAL_VERB_RE` и небазовой лексики `non_core_a2_word_spans(s_text, known_names=known_names)`,
     то все `GLOSSARY_ONLY`-сегменты этого предложения понижаются в `READ_AS_IS` (режим всего блока остаётся `TRANSLATE`).
2. **Уточнить промпты и диагностику без изменения замороженных SHA-256 шаблонов:**
   - В `_reader_level_treatment_guidance("a2")` (`context_builder.py`) уточнить, что `GLOSSARY_ONLY` требует хотя бы 1 незнакомое слово/выражение выше A2, иначе — `READ_AS_IS`.
   - В `_CONTENT_PLAN_PROCESSING_V3` и `_SEGMENT_TRANSLATION_RULE` (`context_builder.py`) явно указать, что в смешанном блоке с сегментом `GLOSSARY_ONLY` обязательно вернуть 1–3 `glossary_hints` именно из сегмента `GLOSSARY_ONLY`.
   - В `validators/treatment.py:_segmented_content()` включать список индексов сегментов `GLOSSARY_ONLY` в оба сообщения об ошибках, сохраняя прежние текстовые префиксы.

### Альтернативы и причины отказа
1. **Разрешить `GLOSSARY_ONLY` без подсказок в `domain/treatment.py` и `validators/treatment.py`:** отклонено — нарушает базовый продуктовый инвариант PRD и позволило бы молча принимать пустые подсказки на действительно сложных сегментах вроде `b#106` (`doorstep`, `cloak`, `tucked`).
2. **Подменять `GLOSSARY_ONLY` → `READ_AS_IS` прямо в `ResultValidator` во время перевода:** отклонено — приведёт к тихому расхождению сохранённого `TreatmentPlan` (хешированного в `root_manifest` и проверяемого в `correct_chapter`) с `BlockProcessingResult`.
3. **Понижать пустые несегментированные `GLOSSARY_ONLY`-блоки целиком:** отклонено на текущем этапе — может задеть устоявшиеся регрессионные тесты A2 на коротких репликах (например, `b_dlg14` в `test_a2_harry_potter_regressions.py`), тогда как зацикливание в смешанных блоках закрывается точечно и безопасно.

### Последствия
- `TreatmentPlan`, `ContentPlan`, `ResultValidator`, `Recovery`, `Correction` и `Layout` остаются полностью согласованными.
- Замороженные SHA-256 шаблоны (`_TRANSLATION_INSTRUCTIONS` и legacy-v1 B2) не затронуты.
- Понижение работает одинаково корректно для A2, B1, B2 и C1 (предложение чисто из базовой лексики A1–A2 тем более элементарно для B1–C1).

---

## ADR-3: Посекционная компактная нарезка и автодробление `EXTRACT` при повторных ошибках валидации

### Контекст
В `IllustrationReference.plan_scopes()` юниты всех секций склеиваются в единый список и упаковываются до 64 юнитов / 23.4 КБ. В Round 4 `extract:p0002` склеил 4 сцены (31 блок) из двух разных глав, перегрузив контекст модели (12–19К токенов рассуждений) и вызвав сбои `subject_mention_id`, сдвиг UUID блока и галлюцинации с потерей `covered_source_unit_ids`. При этом `IllustrationReference.run_task()` не дробил составной скоуп после исчерпания `max_repair_turns`, в отличие от `CANDIDATES` в `IllustrationSelection.run_task()`.

### Решение
1. **Посекционная нарезка с жёстким потолком размера (`application/illustration_reference.py`):**
   - Упаковывать `EXTRACT`-юниты **внутри каждой секции отдельно** (никаких межглавных пакетов; каждый пакет получает конкретный `section_id`, что улучшает повторное использование по `input_sha256`).
   - Ввести явные константы упаковки `EXTRACT`:
     - `_MAX_EXTRACT_TARGET_BYTES = 12_288` (эффективный лимит: `min(policy.initial_target_bytes, _MAX_EXTRACT_TARGET_BYTES)`),
     - `_MAX_EXTRACT_BLOCKS_PER_PAGE = 16`,
     - `_MAX_EXTRACT_SCENE_UNITS_PER_PAGE = 2`,
     - `_MAX_EXTRACT_BLOCK_UNITS_PER_PAGE = 16` (для секций без сцен, где юнит = 1 блок).
   - Одиночный юнит, превышающий лимит блоков или `extract_target_bytes`, выпускается как одноэлементный пакет `(unit,)` — как и прежде, он либо проходит сам, либо дробится через `#L`/`#R` при превышении `policy.max_request_bytes` или повторной ошибке.
   - Сохранить последовательную нумерацию `scope_key = f"extract:p{page_idx:04d}"` по всем секциям для обратной совместимости.
2. **Автодробление составных `EXTRACT`-скоупов при исчерпании ремонтов (`application/illustration_reference.py`):**
   - В `run_task()` в обеих ветках исчерпания `max_repair_turns` / `_can_continue_activation` — для `LLMResponseFormatError` и для `not validation.ok` — если `scope.stage is IllustrationScopeStage.EXTRACT and is_composite`, вызывать `self._handle_post_send_split(..., fail_reason=..., detail=...)` и выходить из цикла (`break`), полностью зеркаля уже проверенный паттерн `CANDIDATES` (`illustration_selection.py:1213-1226, 1303-1320`).
   - Существующие предохранители (`max_split_depth`, `max_scopes_per_generation`, `max_calls_per_generation`, проверки устаревания `STALE`/`SOURCE_CHANGED`/`SETTINGS_CHANGED`) остаются без изменений.
   - `_advance_reference_pipeline()` уже собирает все не-`SPLIT` листья `EXTRACT`, поэтому дочерние скоупы бесшовно интегрируются в `ALIAS`/`DESIGN`/`SEAL`.
3. **Уточнение контракта `subject_mention_id` (`illustration_context.py`, `illustration_reference.py`):**
   - Явно добавить правило в `_STAGE_INSTRUCTIONS[EXTRACT]` и уточнить текст ошибки в `_validate_extract()` (с сохранением прежнего префикса).
4. **Типографско-пробельный запасной поиск в `resolve_quote_anchor()` (`illustration_context.py`):**
   - Сохранить точный поиск по `visible` и `block.text` как быстрый путь.
   - Только если оба точных поиска вернули `missing`, выполнить нормализованный поиск (сведение фигурных кавычек/апострофов `‘’‚‛`→`'`, `“”„‟«»`→`"`, тире `–—−‑`→`-`, многоточия `…`→`...` и сжатие пробельных последовательностей) с точным обратным отображением индексов в `visible` → `origins` → `block.text`, защитой от частичного совпадения внутри развёрнутого `…`, проверкой однозначности (`ambiguous` при `>1` совпадении) и записью строго дословной `block.text[s_start:s_end]` в `SourceAnchor.quote`.

### Альтернативы и причины отказа
1. **Отключить проверку `covered_source_unit_ids`:** отклонено — в Round 4 именно она поймала галлюцинации модели в `#53` и `#58`.
2. **Алиасы `b1..bN` для `block_id` в проволочном JSON `EXTRACT`:** отклонено — ломает проволочный контракт и множество существующих тестов; компактные посекционные пакеты (`<= 16` блоков, `<= 2` сцен) устраняют перегрузку без смены формата.
3. **Молча отбрасывать невалидные факты/упоминания:** отклонено — искажает визуальный справочник и скрывает деградацию вывода.

### Последствия
- 0 миграций БД, 0 изменений схемы `IllustrationExtractSchema`.
- Для 2 глав из Round 4 вместо 3 гигантских межглавных пакетов (`5 + 4 + 4` сцен) получится ~7–8 компактных посекционных пакетов по 1–2 сцены (до 16 блоков), а любой сбойный пакет при необходимости автоматически распадётся до 1 сцены или половин `#L`/`#R` без остановки конвейера.
