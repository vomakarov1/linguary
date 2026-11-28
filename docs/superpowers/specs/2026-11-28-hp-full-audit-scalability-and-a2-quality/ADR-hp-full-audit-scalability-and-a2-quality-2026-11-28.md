# ADR: Архитектурные решения по масштабируемости конвейера полной книги, качеству A2, экспорту EPUB и мосту субагентов

**Дата:** 2026-11-28  
**Статус:** Revised (доработано по итогам двух независимых ревью спецификаций и задетого кода `HEAD d9ff0f9`)  
**Связанный BRIEF:** `BRIEF-hp-full-audit-scalability-and-a2-quality-2026-11-28.md`

---

## ADR-1: Совместная (хоровая) речь через `ContextFact.joint`, строгий контракт `LayoutUnit.speaker` и защитная нормализация (WP1)

- **Контекст:** В `domain/content_plan.py:220-234` `ContentPlan.__post_init__` индексирует факты по `(fact.kind, fact.subject_anchor)` и считает любое второе значение `(value, certainty)` конфликтом атрибуции. В художественной прозе (гл. 6 `‘Harry Potter,’ chorused the twins.`, гл. 10 `‘Hermione!’ they said together.`) одна реплика может принадлежать нескольким говорящим одновременно. Одновременно необходимо:
  1. не ломать `canonical_id`, `ContentPlan.sha256` и сохранённые `content_plan_json` / `analysis_pass_results` версии `3` (0 миграций БД);
  2. пронести `joint` через **обе** точки конструирования `ContextFact` (`validate_narrative_response` и `content_plan_from_response` в `analysis_merge.py:4013-4024`);
  3. сохранить совместимость с OpenAI strict JSON Schema (`adapters/llm/json_schema.py:101`) и с типом `LayoutUnit.speaker: str | None`, на который завязан Web API (`ResultUnitResponse.speaker: str | None = None` в `web/schemas.py:727-752`) и проверка типов `ty`;
  4. не допустить ложного склеивания разных одиночных реплик и ложных срабатываний эвристики совместной речи внутри кавычек или в соседних предложениях.
- **Решение:**
  1. **Домен (`domain/content_plan.py:110-149, 220-234`):**
     - Добавить в `ContextFact` опциональное поле `joint: bool = False`.
     - В `ContextFact.__post_init__`: `type(self.joint) is bool`; `joint=True` разрешён строго при `kind == 'speaker'` и `certainty != 'unknown'` (иначе `DomainError`).
     - В `ContextFact.to_dict()`: включать `'joint': True` **только при `self.joint is True`**. Все одиночные факты (`joint=False`) сохраняют побайтово идентичный `to_dict()`, `canonical_id` и `ContentPlan.sha256`.
     - В `ContentPlan.__post_init__`: для каждой группы фактов с одинаковым `(kind, subject_anchor)`:
       - при `kind != 'speaker'` сохраняется прежний инвариант строго одного `(value, certainty)`;
       - при `kind == 'speaker'`: группа допустима без конфликта, если либо `len(distinct_entries) == 1`, либо **все** факты группы имеют `joint=True`, `certainty != 'unknown'`, и ни одно `value` не повторяется с двумя разными `certainty`. Все остальные случаи (`joint=False` с разными `value`, смесь `joint=True` и `joint=False`, смесь с `unknown`, дубль одного `value` с разными `certainty`) по-прежнему поднимают `DomainError("конфликт атрибуций одного субъекта: ...")`.
  2. **Схема, DTO, кэш и БД:**
     - В `QuotedContextFactSchema` (`adapters/llm/schemas.py:455`) добавить `joint: bool = Field(default=False, strict=True)` с валидацией `joint=True ⇒ kind == "speaker" and certainty != "unknown"`.
     - В `QuotedContextFactResponse` (`src/linguary/application/llm_dto.py:430`, реэкспорт в `ports/llm.py`) добавить `joint: bool = False`; в `adapters/llm/payloads.py:844-852` прокинуть `joint=schema.joint`.
     - В `analysis_pass_cache.py:encode_pass` опускать `"joint"`, если `not fact.get("joint")`; в `decode_pass` читать `bool(f.get("joint", False))`.
     - В `adapters/db/mapping.py:917-928`: добавить строгий хелпер `_boolean(value)` (`if type(value) is not bool: raise ValueError(...)`) и разрешить опциональный ключ `'joint'` в `_context_fact`.
  3. **Защитная нормализация и порядок вызовов в `analysis_merge.py` (`L3563-3625, 3666-3670, 4013-4024`):**
     - Переносить `joint=fact.joint` и в `validate_narrative_response` (`L3564-3571`), и в `_quoted_fact` (`L3666-3670`), и в **`content_plan_from_response` (`L4016-4023`)**.
     - Вызывать `_normalize_joint_speaker_facts` **ДО** цикла `_explicit_speaker_if_remark_names` (сразу после сборки `mapped_facts` на `L3573`):
       - не требовать равенства `certainty` между разными говорящими (достаточно `certainty != "unknown"` и уникальных `value`);
       - ограничивать поиск `_JOINT_SPEECH_RE` исключительно **авторским текстом вне кавычек** внутри `anchors` группы и предложения, содержащего `subject_anchor` (исключив текст внутри кавычек и соседние предложения блока!).
     - Для `joint=True`-групп применять `_explicit_speaker_if_remark_names` **атомарно**: повышать `inferred → explicit` только если **все** участники совместной группы явно названы в ремарке, иначе оставлять всю группу в `inferred` (чтобы `explicit_parts` в `layout/book.py` не скрыл второго говорящего).
  4. **Строгий контракт `layout/book.py` ↔ `layout/units.py` ↔ Web API (`web/schemas.py`):**
     - `LayoutUnit.speaker` **сохраняет тип `str | None`** (для 1 имени — `"Гарри Поттер"`, для 2 — `"Гарри Поттер и Рон"`, для 3+ — `"Гарри, Рон и Гермиона"`): Web API (`ResultUnitResponse.speaker: str | None`) и `ty` не ломаются.
     - В `layout/units.py`: `block_units` и `_speakers_by_part` принимают `speakers: Mapping[tuple[int, int], str | tuple[str, ...]]`; внутри `block_units` исходная запись передаётся в `_speaker_note(entry)` (строка → `«Говорит {name}»`, кортеж из 2+ имён → `«Говорят {a} и {b}»` / `«Говорят {a}, {b} и {c}»`), а в `LayoutUnit.speaker` записывается отформатированная строка `_format_speaker_names(entry)`.
     - В `layout/book.py:_speakers_for_block`: объединять в совместный кортеж строго факты с `fact.joint is True` и **одинаковым** `fact.subject_anchor` (обычные одиночные `joint=False` факты никогда не склеиваются в «Говорят …», сохраняя прежнее правило первого говорящего юнита).
     - **Детерминированный порядок совместных имён:** так как `ContentPlan` сортирует `context_facts` по SHA-256 `id`, в `_speakers_for_block` сортировать имена совместной группы по позиции первого упоминания имени в тексте блока/главы, а при равенстве — по `name.casefold()`.
  5. **Коллизия `visual_facts.subject_anchor`:**
     - Оставить строгий wire-контракт `QuotedSceneVisualFactSchema(anchor, subject, detail)` без переименования, добавив явный анти-паттерн в `_NARRATIVE_ANALYSIS_INSTRUCTIONS` (`context_builder.py:207, 216`).
- **Альтернативы:**
  - *Менять `LayoutUnit.speaker` на `str | tuple[str, ...] | None`* — отвергнуто: ломает `ResultUnitResponse` (`web/schemas.py:727-752`) и `ty`.
  - *Хранить объединённую строку `"Harry Potter, Ron"` в `ContextFact.value`* — отвергнуто: ломает сопоставление с глоссарием и анти-спойлером.
- **Последствия:** Полная обратная совместимость `version=3` и Web API, снятие детерминированного блокера глав 6 и 10, детерминированный рендер «Говорят Гарри Поттер и Рон».

---

## ADR-2: Байтовые бюджеты промптов, компактный бэкфилл, фильтрация `scenes` и батчинг `chapter_correction_review` (WP2)

- **Контекст:** В реальных прогонах сразу два этапа пробили порог 120 КБ UTF-8: `chapter_analysis_narrative` (102,7 КБ в гл. 9; 132 КБ на ремонте гл. 10) и `chapter_correction` (139,9 КБ уже в гл. 3 из-за 36 КБ неотфильтрованных `scenes` с `visual_facts` и чанкинга по 8 КБ голого оригинала). Кроме того, `CorrectChapterV2` запускает ревью строго по одному `group_id` за вызов (8 вызовов на 15 кандидатов).
- **Решение:**
  1. **`chapter_analysis_narrative` (`context_builder.py`, `analyze_chapter.py`, `retry_policy.py`):**
     - Разделить подсказки прошлых глав в `_chapter_input` (`L1940-1985`) в два шага:
       1. сначала обогатить совпавшие с текущим фрагментом `optional_items` (в т. ч. `TermConcept(SUGGESTED)` с `key in used_keys`) полными метаданными (`note`, `description`, `opaque_name=false`) из `raw_accumulated` — сохраняет `test_context_builder.py:3110-3116`;
       2. затем действительно **не упомянутые** в текущем фрагменте сущности прошлых глав передать как `compact_backfill` в конец секции «Необязательные подсказки» (в порядке первых появлений / приоритетных `_KEY_GLOSSARY_BACKFILL_TYPES`) в компактном формате `- {source} [{kind}] -> {translation}` (плюс `; непрозрачное=false` при `opaque_name is False`), **без `note` и `description`**, с жёстким бюджетом `DEFAULT_MAX_BACKFILL_BYTES = 12_000` (и общим бюджетом секции `DEFAULT_MAX_OPTIONAL_CONTEXT_BYTES = 24_000`). Синхронно обновить `tests/unit/application/test_context_builder.py:3119`.
     - В `_fragment_findings_candidates` (`L3208-3231`): дедуплицировать против уже показанных в `_name_context` ключей и ограничивать размер `findings.instructions` (дедупликация строк, потолок 4 000 байт, не более 8 `visual_facts` последней сцены).
     - **Компактные `repair_turns` с единым порогом `65_000` байт UTF-8 (`analyze_chapter.py:1235-1241`, `retry_policy.py:242`):**
       - Учитывать `candidate_raw = (response.raw_content if response is not None else None) or getattr(error, "raw_content", None)` и байтовый размер уже накопленных `request.repair_turns` в `base_bytes`.
       - Если `raw_bytes > 20_000` или `base_bytes + raw_bytes > 65_000`: сбрасывать `assistant_response` в возвращённом `decision = replace(decision, assistant_response=None)` (либо через явный флаг `include_assistant_response=False` в `RetryPolicy.decide`), чтобы `retry_policy.py:242` не подтянул 40 КБ `error.raw_content` при `LLMResponseFormatError`!
     - **Сплит фрагмента при `LLMContextCapacityError` (`analyze_chapter.py:100, 1147-1158, 1231-1234`):**
       - Импортировать `LLMContextCapacityError` в `analyze_chapter.py`.
       - В `_request_staged` (`L1231`) перехватывать `isinstance(error, LLMContextCapacityError)` наравне с `LLMResponseFormatError(truncated=True)` (быстрый проброс без `retry_policy.decide`, фиксация `analysis_attempts = attempts` и накопленного `usage`).
       - В `_run_staged_fragments` (`L1147-1158`) безопасно читать `usage = getattr(error, "usage", None)` (у `LLMContextCapacityError` нет поля `.usage`, что иначе вызвало бы `AttributeError`!) и при `len(fragment.blocks) > 1` делить фрагмент пополам; при `len(fragment.blocks) == 1` пробрасывать ошибку наружу.
  2. **`chapter_correction` & `chapter_correction_review` (`use_cases/correct_chapter.py`):**
     - В `_llm_chapter_analysis` (`L461-575`): **безусловно** (даже при `allowed_block_ids is None`, когда чанк покрывает всю главу!) удалять из `content_plan.scenes` поля `id`, `order`, `visual_facts`, `visual_assumptions` и переводить `start_block_id`/`end_block_id` в алиасы `bN`. При `allowed_block_ids is not None` предварительно фильтровать `scenes` по пересечению диапазона с `allowed_block_ids` через `block_order` (построенный из `ordered_blocks = section.ordered_blocks` по каноническим `block.id` **до** замены на алиасы!).
     - В `_chunk_correction_blocks` (`L303-335`): сохранить позиционную совместимость `(block_ids, inputs, max_chars)` и добавить `*, max_request_bytes: int | None = None, candidates_by_block: Mapping[str, Any] | None = None`. Оценивать размер **поблочно** (не вычитая `chapter_analysis` всей главы целиком!):
       - фиксированный резерв на `instructions + schema + window_glossary + compact_scenes` (`_CORRECTION_FIXED_OVERHEAD_BYTES = 22_000`);
       - поблочный вклад: в режиме `propose` — `len(block.text.encode()) * 2 + result_bytes + per_block_analysis_bytes(bid)`; в режиме `review` (`candidates_by_block` передан) — `len(block.text.encode()) * 2 + candidate_review_bytes(cand) + per_block_analysis_bytes(bid)` (учитывает одновременно `current_result`, `proposed_result`, `changes` и `issues`).
       - По умолчанию в `CorrectChapterV2` (в **обоих** местах — в `__call__` на `L1487, 1492` и внутри `_review_group` на `L1763`!) передавать `max_request_bytes = DEFAULT_CORRECTION_MAX_REQUEST_BYTES = 72_000`, гарантируя минимум 1 блок/кандидат на чанк.
     - **Батчинг `chapter_correction_review` (`CorrectChapterV2.__call__` `L1483-1491`, `_review_group` `L1843-1851`):**
       - Объединять всех текущих `pending` кандидатов главы разных `group_id` в единый батч (в порядке блоков главы; если на один `block_id` > 1 кандидата, второй идёт в следующую итерацию `while True`) и нарезать через `_chunk_correction_blocks(..., max_request_bytes=72_000, candidates_by_block=...)`.
       - В `_review_group` (`L1843-1851`), после записи вердиктов `record_correction_verdict` (`L1815-1820`), для каждого `group_id in {c.group_id for c in live}` проверять готовность группы по `group = tx.work.get_binding('correction_group', group_id)['payload_json']` через `group['candidate_ids']` (`tx.work.get_correction_candidate(cid)` — без новых методов `tx.work`!): фиксировать `unchanged_block_ids` группы только когда среди текущих кандидатов этой группы не осталось `state == 'awaiting_review'`.
     - *Замечание о соседях:* аналогичный балласт `scenes` в `src/linguary/application/quality_context.py:345-389` зафиксирован как отдельный follow-up (вне конвейера анализа/перевода/коррекции книги).
- **Последствия:** Промпты `narrative` на гл. 9–17 держатся в пределах ~35–55 КБ UTF-8; промпты `correction` падают со 140 КБ до ~45–68 КБ без деградации до 1 блока на запрос; вызовы `chapter_correction_review` сокращаются в 2–3 раза (~60% экономии).

---

## ADR-3: Пакетная валидация сегментов, исправление `known_names` + кэшируемый даунгрейд в анти-спойлере и устойчивый фрагментный кэш при `retry` (WP3)

- **Контекст:** Три источника лишних отказов и потерь времени на анализе длинных глав: fail-fast с сырыми UUID в `plan_items_from_response`, отсутствие `known_names` для `mapped_facts` в `_collect_spoiler_name_leaks` и безусловный сброс фрагментного кэша в `RetryTask`. При этом отказ от сброса кэша в `RetryTask` требует защиты от вечного застревания на повреждённом/устаревшем кэше или на ошибке инкрементального слияния внутри `_run_staged_fragments`.
- **Решение:**
  1. **Пакетный валидатор `plan_items_from_response` (`analysis_merge.py:262-350`):**
     - Для односегментного блока (`len(spans) < 2`): если все `segment.treatment is item.treatment` — безопасно нормализовать в `TreatmentPlanItem(block_id=block_id, treatment=item.treatment)`; при расхождении режимов фиксировать ошибку.
     - Собирать все `AnalysisMergeError` по всем блокам чанка в единую ошибку `rule_code="invalid_segment_treatment"`, подставляя алиасы `alias` (`bN`) во все сообщения (включая замену канонического UUID в `str(err)` при `TreatmentPlanError` на `L344-349`), сохраняя поведение `require_all=True` (`KeyError(alias)`).
  2. **Анти-спойлер (`analysis_merge.py:2873-2932`, `analyze_chapter.py:605-610, 1025-1063, 1137-1243, 3126-3160`):**
     - В `_collect_spoiler_name_leaks` (`L2914-2931`) пропускать имена из `known` для `mapped_facts` так же, как для `scenes[].participants`.
     - В `analyze_chapter.py:1025-1039` добавлять `section.title` и тексты `BlockType.HEADING` до начала фрагмента в `fragment_known_names`.
     - В `_downgrade_spoiler_explicit_facts` (`L3126-3160`) добавить `known_names: Iterable[str] = ()` и прокинуть `known_names=known_names` также в существующий вызов уровня главы (`analyze_chapter.py:605-610`).
     - **Кэшируемый пофрагментный даунгрейд (`analyze_chapter.py:1137-1243`):** возвращать обновлённый (пониженный `explicit → inferred`) `payload` **из `_request_staged` наружу** (через нормализующий шаг/коллбэк валидации, возвращающий `(normalized_payload, resolved)`), чтобы именно пониженный `payload` записывался в `analysis_pass_results` (`_save_pass`) и попадал в `parts` — и применять ту же нормализацию на `from_cache=True`.
  3. **Устойчивый фрагментный кэш при `retry` (`retry_task.py:40-85`, `analyze_chapter.py:1085-1099, 1123-1138, 1245-1255`, `cli/app.py`):**
     - Убрать автоматический сброс фрагментного кэша в `RetryTask.__call__` для обычного `RetryTaskCommand(task_id=...)`; добавить явный `clear_cache: bool = False` (и флаг `--clear-cache` в `cli retry`; Web HTTP API `POST /tasks/{id}/retry` в этом скоупе сохраняет поведение по умолчанию `clear_cache=False`).
     - **Защита от вечного застревания на битом кэше в `analyze_chapter.py`:**
       - если `_verify_cached_pass` (`L1253`) или `validate(payload, from_cache=True)` (`L1137`) падает на кэшированном фрагменте, удалять фрагментные кэши текущего прохода и выполнять обычный запрос к LLM (cache-miss);
       - если инкрементальное `merge_chapter_analysis([*parts, candidate_part], ...)` (`L1091`) падает с межфрагментным конфликтом, очищать фрагментные кэши текущего прохода до эскалации задачи в `FAILED`.
     - Синхронно обновить `tests/integration/processing/test_retry_task_recovery_guards.py:299-455`.
- **Последствия:** Ремонт сегментов проходит за 1 попытку; ложные спойлеры исчезают; `retry` экономит 5–10 минут без риска застрять на повреждённом кэше.

---

## ADR-4: Защита ключевых `exempt_type_review` имён в `glossary_snapshot` без дедлока `suggested_note` (WP4)

- **Контекст:** Сущности с `exempt_type_review=True` (`Dursley`, `Privet Drive`, `Muggle`, `Dumbledore`, `McGonagall`, `Hagrid`) уходят в `glossary_snapshot` как `TermStatus.SUGGESTED` и не попадают в `mandatory_entries`. Однако автоматический перенос непроверенного модельного `suggested_note` в `TermStatus.APPROVED` опасен: `block_processing.py:397` на каждом блоке запускает `validate_fact_glossary(content_plan, glossary_context_constraints(mandatory_entries))`, и любое противоречие рода между модельной `suggested_note` одной главы и сохранённым `content_plan` другой главы вызвало бы неисправимый дедлок перевода!
- **Решение:**
  1. **Безопасное автопродвижение ключевых сущностей (`application/glossary_review.py`):**
     - Добавить в `GlossaryReviewItem` поле `exempt_type_review: bool = False` (заполняется из `eligible` на `L436-448`) и свойство `is_auto_mandatory`:
       ```python
       self.decision is None
       and self.exempt_type_review
       and self.type in _AUTO_MANDATORY_TYPES  # PERSON, PLACE, ORGANIZATION, FICTIONAL_CONCEPT
       and bool(self.suggested_translation.strip())
       and (self.is_frequent or self.mention_count >= 2 or self.chapter_count >= 2)
       ```
       *(Ветка `mention_count == 0` намеренно исключена: случай `count_mentions is None` уже даёт `is_frequent = True` на `L398`, а реальные 0 упоминаний не должны продвигаться! Если пользователь нажал `SKIP` → `decision` не `None` → `is_auto_mandatory is False`.)*
     - В `glossary_snapshot(..., promote_exempt: bool = True)` и `_snapshot_entry`: для элементов с `item.is_auto_mandatory` создавать `GlossarySnapshotEntry` со статусом **`TermStatus.APPROVED`** и **`note=""`** (не перенося непроверенную модельную `suggested_note` в жёсткий `glossary_context_constraints`!). Таким образом:
       - канонический перевод ключевых героев и топонимов гарантированно попадает в обязательный блок промпта перевода (`context_builder.py:1579, 1632, 2719`) и валидацию корректора (`correct_chapter.py:1185`);
       - полностью исключён дедлок `validate_fact_glossary` (`block_processing.py:397`) на противоречивой модельной гендерной помете;
       - ручная очередь ревью (`awaiting_decision`) и схема БД не меняются.
     - В `cli glossary list` (`cli/app.py`) вывести признак `auto_mandatory` и сводку по категориям (`decided`, `auto_mandatory`, `awaiting_review`, `suggested`). (Визуальный бейдж в Web UI глоссария зафиксирован в `BRIEF §3 Non-goals` как отдельный UI follow-up.)
  2. **Грамматика склонения титулов и прозрачность `language_notes_per_chapter`:**
     - В `_NARRATIVE_ANALYSIS_INSTRUCTIONS` (`context_builder.py:207`): закрепить склонение мужских титулов/должностей (`профессор`, `доктор`, `директор`, `инспектор`) перед женской фамилией (`голос профессора Макгонагалл`) и отслеживание второстепенных персонажей / говорящих кличек животных.
     - В `adapters/config/settings.py:329` и `analyze_chapter.py`: документировать масштабирование `scaled_chapter_note_limit` и логировать эффективный лимит заметок главы.
- **Последствия:** Главные имена романа защищены как обязательные `mandatory_entries` без риска дедлока `process_chapter` и без раздувания ручного ревью.

---

## ADR-5: Профиль A2 — произношение, порог `GLOSSARY_ONLY` (блоки и сегменты), безопасные анти-дубли, `<sup>`, «Говорит X» и `legend.xhtml` (WP5)

- **Контекст:** Аудит преподавателя и ученика A2 выявил 5 методических пробелов, при этом ревью кодовой базы выявило тонкие ограничения: `--pronunciation-help` и `--pronunciation-variant` уже существуют в CLI (`cli/app.py:139-149`); чтение ZIP EPUB разрешено строго в `adapters/epub/` (Clean Architecture) и недопустимо на каждом polling-запросе `GET /books/{id}`; форма переанализа `partials/quality_review.html` не должна меняться; порог `GLOSSARY_ONLY` должен работать и на сегментах, и после `_promote_a2_stoplist_item`; сдвиг `<sup>` не должен рвать апострофы `flyin’` и сокращения `Mr.`; подавление «Говорит X» не должно ложно срабатывать на однофамильцах; а `legend_document` обязан сохранять `test_learner_book_pages.py:65`.
- **Решение:**
  1. **Фонетическая поддержка (`pronunciation_help`) для A2:**
     - В `domain/document.py`: чистая функция `detect_english_variant(raw_language: str | None) -> Literal["uk", "us"] | None` (`en-gb`/`en-uk`/`en-au`/`en-nz`/`en-ie` → `"uk"`, `en-us`/`en-ca` → `"us"`, иначе `None`).
     - В `adapters/epub/inline.py` (строго в слое `adapters`, соблюдая `DOMAIN_RULE`/`APPLICATION_RULE`): хелпер `read_source_language_tag(source_epub_path: str | None) -> str | None` (читает первый `dc:language` из OPF, 0 миграций БД).
     - В `cli/app.py`: расширить существующий `--pronunciation-variant` (`choices=["uk", "us", "auto"]`); в `_analyze` разрешать `"auto"` через `detect_english_variant(read_source_language_tag(book.source_epub_path)) or "us"` **до** вызова `book_conditions.ensure_initial_for_analysis(...)` (явный fallback `"us"` для чистого `en` документируется в `--help`). Программный дефолт `BookConditionsService.create(pronunciation_help_enabled=False)` не меняется.
     - В Web UI (`web/routes/books.py:get_book` + `web/templates/books/detail.html:34-55`, **НЕ** затрагивая `partials/quality_review.html:70-76`!): читать `dc:language` **строго** при `wants_html(request) and response.book.status == "IMPORTED" and not response.conditions.conditions_id and response.book.source_language == "en"` (нулевая нагрузка на автообновление во время анализа/обработки!) и предвыбирать `pronunciation_help_enabled` + обнаруженный вариант (fallback `"us"`) для `default_reader_level == "a2"` (с аккуратным `onchange` на `<select name="reader_level">`, пока пользователь не переключил чекбокс вручную).
  2. **Порог перегрузки `GLOSSARY_ONLY` (> 4 не-A2 слов) и калибровка A2 (`analysis_merge.py`, `context_builder.py`):**
     - В `normalize_pedagogical_response` (`analysis_merge.py:6029`) добавить шаг `_promote_overloaded_glossary_only` **ПОСЛЕ** `_promote_a2_stoplist_item` (`L6022`) и до Step 3, с передачей `known_names`, применяемый единообразно и к целым `GLOSSARY_ONLY`-блокам, и к отдельным `GLOSSARY_ONLY`-сегментам: если уникальных не-A2 слов (`non_core_a2_words`, вне `known_names`) строго больше `_NARRATIVE_GLOSSARY_NON_CORE_WORD_LIMIT = 4`, повышать блок/сегмент до `TRANSLATE` (пересчитывая `strongest_treatment` у сегментированных блоков).
     - В `_PEDAGOGICAL_ANALYSIS_INSTRUCTIONS` (`context_builder.py:231`): ограничить дробление до 1–2 переходов режима на абзац и **запретить разрывать одну реплику персонажа в кавычках пополам между `TRANSLATE` и `GLOSSARY_ONLY`**.
     - В `_reader_level_treatment_guidance("a2")` (`L2166-2190`): синхронизировать короткие реплики (≤ 10 слов) с 1 покрытым глаголом речи (`single_covered_nonbasic`) и добавить скрытые ловушки B1 (`great`, `cool`, `work`, `hall`, `have a place at`, `No one ever did`, `Found…!`, `Everyone out!`, `should be celebrating`, `sounding relieved`, `looking tired`).
  3. **Безопасное анти-дублирование глосс, точный A1-фильтр и сдвиг `<sup>` (`layout/notes.py`, `a2_core_vocab.py`):**
     - `_suppress_hints_covered_by_notes` в `layout/notes.py:block_notes`: подавлять `hint`, только если перекрывающая его по смещению видимая `language_note` имеет компактный `original_excerpt` (`<= 6` слов) либо разделяет с глоссой `difficulty_id`, либо прямо покрывает перевод глоссы. Если у подавленного `hint` было `ipa`, **не** добавлять его в `excluded_hints`, чтобы `PronunciationOccurrence` сохранилось отдельной фонетической пометкой (без потери IPA и без дубля глоссы!). Инвариант «остаётся ≥ 1 подсказка/заметка» проверять в границах каждого `GLOSSARY_ONLY`-**сегмента**.
     - В `a2_core_vocab.py`: разложить слова по правильным наборам (`enter`, `celebrate` → `_A1_VERBS`; `bacon`, `mistake`, `darkness`, `uniform` → `_A1_NOUNS`/`_A1_INTERNATIONALISMS`; `perfect`, `violet`, `unusual`, `angrily` → `_A1_ADJECTIVES_AND_ADVERBS`; `wrong` уже в `_A1_ADJECTIVES_AND_ADVERBS`) и поддержать 2-словные составные A1-существительные (`school uniform`) в `is_trivial_a1_hint` (без ложных срабатываний на фразовые глаголы/идиомы).
     - В `layout/notes.py`: рефакторинг `_find_end` → `_find_span(...) -> tuple[int, int] | None` + `_trim_terminal_punctuation(text, start, end)`: сдвигать `marker_end` влево **только** через терминальные `.`, `!`, `?`, `…` и закрывающие `”`, `"` (а `’` — **строго** когда перед ним стоит `.`, `!`, `?` или `…`, никогда после буквы вроде `flyin’`/`Dursleys’`!), защищая сокращения (`Mr.`, `Mrs.`, `Ms.`, `Dr.`, `St.`, `Prof.`, `a.m.`, `p.m.`, инициалы `M.`) и сохраняя `end` при `new_end <= start`.
  4. **Безопасное подавление лишнего «Говорит X» (`layout/book.py:923-1030`):**
     - Согласовать очистку кавычек с `_DIALOGUE_QUOTES` (`analysis_merge.py:4811`) с защитой множественного притяжательного `s’ `.
     - В `_part_explicitly_names_speaker`: запретить совпадение по голой фамилии, если в авторском тексте перед ней стоит чужой титул или чужое имя (`Mrs Dursley` ≠ `Mr Dursley`), исключить притяжательные формы (`Harry’s`) и позицию после предлогов (`at/to/for/with/from/about/after/of`).
  5. **Обновление `legend.xhtml` (`layout/legend.py`, `adapters/epub/book_pages.py`, `writer.py:499`):**
     - В `legend.py`: «Пометка в начале примечания называет его вид: …», разведение глосс/заметок + совет не зубрить всё подряд в `_NOTES`, уточнение `Previously in…` как напоминания на языке оригинала.
     - В `book_pages.py:legend_document`: добавить опциональные `source_language: str | None = None, include_example: bool = False`. В `writer.py:499` передавать `source_language=source` и `include_example=bool((layout.features.has_translation or layout.features.has_notes) and source == "en")` — `test_learner_book_pages.py:63-65` (легенда только с `_ORIGINAL` и `_IPA`) остаётся зелёным!
     - В `layout/book.py:542, 556`: добавить явный комментарий о намеренно отложенном статусе `who_is_who=()` и `review=()`.
- **Последствия:** Закрыты все методические пробелы A2 без нарушения Clean Architecture, без регрессий в Web UI/CLI/кэше и без ложных срабатываний эвристик.

---

## ADR-6: Лингвистическое усиление промптов перевода, заметок, пересказа и балансировка корректора (WP6)

- **Контекст:** Лингвистическое и педагогическое ревью выявили промптовые пробелы: сглаживание просторечия персонажей (Хагрид), избыточные притяжательные местоимения, отдельные кальки, скачки рода животных, переводческий уклон и рассогласование якорей в заметках `speech_style`, громоздкие грамматические заметки, географическая неточность (`amigo` у бразильского удава), сбой времён в `Previously in…` и 1 гиперкоррекция автокорректора в несобственно-прямой речи. При этом `_A2_NOTE_DISCIPLINE` и `_A2_DIALOGUE_AND_CALQUE_RULES` входят в `_TRANSLATION_CONTEXT_V2` (`context_builder.py:164`) для всех уровней `content-plan-v3`, поэтому ограничения длины заметок нужно явно помечать «для уровня A2».
- **Решение:**
  1. В `_A2_DIALOGUE_AND_CALQUE_RULES` (`reader_text.py:516-530`): сквозное сохранение разговорного/просторечного регистра во всех репликах маркированного персонажа, единство грамматического рода животных/перевоплощающихся персонажей в сцене/главе, сокращение избыточных притяжательных местоимений, запрет словарного толкования реалии внутри художественной фразы при наличии сноски.
  2. В `_A2_NOTE_DISCIPLINE` (`reader_text.py:508-514`): в `speech_style` расшифровывать только нестандартные формы оригинала (без анализа приёмов русского перевода) и ставить якорь на конец реплики либо охватывать все разбираемые формы; для уровня A2 писать грамматические заметки коротко (15–30 слов, формула + перевод); сверять географический/языковой контекст заметок (например, португальский/испанский для Бразилии); в `TRANSLATE` избегать заметок на прозрачные аналоги и редкую книжную грамматику C1; в `glossary_hints.translation` оставлять только контекстное значение сцены.
  3. В `learner_instructions("a2")` (`reader_text.py:554`): согласование времён в `Previously in…` для событий до описываемой сцены.
  4. В `_PEDAGOGICAL_ANALYSIS_INSTRUCTIONS` (`context_builder.py:237-249`): приоритет реальных ловушек A2→B1 (`had had`, `can't/couldn't help + V-ing`, `had better`, редкие стяжения), привязка к первому появлению и баланс бюджета заметок (≤ 50% на `TRANSLATE` при наличии трудных `GLOSSARY_ONLY`).
  5. В `CORRECTION_PROMPT_V2` и `CORRECTION_REVIEW_PROMPT` (`composition/correction_runtime.py:85-116`): запрет гиперкоррекции времён в несобственно-прямой речи и педантичного дробления авторского продолжения через `<br/>`; расширение п. 2 чек-листа на очевидные лексико-смысловые сдвиги и грубые кальки.
  6. **Защита замороженных SHA-256:** `_TRANSLATION_INSTRUCTIONS`, `_ANALYSIS_INSTRUCTIONS` и `test_baseline_b2_templates_sha256_unchanged` побайтово неизменны.
- **Последствия:** Ровный литературный стиль, компактные точные заметки и сбалансированная корректура без изменения схем и версий.

---

## ADR-7: Качество экспорта EPUB — очистка `dc:description`, `cover-image`, чистая обложка, ведущий `pagebreak`, `.lg-unit-long` и front matter (WP7)

- **Контекст:** Аудит распакованного EPUB и проверка задетого кода выявили 6 дефектов вёрстки/метаданных и 4 критических краевых случая: повторный экспорт из БД без очистки `book_metadata.description`, вызов `document.section()` на синтетических главах (`linguary-legend`), `_reserve_opaque_ids` до удаления ведущего `pagebreak`, opaque SVG-обложки и multi-book front matter.
- **Решение:**
  1. **Единый хелпер `clean_dc_text` (`adapters/epub/inline.py`):** размещается в `inline.py` (исключает связность `reader → writer`), очищает HTML-теги/сущности и применяется в **трёх** точках: `reader.py:_metadata`, `writer.py:_read_source_dc_metadata` и `writer.py:_append_metadata` (`L1503-1508`, покрывая уже импортированные в SQLite книги).
  2. **`properties="cover-image"` (`writer.py:1515-1535`):** в `_cover_image_href` искать секцию строго через словарь `sections_by_id = {s.id: s for s in document.ordered_sections}` (безопасно пропуская синтетические `linguary-legend`, `linguary-ipa-key` и т. п.) и проставлять `properties="cover-image"` первому локальному image-ресурсу `cover`-секции.
  3. **Чистая обложка `.lg-cover` (`writer.py`, `layout_render.py`):** для `is_cover_section(section)` рендерить `<div class="lg-cover">...</div>` без `.lg-unit` и `<p class="lg-src">` **как** для `BlockType.IMAGE` (в `_render_units`), **так и** для `PRESERVE_OPAQUE` SVG-обложек (в `_append_opaque`).
  4. **Ведущий пустой `pagebreak` (`writer.py:992-1039`):** в `_append_blocks` извлекать и удалять ведущий пустой `pagebreak` (и `BlockType.PAGE_BREAK`, и `PRESERVE_OPAQUE` `p.pagebreak`/`div.pagebreak` без текста и без `<img>`/`<svg>`) **ДО** вызова `self._reserve_opaque_ids(blocks, state)`! Переносить первый его `id` на `<section epub:type="chapter">` (если у `container` ещё нет `id`), остальные — в `<div class="linguary-anchors"><span id="..."/></div>`, и регистрировать перенесённые `id` в `state.used_ids`.
  5. **Единый критерий `.lg-unit-long` (`layout_render.py`, `writer.py`, `web/static/linguary.css`):** в `render_unit` добавлять класс `lg-unit-long` при наличии `unit.translation`, когда `(unit.end - unit.start) + len(unit.translation) + sum(len(p) for n in unit.notes for p in n.note.body) > 450` (или в срезе оригинала ≥ 4 предложений). В `linguary.css` задать `.lg-unit.lg-unit-long { page-break-inside: auto; break-inside: auto; }`. Шаблон Web UI `result_block.html` не меняется (веб-страница прокручивается непрерывно без постраничной вёрстки).
  6. **Политика front matter (`writer.py:539-556`) и multi-book EPUB:** предикат `_is_dropped_front_matter(section, document)` отбрасывает дублирующие HTML-оглавления (`contents`/`toc`) и рекламные/издательские вставки, сохраняя `cover`, `part_divider`, `title`, `dedication`, `copyright` как `source-only` (без перевода и без `Previously in…`, не смещая нумерацию глав и `previous` в multi-book EPUB). Синхронно обновляются **все три** набора тестов: `test_epub_export_content.py:769`, `test_learner_book_pages.py:59,76,181` и `test_epub_writer.py:627-642`.
- **Последствия:** Юридически и типографически чистый EPUB 3.3, совместимый с multi-book изданиями и синтетическими страницами.

---

## ADR-8: Локальный скилл `linguary-subagent-run` — превентивный потолок UTF-8, единичная запись метрик и прогресс-монитор (WP8-local, вне git)

- **Контекст:** `.agents/skills/linguary-subagent-run/` не отслеживается git (`.agents/` в `.gitignore`). В реальных прогонах `run_workflow` усекает промпты по байтам UTF-8 (~120 КБ), причём на `chapter_correction` усечение середины JSON способно проходить **тихо** (модель возвращает валидный JSON, помечая выпавшие блоки как `unchanged`, и текстовый детектор отказа не сработает). Кроме того, `handle_completion` имеет внешний `except Exception`, что при неосторожном поднятии `ContextLengthExceededError` привело бы к двойной записи в `bridge_metrics.json`.
- **Решение:**
  1. **Байтовый контроль UTF-8 и превентивный потолок (`bridge_core.py`):**
     - Константы: `PROMPT_WARN_BYTES = 95_000`, `PROMPT_TRUNCATION_RISK_BYTES = 112_000`, `PROMPT_HARD_LIMIT_BYTES = 118_000` (конфигурируемо через `SubagentBridgeServer(..., hard_prompt_limit_bytes=PROMPT_HARD_LIMIT_BYTES)`, `None` отключает).
     - Если `prompt_bytes > hard_prompt_limit_bytes` — **до отправки в `worker`** немедленно поднимать `ContextLengthExceededError` → HTTP 400 `context_length_exceeded`! Это на 100% исключает тихое усечение и сразу запускает штатный сплит чанка в Linguary (`correct_chapter.py` и `analyze_chapter.py`).
  2. **Компактизация JSON Schema (`build_agent_prompt`):** удалять метаданные `"title"` и служебные `"description"` только на узлах схемы (никогда не удаляя ключи внутри словаря `"properties"`!), передавая исходную `schema_obj` в `validate_json_schema`.
  3. **Детекция отказа из-за усечения на всех путях + отсутствие двойной записи метрик:**
     - Проверять `_is_truncated_prompt_refusal(raw)` на **каждом** ответе `worker(...)` — до `if not expect_json:` и на старте каждой итерации `for attempt in range(3)` (включая ремонтные ходы).
     - В `handle_completion`: выделить отдельный `except ContextLengthExceededError:` **перед** `except Exception as exc:`, записывающий метрику ровно **один** раз со `status="context_length_exceeded"` и пробрасывающий исключение в `handle_client` (где оно сериализуется в HTTP `400` с `error.code = "context_length_exceeded"`).
  4. **Наблюдаемость (`bridge_metrics.json` + read-only `progress_monitor`):**
     - В `_record_metric` записывать `prompt_bytes`, `truncation_risk`, `repair_errors: list[str]`.
     - В `run_cmd(..., db_path: Path | None = None, progress_interval_seconds: float = 90.0)`: при переданном `db_path` запускать фоновую корутину-монитор, открывающую SQLite строго в read-only режиме (`sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)`), подавляющую `sqlite3.OperationalError` (если БД не создана или занята) и выводящую 1 строку прогресса по `processing_tasks` + счётчики `bridge.metrics` с гарантированной отменой в `finally`.
  5. **Документация (`SKILL.md`):** описать байтовый потолок UTF-8, новые поля `bridge_metrics.json`, прогресс-монитор и рекомендацию `--pronunciation-help --pronunciation-variant auto` для A2.
- **Последствия:** Полностью исключены тихое усечение и дублирование метрик; длинные прогоны прозрачны в реальном времени.
