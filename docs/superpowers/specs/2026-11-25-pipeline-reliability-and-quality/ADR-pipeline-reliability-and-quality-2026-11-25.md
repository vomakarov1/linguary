# ADR: Архитектурные решения по надёжности конвейера иллюстраций, бюджету рассуждения и качеству вывода

> **Статус:** Accepted (проверка спецификаций пройдена)

- Дата: 2026-11-25
- Тема: совместимость рантайма иллюстраций, конечный автомат `CHAPTER`, устойчивость `CANDIDATES`, бюджет рассуждения перевода, диагностика и редакционные контракты
- BRIEF: [BRIEF-pipeline-reliability-and-quality-2026-11-25.md](BRIEF-pipeline-reliability-and-quality-2026-11-25.md)
- SDD: [SDD-backend-pipeline-reliability-and-quality-2026-11-25.md](SDD-backend-pipeline-reliability-and-quality-2026-11-25.md)

## 0. Коротко

- **Зачем.** Шесть подтверждённых дефектов ломают прохождение конвейера на реальных моделях OpenRouter и портят редакционное качество иллюстраций и анализа.
- **Решение.** Все исправления выполняются строго внутри существующих слоёв Clean Architecture (`domain <- application <- adapters / composition`) без миграций БД и без нарушения обратной совместимости сериализованных JSON-полезных нагрузок.
- **Чем платим.** Часть старых кэшированных `DESIGN`- и `CANDIDATES`-областей из прошлых поколений получит новый `input_sha256` (из-за включения `style` в `DESIGN` и упрощения `source_unit_id` в `CANDIDATES`), что намеренно предотвращает переиспользование дефектных результатов прежних прогонов.
- **Что не трогаем.** Таблицы SQLite, миграции Alembic, публичные HTTP/CLI-контракты и протоколы очередей задач остаются прежними.

## 1. Архитектурные границы

Соблюдается строгое правило зависимостей (`tests/architecture/test_layer_dependencies.py`):
- `domain` (`domain/illustrations.py`) — только чистый Python: ослабление типа `IllustrationPolicy.temperature: float | None = None`, повышение значения по умолчанию `max_completion_tokens: int = 16384`, чистые функции `executable_scopes()` и `derive_chapter_state()`.
- `application` (`illustrations.py`, `illustration_coordinator.py`, `illustration_view.py`, `illustration_context.py`, `illustration_reference.py`, `illustration_selection.py`, `illustration_chapter.py`, `illustration_validation.py`, `token_budget.py`, `block_processing.py`, `context_builder.py`) — без сторонних библиотек и прямого I/O.
- `adapters` (`adapters/llm/illustration_payloads.py`, `adapters/db/illustration_repository.py`) — лимит схемы `covered_source_unit_ids` и проекция заголовков глав.
- `composition` (`composition/illustration_runtime.py`) — построение `LLMSettings` и ключ пула HTTP-клиентов.

## 2. Ключевые решения и отброшенные альтернативы

### Решение 1: Параметры генерации в `IllustrationRuntime._clean_llm_settings()`

**Выбрано:**
1. `IllustrationPolicy.temperature` становится `float | None = None` (значение по умолчанию — `None`; в `from_payload()` сохраняется обратная совместимость с уже сохранёнными записями, где `"temperature": 0.2` или `None`). Примечание по старым ревизиям в SQLite: если у существующей книги уже сохранена старая ревизия с `"temperature": 0.2`, её нужно однократно пересохранить (новая ревизия получит `temperature: None` и отзовёт старое поколение); при этом явная `policy.temperature` в тестах и политиках уважается без скрытых подмен.
2. `IllustrationPolicy.max_completion_tokens` по умолчанию повышается с `4096` до `16384`. В `_clean_llm_settings()` используется `base_llm.max_output_tokens` (если задан пользователем в конфиге — уважает как высокий лимит `32768`, так и ограничение меньших моделей), иначе `policy.max_completion_tokens`.
3. `reasoning_effort` и `reasoning_max_tokens` из `base_llm` наследуются **только когда `model == base_llm.model`** (причём `reasoning_max_tokens` — дополнительно только при `_is_openrouter_url(base_llm.base_url)` и `reasoning_max_tokens < effective_max_tokens`). Для любой другой модели (`model != base_llm.model`) они остаются `None`, чтобы не вызывать HTTP 404 на OpenRouter при `require_parameters=true` у не-рассуждающих моделей.
4. Ключ пула `IllustrationRuntime._pool` разделяется на `(eviction_fp, model, policy_gen_fp)`, где `eviction_fp` включает транспортные и глобальные параметры `base_llm` (`max_output_tokens`, `reasoning_effort`, `reasoning_max_tokens`) — смена `/system/config` корректно вытесняет и закрывает устаревшие клиенты, а разные ревизии политик разных книг мирно сосуществуют в пуле.
5. Постоянные ошибки несовместимости параметров/маршрута (HTTP 404 или HTTP 400/422 с признаками неподдерживаемого параметра / отсутствующего эндпоинта, не относящиеся к `context_limit`) классифицируются как `IllustrationReasonCode.ROUTE_UNAVAILABLE` без повторов внутри активации.

**Отброшенные альтернативы:**
- *Безусловно наследовать `base_llm.reasoning_effort` для всех моделей иллюстраций.* Отброшено: если в `config.toml` основная модель — DeepSeek с `reasoning_effort="low"`, а для иллюстраций выбрана `openai/gpt-4.1-mini` (без поддержки `reasoning`), OpenRouter с `require_parameters=true` вернёт HTTP 404.
- *Убирать `require_parameters=true` для OpenRouter.* Отброшено: `require_parameters=true` защищает от тихого игнорирования `response_format` (strict JSON schema) на сторонних бэкендах OpenRouter.
- *Добавлять `ROUTE_UNAVAILABLE` в `DETERMINISTIC_NON_RETRYABLE_REASONS`.* Отброшено: `ROUTE_UNAVAILABLE` используется и при временно ненастроенном ключе/маршруте, после исправления которого пользователь вправе нажать `retry`. Смена модели в настройках и так создаёт новую ревизию и отзывает старое поколение.

### Решение 2: Исключение виртуального агрегата `CHAPTER` из расчёта активной работы

**Выбрано:**
- В `finalize_generation_state_in_tx()`, `IllustrationCoordinator.recover_interrupted()` и `build_illustration_view()` виртуальные области `stage == IllustrationScopeStage.CHAPTER` исключаются из проверок «есть ли активные/ожидающие исполнимые области» (`has_active`, `has_pending`, `has_running`).
- Область `CHAPTER` в БД **не переводится в `FAILED`** при падении дочерних областей — она остаётся `PENDING` до успешной сборки всех моментов главы. В `requeue_failed_scopes()` добавляется защитный фильтр `stage != CHAPTER`, гарантирующий, что виртуальному `CHAPTER` никогда не назначат `ProcessingTask`.
- Состояние заголовка главы (`get_chapter_header_page()`) и выбранной главы (`build_illustration_view()`) при `CHAPTER.state == PENDING` вычисляется детерминированно по её исполнимым дочерним областям (`stage != CHAPTER`): `running` > `queued` (если есть `PENDING` дети) > `failed` (если все активные дети завершены и есть `FAILED`, с `reason_code` первого упавшего ребёнка) > `queued` (если дети ещё не запланированы).

**Отброшенные альтернативы:**
- *Помечать сам `CHAPTER` как `FAILED` при падении ребёнка.* Отброшено: `requeue_failed_scopes()` + `IllustrationService.retry()` перебирают все `FAILED`-области и вызывают `dispatch_scope_task()` для каждой — это создало бы бессмысленную задачу `PREPARE_CHAPTER_ILLUSTRATIONS` для `CHAPTER`, которую `IllustrationChapter.run_task()` молча игнорирует.

### Решение 3: Упрощение `source_unit_id` и устойчивость `CANDIDATES`

**Выбрано:**
- В `IllustrationCandidatesSchema` лимит `covered_source_unit_ids` повышается с `_MAX_LIST` (32) до `_MAX_COVERED_IDS` (256), синхронно с `IllustrationExtractSchema`.
- В `pack_items_by_wire_budget()` добавляется необязательный параметр `max_items: int | None = None`. Этап `CANDIDATES` вызывает его с `max_items=24`, этап `EXTRACT` — с `max_items=64` (заведомо меньше 256 и комфортно для внимания LLM).
- В `_scene_candidate_units()` `source_unit_id` равен `b.id` (36 символов вместо 103-символьного `{scene_sha256}:b:{block_uuid}`), что совпадает с `block_id` того же блока и исключает путаницу у модели.
- В `_validate_candidates()` дополнительно поддерживается 1-к-1 нормализация голых `block_id` для обратной совместимости со старыми/унаследованными scope; при расхождении в `repair_reason` передаются конкретные `пропущены=...` и `лишние=...` (до 4 шт.) и явное напоминание перечислять **все** переданные блоки чанка.
- В `IllustrationSelection.run_task()` при исчерпании раундов ремонта (`repairs_used >= max_repair_turns`) на **составной** (`is_composite=True`) области `CANDIDATES` из-за ошибки формата/покрытия (`LLMResponseFormatError` или провал покрытия `covered_source_unit_ids`) выполняется `_handle_post_send_split()` вместо немедленного провала области.

### Решение 4: Адаптивный бюджет рассуждения и эскалация при `provider_truncated` в переводе

**Выбрано:**
- В `token_budget.py` вводится таблица минимального резерва рассуждения `REASONING_EFFORT_MIN_TOKENS` (`none: 0`, `minimal: 1024`, `low: 4096`, `medium: 8192`, `high: 12288`). Для активного `reasoning_effort` итоговый потолок равен `min(max_output_tokens, max(pct_total, visible + min_reasoning))`.
- В `block_processing.py` при получении `provider_truncated`, если текущий `request.max_completion_tokens` строго меньше настроенного в `loaded_conditions` значения `max_output_tokens` и бюджет вызовов текущей области (`max_scope_calls` / `max_calls` в recovery или `max_attempts` вне recovery) ещё позволяет попытку, лимит немедленно эскалируется до `max_output_tokens` в рамках той же области — **до** дробления и **до** остановки корня с `output_truncated`. Если ответ усечён даже на полном `max_output_tokens`, срабатывает прежняя логика дробления / остановки.

### Решение 5: Финализация диагностики иллюстраций

**Выбрано:**
- Переиспользуем существующие хелперы `record_rejection()` и `finalize_response()` из `application/llm_rejections.py` во всех трёх модулях иллюстраций (`illustration_reference.py`, `illustration_selection.py`, `illustration_chapter.py`):
  - на принятии области — `finalize_response(uow, response, "accepted", transaction=tx)`;
  - на устаревшем ответе (`stale_reason` / `IllustrationOwnershipError`) — `finalize_response(uow, response, "stale", transaction=tx)`;
  - на семантическом отклонении (`not validation.ok`) и на `validation.request_split` (`complete=False`) — `record_rejection(uow, response, ResultValidationError(detail))`.
- При записи `illustration_attempts.outcome_json` на семантическом отказе (и через уже существующий параметр `outcome` метода `fail_scope(...)`, без изменения порта и репозитория) добавляется компактное безопасное поле `"detail"` (чистая обрезка до 256 байт UTF-8 без префикса «Исправь ответ:», без сырого вывода модели и без секретов).

### Решение 6: Редакционные контракты `DESIGN`, `ALIAS`, `PROMPT`, `compose_monochrome_prompt` и промпты анализа/перевода

**Выбрано:**
- **Антиспойлер и `monochrome_ink` в `DESIGN`:** в `request_input` этапа `DESIGN` добавляется поле `"style": revision.settings.style` (что также инвалидирует кэш старых дефектных `DESIGN`-областей). В `_STAGE_INSTRUCTIONS[DESIGN]` добавляются жёсткие правила: запрет внешних знаний о книге/серии/экранизациях и будущих главах; для `monochrome_ink` черта `palette_value` описывается только тональностью/контрастом/штриховкой без цветовых оттенков. В `_validate_design()` добавляется детерминированная проверка: при `style == "monochrome_ink"` черта `palette_value` не должна содержать слов-оттенков (red, green, blue, yellow, orange, purple, violet, pink, beige, brown, crimson, emerald, scarlet, turquoise, teal, amber, gold/golden, magenta и т. п.) — нарушение запускает repair-раунд.
- **Приоритет явных фактов над `DESIGN`:** расширяется `_TRAIT_KEYWORDS` в `illustration_reference.py` (добавляются маркеры бороды, тёмных/чёрных/седых волос, младенческого/детского возраста, очков, шрамов и др. на en/ru).
- **Англоязычные имена в `prompt_en`:** в `IllustrationAppearance` добавляется необязательное поле `name_en: str | None = None` (обратно совместимо с существующими сохранёнными JSON). При построении `IllustrationAppearance` выбирается латинское имя из `entity["canonical_name"]` / `entity["aliases"]`; если все имена сущности кириллические, `name_en=None`. В `compose_monochrome_prompt()` используется `name_en` (или `name_ru`, если оно не содержит кириллицы), иначе префикс имени опускается — в `prompt_en` не утекает кириллица.
- **Усиление `ALIAS`:** в `left`/`right` пар `ALIAS` передаются `alias_of_hint` (если есть) и `glossary_concept_id` (если совпало с глоссарием книги), а инструкция `_STAGE_INSTRUCTIONS[ALIAS]` поясняет, как использовать эти сигналы без слепого слияния однофамильцев.
- **Точность композиции и дедупликация в `PROMPT`:** в `_STAGE_INSTRUCTIONS[PROMPT]` добавляются требования строгой верности позе/действию и физически возможной точке зрения камеры из `event_ru`; в `illustration_chapter.py` выполняется нормализованная дедупликация `assumptions_ru` и устранение дублей при склейке `FACT_SCOPE` + `DESIGN`.
- **Промпты анализа и перевода (`context_builder.py`):** точечные уточнения в `_NARRATIVE_ANALYSIS_INSTRUCTIONS` (ориентир гранулярности сцен 5–25 блоков и запрет сваливать 40+ блоков разных событий в одну сцену; личные имена всегда `person`, не `term`; точность грамматических помет) и в `_TRANSLATION_CONTEXT_V2` / `_CONTENT_PLAN_PROCESSING_V3` (сохранение разговорного/диалектного регистра персонажей, единообразная форма `glossary_hints.original`, отнесение `help_kind=explanation` в `language_notes`).

## 3. Влияние на совместимость и производительность

- **Схема БД:** изменений нет, миграции Alembic не требуются.
- **Сохранённые ревизии настроек:** `IllustrationPolicy.from_payload()` по-прежнему читает старые записи с `"temperature": 0.2` и `"max_completion_tokens": 4096`.
- **Сохранённые `ChapterIllustrationPlan` / `IllustrationAppearance`:** `from_dict()` возвращает `name_en=None`, если ключа нет в старом JSON.
- **Стоимость и скорость:** сокращается число бесполезных ретраев, устраняется лавина дроблений DeepSeek и 404-ошибки OpenRouter, сокращается длина `CANDIDATES`-запросов за счёт коротких `source_unit_id`.
