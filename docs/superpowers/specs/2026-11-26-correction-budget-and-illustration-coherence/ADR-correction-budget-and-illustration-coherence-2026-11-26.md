# ADR: Архитектурные решения по коррекции сегментов, бюджету рассуждений и связности иллюстраций

**Дата:** 2026-11-26  
**Статус:** Accepted  
**Связанный BRIEF:** `BRIEF-correction-budget-and-illustration-coherence-2026-11-26.md`

---

## ADR-1: Единое разрешение `segments` в `_evidence_value()` и безопасный белый список диагностики `parse_payload()`

- **Контекст:** В `correct_chapter.py` на вход LLM отправляется сериализованный вид `_llm_block_result()` с 1-based массивом `segments: [{"index": M, "translated_text": ...}]`, тогда как `valid_for()` вызывается с `asdict(BlockProcessingResult)`, где хранится 0-based кортеж `segment_translations: tuple[str | None, ...]`. При этом `_evidence_value()` запрещает корень `segments`, а `parse_payload()` скрывает причину `LLMPayloadError`.
- **Решение:**
  1. Добавить `'segments'` в белый список корней `_evidence_value()` (`segments`, `segments.<N>`, `segments.<N>.translated_text`).
  2. Выделить в `correction_dto.py` чистую функцию `_resolve_segment_entries(current)`, принимающую как `segments`, так и `segment_translations` и строящую единое представление 1-based переведённых сегментов `[{'index': i, 'translated_text': text}]`.
  3. При индексации `segments.<N>` искать **строго по `entry['index'] == N`** (канонический 1-based номер из `CORRECTION_PROMPT_V2`), допуская позиционный fallback на `entries[0]` **только при `N == 0`** (где коллизия с 1-based `index >= 1` невозможна). Это исключает ложное попадание на чужой сегмент на смешанном блоке `[READ_AS_IS, TRANSLATE, TRANSLATE]`. Для `CurrentQuoteEvidence.valid_for()` дополнительно допустим позиционный fallback `entries[N]` **только если по `entry['index'] == N` цитата не нашлась**, тогда как `CurrentAbsenceEvidence` никогда не использует неоднозначный fallback.
  4. В `parse_payload()` (`adapters/llm/payloads.py`) не передавать сырой `str(error)` (так как часть ошибок `llm_dto.py` / `domain` содержит фрагменты текста книги или `source_block_id`), а ввести статический белый список `_STATIC_CONTRACT_ERROR_HINTS` (по аналогии с `_STATIC_VALUE_ERROR_HINTS`), покрывающий безопасные статические сообщения `correction_dto.py` и контрактов. Для незарегистрированных ошибок сохранять только имя типа.
- **Альтернативы:**
  - *Менять структуру `BlockProcessingResult` в `domain`* — отвергнуто: ломает слой домена и сериализацию.
  - *Передавать сырой `str(error)[:220]`* — отвергнуто: нарушает политику приватности `LLMResponseFormatError`.
- **Последствия:** Защита служебных полей (`block_id`, `treatment`, `segment_treatments`, `segments.<N>.index`) и приватность ошибок сохранены; миграции не требуются.

---

## ADR-2: Совместный учёт `reasoning_effort` и адаптивного `_implicit_reasoning_floor`

- **Контекст:** В `recommended_unit_max_completion_tokens()` `implicit_reasoning_tokens` применялся только в ветке `elif` (когда `reasoning_effort is None`). У провайдеров, где режим `"low"` тратит больше статического резерва `min_reasoning=4096`, каждый первый вызов нового юнита обрывался на потолке `low`, несмотря на накопленный `_implicit_reasoning_floor`. Кроме того, в `block_processing.py` проверка `can_escalate_recovery = can_escalate_tokens and not used_implicit_escalation` делала последующее присваивание `implicit_token_escalated = True` недостижимым.
- **Решение:**
  1. В `recommended_unit_max_completion_tokens()` при активном `effort` (`effort is not None and effort != "none"`) вычислять `total = max(effort_total, implicit_total)` при `implicit_reasoning_tokens > 0` (с сохранением финального `min(max_output_tokens, total)`).
  2. Явные режимы `reasoning_effort="none"` и `reasoning_max_tokens=<N>` не переопределяются `implicit_reasoning_tokens`.
  3. В `block_processing.py` разрешить implicit-эскалацию при `configured_max_output_tokens is None` для активного или незаданного `reasoning_effort` (не `"none"` и при `configured_reasoning_max_tokens is None`), и разрешить `can_escalate_recovery` при `used_implicit_escalation` только когда `observed_reasoning > 0` (сохраняя немедленный сплит / `output_truncated` при обычном усечении без `reasoning_tokens` и однократность через флаг `implicit_token_escalated`).
- **Последствия:** Первый вызов последующих юнитов сразу учитывает реально наблюдаемый расход рассуждений текущей модели; существующие инварианты для `"none"` и `reasoning_max_tokens` не меняются.

---

## ADR-3: Самодокументируемые двунаправленные ошибки `segments` ↔ `translated_text` без изменения замороженных шаблонов

- **Контекст:** При одновременном заполнении `translated_text` и `segments` схема `BlockResultSchema` сообщала лишь «полный перевод блока недопустим», толкая модель удалить `translated_text` на несегментированном блоке и упасть в `treatment.py`. При этом `_TRANSLATION_INSTRUCTIONS`, `_SEGMENT_TRANSLATION_RULE` и `legacy-v1`/`b2` заморожены по SHA-256.
- **Решение:**
  1. Не запрещать `len(segments) == 1` на уровне Pydantic (смешанный блок `[READ_AS_IS, TRANSLATE]` законно имеет 1 переводимый сегмент).
  2. Сделать ошибку в `BlockResultSchema._check_treatment_composition` и `_STATIC_VALUE_ERROR_HINTS` двунаправленной: явно разделить случай без меток `[bN.M]` (`translated_text` + `segments: []`) и случай с метками `[bN.M]` (`translated_text: null` + `segments`).
  3. Уточнить сообщение `_block_content_violation()` в `validators/treatment.py` (перенести перевод в `translated_text` и вернуть `segments: []`).
  4. Добавить правило про несегментированные блоки в незамороженные шаблоны `_TRANSLATION_CONTEXT_V2` и `_CONTENT_PLAN_PROCESSING_V3`, не трогая замороженные шаблоны.
- **Последствия:** Замороженные SHA-256 (`test_baseline_b2_templates_sha256_unchanged`) не меняются; цикл пинг-понга разрывается за один ход.

---

## ADR-4: Неспойлерный выбор `canonical_name` (3 уровня) и защита `temporal_scope == "state_change"` в `DESIGN`/`FACT_SCOPE`

- **Контекст:** Выбор `members[0]["name"]` закреплял за персонажем ранний родовой дескриптор (`tabby cat`), что проникало в `DESIGN` и приводило к изображению человека как животного после трансформации. В `resolve_appearance_with_design_fallback()` защита не срабатывала из-за порога длины токенов `>= 5` и неполного списка маркеров `build`.
- **Решение:**
  1. Ввести детерминированный нейтральный к книге выбор `_choose_canonical_name(members)` с 3-уровневым рангом (`tier`):
     - `tier = 2` (сильное имя собственное): все значимые слова (кроме ведущего артикля `the` и коротких служебных предлогов/союзов) с заглавной (`"Professor McGonagall"`, `"McGonagall"`, `"Hogwarts"`, `"The Leaky Cauldron"`), **или** есть слово с заглавной после первого слова (`"number four, Privet Drive"`).
     - `tier = 1` (слабое/смешанное): начинается с заглавной не из `_LEADING_GENERIC_WORDS` (`"Tabby cat"`, `"Тисовая улица"`).
     - `tier = 0` (родовое): строчные и фразы с ведущим служебным словом без заглавных далее (`"tabby cat"`, `"the boy"`, `"a dark cupboard"`).
  2. Сохранять в JSON-полезной нагрузке `ENTITY` карту `alias_first_positions: dict[str, str]` и список `section_orders: list[int]`.
  3. В `illustration_chapter.py` выбирать отображаемое имя `_display_name_at_position(entity, event_position_key)` из вариантов, уже встретившихся к моменту `event_position_key` (fallback на `canonical_name` для старых записей), и применять его **во всех трёх точках**: `FACT_SCOPE` `entity_name`, `IllustrationAppearance.name_ru` и `IllustrationAppearance.name_en` (с сохранением проверки `not has_cyrillic`).
  4. Передавать `aliases` в `_build_bounded_design_input()`; в `_validate_design()` снимать эхо-префикс `^<trait>\s*:\s*` **до** проверок `has_cyrillic()` и `find_monochrome_color_word()`; в `resolve_appearance_with_design_fallback()` проверять наличие факта с **`temporal_scope == "state_change"`** среди **обоих** списков `(*explicit_facts, *excluded_facts)` и при его наличии подавлять анатомические/видовые признаки `DESIGN` (`build`, `distinctive_mark`, `hair`, `eyes`, `age_appearance`), а также расширить `_TRAIT_PATTERNS["build"]` и снизить порог `_echo_content_tokens` до 3 символов со стоп-листом.
- **Последствия:** Нет спойлеров имён ни в `FACT_SCOPE`, ни в `PROMPT`, ни в UI до их появления в тексте; нет утечки старой формы после `state_change`; нет миграций БД.

---

## ADR-5: Структурное разделение присутствующих и отсутствующих сущностей (`depicted_entity_ids`)

- **Контекст:** В Round 3 отсутствующий персонаж (`Harry` на сцене с фотографиями Дадли) попадал в `participant_names` на этапе `CANDIDATES`, а затем `compose_monochrome_prompt()` внутри `_validate_render_response()` принудительно дописывал его в `Visual details:`.
- **Решение (двухуровневая защита):**
  1. **Уровень `CANDIDATES`:** Явно определить `participant_names` как **только физически изображаемых в кадре** персонажей (исключая отсутствующих, упомянутых через отрицание, находящихся за кадром или в мыслях).
  2. **Уровень `PROMPT` (структурная гарантия):** Добавить обязательное поле `depicted_entity_ids: list[str]` в `IllustrationPromptSchema` и `depicted_entity_ids: tuple[str, ...] | None = None` в `IllustrationPromptResponse` (`None` при чтении старых `payload_json` -> полная обратная совместимость).
     - В `_validate_render_response()`: проверить, что `depicted_entity_ids` — подмножество `allowed_entity_ids` без дублей и все владельцы `used_fact_ids` и `used_design_keys` входят в `depicted_set`; **отфильтровать `base_characters` и `base_locations` по `depicted_set` до `_materialize_appearances_for_prompt()` и `compose_monochrome_prompt()`**; сохранить `"depicted_entity_ids": list(depicted)` в `payload` скоупа `PROMPT`.
     - В `_advance_chapter_rendering()`: читать `p_pay.get("depicted_entity_ids")` и при `not None` тем же `depicted_set` фильтровать `char_apps`/`loc_apps` (и legacy `summary["characters"/"locations"]`) перед сборкой `IllustrationMoment`.
- **Последствия:** `prompt_en` и `IllustrationMoment` строго согласованы; `validate_moment()` остаётся строгим, но отсутствующие герои больше не попадают в `Visual details:`.

---

## ADR-6: Совместное исправление языкового разрыва локаций и защиты от ложного `ambiguous`

- **Контекст:** `CANDIDATES` получал русские названия сцен из `NARRATIVE` и возвращал `location_names` на русском, а реестр `ENTITY` хранит английские имена -> `locations: []` во всех 5 моментах. При этом в `ALIAS` не было правил для локаций (часть ↔ целое), из-за чего одна улица распалась на 2 сущности. Если исправить только язык `CANDIDATES`, `_resolve_entity_name()` начнёт возвращать `"ambiguous"` и отбрасывать моменты.
- **Решение (строго парное):**
  1. В `_STAGE_INSTRUCTIONS[EXTRACT]` и `_STAGE_INSTRUCTIONS[ALIAS]` добавить явные правила для `entity_kind="location"`: точное различающее имя места (улица ≠ дом ≠ комната), склейка повторных упоминаний одного места и разделение отношения «контейнер ↔ часть».
  2. В `plan_scopes_in_tx()` (`illustration_selection.py`, где реестр уже запечатан и `registry_sha256` входит в fingerprint главы) читать `ENTITY` через `tx.illustrations.query_registry_items(...)`, фильтровать по `first_position_key <= chapter_end` (без спойлеров будущих глав!), приоритизировать `entry.order in section_orders` и передавать `known_characters` (`entity_kind == "person"`) и `known_locations` (`entity_kind == "location"`) внутри `_build_req` в `pack_items_by_wire_budget()`, сохраняя их также в `_split_children_for_scope()`.
  3. В `_resolve_entity_name()` (`illustration_chapter.py`) перейти к двухступенчатому детерминированному сопоставлению с возвратом `"resolved"` / `"ambiguous"` / `"unmatched"` через `normalize_name_token()`: сначала `canonical_name`, и только при 0 совпадений — `aliases`.
  4. В `_build_prompt_scope_spec()`: если `locations` всё же остался пустым (безымянная локация вне реестра), находить сцену в `entry.scenes` по `s.id == card.scene_id` или `f"{section_id}:scene:{idx}"`, передавать справочное `scene_location_ru` (без создания фиктивного `IllustrationAppearance`) и сбалансировать инструкцию `PROMPT` про интерьер/экстерьер.
- **Последствия:** Сохраняется `test_ambiguous_identity_remains_separate`, нет спойлеров будущих глав в `CANDIDATES`, нет потери `known_*` при сплите `CANDIDATES`, `locations` надёжно наполняются для именованных локаций.
