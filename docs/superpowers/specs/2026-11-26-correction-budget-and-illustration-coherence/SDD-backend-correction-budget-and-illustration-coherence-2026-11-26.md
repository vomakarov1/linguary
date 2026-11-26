# SDD (Backend): Контракты, алгоритмы и тестовая матрица

**Дата:** 2026-11-26  
**Статус:** Approved (актуализировано по итогам глубокого ревью)  
**Связанные документы:** `BRIEF-correction-budget-and-illustration-coherence-2026-11-26.md`, `ADR-correction-budget-and-illustration-coherence-2026-11-26.md`

---

## Глобальные ограничения

- **Границы слоёв:** `domain ← application ← adapters / web / cli` (проверяется `tests/architecture/test_layer_dependencies.py`). В `domain` и `application` нет сторонних библиотек и I/O.
- **Схема БД:** 0 миграций Alembic. Новые поля хранятся в JSON (`payload_json`) и имеют обратную совместимость при чтении старых записей.
- **Замороженные шаблоны:** `_TRANSLATION_INSTRUCTIONS`, `_SEGMENT_TRANSLATION_RULE`, `snapshot_prompt_templates(..., reader_level="b2", profile="legacy-v1")` не изменяются.
- **Универсальность:** запрещены любые привязки к названиям книг, персонажей, локаций или конкретных моделей LLM.

---

## WP1 — Коррекция сегментированных блоков и диагностика `LLMPayloadError`

### 1.1 `src/linguary/application/correction_dto.py`

Добавить чистый хелпер `_resolve_segment_entries(current: Mapping[str, Any]) -> list[dict[str, Any]] | None`:
- Если `'segments'` присутствует в `current` и является `(list, tuple)` — вернуть `[dict(item) for item in current['segments'] if isinstance(item, Mapping)]`.
- Иначе если `'segment_translations'` присутствует в `current` и является `(list, tuple)` — построить `[{'index': idx, 'translated_text': text} for idx, text in enumerate(current['segment_translations'], start=1) if isinstance(text, str) and text.strip()]`.
- Иначе вернуть `None`.

Обновить `_evidence_value(current, path)`:
1. Белый список корней: `{'translated_text', 'segments', 'glossary_hints', 'language_notes'}`.
2. Разрешённые подполя при `len(parts) == 3`:
   - `glossary_hints`: `{'original', 'translation', 'note'}`
   - `language_notes`: `{'original_excerpt', 'translated_excerpt', 'note'}`
   - `segments`: `{'translated_text'}`
3. Проверка синтаксиса пути остаётся прежней (`r'[a-z_]+(?:\.[a-z_0-9]+|\[\d+\])*'`, `len(parts) <= 3`, `parts[1].isdigit()` при `len(parts) > 1`). Таким образом `segments.1.index`, `segments.foo.translated_text`, `block_id`, `treatment`, `segment_treatments` по-прежнему отклоняются `LLMPayloadError`.
4. При `parts[0] == 'segments'`:
   - `entries = _resolve_segment_entries(current)`
   - Если `entries is None`: вернуть `None`.
   - Если `len(parts) == 1`: вернуть `entries`.
   - Если `len(parts) >= 2`: `req_idx = int(parts[1])`.
     - Строгий поиск элемента `entry` с `entry.get('index') == req_idx` (канонический 1-based номер сегмента).
     - Если не найден и `req_idx == 0` и `entries`: вернуть `entries[0]` (коллизия с 1-based `index >= 1` невозможна).
     - При `req_idx >= 1` позиционного fallback в `_evidence_value()` **нет** (возвращается `None`, что исключает ложное попадание на чужой сегмент на смешанном блоке `[READ_AS_IS, TRANSLATE, TRANSLATE]`).
     - При `len(parts) == 2` вернуть `entry`, при `len(parts) == 3` вернуть `entry.get('translated_text')`.
5. В `CurrentQuoteEvidence.valid_for(current)`:
   - В `texts(item)` добавить `'index'` в игнорируемые ключи.
   - Если проверка по `_evidence_value(current, self.field_path)` не нашла цитату, а `parts[0] == 'segments'` и `len(parts) >= 2` и `req_idx >= 1`: выполнить безопасный толерантный fallback — проверить позиционный элемент `entries[req_idx]` (если `req_idx < len(entries)`) на реальное совпадение `find_quote_span(..., self.quote)`. В `CurrentAbsenceEvidence.valid_for()` такого fallback нет.

### 1.2 `src/linguary/composition/correction_runtime.py`

В `CORRECTION_PROMPT_V2` уточнить правило `field_path`:
- Для несегментированного блока — `translated_text`, `glossary_hints.<индекс>.<поле>`, `language_notes.<индекс>.<поле>`.
- Для сегментированного блока (где `translated_text` равен `null` и передан массив `segments`) — `segments.<index>.translated_text`, где `<index>` — номер `index` из переданного элемента `segments` (например, `segments.2.translated_text`), а также `glossary_hints.<индекс>.<поле>` и `language_notes.<индекс>.<поле>`.

### 1.3 `src/linguary/adapters/llm/payloads.py`

В `parse_payload()` (обработчик `except (LLMPayloadError, DomainError, ValueError) as error:`):
- Ввести статический белый список `_STATIC_CONTRACT_ERROR_HINTS: Mapping[str, str]` (по аналогии с `_STATIC_VALUE_ERROR_HINTS`), содержащий безопасные статические сообщения без интерполяции текста книги или `source_block_id` (все статические ошибки `correction_dto.py` — путь доказательства, защищённое поле, пустая цитата, свидетельство отсутствия, `expected_field`, `issues`, `segments` и пр., а также безопасные статические ошибки `llm_dto.py`).
- Если `str(error).strip()` присутствует в `_STATIC_CONTRACT_ERROR_HINTS` (или его нормализованный ключ), сформировать:
  `message = f"ответ модели ({kind.value}) не соответствует внутреннему контракту ({type(error).__name__}): {hint}"`
  иначе сохранить прежний безопасный формат:
  `message = f"ответ модели ({kind.value}) не соответствует внутреннему контракту: {type(error).__name__}"`.
- В `Violation(..., safe_message='Ответ не соответствует внутреннему контракту')` поле `safe_message` оставить неизменным.

---

## WP2 — Адаптивный бюджет рассуждений, контракт `segments`/`translated_text` и паритет многоточия

### 2.1 `src/linguary/application/token_budget.py`

В `recommended_unit_max_completion_tokens(...)`:
```python
implicit_total = (
    visible + max(IMPLICIT_REASONING_MIN_TOKENS, implicit_reasoning_tokens * 2)
    if implicit_reasoning_tokens > 0
    else visible
)
if effort is not None:
    reserve = REASONING_EFFORT_RESERVE_PCT.get(effort, 0)
    min_reasoning = REASONING_EFFORT_MIN_TOKENS.get(effort, 0)
    pct_total = (visible * 100 + (99 - reserve)) // (100 - reserve) if reserve > 0 else visible
    total = max(pct_total, visible + min_reasoning)
    if effort != "none" and implicit_reasoning_tokens > 0:
        total = max(total, implicit_total)
elif reasoning_max_tokens is not None:
    total = visible + reasoning_max_tokens
elif implicit_reasoning_tokens > 0:
    total = implicit_total
else:
    total = visible
```
Финальный clamp по `max_output_tokens` не меняется.

### 2.2 `src/linguary/application/block_processing.py`

1. При обработке `is_truncated` (L515–542):
   - Определить `effort_allows_implicit = configured_reasoning_max_tokens is None and (configured_reasoning_effort is None or configured_reasoning_effort.strip().lower() != "none")`.
   - При `observed_reasoning > 0`: сохраняется `self._raise_implicit_reasoning_floor(observed_reasoning)`.
   - При `observed_reasoning == 0`: fallback-повышение floor по `request.max_completion_tokens or IMPLICIT_REASONING_MIN_TOKENS` сохраняется только при `configured_reasoning_effort is None and configured_reasoning_max_tokens is None`.
   - В ветке `elif not implicit_token_escalated and unit is not None and effort_allows_implicit:` (когда `configured_max_output_tokens is None`): вызывать `recommended_unit_max_completion_tokens(..., reasoning_effort=configured_reasoning_effort, reasoning_max_tokens=None, implicit_reasoning_tokens=self._get_implicit_reasoning_floor())` и ставить `used_implicit_escalation = True`.
2. На L592 заменить `can_escalate_recovery = can_escalate_tokens and not used_implicit_escalation` на `can_escalate_recovery = can_escalate_tokens and (not used_implicit_escalation or observed_reasoning > 0)`. Флаг `implicit_token_escalated = True` внутри `if within_budget:` (L638–639) становится достижимым при реально наблюдаемых `reasoning_tokens > 0` и ограничивает implicit-эскалацию ровно одной попыткой на recovery-цикл, не нарушая немедленный сплит / `output_truncated` при обычном усечении без `reasoning_tokens`.

### 2.3 `src/linguary/adapters/llm/schemas.py`, `payloads.py`, `validators/treatment.py`, `context_builder.py`

1. `schemas.py` (`BlockResultSchema._check_treatment_composition`):
   Если `self.segments` непуст и `self.translated_text is not None`:
   поднять `ValueError("нельзя одновременно заполнять translated_text и segments: для блока без меток сегментов верни перевод в translated_text и segments: []; для блока с метками сегментов верни translated_text: null и заполни segments")`.
2. `payloads.py` (`_STATIC_VALUE_ERROR_HINTS`):
   Обновить запись под новое двунаправленное сообщение (и обновить соответствующее ожидание в `tests/unit/llm/test_llm_payload_mapping.py:966`).
3. `validators/treatment.py` (`_block_content_violation`, L255–260):
   Сообщение: `"для блока без нескольких сегментов список segments недопустим: перенеси полный перевод в translated_text и верни segments: []"`.
4. `context_builder.py`:
   В `_TRANSLATION_CONTEXT_V2` (L162) и `_CONTENT_PLAN_PROCESSING_V3` (L314) добавить предложение:
   `"Для обычного блока без меток сегментов [bN.M] всегда возвращай перевод в translated_text и оставляй segments: []."`
   (Не трогать `_TRANSLATION_INSTRUCTIONS`, `_SEGMENT_TRANSLATION_RULE` и шаблоны `legacy-v1`/`b2`!)

### 2.4 `src/linguary/application/analysis_merge.py`

В `_QUOTE_LENIENT_TOKEN_RE` (L1848) и `_lenient_quote_pattern()` (L1859):
- Добавить группу `(?P<ellipsis>…|\.{3})` перед `(?P<ch>.)`.
- В цикле сборки паттерна при `kind == 'ellipsis'` добавлять `r'(?:…|\.{3})'`.

---

## WP3 — Надёжность протокола и ремонта этапов иллюстраций

### 3.1 `src/linguary/application/illustration_context.py`

1. `_STAGE_INSTRUCTIONS[IllustrationScopeStage.SELECTION]`:
   Явно указать:
   - для каждого `candidate_id` из `selected_candidate_ids` обязательно вернуть объект `{candidate_id, reason_ru}` в `selection_reasons_ru` с кратким обоснованием выбора на русском;
   - если `selected_candidate_ids` пуст, заполнить `zero_reason_ru` и оставить `selection_reasons_ru: []`.
2. `compact_repair_summary(reason: str, details: Sequence[str] = (), *, max_bytes: int = MAX_REPAIR_SUMMARY_BYTES, stage: IllustrationScopeStage | None = None) -> str`:
   - Сохранить существующую сигнатуру и **идемпотентность** (если `reason` уже начинается с ремонтного префикса, не оборачивать повторно — сохраняет `test_illustration_reference.py:486-500`).
   - При первичной сборке использовать полную преамбулу:
     - по умолчанию: `"Исправь ошибку и верни ПОЛНЫЙ JSON-ответ целиком для всех элементов запроса (не частичный фрагмент): "`
     - для `stage in {IllustrationScopeStage.EXTRACT, IllustrationScopeStage.CANDIDATES}`: `"Исправь ошибку и верни ПОЛНЫЙ JSON-ответ для ВСЕХ переданных source_unit_id в covered_source_unit_ids (не частичный фрагмент): "`
   - Аккуратно обрезать остаток так, чтобы итог строго укладывался в `max_bytes` (UTF-8).
   - В `IllustrationContext.build_for_kind()` (где `stage` уже передаётся параметром `L417`) передавать `stage=stage` в `compact_repair_summary(...)`.

### 3.2 `src/linguary/adapters/llm/illustration_payloads.py`

В `IllustrationSelectionSchema`:
- Добавить информативные `Field(description=...)` для `selected_candidate_ids`, `selection_reasons_ru` («Обязательно: по одной записи `{candidate_id, reason_ru}` для КАЖДОГО элемента из `selected_candidate_ids`»), `rejected_reasons_ru`, `zero_reason_ru` (попадает в `strict_json_schema`).
- `@model_validator` на схеме **не добавляем**, сохраняя точную доменную диагностику `_validate_selection()` с перечислением конкретных пропущенных `candidate_id`.

### 3.3 `src/linguary/domain/illustrations.py`

- В `IllustrationPolicy` и `IllustrationPolicy.from_payload()` (L311): изменить значение по умолчанию `max_repair_turns` с `1` на `2` (в пределах `max_calls_per_scope_activation = 3`).
- В существующем тесте `test_hallucinated_and_duplicate_candidate_ids_rejected` (`tests/unit/application/test_illustration_selection.py:802-822`) явно фиксировать `max_repair_turns=1` (или проверять `selection_attempts == 3` для дефолта), добавив отдельный тест на успех на 2-м ремонте при дефолтной политике.

---

## WP4 — Сущности, смена облика, `depicted_entity_ids` и привязка локаций

### 4.1 Выбор `canonical_name`, `alias_first_positions` и `section_orders` (`illustration_reference.py`, `illustration_chapter.py`)

1. **`illustration_reference.py`:**
   - Ввести чистую функцию `_choose_canonical_name(members: Sequence[Mapping[str, Any]]) -> str`:
     - Группировать варианты имени по `normalize_name_token(name)`, сохраняя: самое частое оригинальное написание, `count`, `min_position_key`.
     - Ввести 3-уровневую оценку имени (универсальная, без привязки к книге):
       - `_LEADING_GENERIC_WORDS`: `{"the", "a", "an", "this", "that", "these", "those", "his", "her", "their", "my", "our", "your", "some", "one", "old", "young", "little", "small", "big", "large", "tall", "short", "thin", "fat", "tiny", "giant", "strange", "dark"}`.
       - `_INNER_CONNECTOR_WORDS`: `{"of", "in", "on", "at", "by", "to", "for", "and", "or", "the", "a", "an", "de", "van", "von", "la", "le", "du"}`.
       - Разбить на слова (`re.findall(r"[A-Za-zА-Яа-яЁё0-9'-]+", name)`).
       - `tier = 2` (сильное имя собственное), если `1 <= len(words) <= 5` и выполняется хотя бы одно из:
         - (a) после снятия ведущего `"the"` (если оно есть и остаются слова) все не-служебные слова (не в `_INNER_CONNECTOR_WORDS`) начинаются с заглавной буквы (`w[0].isupper()`), и первое из них не в `_LEADING_GENERIC_WORDS` (кроме случая когда `len(content_words) >= 2` и все с заглавной — покрывает `"The Leaky Cauldron"`, `"Professor McGonagall"`, `"McGonagall"`, `"Hogwarts"`, `"Mr. Dursley"`);
         - (b) есть хотя бы одно слово с заглавной буквы после первого слова (`any(w[0].isupper() for w in words[1:])`), покрывает адреса вида `"number four, Privet Drive"`.
       - `tier = 1` (слабое/смешанное): `1 <= len(words) <= 5`, `words[0][0].isupper()`, и `words[0].lower() not in _LEADING_GENERIC_WORDS` (например `"Tabby cat"`, `"Тисовая улица"`).
       - `tier = 0` (родовое): всё остальное (`"tabby cat"`, `"the boy"`, `"a dark cupboard"`).
       - Ключ сортировки по убыванию: `(tier, 1 if len(words) <= 4 else 0, min(proper_word_count, 3), count)` с детерминированным tie-breaker по минимальному `(min_position_key, name.casefold())`.
   - При формировании `ENTITY`-записи (L2290–2306):
     - `canonical_name = _choose_canonical_name(members)`
     - `aliases`: все остальные уникальные имена кластера, отсортированные по `(min_position_key, name.casefold())`.
     - `alias_first_positions`: `{name: min_pos}` для всех имён кластера (включая `canonical_name`).
     - `section_orders`: отсортированный список уникальных `int` номеров глав, извлечённых из префикса `position_key` участников кластера (например `"0001:..."` -> `1`).
2. **`illustration_chapter.py`:**
   - Ввести `_display_name_at_position(entity: Mapping[str, Any], event_position_key: str) -> str`:
     - Если в `entity` есть непустой словарь `alias_first_positions` и `event_position_key` непуст:
       - Отфильтровать кандидатов `{name: pos for name, pos in alias_first_positions.items() if isinstance(pos, str) and pos <= event_position_key}`.
       - Если отфильтрованный набор непуст — выбрать лучший вариант через `_choose_canonical_name([{"name": n, "position_key": p} for n, p in eligible.items()])`.
     - Иначе вернуть `entity["canonical_name"]`.
   - Применять `_display_name_at_position(entity, event_position_key)` во всех трёх точках (`L930`, `L957-962`, `L981`, `L1132`, `L1323-1329`):
     - `entity_name` на входе `FACT_SCOPE`;
     - `name_ru` в `IllustrationAppearance`;
     - `name_en` в `IllustrationAppearance` (с сохранением проверки `not has_cyrillic(display_name)`).

### 4.2 `DESIGN`, защита `state_change`, очистка `palette_value:` и пунктуация (`illustration_reference.py`, `illustration_context.py`, `illustration_chapter.py`, `illustration_validation.py`)

1. **`illustration_reference.py`:**
   - `_build_bounded_design_input()` (L573): добавить `"aliases": list(entity_payload.get("aliases") or [])[:8]` в описание сущности, передаваемой в `DESIGN`.
   - `_validate_design()` (L1646): для каждого `trait_item` снимать эхо-префикс `re.sub(rf"^{re.escape(trait)}\s*:\s*", "", text, flags=re.IGNORECASE).strip()` для `description_en` и `description_ru` **до** проверок `has_cyrillic()` и `find_monochrome_color_word()`. Если после снятия строка пуста — отклонять. Сохранять очищенные строки в валидированный payload.
   - `resolve_appearance_with_design_fallback()` (L2618–2715):
     - Ввести `_ANATOMICAL_DESIGN_TRAITS = frozenset({"build", "distinctive_mark", "hair", "eyes", "age_appearance"})`.
     - Проверить `has_state_change = any(str(f.get("temporal_scope", "")) == "state_change" for f in (*explicit_facts, *excluded_facts))`.
     - Если `has_state_change`: исключать из `available_design_traits` все признаки из `_ANATOMICAL_DESIGN_TRAITS`, сохраняя нейтральные (например, `palette_value`), если они не противоречат `compact_description_en`.
     - Расширить `_TRAIT_PATTERNS["build"]` словами облика/тела: `woman|man|boy|girl|human|person|cat|dog|owl|bird|animal|creature|quadruped|silhouette|figure|женщин|мужчин|человек|кошк|кот\b|животн`.
     - В `_echo_content_tokens()`: снизить порог длины токена с `{5,}` до `{3,}` со стоп-листом частых служебных слов (en/ru).
2. **`illustration_context.py`:**
   - `_STAGE_INSTRUCTIONS[DESIGN]`: если у сущности есть факты `state_change` (смена облика/формы/маскировки), не фиксировать в `DESIGN` анатомические черты только одной из форм; оставлять только нейтральные признаки или `traits: []`.
   - `_STAGE_INSTRUCTIONS[PROMPT]`:
     - `characters[].description_en` — единственный авторитетный облик на данный момент; запрещено выбирать `used_design_keys`, противоречащие `characters[].description_en` или `event_ru`.
     - Запретить описывать в `scene_composition_en` отсутствующих персонажей через отрицания («no boy with a scar», «without X»); описывать только реально изображаемое в кадре.
     - Сбалансировать правило про интерьер/экстерьер: для комнатных сцен не рисовать внешний фасад; для уличных/наружных сцен (улица, крыльцо, двор, сад, дорога) изображать именно наружную обстановку, а не интерьер.
3. **`illustration_chapter.py` и `illustration_validation.py`:**
   - В `_dedupe_semicolon_parts()` и `_materialize_appearances_for_prompt()` нормализовать каждый сегмент: `cleaned = part.strip().rstrip(".;").strip()`.
   - В `compose_monochrome_prompt()`:
     - У `comp`: снимать дублирующиеся точки на конце (`re.sub(r"\.{2,}$", ".", comp)`), гарантируя ровно один завершающий знак (`.`, `!` или `?`).
     - У каждого элемента `Visual details:`: очищать `line.strip().rstrip(".;").strip()`, склеивать через `"; "` и завершать одной точкой `"."`.
   - В `_clause_covered_in_text()` (`illustration_validation.py:208-221`): применять `.rstrip(".;").strip()` к `norm_clause` (чтобы однословные клаузы без хвостовой точки не вызывали ложный отказ `validate_moment()`).

### 4.3 Фильтрация изображаемых сущностей (`depicted_entity_ids`)

1. **`illustration_context.py` (`_STAGE_INSTRUCTIONS[CANDIDATES]`):**
   - `participant_names` — **только персонажи, физически видимые/изображаемые в кадре** (включая показанных на фотографиях/портретах, если кадр про них). Запрещено включать отсутствующих персонажей, упомянутых через отрицание («нет следов X»), находящихся за кадром или упоминаемых только в речи/мыслях.
2. **`adapters/llm/illustration_payloads.py` (`IllustrationCandidateEventSchema`, `IllustrationPromptSchema`) и `application/illustration_dto.py` (`IllustrationPromptResponse`):**
   - В `IllustrationCandidateEventSchema`: уточнить `Field(description=...)` для `participant_names` (только физически изображаемые в кадре, на языке оригинала) и `location_names` (место действия на языке оригинала, из `known_locations` если подходит).
   - В `IllustrationPromptSchema`: добавить обязательное поле `depicted_entity_ids: list[str] = Field(description=...)` — список `entity_id` из переданных `characters` и `locations`, которые реально изображены в `scene_composition_en`.
   - В `IllustrationPromptResponse`: добавить `depicted_entity_ids: tuple[str, ...] | None = None` (`None` означает старый сохранённый payload / Python-мок без этого поля -> обратная совместимость). Обновить raw-JSON фикстуры в `tests/unit/llm/test_illustration_payloads.py:144` и `tests/unit/composition/test_illustration_runtime.py:159`.
3. **`illustration_chapter.py` (`_validate_render_response`, `_advance_chapter_rendering`):**
   - В `_validate_render_response()` (`L2074-2164`):
     - Проверить `depicted = response.depicted_entity_ids`:
       - Если `depicted is not None`:
         - Проверить отсутствие дублей и что `set(depicted) <= allowed_entity_ids` (иначе отклонить).
         - Обязательный инвариант: каждый `used_fact_id` должен принадлежать сущности из `depicted_set`, и каждый `entity_id` из `used_design_keys` (`entity_id:trait`) должен входить в `depicted_set` (иначе отклонить).
         - Отфильтровать `base_characters` и `base_locations` по `entity_id in depicted_set` **до вызова `_materialize_appearances_for_prompt()` и `compose_monochrome_prompt()`**!
         - Сохранить `"depicted_entity_ids": list(depicted)` в `payload` скоупа `PROMPT` (`L2153-2163`).
       - Если `depicted is None` (старые записи / моки): сохранить прежнее поведение (все `base_characters` и `base_locations`).
   - В `_advance_chapter_rendering()` (`L1446-1477`):
     - Читать `depicted_raw = p_pay.get("depicted_entity_ids")`.
     - Если `isinstance(depicted_raw, list)`: построить `depicted_set = {str(x) for x in depicted_raw}` и отфильтровать `char_apps`/`loc_apps` (и legacy `summary["characters"/"locations"]`) тем же `depicted_set` перед сборкой `IllustrationMoment`.

### 4.4 Привязка локаций и правила `EXTRACT` / `ALIAS` / `CANDIDATES`

1. **`illustration_context.py`:**
   - `_STAGE_INSTRUCTIONS[EXTRACT]`: для `entity_kind="location"` использовать точное различающее имя места из оригинального текста (улица — именем улицы; дом — адресом/названием дома; комната — названием комнаты; не подписывать дом голым названием улицы).
   - `_STAGE_INSTRUCTIONS[ALIAS]`: добавить правила для `entity_kind="location"`:
     - повторные упоминания одного и того же места/улицы/дома под тем же именем или явным вариантом — `same_entity`, даже если в разных сценах упомянуты разные детали окружения;
     - контейнер и его часть (улица ≠ конкретный дом на ней ≠ отдельная комната/дверь внутри дома) — строго `different`.
   - `_STAGE_INSTRUCTIONS[CANDIDATES]`:
     - `participant_names` и `location_names` **всегда на языке оригинала книги** (никогда не переводить на язык перевода, даже если `scene.location` дан на языке перевода).
     - При совпадении использовать точные имена из `known_characters` / `known_locations`.
     - Обязательно заполнять `location_names`, если место действия известно из текста сцены.
2. **`illustration_selection.py` (`plan_scopes_in_tx`, `_build_req` в `pack_items_by_wire_budget`, `_split_children_for_scope`):**
   - Загружать `ENTITY`-записи запечатанного реестра постранично через `tx.illustrations.query_registry_items(generation.id, kind=IllustrationRegistryKind.ENTITY, offset=..., limit=...)`.
   - Для текущей главы `entry` вычислять верхнюю границу `chapter_end_pos` (чтобы сущности из будущих глав не попадали в `CANDIDATES` и не создавали спойлеров): оставлять сущности с `first_position_key <= chapter_end_pos` (или `entry.order in section_orders` если `section_orders` присутствует).
   - Собрать два компактных списка (до 40 имён каждый, сначала сущности с `entry.order in section_orders`, затем прочие встретившиеся до текущей главы, в детерминированном порядке):
     - `known_characters`: `canonical_name` сущностей с `entity_kind == "person"` (использовать `_display_name_at_position(payload, chapter_end_pos)` для неспойлерного имени к концу главы!).
     - `known_locations`: `canonical_name` сущностей с `entity_kind == "location"`.
   - Включить `known_characters` и `known_locations` в `_build_req` внутри `pack_items_by_wire_budget()` (L680–689) для точного учёта wire byte budget, а также переносить их в `_split_children_for_scope()` (L2089–2115).
3. **`illustration_chapter.py` (`_resolve_entity_name`, `_build_prompt_scope_spec`):**
   - В `_resolve_entity_name(name, entities)` (L252–282):
     - Проход 1: точное совпадение `normalize_name_token(name) == normalize_name_token(entity["canonical_name"])`. Если 1 совпадение — `("resolved", entity)`; если `>= 2` — `("ambiguous", None)`.
     - Проход 2 (только если Проход 1 дал 0 совпадений): точное совпадение по `entity["aliases"]` через `normalize_name_token()`. Если 1 совпадение — `("resolved", entity)`; если `>= 2` — `("ambiguous", None)`.
     - Иначе `("unmatched", None)`.
   - В `_build_prompt_scope_spec()` (L481):
     - Если `locations` пуст, находить сцену в `entry.scenes` по `s.id == card.scene_id` или `f"{section_id}:scene:{idx}"` и, если у сцены непустой `location`, передавать справочное поле `scene_location_ru` (ограниченное по длине) для `scene_composition_en` (без создания фиктивного `IllustrationAppearance`).

---

## Тестовая матрица

| WP | Файл тестов | Проверяемые сценарии |
|---|---|---|
| WP1 | `tests/unit/application/test_correction_inputs.py` | `_evidence_value` и `CorrectionIssue.valid_for` на `asdict(BlockProcessingResult)` и `_llm_block_result`: 1-based `segments.2.translated_text`, `segments.0` fallback, запрет коллизии `segments.1` на `[READ_AS_IS, TRANSLATE, TRANSLATE]` (в т. ч. для `CurrentAbsenceEvidence`), толерантный quote fallback в `CurrentQuoteEvidence`, отказ на `segments.1.index` / `block_id` / `treatment` |
| WP1 | `tests/unit/llm/test_llm_payload_mapping.py` | `parse_payload()` возвращает статическую подсказку из `_STATIC_CONTRACT_ERROR_HINTS` для известных `LLMPayloadError`, но не утекает произвольный текст книги / `source_block_id` для незарегистрированных ошибок; `Violation.safe_message` статичен |
| WP1 | `tests/integration/processing/test_segment_correction_lifecycle.py` | Полный V2-цикл `propose → review → apply` на смешанном сегментированном блоке `[READ_AS_IS, TRANSLATE, TRANSLATE]` с `field_path="segments.2.translated_text"` |
| WP2 | `tests/unit/application/test_token_budget.py` | `reasoning_effort="low"` + `implicit_reasoning_tokens=8000` повышает бюджет; `reasoning_effort="none"` и `reasoning_max_tokens=4096` не меняются; однократная implicit-эскалация в recovery при `max_output_tokens=None` и `reasoning_effort="low"` |
| WP2 | `tests/unit/llm/test_segment_translation_contract.py`, `tests/unit/llm/test_llm_payload_mapping.py`, `tests/unit/application/test_result_validators.py` | Двунаправленное сообщение при одновременных `translated_text` и `segments`; уточнённое сообщение в `treatment.py`; замороженные SHA-256 (`test_baseline_b2_templates_sha256_unchanged`) не задеты |
| WP2 | `tests/unit/application/test_analysis_merge_scenes.py` | `find_quote_span` уравнивает `…` и `...` в обе стороны, но не матчит произвольный текст |
| WP3 | `tests/unit/application/test_illustration_selection.py`, `tests/unit/application/test_illustration_reference.py` | Инструкция и `Field.description` схемы `SELECTION` требуют `selection_reasons_ru`; `compact_repair_summary` идемпотентен, на `EXTRACT` требует полный ответ для всех `source_unit_id` и укладывается в `MAX_REPAIR_SUMMARY_BYTES`; дефолт `max_repair_turns=2` допускает 2 ремонта в пределах 3 вызовов |
| WP4 | `tests/unit/application/test_illustration_reference.py` | 3-уровневый `_choose_canonical_name` (`"Professor McGonagall"` > `"McGonagall"` > `"Tabby cat"` > `"tabby cat"`, `"The Leaky Cauldron"` и `"number four, Privet Drive"` в `tier=2`), сохранение `alias_first_positions` и `section_orders`; `_validate_design` снимает префикс `palette_value:`/`build:` до проверок; `resolve_appearance_with_design_fallback` подавляет анатомические черты `DESIGN` при `temporal_scope == "state_change"` в `explicit_facts` или `excluded_facts` и распознаёт 3-символьные токены (`cat`, `tail`, `fur`) |
| WP4 | `tests/unit/application/test_illustration_chapter.py`, `tests/unit/application/test_illustration_reliability.py`, `tests/unit/llm/test_illustration_payloads.py`, `tests/unit/composition/test_illustration_runtime.py` | `_display_name_at_position` не спойлерит имя в `FACT_SCOPE`, `name_ru` и `name_en`; `_resolve_entity_name` возвращает `"resolved"` с приоритетом `canonical_name` > `aliases` при сохранении `ambiguous` для двух одинаковых `canonical_name`; `depicted_entity_ids` фильтрует `compose_monochrome_prompt()` в `_validate_render_response()` и `IllustrationMoment` в `_advance_chapter_rendering()`, отклоняет чужие `used_fact_ids`/`used_design_keys`, сохраняет обратную совместимость `None`; `compose_monochrome_prompt` и `_clause_covered_in_text` без `.;` и `..` (вкл. однословные клаузы); `scene_location_ru` при пустых `locations` |
| WP4 | `tests/unit/application/test_illustration_selection.py` | `plan_scopes_in_tx` пробрасывает `known_characters` (`person`) и `known_locations` (`location`) с фильтром `first_position_key <= chapter_end` и сохраняет их в `_split_children_for_scope()` |
| WP4 | `tests/integration/processing/test_illustration_reliability_integration.py` | Сквозной сценарий главы: смена облика (`state_change`), уличная локация + дом, сцена с отсутствующим персонажем -> непустые `locations`, отсутствие отсутствующего героя и старой формы в `prompt_en` и `IllustrationMoment`, чистая пунктуация |
