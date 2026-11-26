# SDD (Backend): `reason` в проверке коррекции, пустые `GLOSSARY_ONLY`-сегменты и устойчивость `EXTRACT`

**Дата:** 2026-11-26  
**Статус:** Approved (после глубокого ревью спецификаций и устранения всех замечаний)  
**Слойная модель:** `domain ← application ← adapters / web / cli` (без новых межслойных зависимостей и без миграций БД).

---

## WP1: Допуск пустого `reason` при подтверждении в `chapter_correction_review`

### 1.1 `src/linguary/adapters/llm/correction_payloads.py`

В `CorrectionVerdictSchema` (строки 123–132):
- Изменить поле `reason`:
  ```python
  reason: str
  ```
  (отсутствие значения по умолчанию гарантирует, что пропуск ключа `reason` всё равно отклоняется, а `""` принимается при подтверждении).
- Добавить валидатор `@model_validator(mode="after")` (импортировать `model_validator` из `pydantic`):
  ```python
  @model_validator(mode="after")
  def _require_reason_unless_confirmed(self) -> CorrectionVerdictSchema:
      confirmed = (
          self.verdict == "confirmed"
          and self.error_supported
          and self.evidence_valid
          and self.minimal
          and self.no_semantic_loss
      )
      if not confirmed and not self.reason.strip():
          raise ValueError("для вердикта needs_human нужна непустая причина")
      if confirmed and not self.reason.strip():
          self.reason = ""
      return self
  ```
- Проверка совместимости с `strict_json_schema()`:
  - `strict_json_schema(ChapterCorrectionReviewSchema)` продолжает выдавать `"reason": {"type": "string"}` в `required` и `additionalProperties: false` — проволочный контракт провайдера не меняется.
  - При `verdict="confirmed"` и всех 4 флагах `True` принимаются как `reason=""`, так и пробельный `reason` (нормализуется в `""`, чтобы в `awaiting_human` при отказе по `evidence_valid` шаблон `{% if candidate.review_reason %}` не выводил пустую строку), так и непустой `reason`.
  - При `verdict="needs_human"` ИЛИ при `verdict="confirmed"` с хотя бы одним ложным флагом (`error_supported`/`evidence_valid`/`minimal`/`no_semantic_loss`) пустая или пробельная строка `reason` отклоняется с `ValueError`.

### 1.2 `src/linguary/adapters/llm/payloads.py`

В `_STATIC_VALUE_ERROR_HINTS` (строки 654–687) добавить элемент:
```python
(
    "для вердикта needs_human нужна непустая причина",
    "для вердикта needs_human нужна непустая причина",
),
```
Это гарантирует, что `_schema_hints()` вернёт модели конкретную безопасную подсказку без интерполяции текста книги или ответа модели.

### 1.3 `src/linguary/application/correction_dto.py`

В `CorrectionVerdict.__post_init__` (строки 581–594):
- Сначала проверить базовые поля и типы (включая `isinstance(self.reason, str)` и `type(getattr(self, n)) is bool` для всех четырёх флагов), затем проверить `not self.confirmed and not self.reason.strip()`:
  ```python
  def __post_init__(self):
      if (
          not self.candidate_id
          or (
              self.proposal_sha256 is not None
              and (not isinstance(self.proposal_sha256, str) or len(self.proposal_sha256) != 64)
          )
          or self.verdict not in {'confirmed', 'needs_human'}
          or not isinstance(self.reason, str)
          or any(
              type(getattr(self, n)) is not bool
              for n in ('error_supported', 'evidence_valid', 'minimal', 'no_semantic_loss')
          )
          or (not self.confirmed and not self.reason.strip())
      ):
          raise LLMPayloadError('неполный вердикт проверки кандидата')
  ```
- Проверка legacy-конструктора `CorrectionVerdict.__init__` (строки 558–570):
  - При 7 позиционных аргументах `CorrectionVerdict('c1', 'confirmed', True, True, True, True, '')`:
    - `proposal_sha256 == 'confirmed'`, `type(verdict) is bool`, `reason == ''`, `isinstance(no_semantic_loss, str)` (`isinstance('', str) is True`) → аргументы корректно сдвигаются, `self.confirmed is True`, валидация проходит.
  - При 8 позиционных/именованных аргументах с `proposal_sha256=None` и `reason=""`: работает штатно.

### 1.4 Проверка потребителей (без изменений кода)
- `src/linguary/composition/correction_runtime.py:90-101` — `CORRECTION_REVIEW_PROMPT` не меняется.
- `src/linguary/adapters/db/work_state.py:426-515` — `record_correction_verdict` и `_confirmed_candidate` сравнивают `review` с архивом по равенству словарей, пустой `reason=""` сохраняется и сверяется идентично.
- `src/linguary/application/book_overview.py:1475` и `src/linguary/web/templates/partials/correction_candidate.html:40` — `awaiting_human` всегда имеет `not confirmed` (значит непустой `reason`), а шаблон дополнительно защищён `{% if candidate.review_reason %}`.

---

## WP2: Понижение пустых `GLOSSARY_ONLY`-сегментов и явные правила для смешанных блоков

### 2.1 `src/linguary/application/analysis_merge.py`

Добавить приватную функцию `_demote_vacuous_mixed_glossary_segments`:
```python
def _demote_vacuous_mixed_glossary_segments(
    item: TreatmentPlanItemResponse,
    block: DocumentBlock,
    difficulties,
    owners: frozenset[str],
    *,
    known_names: Sequence[str] = (),
) -> TreatmentPlanItemResponse:
```
Точная логика:
1. Ранний выход (вернуть `item` без изменений), если:
   - `block.type is BlockType.POEM`, или
   - `not item.segments`, или
   - в `item.segments` нет одновременно хотя бы одного `BlockTreatment.TRANSLATE` и хотя бы одного `BlockTreatment.GLOSSARY_ONLY`.
2. Сгруппировать `item.segments` по `sentence_index` через `spans = block.segment_spans` и `sentence_bounds = _sentence_bounds_by_index(block)` (аналогично `_promote_a2_stoplist_item`), с защитой `1 <= segment.index <= len(spans)` и `sentence_index in sentence_bounds`.
3. Для каждой группы `(sentence_index, segs)`:
   - Проверить `strongest_treatment(seg.treatment for seg in segs) is BlockTreatment.GLOSSARY_ONLY`; иначе пропустить.
   - Получить `s_start, s_end = sentence_bounds[sentence_index]` и `s_text = block.text[s_start:s_end]`.
   - Если `dominant_script(s_text) != "LATIN"` → пропустить (не понижать иноязычные/нелатинские вставки).
   - Если `_has_difficulty_in_range(difficulties, owners, block.text, s_start, s_end, whole_block=False)` → пропустить (есть явная привязанная трудность).
   - Если `_has_a2_nonlexical_stoplist(s_text)` → пропустить.
   - Вычислить `words = expression_words(s_text)`.
   - Если `words` содержит слово из `_SHORT_B1_WORDS`, слово из `(_NONBASIC_REPORTING - _BASIC_REPORTING)`, или фразу из `_SHORT_B1_PHRASES` (`_contains_phrase(words, phrase)`) → пропустить.
   - Если `_A2_B1_PHRASAL_VERB_RE.search(s_text) is not None` → пропустить.
   - Если `non_core_a2_word_spans(s_text, known_names=known_names)` непуст → пропустить.
   - Иначе добавить `sentence_index` в `demoted_sentences`.
4. Если `not demoted_sentences`: вернуть `item`.
5. Заменить у сегментов, чей `sentence_index in demoted_sentences` и `segment.treatment is BlockTreatment.GLOSSARY_ONLY`, режим на `BlockTreatment.READ_AS_IS`.
6. Вычислить `strongest = strongest_treatment(seg.treatment for seg in new_segments)` (остаётся `BlockTreatment.TRANSLATE`) и вернуть `replace(item, treatment=strongest, segments=new_segments)`.

Интеграция в `normalize_pedagogical_response()` (`analysis_merge.py`, сразу после Шага 3, перед Шагом 4, внутри `if not english_help:`):
```python
if not english_help:
    for idx, item in enumerate(items):
        block = _block_for_plan_item(item.source_block_id, blocks_by_id, block_ids_by_alias)
        if block is None:
            continue
        owners = frozenset(part for part in (item.source_block_id, block.id) if part)
        updated = _demote_vacuous_mixed_glossary_segments(
            item, block, difficulties, owners, known_names=known_names
        )
        if updated is not item:
            items[idx] = updated
            changed = True
```
Инварианты:
- Запускается после Шага 2 (повышение стоп-листа A2 уже выполнено) и Шага 3 (стихи/списки уже унифицированы), и до Шага 5 (`_materialize_a2_glossary_difficulties` работает строго по финальным `GLOSSARY_ONLY`-срезам).
- Не влияет на `_compute_promoted_block` в Шаге 4 (понижение `GLOSSARY_ONLY` → `READ_AS_IS` не является повышением в `TRANSLATE`).
- Проверяет `difficulties` (исходные трудности после Шага 1) — если трудность была на этом сегменте, сегмент не понижается. (Замечание: трудность на сегменте `GLOSSARY_ONLY` в Шаге 4 не удаляется и не переключается, потому что сегмент не повышался в `TRANSLATE`.)

### 2.2 `src/linguary/application/context_builder.py`

1. В `_reader_level_treatment_guidance("a2")` (строки 2181–2184, **не** в замороженном SHA-256):
   - Дополнить правило в строке 2181:
     уточнить, что `GLOSSARY_ONLY` выбирается только при наличии хотя бы 1 слова или выражения выше A2 (если в предложении или сегменте нет слов выше A2, ставь `READ_AS_IS`).
2. В `_CONTENT_PLAN_PROCESSING_V3` (строка 308, **не** в замороженном SHA-256):
   - Сразу после предложения про блоки `TRANSLATE (без сегментов GLOSSARY_ONLY)` добавить:
     `В смешанном блоке с сегментом GLOSSARY_ONLY (даже если режим блока — TRANSLATE) обязательно верни в glossary_hints 1–3 трудных слова или выражения именно из сегмента GLOSSARY_ONLY.`
3. В `_SEGMENT_TRANSLATION_RULE` (строки 2976–2983, **не** в замороженном SHA-256; обновить зеркальную константу в `tests/unit/application/test_translation_context.py:2271`):
   - Добавить краткое предложение:
     `Если среди сегментов блока есть GLOSSARY_ONLY, обязательно верни в glossary_hints 1–3 трудных слова или выражения именно из этого сегмента.`
4. `_TRANSLATION_INSTRUCTIONS` и `snapshot_prompt_templates(..., reader_level="b2", profile="legacy-v1")` **не изменять**.

### 2.3 `src/linguary/application/validators/treatment.py`

В `_segmented_content()` (строки 309–324):
- Вычислить индексы `GLOSSARY_ONLY`-сегментов (1-based):
  ```python
  glossary_indices = ", ".join(
      str(i) for i, mode in enumerate(modes, start=1) if mode is BlockTreatment.GLOSSARY_ONLY
  )
  ```
- Обновить три сообщения об ошибках для `GLOSSARY_ONLY`, сохранив существующие префиксы:
  - Строка 311:
    `f"для блока {block.source_block_id} обязательны словарные подсказки (для сегмента GLOSSARY_ONLY: {glossary_indices})"`
  - Строки 321–324:
    `f"для блока {block.source_block_id} ни одна словарная подсказка не относится к сегменту GLOSSARY_ONLY ({glossary_indices})"`
  - Строки 330–333:
    `f"для блока {block.source_block_id} словарная подсказка «{hint.original}» без note не относится к сегменту GLOSSARY_ONLY ({glossary_indices})"`
- Проверка совместимости с `test_result_validators.py`: существующие тесты проверяют подстроки префиксов (`in message`), оба префикса сохранены дословно.

---

## WP3: Компактная посекционная нарезка, автодробление и защита `EXTRACT`

### 3.1 Константы и посекционная упаковка в `src/linguary/application/illustration_reference.py`

Добавить константы (рядом со строкой 195):
```python
_MAX_EXTRACT_TARGET_BYTES = 12_288
_MAX_EXTRACT_BLOCKS_PER_PAGE = 16
_MAX_EXTRACT_SCENE_UNITS_PER_PAGE = 2
_MAX_EXTRACT_BLOCK_UNITS_PER_PAGE = 16
```
(Старая константа `_MAX_EXTRACT_UNITS_PER_PAGE = 64` заменяется явными лимитами для сцен и одноблочных юнитов.)

Добавить чистую функцию `_pack_extract_section_units`:
```python
def _pack_extract_section_units(
    section_units: Sequence[dict[str, Any]],
    *,
    build_request: Callable[[Sequence[dict[str, Any]]], LLMRequest[Any]],
    sizer: RequestWireSizer,
    target_bytes: int,
    max_units: int,
    max_blocks: int,
) -> tuple[tuple[dict[str, Any], ...], ...]:
```
Алгоритм:
1. Если `not section_units`: вернуть `()`.
2. Поддерживать `pages: list[tuple[dict[str, Any], ...]] = []`, `current: list[dict[str, Any]] = []`, `current_blocks = 0`.
3. Для каждого `unit` в `section_units`:
   - `u_blocks = len(unit.get("blocks", ()))`
   - Проверить, можно ли добавить `unit` в непустой `current`:
     - `len(current) < max_units`
     - `current_blocks + u_blocks <= max_blocks`
     - `sizer.measure(build_request([*current, unit])).size_bytes <= target_bytes`
   - Если `current` непуст и все 3 условия выполнены: добавить `unit` в `current`, `current_blocks += u_blocks`, `continue`.
   - Если `current` непуст (не поместилось): `pages.append(tuple(current))`, `current = []`, `current_blocks = 0`.
   - Теперь `current` пуст. Проверить, может ли `unit` быть началом новой страницы с последующим добавлением:
     - Если `max_units <= 1` или `u_blocks >= max_blocks` или `sizer.measure(build_request([unit])).size_bytes > target_bytes`:
       выпустить одиночную страницу `pages.append((unit,))` (чтобы не терять крупную сцену и не делать лишний замер).
     - Иначе: `current = [unit]`, `current_blocks = u_blocks`.
4. В конце, если `current`: `pages.append(tuple(current))`.
5. Вернуть `tuple(pages)`.

Изменение в `IllustrationReference.plan_scopes()` (строки 815–860):
- Собирать `section_unit_groups: list[list[dict[str, Any]]] = []` по не пропущенным секциям `snapshot.entries`, добавляя группу только при непустом результате (`sec_units = _section_extract_units(entry, sec); if sec_units: section_unit_groups.append(sec_units)` — защита от пустых секций, сохраняющая ветку пустого реестра).
- Если `not section_unit_groups`: оставить текущую ветку пустого реестра (строки 824–834).
- Вычислить `extract_target_bytes = min(policy.initial_target_bytes, _MAX_EXTRACT_TARGET_BYTES)`.
- Для каждой `section_units` в `section_unit_groups`:
  - `has_scenes = any(u.get("scene_id") is not None for u in section_units)`
  - `max_units = _MAX_EXTRACT_SCENE_UNITS_PER_PAGE if has_scenes else _MAX_EXTRACT_BLOCK_UNITS_PER_PAGE`
  - Вызвать `_pack_extract_section_units(...)` и добавить полученные страницы в общий список `pages`.
- По `enumerate(pages)` строить `IllustrationScopeSpec` с `scope_key = f"extract:p{page_idx:04d}"` как раньше (с проверкой `find_reusable_scope`).

Связанное уточнение в `src/linguary/application/illustrations.py` (строки 454–462):
- Поскольку посекционные `EXTRACT`-скоупы теперь имеют конкретный `section_id`, в проверке `has_failed` добавить `or s.stage in _REFERENCE_STAGE_SET`, чтобы упавший `EXTRACT` (общекнижная предпосылка) продолжал блокировать `prepare(..., section_id=...)` и требовал `retry`.

Гарантии:
- Ни один `EXTRACT`-пакет никогда не пересекает границу секции/главы (`section_id` всегда конкретен).
- Каждый многосценовый пакет содержит не более 2 сцен, не более 16 блоков и не превышает `min(policy.initial_target_bytes, 12_288)` байт.
- Если пользователь/тест выставил меньший `policy.max_request_bytes` (например, 4096), `extract_target_bytes` строго соблюдает `policy.initial_target_bytes`.
- Сверхкрупная одиночная сцена (>16 блоков или >12 КБ) выпускается как `(unit,)` и при `> policy.max_request_bytes` дробится существующим `_abort_oversized_or_split`, а при повторной ошибке — новым пост-ремонтным автодроблением через `_split_extract_unit_blocks` (`#L`/`#R`).

### 3.2 Пост-ремонтное автодробление `EXTRACT` в `src/linguary/application/illustration_reference.py`

В `IllustrationReference.run_task()`:
1. Ветка `isinstance(call_error, LLMResponseFormatError)` (строки 1218–1245):
   - После исчерпания `repairs_used < policy.max_repair_turns and self._can_continue_activation(...)`:
   ```python
   if scope.stage is IllustrationScopeStage.EXTRACT and is_composite:
       notify_after = self._handle_post_send_split(
           book_id=book_id,
           generation_id=generation_id,
           task_id=task.id,
           scope=scope,
           attempt_id=attempt.id,
           revision=revision,
           depth=depth,
           policy=policy,
           fail_reason=IllustrationReasonCode.MODEL_OUTPUT_INVALID.value,
           detail=err_detail,
       )
       break
   ```
   - Иначе (атомарный скоуп или не-`EXTRACT` стадия) — прежний `_finalize_scope_failure(...)`.
2. Ветка `if not validation.ok:` (строки 1292–1326):
   - После исчерпания `repairs_used < policy.max_repair_turns and self._can_continue_activation(...)`:
   ```python
   if scope.stage is IllustrationScopeStage.EXTRACT and is_composite:
       notify_after = self._handle_post_send_split(
           book_id=book_id,
           generation_id=generation_id,
           task_id=task.id,
           scope=scope,
           attempt_id=attempt.id,
           revision=revision,
           depth=depth,
           policy=policy,
           fail_reason=validation.reason_code,
           detail=detail_text,
       )
       break
   ```
   - Иначе — прежний `_finalize_scope_failure(...)`.

Проверка граничных случаев:
- При `notify_after` после `break` в конце `run_task()` (строки 1408–1411) вызывается `self._notify_work_available()` — дочерние задачи немедленно подхватываются воркером.
- При достижении `depth >= policy.max_split_depth` или `max_scopes_per_generation` `_handle_post_send_split()` штатно переводит скоуп в `FAILED` (`SPLIT_DEPTH_EXHAUSTED` / `SCOPE_LIMIT_EXCEEDED`).
- Для атомарного `EXTRACT`-скоупа (1 юнит из 1 блока, `is_composite == False`) сохраняется немедленный отказ `MODEL_OUTPUT_INVALID` (полная совместимость с `test_complete_true_missing_covered_units_rejected`).
- Для `ALIAS`/`DESIGN` поведение не меняется.

### 3.3 Контракт `subject_mention_id` в `EXTRACT`

1. `src/linguary/application/illustration_context.py` (`_STAGE_INSTRUCTIONS[IllustrationScopeStage.EXTRACT]`, строки 353–364):
   - Добавить явное предложение:
     `"Каждый элемент facts обязан ссылаться в subject_mention_id на существующий mention_id из массива mentions этого же ответа (только person или location; не создавай факты для предметов без субъекта в mentions). "`
   - Сохранить все существующие ключевые фразы (`alias_of_hint`, `entity_kind="location"`, `complete=false`, `covered_source_unit_ids`).
2. `src/linguary/application/illustration_reference.py` (`_validate_extract`, строка 1547):
   - Расширить сообщение ремонта с сохранением префикса:
     `f"факт {f.fact_id} ссылается на неизвестный subject_mention_id {f.subject_mention_id}; укажи mention_id из массива mentions этого же ответа"`

### 3.4 Типографско-пробельный запасной поиск в `resolve_quote_anchor()` (`src/linguary/application/illustration_context.py`)

Добавить вспомогательные функции (внутри `illustration_context.py`, без сторонних зависимостей):
```python
_QUOTE_CHAR_MAP: Mapping[str, str] = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "`": "'", "\u00b4": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u00ab": '"', "\u00bb": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2011": "-",
}
```
Функция `_normalize_quote_with_spans(text: str) -> tuple[str, list[int], list[int]]`:
- Проходит `text` посимвольно (`idx`, `ch`):
  - Если `ch.isspace()`:
    - если последний добавленный нормализованный символ уже `' '`, расширить `ends[-1] = idx + 1`;
    - иначе добавить `' '`, `starts.append(idx)`, `ends.append(idx + 1)`.
  - Иначе если `ch == "\u2026"` (`…`):
    - добавить три `'.'` с одинаковыми `(idx, idx + 1)` в `starts`/`ends`.
  - Иначе:
    - добавить `_QUOTE_CHAR_MAP.get(ch, ch)`, `starts.append(idx)`, `ends.append(idx + 1)`.
- Срезать ведущий и замыкающий одиночный пробел из `(norm, starts, ends)` (чтобы пробелы по краям цитаты не мешали).

Функция `_normalized_occurrences(haystack: str, needle: str) -> list[tuple[int, int]]`:
- Вычислить `h_norm, h_starts, h_ends = _normalize_quote_with_spans(haystack)` и `n_norm, _, _ = _normalize_quote_with_spans(needle)`.
- Если `not n_norm`: вернуть `[]`.
- Найти все позиции `pos` в `_all_occurrences(h_norm, n_norm)`.
- Отфильтровать частичные совпадения внутри многосимвольного развёртывания (`…` → `...`):
  - пропустить, если `pos > 0 and h_starts[pos] == h_starts[pos - 1]` или `end_pos < len(h_ends) and h_ends[end_pos - 1] == h_ends[end_pos]`.
- Вернуть уникальные пары `(h_starts[pos], h_ends[end_pos - 1])`.

В `resolve_quote_anchor()` (после двух точных поисков `visible` и `block.text`, перед `return QuoteResolutionResult(status="missing")`):
1. Проверить `norm_visible = _normalized_occurrences(visible, quote)`:
   - если `len(norm_visible) > 1` → `return QuoteResolutionResult(status="ambiguous")`
   - если `len(norm_visible) == 1` → перевести `(v_start, v_end)` через `origins` в `(s_start, s_end)` и вернуть `resolved` с `quote=block.text[s_start:s_end]`.
2. Проверить `norm_raw = _normalized_occurrences(block.text, quote)`:
   - если `len(norm_raw) > 1` → `return QuoteResolutionResult(status="ambiguous")`
   - если `len(norm_raw) == 1` → вернуть `resolved` с `quote=block.text[s_start:s_end]`.
3. Иначе → `return QuoteResolutionResult(status="missing")`.

---

## Матрица тестирования

| Пакет | Файл тестов | Проверяемые сценарии |
|---|---|---|
| WP1 | `tests/unit/application/test_correction_inputs.py` | (1) `CorrectionVerdictSchema` + `parse_payload`: `confirmed` + 4 `True` + `reason=""` проходит; (2) `needs_human` + `reason=""` падает с понятной подсказкой `_STATIC_VALUE_ERROR_HINTS`; (3) `confirmed` + один `False`-флаг + `reason=""` падает; (4) `CorrectionVerdict` DTO (именованный и 7-аргументный legacy-вызов) соответствует тем же правилам; (5) `strict_json_schema(ChapterCorrectionReviewSchema)` содержит `reason` в `required`. |
| WP2 | `tests/unit/application/test_analysis_merge.py`, `tests/unit/application/test_result_validators.py`, `tests/unit/application/test_translation_context.py`, `tests/unit/application/test_context_builder.py` | (1) Смешанный блок (`TRANSLATE` + пустой элементарный `GLOSSARY_ONLY` вроде `b#5`) понижает элементарный сегмент в `READ_AS_IS` для A2 и B2; (2) Смешанный блок с небазовым словом (`doorstep`/`cloak` как `b#106`) или привязанной трудностью или нелатинским текстом сохраняет `GLOSSARY_ONLY`; (3) Несегментированный `GLOSSARY_ONLY` и `POEM` не затрагиваются; (4) Сообщения `_segmented_content` содержат индекс сегмента `GLOSSARY_ONLY` и прежний префикс; (5) `_CONTENT_PLAN_PROCESSING_V3`, `_SEGMENT_TRANSLATION_RULE`, `_reader_level_treatment_guidance("a2")` содержат уточнённые правила, а `test_baseline_b2_templates_sha256_unchanged` зелёный. |
| WP3 | `tests/unit/application/test_illustration_reference.py`, `tests/unit/application/test_illustration_reliability.py` | (1) `plan_scopes` не объединяет сцены из разных секций, ограничивает пакет ≤2 сценами / ≤16 блоками / ≤12 КБ, но позволяет до 16 одноблочных юнитов в секции без сцен и выпускает сцену >16 блоков одиночным пакетом; (2) Составной `EXTRACT` после исчерпания `max_repair_turns` на `LLMResponseFormatError` и на ошибке валидации (неизвестный `subject_mention_id` / неполный `covered_source_unit_ids`) автоматически дробится на дочерние скоупы и завершает пайплайн; (3) Атомарный `EXTRACT` и исчерпание `max_split_depth` падают штатно; (4) `resolve_quote_anchor` разрешает фигурные/прямые кавычки, тире, многоточие и сжатые пробелы, возвращая дословный `block.text[s_start:s_end]`, и возвращает `ambiguous` при дубликате; (5) `_STAGE_INSTRUCTIONS[EXTRACT]` содержит правило `subject_mention_id`. |
