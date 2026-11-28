# SDD (Backend & Skill): Контракты, алгоритмы и тестовая матрица (HP Full Audit — Scalability & A2 Quality)

**Дата:** 2026-11-28  
**Статус:** Revised (доработано по итогам двух независимых ревью спецификаций и проверки задетого кода `HEAD d9ff0f9`)  
**Связанные документы:** `BRIEF-hp-full-audit-scalability-and-a2-quality-2026-11-28.md`, `ADR-hp-full-audit-scalability-and-a2-quality-2026-11-28.md`

---

## Глобальные ограничения и инварианты

- **Границы слоёв:** `domain ← application ← adapters / web / cli` (`uv run pytest tests/architecture`). В `domain` и `application` нет сторонних библиотек и прямого I/O (чтение ZIP/OPF исходного EPUB размещается строго в `src/linguary/adapters/epub/`).
- **Схема БД:** 0 миграций Alembic. Существующие записи `content_plan_json` (`version=3`), `analysis_pass_results`, `books` и `processing_runs` остаются валидными и обратно совместимыми.
- **Замороженные SHA-256 и версии:**
  - `_TRANSLATION_INSTRUCTIONS`, `_ANALYSIS_INSTRUCTIONS` и `snapshot_prompt_templates(source_language="en", target_language="ru", reader_level="b2", profile="legacy-v1")` (с/без `pronunciation_help_enabled`, `name_translation_policy=None`) остаются побайтово неизменными (`tests/unit/application/test_context_builder.py:2784` `test_baseline_b2_templates_sha256_unchanged`).
  - `A2_CONTENT_PLAN_V8_VERSIONS` (`{'runtime': 1, 'context_builder': 6, 'validators': 4, 'schemas': 5, 'prompts': 8}`) не меняется.
  - *Кэш `content-plan-v3`:* обновление промптов и схемы `QuotedContextFactSchema` законно изменяет `snapshot_prompt_templates(..., profile="content-plan-v3")` и `input_sha256` проходов анализа для новых/переанализируемых книг; подгонять старые хэши `content-plan-v3` не нужно.
- **Универсальность:** 0 новых хардкодов под конкретные книги, персонажей, топонимы или модели LLM.
- **Статанализ и покрытие:** `uv run ruff check src tests`, `uv run ty check`, `uv run pytest tests/architecture`, `uv run pytest -m "not slow"`, `uv run pytest --cov=linguary --cov-fail-under=90`.
- **Разделение треков:**
  - **Трек A (git репозитория):** WP1–WP7 (`src/linguary/`, `tests/`, `docs/superpowers/`).
  - **Трек B (локальный, вне git):** WP8-local (`.agents/skills/linguary-subagent-run/`).

---

## WP1 — Совместная (хоровая) речь и устранение коллизии `visual_facts.subject_anchor`

### 1.1 `src/linguary/domain/content_plan.py`

1. **`ContextFact` (`L110-149`):**
   - Добавить опциональное поле:
     ```python
     joint: bool = False
     ```
   - В `__post_init__`:
     ```python
     if type(self.joint) is not bool:
         raise DomainError('признак совместной речи должен быть булевым')
     if self.joint and (self.kind != 'speaker' or self.certainty == 'unknown'):
         raise DomainError('совместная атрибуция допустима только для известного speaker')
     ```
   - В `to_dict()`:
     ```python
     value = asdict(self)
     if self.certainty == 'unknown':
         value.pop('value')
     if not self.joint:
         value.pop('joint')
     return value
     ```
     *Инвариант:* для всех одиночных фактов (`joint=False`) `to_dict()`, `canonical_id` и `ContentPlan.sha256` побайтово неизменны.
2. **`ContentPlan.__post_init__` (`L220-234`):**
   - Группировать факты по `(fact.kind, fact.subject_anchor)` в уникальные тройки `(fact.value, fact.certainty, fact.joint)`.
   - Для пары `(kind, anchor)` с `len(entries) > 1`:
     - при `kind != 'speaker'` → конфликт;
     - при `kind == 'speaker'`: группа валидна **только если**:
       1. `all(joint for _, _, joint in entries)`;
       2. `all(certainty != 'unknown' for _, certainty, _ in entries)`;
       3. `len({value for value, _, _ in entries}) == len(entries)` (один и тот же говорящий не указан дважды с разными `certainty`);
     - иначе → конфликт (формат сообщения об ошибке `конфликт атрибуций одного субъекта: ...` сохраняется).

### 1.2 Схемы, DTO, кэш и маппинг БД

1. **`src/linguary/application/llm_dto.py:430` (`QuotedContextFactResponse`, реэкспортирован в `ports/llm.py`):** добавить `joint: bool = False`.
2. **`src/linguary/adapters/llm/schemas.py:455-473` (`QuotedContextFactSchema`):**
   - Добавить `joint: bool = Field(default=False, strict=True)`.
   - В `@model_validator(mode="after")` проверять: при `self.joint` и (`self.kind != "speaker"` или `self.certainty == "unknown"`) поднимать `ValueError("joint=true допустимо только для speaker с непустым значением")`.
3. **`src/linguary/adapters/llm/payloads.py:844-852` (`_quoted_context_fact`):** прокинуть `joint=schema.joint`.
4. **`src/linguary/application/analysis_pass_cache.py:125-135, 224-230`:**
   - В `encode_pass`: удалять `"joint"` на элементах `context_facts`, если `not fact.get("joint")`.
   - В `decode_pass`: безопасно передавать `joint=bool(f.get("joint", False))`.
5. **`src/linguary/adapters/db/mapping.py:917-928` (`_context_fact`):**
   - Добавить строгий хелпер:
     ```python
     def _boolean(value: object) -> bool:
         if type(value) is not bool:
             raise ValueError("ожидалось булево значение")
         return value
     ```
   - В `_context_fact`: добавить `'joint'` в `fields`, если `isinstance(value, dict) and 'joint' in value`, и передать `joint=_boolean(data['joint']) if 'joint' in data else False`.

### 1.3 `src/linguary/application/analysis_merge.py` и `context_builder.py`

1. **`analysis_merge.py` — все точки переноса `joint` (`L3272-3284, 3563-3625, 3666-3670, 4013-4024`):**
   - Передать `joint=fact.joint` в:
     - `validate_narrative_response` (`L3564-3571`);
     - `_quoted_fact` (`L3666-3670`);
     - **`content_plan_from_response` (`L4016-4023`)** (критично для финальной сборки `ContentPlan` главы!).
   - Добавить `_normalize_joint_speaker_facts(facts: Sequence[ContextFact], texts: Mapping[str, str]) -> list[ContextFact]` и вызывать его **ДО** цикла `_explicit_speaker_if_remark_names` (сразу после `L3573` в `validate_narrative_response` и в `content_plan_from_response` до `L4055`):
     - группировать `kind == "speaker"` по `subject_anchor`;
     - если в группе ≥ 2 фактов с уникальными непустыми `value` и `all(f.certainty != "unknown")` (без требования равенства `certainty` между разными говорящими!) и хотя бы у части `joint=False`:
       - извлечь **только авторский текст вне кавычек** из `anchors` группы и предложения, содержащего `subject_anchor` (используя очистку цитат, чтобы слова внутри прямой речи вроде `‘Let’s go together,’` и соседние предложения того же блока никогда не попадали под матч!);
       - если в этом авторском тексте найден `_JOINT_SPEECH_RE` (`\b(?:together|chorused|in\s+unison|at\s+once|both\s+(?:said|cried|shouted|replied|whispered|asked)|all\s+(?:said|cried|shouted)|(?:said|cried|shouted|asked|replied)\s+[A-Z][\w'’.-]+\s+and\s+[A-Z][\w'’.-]+|[A-Z][\w'’.-]+\s+and\s+[A-Z][\w'’.-]+\s+(?:said|cried|shouted|asked|replied|chorused))\b`) — нормализовать всю группу в `replace(f, joint=True)`.
   - **Атомарное повышение `joint=True`-групп в `_explicit_speaker_if_remark_names` (`L3597-3609, L4055-4069`):**
     - Для фактов с `joint=True`, делящих один `subject_anchor`: вычислять кандидатов повышения для всех участников группы; применять повышение `inferred → explicit` **только если все участники группы** стали `explicit`, иначе оставлять всю группу в `inferred` (предотвращает рассогласование и скрытие второго говорящего в `explicit_parts` `layout/book.py`).
   - В `_conflicting_subject_block_ids` (`L3272-3284`): использовать ту же проверку допустимости совместных `speaker`-фактов, что и в `ContentPlan.__post_init__`.
2. **`context_builder.py` (`_NARRATIVE_ANALYSIS_INSTRUCTIONS`, `L206-207, 216`):**
   - `L206`: явно описать хоровую речь (`joint: true`, по одному факту `speaker` на каждого говорящего с тем же `subject_anchor`; для одиночных реплик и `gender`/`pronoun_referent` всегда `joint: false`).
   - `L207, 216`: явный анти-паттерн: *«в `scenes[].visual_facts[]` поля строго `anchor` (`{block_id, quote}`), `subject` (строка) и `detail` (строка) — никогда не используй `subject_anchor` внутри `visual_facts`»*.

### 1.4 Строгий контракт рендера совместных говорящих (`layout/book.py`, `layout/units.py`, `web/schemas.py`)

1. **`src/linguary/application/layout/book.py` (`_speakers_for_block`, `L957-1030`):**
   - Возвращать `dict[tuple[int, int], str | tuple[str, ...]]`.
   - Группировать `block_facts` по `fact.subject_anchor`:
     - если группа имеет `joint=True` и ≥ 2 разных разрешённых имени: отсортировать их **детерминированно** — по позиции первого упоминания имени (или его токенов) в тексте блока/главы (при отсутствии/равенстве — по `name.casefold()`), и сохранить в `speakers.setdefault(bounds, tuple(ordered_names))`;
     - для обычных `joint=False` фактов сохранять строго прежнее правило: `speakers.setdefault(bounds, name)` (две разные одиночные реплики в одном юните **никогда** не склеиваются в «Говорят A и B»!).
2. **`src/linguary/application/layout/units.py` (`L48-149, 252-270, 429-443`):**
   - **`LayoutUnit.speaker` сохраняет тип `str | None`** (полная совместимость с `ResultUnitResponse.speaker: str | None = None` в `web/schemas.py:727-752` и `ty`!).
   - В `block_units` (`L68`) и `_speakers_by_part` (`L252`): расширить тип аргумента `speakers: Mapping[tuple[int, int], str | tuple[str, ...]]`.
   - Добавить чистый хелпер `_format_speaker_names(entry: str | tuple[str, ...]) -> str`:
     - `str` → `entry`;
     - 1 элемент → `entry[0]`;
     - 2 элемента → `f"{entry[0]} и {entry[1]}"`;
     - 3+ элементов → `f"{', '.join(entry[:-1])} и {entry[-1]}"`.
   - В `block_units` (`L131-141`):
     ```python
     speaker_entry = speaker_names[index]
     speaker = None if speaker_entry is None else _format_speaker_names(speaker_entry)
     if speaker_entry is not None and translation is None:
         part_notes.append(_speaker_note(speaker_entry))
     ```
   - В `_speaker_note(entry: str | tuple[str, ...])` (`L429-443`):
     - если `isinstance(entry, str)` или `len(entry) == 1` → `body=(f"Говорит {entry if isinstance(entry, str) else entry[0]}",)`;
     - иначе → `body=(f"Говорят {_format_speaker_names(entry)}",)`.

### 1.5 Тестовая матрица WP1
- `tests/unit/application/test_analysis_merge.py`:
  - `ContextFact(..., joint=True)` допустим для `speaker` (`explicit`/`inferred`), запрещён для `gender`/`pronoun_referent`/`unknown`; `joint=False` не меняет `to_dict()`, `canonical_id` и `ContentPlan.sha256`;
  - полный сквозной прогон `validate_narrative_response` → `merge_narrative_fragments` → `combine_staged_analysis_responses` → `content_plan_from_response` для `‘Hermione!’ they said together.` и `‘Harry Potter,’ chorused the twins.` (с `joint=True` и с защитной нормализацией);
  - негативные тесты `_JOINT_SPEECH_RE`: `‘Let’s go together,’ said Harry.` (слово внутри кавычек) и `‘Hermione!’ they said together. ‘Run!’ said Harry.` (соседнее предложение) НЕ маскируют реальный конфликт атрибуции;
  - атомарность `_explicit_speaker_if_remark_names` для `joint=True`-группы (один участник в ремарке, второй местоимением → вся группа остаётся `inferred`);
  - round-trip `encode_pass`/`decode_pass` и `_encode_content_plan`/`_decode_content_plan` (со старыми записями без `'joint'`, с `joint=True` и отказ `_boolean` на `0`/`1`).
- `tests/unit/llm/test_llm_response_schemas.py` и `tests/unit/llm/test_analysis_segment_contract.py`: синхронизировать ассерты `QuotedContextFactSchema` (`joint`).
- `tests/unit/application/layout/test_book_layout.py` и `tests/integration/web/`:
  - совместная реплика рендерится как `«Говорят Гарри Поттер и Рон»` в детерминированном порядке;
  - две разные `joint=False` реплики в одном юните НЕ склеиваются в «Говорят …»;
  - `LayoutUnit.speaker` имеет тип `str`, `ResultUnitResponse.of_unit(unit)` валидируется без ошибок Pydantic.

---

## WP2 — Байтовые бюджеты промптов `narrative` / `correction` и батчинг `chapter_correction_review`

### 2.1 `src/linguary/application/context_builder.py`

1. **Константы байтового бюджета (`L115-120`):**
   ```python
   DEFAULT_MAX_ACCUMULATED_CANDIDATES = 200
   DEFAULT_MAX_BACKFILL_BYTES = 12_000
   DEFAULT_MAX_OPTIONAL_CONTEXT_BYTES = 24_000
   ```
2. **Разделение обогащения совпавших терминов и компактного бэкфилла (`_chapter_input` `L1940-1985`, `_backfilled_prior_candidates` `L2027-2066`, `_name_context` `L3341-3405`):**
   - В `_backfilled_prior_candidates`: разделить возврат на `(enrichment_for_selected, compact_backfill)`:
     - `enrichment_for_selected` — записи из `raw_accumulated` с `key in used_keys`, которые по-прежнему добавляются в `optional_items` для обогащения совпавших `TermConcept(SUGGESTED)` полными `note`/`description`/`opaque_name=false` (сохраняет `test_context_builder.py:3110-3116`);
     - `compact_backfill` — действительно **не упомянутые** в текущем фрагменте ключевые кандидаты прошлых глав (`_KEY_GLOSSARY_BACKFILL_TYPES`, в порядке первых появлений).
   - В `_name_context(..., compact_backfill=(), max_optional_bytes=DEFAULT_MAX_OPTIONAL_CONTEXT_BYTES, max_backfill_bytes=DEFAULT_MAX_BACKFILL_BYTES)`:
     - `optional_items` (совпавшие с фрагментом) выводятся в полном формате в пределах `max_optional_bytes` (при превышении сначала опускается `description`, затем `note`, сохраняя `source [kind] -> translation`);
     - `compact_backfill` выводится в конце секции «Необязательные подсказки» (после совпавших записей) компактными строками без `note` и без `description`:
       `- {source} [{kind}] -> {translation}` (плюс `; непрозрачное=false` при `opaque_name is False`), с жёсткой отсечкой по `max_backfill_bytes` в UTF-8 (`len(line.encode("utf-8"))`).
3. **Дедупликация `_fragment_findings_candidates` (`L1970-1978, 3208-3231`):**
   - Прокинуть `already_shown_keys` (из `decisions` и `optional_items` текущего `_name_context`): выводить в полном виде только новые сущности текущей главы, не дублируя уже показанные.
   - В `analyze_chapter.py:966-991`: дедуплицировать строки `instruction_lines` (`dict.fromkeys`), ограничить `findings.instructions` потолком 4 000 байт UTF-8 и брать не более 8 последних `visual_facts` у `last_scene`.

### 2.2 `src/linguary/application/use_cases/analyze_chapter.py` и `retry_policy.py`

1. **Компактные `repair_turns` с единым порогом `65_000` байт UTF-8 (`analyze_chapter.py:1235-1241`, `retry_policy.py:223-245`):**
   - В `_request_staged`: вычислять:
     ```python
     candidate_raw = (response.raw_content if response is not None else None) or getattr(error, "raw_content", None)
     existing_turns_bytes = sum(
         len((a or "").encode("utf-8")) + len(instr.encode("utf-8"))
         for a, instr in request.repair_turns
     )
     base_bytes = len(request.instructions.encode("utf-8")) + len(request.input_text.encode("utf-8")) + existing_turns_bytes
     raw_bytes = len((candidate_raw or "").encode("utf-8"))
     omit_assistant = bool(candidate_raw) and (raw_bytes > 20_000 or base_bytes + raw_bytes > 65_000)
     ```
   - При `omit_assistant is True`: сбрасывать `decision = replace(decision, assistant_response=None)` **после** `self._retry_policy.decide(...)` (либо добавить параметр `include_assistant_response: bool = True` в `RetryPolicy.decide`), чтобы `retry_policy.py:242` (`raw_content or getattr(error, "raw_content", None)`) не восстановил 40 КБ JSON при `LLMResponseFormatError`!
2. **Сплит фрагмента при `LLMContextCapacityError` (`analyze_chapter.py:100, 1147-1158, 1231-1234`):**
   - Добавить `LLMContextCapacityError` в импорт на `analyze_chapter.py:100`.
   - В `_request_staged` (`L1231-1234`):
     ```python
     if (isinstance(error, LLMResponseFormatError) and error.truncated) or isinstance(error, LLMContextCapacityError):
         cast(Any, error).analysis_attempts = attempts
         err_usage = getattr(error, 'usage', None)
         combined_usages = [*usages, err_usage] if (response is None and err_usage is not None) else usages
         cast(Any, error).usage = _total_usage(combined_usages)
         raise
     ```
   - В **`_run_staged_fragments`** (`L1147-1158`):
     ```python
     except (LLMResponseFormatError, LLMContextCapacityError) as error:
         is_trunc = isinstance(error, LLMResponseFormatError) and error.truncated
         is_cap = isinstance(error, LLMContextCapacityError)
         if not (is_trunc or is_cap) or len(fragment.blocks) <= 1:
             raise
         attempts += getattr(error, 'analysis_attempts', 1)
         err_usage = getattr(error, 'usage', None)
         if err_usage is not None:
             usages.append(err_usage)
         midpoint = len(fragment.blocks) // 2
         queue[:0] = [
             (ChapterFragment(0, fragment.blocks[:midpoint]), (byr_quota + 1) // 2, (note_quota + 1) // 2),
             (ChapterFragment(0, fragment.blocks[midpoint:]), byr_quota // 2, note_quota // 2),
         ]
         continue
     ```
     *(При `len(fragment.blocks) == 1` ошибка чисто пробрасывается наружу без повторных попыток с раздутым `repair_turns`.)*

### 2.3 `src/linguary/application/use_cases/correct_chapter.py`

1. **Безусловная очистка и оконная фильтрация `scenes` (`_filtered_chapter_context` `L200-277`, `_llm_chapter_analysis` `L461-575`):**
   - В `_filtered_chapter_context`: построить `block_order = {b.id: idx for idx, b in enumerate(ordered_blocks)}` по **`ordered_blocks`** (`section.ordered_blocks`, а не только `inputs.blocks`!) и передать `block_order=block_order` в `_llm_chapter_analysis`.
   - В `_llm_chapter_analysis(..., block_order: Mapping[str, int] | None = None)`:
     - Если в `cp` есть `'scenes'` (`list | tuple`):
       - при `allowed_block_ids is not None` сначала (по каноническим `block.id` **ДО** замены на алиасы!) отфильтровать сцены по пересечению `[start_pos..end_pos]` с позициями `allowed_block_ids` в `block_order` (при отсутствии `block_order` — по прямому вхождению `start_block_id`/`end_block_id` в `allowed`);
       - **безусловно** (даже если `allowed_block_ids is None`, когда чанк покрывает всю главу!) заменить каждую оставленную сцену компактным словарём строго из 5 ключей: `start_block_id` (через `aliases`), `end_block_id` (через `aliases`), `summary`, `location`, `participants` — полностью удаляя `id`, `order`, `visual_facts`, `visual_assumptions`. Если список пуст — удалить `'scenes'`.
2. **Точный поблочный байтовый чанкинг (`_chunk_correction_blocks`, `L303-335`):**
   - Константы:
     ```python
     DEFAULT_CORRECTION_MAX_REQUEST_BYTES = 72_000
     _CORRECTION_FIXED_OVERHEAD_BYTES = 22_000  # instructions + JSON schema + filtered glossary + compact scenes
     ```
   - Сигнатура (сохраняет полную обратную совместимость с существующими тестами):
     ```python
     def _chunk_correction_blocks(
         block_ids: Sequence[str],
         inputs: Any,
         max_chars: int | None,
         *,
         max_request_bytes: int | None = None,
         candidates_by_block: Mapping[str, Any] | None = None,
     ) -> tuple[tuple[str, ...], ...]:
     ```
   - Поблочный расчёт байтового веса (`_estimate_block_bytes(bid)`):
     - **не** вычитать `chapter_analysis` всей главы целиком! Вместо этого проиндексировать байтовый вес per-block записей `chapter_analysis` (`treatment_plan.items`, `difficulties` по `owner_block_id`, `context_facts` по `subject_anchor.block_id`) один раз за O(N);
     - в режиме `propose` (`candidates_by_block is None`): `len(block.text.encode()) * 2 + serialized_result_bytes(inputs.results[bid]) + per_block_analysis_bytes[bid] + 256`;
     - в режиме `review` (`candidates_by_block` задан): `len(block.text.encode()) * 2 + serialized_candidate_review_bytes(candidates_by_block[bid]) + per_block_analysis_bytes[bid] + 384` (учитывает одновременно `current_result`, `proposed_result`, `changes` и `issues`).
   - Закрывать чанк при превышении `max_chars` (по `block.text`) **либо** `_CORRECTION_FIXED_OVERHEAD_BYTES + current_bytes + block_bytes > max_request_bytes` (гарантируя минимум 1 блок на чанк).
3. **Батчинг `chapter_correction_review` по главе (`CorrectChapterV2.__call__` `L1483-1495`, `_review_group` `L1750-1853`):**
   - В `CorrectChapterV2.__call__`:
     - Объединять всех текущих `pending` кандидатов главы (из любых `group_id`) в порядке блоков главы (если на один `block_id` > 1 кандидата, брать первый в эту волну, остальные — в следующую итерацию `while True`);
     - Нарезать через `_chunk_correction_blocks(tuple(by_block), inputs, max_chars, max_request_bytes=DEFAULT_CORRECTION_MAX_REQUEST_BYTES, candidates_by_block=candidates_by_block)`;
     - Аналогично в ветке `elif missing:` передавать `max_request_bytes=DEFAULT_CORRECTION_MAX_REQUEST_BYTES`.
   - В `_review_group` (`L1763`): во второй вызов `_chunk_correction_blocks` также прокинуть `max_request_bytes=DEFAULT_CORRECTION_MAX_REQUEST_BYTES, candidates_by_block={c.block_id: c for c in candidates}`!
   - В `_review_group` (`L1843-1851`): после цикла `record_correction_verdict` (`L1815-1820`), для каждого `group_id in {c.group_id for c in live}`:
     - получить `group = tx.work.get_binding('correction_group', group_id)['payload_json']`;
     - проверить по `group['candidate_ids']` через `tx.work.get_correction_candidate(cid)`: остались ли в этой группе текущие кандидаты (`candidate_is_current(tx, after, c)`) со `state == 'awaiting_review'`;
     - только если не осталось — проверить контрольные суммы и записать `CorrectionCheck(..., 'unchanged')` для `group['unchanged_block_ids']`.

### 2.4 Тестовая матрица WP2
- `tests/unit/application/test_context_builder.py`:
  - обновить `L3119` (`Alexandra` compact backfill без `description`), убедившись, что `L3110-3116` (`Allie`/`Lief`, обогащение совпавших `TermConcept(SUGGESTED)`) остаются зелёными;
  - тест на 200 прошлых кандидатов с длинными `note`/`description`: совпавшие сохраняют полные метаданные, несовпавшие укладываются в `DEFAULT_MAX_BACKFILL_BYTES`, итоговый промпт < 45 КБ UTF-8;
  - `test_baseline_b2_templates_sha256_unchanged` зелёный.
- `tests/unit/application/test_staged_analyze_chapter.py`:
  - при `LLMContextCapacityError` (без атрибута `.usage`) фрагмент из > 1 блока делится пополам без `AttributeError` и с сохранением накопленного `usage`; при 1 блоке ошибка пробрасывается;
  - при `LLMResponseFormatError` с 40 КБ `error.raw_content` ремонтный запрос не содержит 40 КБ JSON (`assistant_response is None`).
- `tests/unit/application/test_correction_inputs.py`, `tests/integration/processing/test_segment_correction_lifecycle.py`, `tests/e2e/test_correction_v2_pipeline.py`:
  - `_llm_chapter_analysis` очищает `scenes` от `visual_facts`/`visual_assumptions`/UUID **и** при `allowed_block_ids=None`, **и** с фильтрацией по `section.ordered_blocks` при заданном окне;
  - `_chunk_correction_blocks` с `max_request_bytes` не деградирует до 1 блока на запрос при крупном `chapter_analysis` главы и корректно учитывает двойной вес кандидатов в режиме `review`;
  - `CorrectChapterV2` объединяет 3 кандидатов из 3 разных `group_id` в 1 вызов `chapter_correction_review` и корректно фиксирует `unchanged_block_ids` (в т. ч. при разбиении 1 группы на 2 чанка ревью).

---

## WP3 — Пакетный валидатор сегментов, анти-спойлер и устойчивый кэш фрагментов при `retry`

### 3.1 Пакетный валидатор педагогического плана (`analysis_merge.py:262-350`)

- В `_plan_item_from_response(..., alias: str | None = None)`:
  - `label = alias or block_id`;
  - при `len(spans) < 2`: если `all(segment.treatment is item.treatment for segment in item.segments)` — вернуть `TreatmentPlanItem(block_id=block_id, treatment=item.treatment)` без ошибки; иначе — ошибка с `label`;
  - при `TreatmentPlanError as err` (`L344-349`): заменять канонический `block_id` на `label` в `str(err)`.
- В `plan_items_from_response`:
  - сохранить `if response_item is None: if require_all: raise KeyError(alias); continue`;
  - собирать все `AnalysisMergeError` по всем блокам чанка в список `(message, block_id)` и выбрасывать единый `AnalysisMergeError("; ".join(...), violation=ValidationViolation(rule_code="invalid_segment_treatment", block_ids=...))`.

### 3.2 Анти-спойлер и кэшируемый даунгрейд (`analysis_merge.py:2873-2932`, `analyze_chapter.py:605-610, 1025-1063, 1137-1243, 3126-3160`)

1. В `_collect_spoiler_name_leaks` (`analysis_merge.py:2914-2931`): добавить для `mapped_facts` пропуск `if any(_same_spoiler_name(fact.value, kn, glossary_candidates) for kn in known): continue`.
2. В `analyze_chapter.py:1025-1039`: включать `section.title` и тексты всех `BlockType.HEADING` секции до начала фрагмента в `fragment_known_names`.
3. В `_downgrade_spoiler_explicit_facts` (`L3126-3160`): добавить `known_names: Iterable[str] = ()` (не понижать имена из `known_names`); передать `known_names=known_names` и в вызов уровня главы (`analyze_chapter.py:605-610`).
4. **Возврат пониженного `payload` наружу (`analyze_chapter.py:1123-1243`):**
   - В пофрагментном `validate(payload, ...)` возвращать `(normalized_payload, resolved)` (либо выполнять пофрагментный даунгрейд `spoiler_name_leak` в явном шаге нормализации, возвращающем обновлённый `payload`), чтобы `_request_staged` вернул именно **пониженный** `payload` для `_save_pass` (`L1166`) и `parts.append` (`L1174`), и тот же путь работал при `from_cache=True` (`L1137`).

### 3.3 Устойчивый фрагментный кэш при `retry` (`retry_task.py:40-85`, `analyze_chapter.py:1085-1099, 1123-1138, 1245-1255`, `cli/app.py`)

1. **`RetryTaskCommand` (`retry_task.py:41-46`):** добавить `clear_cache: bool = False`. В `RetryTask.__call__` (`L73-82`) вызывать `delete_fragment_passes_for_section` **только** при `command.clear_cache is True` (на `ANALYZE_CHAPTER` с `scope_id`). В `cli retry` добавить флаг `--clear-cache` (Web HTTP API `POST /tasks/{id}/retry` в этом скоупе сохраняет `clear_cache=False`).
2. **Авто-самовосстановление при битом кэше или сбое инкрементального слияния (`analyze_chapter.py`):**
   - На `L1118-1138` (`_cached_pass` / `validate(payload, from_cache=True)`): обернуть чтение/декодирование/валидацию кэша в `try ... except (ValueError, PersistenceError, ApplicationError, DomainError, ResultValidationError):` → вызвать `self._clear_fragment_passes(command.book_id, command.section_id, pass_kind)` и трактовать как `cached_frag = None` (cache-miss с переходом к нормальному запросу LLM).
   - На `L1091` (инкрементальное `merge_chapter_analysis([*parts, candidate_part], ...)`): при `AnalysisMergeError` / `DomainError` помечать конфликт межфрагментного слияния так, чтобы при исчерпании попыток фрагмента очищались фрагментные кэши текущего прохода до перевода задачи в `FAILED`.

### 3.4 Тестовая матрица WP3
- `tests/unit/application/test_analysis_merge.py`: нормализация согласованных односегментных `segments`; пакетная ошибка по `b16, b28, b51` без сырых UUID (включая `TreatmentPlanError` и сохранение `require_all=True`); пропуск `known_names` для `mapped_facts` (`Dudley`, `Sorting Hat`, `Professor McGonagall`).
- `tests/unit/application/test_staged_analyze_chapter.py`:
  - пофрагментный даунгрейд `explicit → inferred` сохраняет в `analysis_pass_results` и `parts` именно пониженный `payload` (повторное чтение из кэша проходит без ошибки);
  - битый/невалидный кэш фрагмента автоматически сбрасывается и переходит в cache-miss; сбой инкрементального `merge_chapter_analysis` очищает фрагментный кэш;
  - вызов уровня главы `L605` не понижает уже известные имена из `known_names`.
- `tests/integration/processing/test_retry_task_recovery_guards.py`: обновить `test_retry_task_analyze_chapter_fragment_cache_cleared_on_validation_preserved_on_transient` (обычный `retry` сохраняет кэш и при `VALIDATION`, `clear_cache=True` очищает фрагменты, сохраняя целый `NARRATIVE`).
- `tests/integration/cli/test_cli.py`: тест `cli retry <task_id> --clear-cache`.

---

## WP4 — Обязательная защита ключевых имён глоссария и грамматические правила анализа

### 4.1 `src/linguary/application/glossary_review.py` и `cli/app.py`

1. **`GlossaryReviewItem` (`L75-153`):**
   - Добавить поле `exempt_type_review: bool = False`, заполнять из `eligible` (`L436-448`).
   - Константа `_AUTO_MANDATORY_TYPES = frozenset({ConceptType.PERSON, ConceptType.PLACE, ConceptType.ORGANIZATION, ConceptType.FICTIONAL_CONCEPT})` и свойство:
     ```python
     @property
     def is_auto_mandatory(self) -> bool:
         """Key entity exempted from manual review by NameTranslationPolicy whose translation is enforced."""
         if self.decision is not None or not self.exempt_type_review:
             return False
         if self.type not in _AUTO_MANDATORY_TYPES:
             return False
         if not self.suggested_translation.strip():
             return False
         return self.is_frequent or self.mention_count >= 2 or self.chapter_count >= 2
     ```
2. **`glossary_snapshot` и `_snapshot_entry` (`L255-269, 550-571`):**
   - Добавить `promote_exempt: bool = True`.
   - Если `item.decision is None` и `promote_exempt and item.is_auto_mandatory`:
     ```python
     return GlossarySnapshotEntry(
         source_text=item.source_text,
         translation=item.suggested_translation,
         type=item.type,
         status=TermStatus.APPROVED,
         note="",  # непроверенная модельная suggested_note не переносится в mandatory constraints!
     )
     ```
     *Защита от дедлока `block_processing.py:397`:* пустой `note=""` гарантирует, что `glossary_context_constraints` не получит противоречивую модельную гендерную помету из другой главы, при этом канонический перевод ключевого имени защищён в `mandatory_entries` промпта и корректора!
   - Если пользователь нажал `SKIP` (`item.decision` имеет `TermStatus.SUGGESTED`), `item.decision is not None` ⇒ `is_auto_mandatory is False` ⇒ запись остаётся `TermStatus.SUGGESTED`.
3. **`src/linguary/cli/app.py` (`glossary list`):** добавить поле `auto_mandatory` и сводку счётчиков (`decided`, `auto_mandatory`, `awaiting_review`, `suggested`).

### 4.2 `src/linguary/application/context_builder.py`, `adapters/config/settings.py`, `analyze_chapter.py`

- В `_NARRATIVE_ANALYSIS_INSTRUCTIONS` (`context_builder.py:198, 207`): закрепить норму склонения мужских титулов/должностей (`профессор`, `доктор`, `директор`, `инспектор`) перед женской фамилией (`голос профессора Макгонагалл`, `к профессору Макгонагалл`), отслеживание второстепенных сквозных персонажей и смысловую/пояснительную обработку говорящих кличек животных.
- В `adapters/config/settings.py:329`: уточнить описание `language_notes_per_chapter` (базовая норма с авто-масштабированием до 18 на длинных главах). В `analyze_chapter.py` логировать эффективный лимит заметок главы (`effective_language_notes_limit`).

### 4.3 Тестовая матрица WP4
- `tests/unit/domain/test_glossary_review.py`, `tests/unit/application/test_book_glossary.py`, `tests/unit/application/test_approve_glossary.py`, `tests/unit/application/test_name_translation_analysis.py`, `tests/unit/application/test_name_translation_evaluation.py`:
  - `exempt_type_review=True` для `PERSON`/`PLACE`/`ORGANIZATION`/`FICTIONAL_CONCEPT` (`is_frequent` или `>= 2` упоминаний/глав) попадает в `glossary_snapshot().mandatory_entries` со статусом `TermStatus.APPROVED` и `note=""` (не вызывая `glossary_fact_conflict` в `validate_fact_glossary` при противоречивой `suggested_note`!);
  - `mention_count == 0` при `is_frequent=False`, пустой `suggested_translation`, разовый `TERM` и пользовательский `SKIP` (`TermStatus.SUGGESTED`) остаются `TermStatus.SUGGESTED`;
  - синхронизировать существующие тесты `glossary_snapshot`, если какой-то из них ожидал `SUGGESTED` у `exempt_type_review`.
- `tests/unit/application/test_context_builder.py` и `tests/integration/cli/test_cli.py`: правило склонения титулов, зелёный `test_baseline_b2_templates_sha256_unchanged`, вывод `auto_mandatory` в `cli glossary list --json`.

---

## WP5 — Педагогическая калибровка A2, произношение, анти-дубли глосс, `<sup>`, «Говорит X» и `legend.xhtml`

### 5.1 Автоопределение варианта (`uk`/`us`) и UX произношения для A2

1. **`src/linguary/domain/document.py`:** чистая функция `detect_english_variant(raw_language: str | None) -> Literal["uk", "us"] | None` (`en-gb`/`en-uk`/`en-au`/`en-nz`/`en-ie` → `"uk"`, `en-us`/`en-ca` → `"us"`, иначе `None`).
2. **`src/linguary/adapters/epub/inline.py` (строго в `adapters/epub/`!):** хелпер `read_source_language_tag(source_epub_path: str | None) -> str | None` (читает первый `dc:language` из OPF архива, при ошибке возвращает `None`).
3. **`src/linguary/cli/app.py` (`L144-149, 480-508`):**
   - В уже существующий аргумент `--pronunciation-variant` добавить `"auto"` в `choices=["uk", "us", "auto"]` с описанием fallback (`"us"` при `en` без регионального тега).
   - В `_analyze` **до** вызова `application.book_conditions.ensure_initial_for_analysis(...)`: если `pronunciation_variant == "auto"`, загрузить `book.source_epub_path` и вычислить `detect_english_variant(read_source_language_tag(book.source_epub_path)) or "us"` (потому что `book_conditions.py:235, 429` принимает строго `{None, 'uk', 'us'}`).
4. **Web UI (`src/linguary/web/routes/books.py:112-143`, `web/templates/books/detail.html:34-55` — НЕ затрагивая `partials/quality_review.html:70-76`!):**
   - В `get_book`: вызывать `read_source_language_tag(source_epub_path)` **строго** при `wants_html(request) and response.book.status == "IMPORTED" and not response.conditions.conditions_id and response.book.source_language == "en"` (нулевая нагрузка при polling `auto_refresh=True`!).
   - Передать `recommended_pronunciation_variant = detect_english_variant(raw_lang) or "us"` в `detail.html`: при `default_reader_level == "a2"` (и отсутствии зафиксированных условий) предвыбирать чекбокс `pronunciation_help_enabled` и `recommended_pronunciation_variant`, а при смене `<select name="reader_level">` синхронизировать чекбокс (пока пользователь не кликнул его вручную).

### 5.2 Порог перегрузки `GLOSSARY_ONLY` (блоки + сегменты) и калибровка A2 (`analysis_merge.py`, `context_builder.py`)

1. **`src/linguary/application/analysis_merge.py` (`L4832-4835, 6012-6035`):**
   - Константа `_NARRATIVE_GLOSSARY_NON_CORE_WORD_LIMIT = 4`.
   - Добавить шаг `_promote_overloaded_glossary_only(item, block, known_names=known_names)` и вызывать его в `normalize_pedagogical_response` **ПОСЛЕ** `_promote_a2_stoplist_item` (`L6022`) и до Step 3 (`L6036`):
     - для целого `GLOSSARY_ONLY`-блока (без `segments`): если число уникальных не-A2 слов (`non_core_a2_words(block.text, known_names=known_names)`) строго `> 4` → повысить `treatment` до `BlockTreatment.TRANSLATE`;
     - для сегментированного блока (`item.segments` непуст): проверить каждый сегмент с `segment.treatment is BlockTreatment.GLOSSARY_ONLY` по его срезу `block.text[span.start:span.end]`: если уникальных не-A2 слов `> 4` → повысить этот сегмент до `TRANSLATE` и пересчитать `strongest_treatment` блока.
2. **`src/linguary/application/context_builder.py` (`L231, 2166-2190`):**
   - В `_PEDAGOGICAL_ANALYSIS_INSTRUCTIONS` (`L231`): ограничить дробление до 1–2 переходов режима на абзац и **запретить разрывать одну реплику персонажа в кавычках пополам между `TRANSLATE` и `GLOSSARY_ONLY`**.
   - В `_reader_level_treatment_guidance("a2")` (`L2166-2190`): разрешить `GLOSSARY_ONLY` для коротких (≤ 10 слов) прозрачных реплик A1–A2 с 1 покрытым глаголом речи (`single_covered_nonbasic`); добавить скрытые ловушки B1 (`great`, `cool`, `work`, `hall`, `have a place at`, `No one ever did`, `Found…!`, `Everyone out!`, `should be celebrating`, `sounding relieved`, `looking tired`).

### 5.3 Анти-дублирование глосс, A1-фильтр и сдвиг `<sup>` (`layout/notes.py`, `a2_core_vocab.py`)

1. **`src/linguary/application/layout/notes.py` (`block_notes`, `L544-626, 1000-1018`):**
   - Добавить `_suppress_hints_covered_by_notes(...)`:
     - подавлять `hint`, только если в том же блоке есть видимая `language_note`, перекрывающаяся по смещению с `hint`, при выполнении хотя бы одного условия: (а) `len(expression_words(note.original_excerpt)) <= 6`, (б) общие `difficulty_id`, (в) перевод глоссы покрыт текстом заметки;
     - если у подавленного `hint` было `ipa`, **не** включать этот `hint` в `excluded_hints` (чтобы `PronunciationOccurrence` осталось доступно как отдельная фонетическая пометка без потери IPA и без дубля глоссы);
     - проверять инвариант «остаётся ≥ 1 видимая подсказка/заметка» в границах каждого `GLOSSARY_ONLY`-**сегмента** (через `block.segment_spans` и `result.segment_treatments`).
   - **Точный сдвиг `<sup>` до финальной пунктуации:**
     - Заменить `_find_end(text, excerpt)` на `_find_span(text, excerpt) -> tuple[int, int] | None` (сохранив `_find_end` как обёртку для совместимости, если используется снаружи).
     - Добавить `_trim_terminal_punctuation(text: str, start: int, end: int) -> int`:
       - если `end <= start` → вернуть `end`;
       - проверять с конца среза `text[start:end]`:
         - снимать закрывающую кавычку `”` / `"` (и `’` **строго** если непосредственно перед ней стоит `.`, `!`, `?` или `…` — никогда после буквы вроде `flyin’` / `Dursleys’`!);
         - снимать финальный знак конца предложения `.`, `!`, `?`, `…`, **кроме** случая, когда точка завершает известное сокращение (`_TITLE_ABBREV_RE`: `Mr.`, `Mrs.`, `Ms.`, `Dr.`, `St.`, `Prof.`, `a.m.`, `p.m.` или одиночный инициал `[A-Z].`);
         - если `new_end <= start` (якорь состоял целиком из пунктуации) — вернуть исходный `end`.
     - Применять `_trim_terminal_punctuation` в `_note_draft` (и `_hint_draft` при совпадении с финальной пунктуацией).
2. **`src/linguary/application/a2_core_vocab.py` (`L766-960, 1138-1170`):**
   - Добавить в профильные наборы:
     - `_A1_VERBS`: `enter`, `celebrate`;
     - `_A1_NOUNS` / `_A1_INTERNATIONALISMS`: `bacon`, `mistake`, `darkness`, `uniform`;
     - `_A1_ADJECTIVES_AND_ADVERBS`: `perfect`, `violet`, `unusual`, `angrily` (`wrong` уже присутствует на `L787`);
   - В `is_trivial_a1_hint`: для 2-словного именного словосочетания (без префикса `to `, не содержащего предлогов/частиц) вида `school uniform` возвращать `True`, если оба слова входят в `(_A1_NOUNS | _A1_INTERNATIONALISMS | _A1_ADJECTIVES_AND_ADVERBS) - _NON_CORE_OVERRIDE` (и не являются глагольными фразами).

### 5.4 Безопасное подавление лишнего «Говорит X» (`layout/book.py:923-1030`)

- Добавить `_strip_quoted_speech(text: str) -> str`, согласованный с `_DIALOGUE_QUOTES` (`analysis_merge.py:4811`) с защитой множественного притяжательного `s’ ` (`(?<!\w)’` / учет открытых кавычек и апострофа после `s`).
- Заменить `_part_self_identifies_speaker` на вызов `_part_explicitly_names_speaker(part_text, speaker_tokens, fact_value)` на `L1024`:
  - возвращает `True` при `_part_self_identifies_speaker(part_text, speaker_tokens)`;
  - иначе анализирует `narrative = _strip_quoted_speech(part_text)`:
    - исключает притяжательные формы (`Name’s` / `Name's`) и позицию после предлогов (`\b(?:at|to|for|with|from|about|after|of|behind|beside|toward|towards)\s+`);
    - если имя говорящего содержит титул или имя+фамилию (`Petunia Dursley`, `Mr Dursley`), **запрещает** срабатывание по голой фамилии (`Dursley`), если перед ней в `narrative` стоит другой титул или другое имя (`Mr Dursley` ≠ `Petunia Dursley`);
    - при чистом субъектном упоминании имени говорящего в `narrative` (как `Dudley sat down heavily`) возвращает `True`.
- В `layout/book.py:542, 556` добавить краткий комментарий о намеренно отложенном статусе `who_is_who=()` и `review=()`.

### 5.5 Обновление `legend.xhtml` (`layout/legend.py`, `adapters/epub/book_pages.py`, `writer.py:499`)

1. **`src/linguary/application/layout/legend.py`:**
   - `L97`: `"Пометка в начале примечания называет его вид: "`.
   - `_NOTES` (`L24-27`): добавить разведение «Слово жирным через тире — быстрый перевод; текст без тире — пояснение выражения или грамматики. Не пытайтесь заучивать все подсказки подряд: достаточно понять смысл и читать дальше.»
   - `_SECTION_ORDER` (`L41`): `("previously", "«Previously in…» — короткое напоминание на языке оригинала перед главой")`.
2. **`src/linguary/adapters/epub/book_pages.py:52-68` (`legend_document`) и `writer.py:499`:**
   - Сигнатура:
     ```python
     def legend_document(
         lines: Sequence[str],
         *,
         language: str,
         stylesheet: str,
         ipa_key_href: str | None = None,
         source_language: str | None = None,
         include_example: bool = False,
     ) -> etree._Element:
     ```
   - При `include_example=True` и `source_language == "en"` (выставляется из `writer.py:499` как `include_example=bool((layout.features.has_translation or layout.features.has_notes) and source == "en")`) добавлять в конец `<section id="legend">` компактный нейтральный мини-пример (`.lg-unit` с `.lg-src lang="en"`, `.lg-tr lang="{language}"` и `.lg-notes` с 1 словарной подсказкой `.lg-term` и 1 кратким пояснением).

### 5.6 Тестовая матрица WP5
- `tests/unit/domain/test_canonical_document.py`: `detect_english_variant` (`en-GB`, `en-US`, `en`, `None`).
- `tests/unit/application/test_analysis_merge.py`:
  - блок `GLOSSARY_ONLY` и сегмент `GLOSSARY_ONLY` с ≥ 5 не-A2 словами (в т. ч. попавшие в `GLOSSARY_ONLY` после `_promote_a2_stoplist_item`) повышаются до `TRANSLATE`; эталонные `GLOSSARY_ONLY` из `test_a2_harry_potter_regressions.py` не ломаются.
- `tests/unit/application/layout/test_notes.py` (и проверка `test_a2_harry_potter_regressions.py:2475-2489`):
  - подавление дублей `Fancy`/`Fancy seeing`, `wrong`/`How very wrong he was.`, `to get`/`Make Harry get it.`;
  - НЕ-подавление глоссы под длинной несвязанной заметкой (> 6 слов); сохранение `ipa` при подавлении дубля; сегментный инвариант `GLOSSARY_ONLY`;
  - `_trim_terminal_punctuation`: `How very wrong he was.` → до `.`; `as we was flyin’` → `’` НЕ отрывается; `Mr.` → `.` НЕ отрывается; `‘No.’` → до `.` и `’`.
- `tests/unit/application/test_a2_core_vocab.py`: `is_trivial_a1_hint` для `bacon`, `perfect`, `mistake`, `wrong`, `to enter`, `to celebrate`, `a mistake`, `violet`, `unusual`, `darkness`, `angrily`, `school uniform`.
- `tests/unit/application/layout/test_book_layout.py`:
  - `‘Oh.’ Dudley sat down heavily.` не получает «Говорит Дадли»;
  - контрпримеры сохраняют плашку: `‘Don’t be silly,’ she said, looking at Mr Dursley.` (однофамилец, говорит Петуния), `‘Where is it?’ he asked. Harry’s aunt didn’t answer.` (притяжательное `Harry’s`), `‘Oh,’ he muttered, looking at Dudley.` (предложное дополнение), `‘The Dursleys’ car is outside,’ he said.` (апостроф `Dursleys’` внутри реплики).
- `tests/unit/application/layout/test_legend.py` и `tests/integration/epub/test_learner_book_pages.py`: синхронизировать `legend_lines`, подтвердить зелёный `test_learner_book_pages.py:63-65` (без примера при `_ORIGINAL`+`_IPA`) и добавить тест мини-примера при `has_translation=True, source_language="en"`.
- `tests/integration/web/test_conditions_choices.py` и `tests/integration/cli/test_cli.py`: предвыбор `pronunciation_help` в `detail.html` (не затрагивая `partials/quality_review.html`) и `--pronunciation-variant auto` в `cli analyze`.

---

## WP6 — Лингвистическое усиление промптов перевода, заметок, пересказа и балансировка корректора

### 6.1 `src/linguary/application/reader_text.py` и `src/linguary/application/context_builder.py`

1. **`_A2_NOTE_DISCIPLINE` (`reader_text.py:508-514`):**
   - Добавить правила (явно маркируя ограничения длины «для уровня A2», чтобы не пережимать B2/C1 в общем `_TRANSLATION_CONTEXT_V2`):
     - в заметках `speech_style` («Речь персонажа.») расшифровывать только нестандартные формы оригинала (`flyin’ → flying`, `we was → we were`), не анализируя приёмы русского перевода, и охватывать якорем разбираемую форму либо ставить якорь в конце реплики при разборе нескольких её элементов;
     - для уровня A2 писать грамматические заметки коротко (формула + прямой перевод куска, ориентир 15–30 слов);
     - сверять географический и языковой контекст заметок с текстом сцены (например, португальский/испанский для Бразилии);
     - в блоках `TRANSLATE` не тратить заметки на прозрачные идиомы с прямым русским аналогом и на редкую книжную грамматику C1;
     - в `glossary_hints.translation` указывать точное значение сцены без лишних синонимов из других контекстов.
2. **`_A2_DIALOGUE_AND_CALQUE_RULES` (`reader_text.py:516-530`):**
   - Сквозное сохранение разговорного/просторечного регистра во всех репликах маркированного персонажа (не только в блоке с `speech_style`); единство грамматического рода животных и перевоплощающихся персонажей в пределах сцены/главы; сокращение избыточных притяжательных местоимений (`his/her/their`); запрет вставки словарного толкования реалии внутрь художественной фразы при наличии сноски.
3. **`learner_instructions("a2")` (`reader_text.py:554`):**
   - Согласование времён в `reader_chapter_summary` (`Previously in…`): события, произошедшие *до* описываемой сцены, передавать в прошедшем времени.
4. **`_PEDAGOGICAL_ANALYSIS_INSTRUCTIONS` (`context_builder.py:237-249`):**
   - Приоритетные ловушки A2→B1 (`had had`, `can't/couldn't help + V-ing`, `had better`, редкие стяжения), привязка к первому появлению и баланс бюджета заметок (≤ 50% на `TRANSLATE` при наличии трудных `GLOSSARY_ONLY`).

### 6.2 `src/linguary/composition/correction_runtime.py`

- **`CORRECTION_PROMPT_V2` (`L85-93`) и `CORRECTION_REVIEW_PROMPT` (`L103-116`):**
  - В п. 2 чек-листа добавить очевидные лексико-смысловые сдвиги и грубые бытовые/телесные кальки;
  - Добавить запрет гиперкоррекции времён в несобственно-прямой речи и педантичного дробления авторского продолжения через `<br/>` после закрытой реплики со словами автора при нормативной пунктуации.

### 6.3 Тестовая матрица WP6
- `tests/unit/application/test_context_builder.py` и `tests/unit/composition/test_correction_runtime.py`: проверить новые инструкции и подтвердить зелёный `test_baseline_b2_templates_sha256_unchanged`.

---

## WP7 — Полировка экспорта EPUB, метаданных, CSS и политики front matter

### 7.1 Единая очистка `dc:description` и `dc:rights` (`adapters/epub/inline.py`, `reader.py`, `writer.py`)

- В **`src/linguary/adapters/epub/inline.py`**: добавить `clean_dc_text(raw: str | None) -> str | None` (при наличии HTML-тегов/сущностей парсит безопасным `lxml.html` без сети, извлекает `" ".join(node.itertext())`, сворачивает пробелы).
- Применять `clean_dc_text` во **всех трёх** точках:
  1. `adapters/epub/reader.py:547-566` (`_metadata`);
  2. `adapters/epub/writer.py:705-734` (`_read_source_dc_metadata`);
  3. **`adapters/epub/writer.py:1503-1508` (`_append_metadata`)** — очищает метаданные и для уже импортированных в SQLite книг!

### 7.2 `properties="cover-image"` и чистая обложка `.lg-cover` (`writer.py`, `layout_render.py`, `web/static/linguary.css`)

1. **`_cover_image_href(chapters, document, payloads, media_types) -> str | None` (`writer.py:1462-1535`):**
   - Строить `sections_by_id = {s.id: s for s in document.ordered_sections}`.
   - Обходить `chapters`: пропускать синтетические документы (`linguary-legend`, `linguary-ipa-key`, `linguary-who-is-who`, `linguary-pronunciation-sources`), если `(section := sections_by_id.get(ch.section_id)) is None` или `not document.is_cover_section(section)`.
   - Найти первый локальный `<img src>` или `<svg:image xlink:href>`, разрешить через `_resolve_target`, проверить наличие в `payloads` и `media_type.startswith("image/")`.
   - Передать `cover_image_href` в `_manifest_items`: для `href == cover_image_href` выставить `attributes["properties"] = "cover-image"`.
2. **Чистая обложка `.lg-cover`:**
   - В `_ChapterRenderer`: если `self._document.is_cover_section(section)`:
     - для `BlockType.IMAGE` без текста — рендерить `<div class="lg-cover"><img src="..." alt="..."/></div>` (без `.lg-unit`, `.lg-end`, `<p class="lg-src">`);
     - для `PRESERVE_OPAQUE` обложки (например, `<svg><image .../></svg>` в `_append_opaque`) — заворачивать в `<div class="lg-cover">...</div>` без `.lg-unit`.
   - В `_STYLESHEET` и `web/static/linguary.css` добавить `.lg-cover` и `.lg-cover img { max-width: 100%; max-height: 100%; }`.

### 7.3 Ведущий пустой `pagebreak` и длинные `.lg-unit-long`

1. **`_ChapterRenderer._append_blocks` (`writer.py:992-1039`):**
   - Для сюжетной главы (`not is_front_matter` и `not section.is_notes_only`): проверять первый корневой блок `roots[0]` **ДО** вызова `self._reserve_opaque_ids(blocks, state)`!
   - Если `roots[0]` не имеет детей и является пустым `pagebreak` (`BlockType.PAGE_BREAK`, либо `PRESERVE_OPAQUE` `p.pagebreak`/`div.pagebreak` без текста после удаления пробелов/`\xa0` и без `<img>`/`<svg>`):
     - извлечь все его `id` (из `_anchors(roots[0])` и атрибутов opaque-элемента);
     - исключить `roots[0]` из `blocks_for_reserve` (передаваемого в `_reserve_opaque_ids`) и из `roots`;
     - повесить первый `id` прямо на `container` (`<section epub:type="chapter">`), остальные — в `<div class="linguary-anchors"><span id="..."/></div>` в начале `container`, и зарегистрировать все перенесённые `id` в `state.used_ids`.
2. **`.lg-unit-long` (`layout_render.py:59-77`, `writer.py`, `web/static/linguary.css`):**
   - Единый критерий: `unit.translation is not None` и (`(unit.end - unit.start) + len(unit.translation) + sum(len(p) for n in unit.notes for p in n.note.body) > 450` **или** ≥ 4 предложений в срезе оригинала `block.text[unit.start:unit.end]`) → добавить `"lg-unit-long"` в `classes`.
   - В `_STYLESHEET` и `web/static/linguary.css`: `.lg-unit.lg-unit-long { page-break-inside: auto; break-inside: auto; }`. (`web/templates/partials/result_block.html` не меняется.)

### 7.4 Политика front matter (`writer.py:539-582`)

- В `_exported_sections` заменить безусловное удаление не-обложечного front matter на `_is_dropped_front_matter(section, document)`:
  - `cover` и `part_divider` всегда сохраняются;
  - дублирующие HTML-оглавления (`contents`/`toc`/`table of contents`/`оглавление`/`содержание`) и рекламные/издательские вставки (`also by`, `about the publisher`, `advert`) отбрасываются;
  - `title`, `dedication`, `copyright` сохраняются как `source-only` (без перевода и без `Previously in…`, не смещая нумерацию глав и `previous` даже для front matter в середине multi-book EPUB).

### 7.5 Тестовая матрица WP7
- Синхронизировать **все три** существующих набора тестов front matter:
  1. `tests/integration/epub/test_epub_export_content.py:769`;
  2. `tests/integration/epub/test_learner_book_pages.py:59, 76, 181`;
  3. `tests/unit/test_epub_writer.py:627-642` (`sec-ded` / `dedication.xhtml`).
- Новые тесты в `tests/unit/test_epub_writer.py`, `tests/unit/adapters/epub/test_layout_render.py`, `tests/integration/epub/test_epub_export_content.py`:
  - `clean_dc_text` очищает `<p>Text <em>here</em>` при импорте, fallback-чтении и повторном экспорте из БД (`_append_metadata`);
  - `properties="cover-image"` проставляется в `content.opf` и не падает на книгах без обложки с синтетическими страницами (`linguary-legend`, `linguary-ipa-key`);
  - `.lg-cover` применяется и к `BlockType.IMAGE`, и к opaque SVG-обложке;
  - ведущий пустой `pagebreak` (и `BlockType.PAGE_BREAK`, и opaque `p.pagebreak`) удаляется до `_reserve_opaque_ids`, а его `id` оказывается на `<section epub:type="chapter">` и остаётся валидной целью ссылок;
  - `.lg-unit-long` срабатывает по объёму оригинал+перевод+заметки > 450 или ≥ 4 предложений;
  - multi-book EPUB с `title`/`dedication` между двумя книгами сохраняет корректную нумерацию глав и `previous` для `Previously in…`.

---

## WP8-local — Локальный скилл `linguary-subagent-run` (вне git, `.agents/`)

**Файлы (не отслеживаются git):**
- `.agents/skills/linguary-subagent-run/scripts/bridge_core.py`
- `.agents/skills/linguary-subagent-run/tests/test_subagent_bridge.py`
- `.agents/skills/linguary-subagent-run/SKILL.md`

### 8.1 Изменения в `bridge_core.py`

1. **Байтовый контроль UTF-8 и превентивный потолок:**
   - Константы:
     ```python
     PROMPT_WARN_BYTES = 95_000
     PROMPT_TRUNCATION_RISK_BYTES = 112_000
     PROMPT_HARD_LIMIT_BYTES = 118_000
     ```
   - Добавить `hard_prompt_limit_bytes: int | None = PROMPT_HARD_LIMIT_BYTES` в `SubagentBridgeServer.__init__`.
   - В `handle_completion`: вычислить `prompt_bytes = len(prompt.encode("utf-8"))`, `truncation_risk = prompt_bytes >= PROMPT_TRUNCATION_RISK_BYTES`; при `prompt_bytes >= PROMPT_WARN_BYTES` писать `WARNING` в `log_fn`.
   - Если `self.hard_prompt_limit_bytes is not None and prompt_bytes > self.hard_prompt_limit_bytes`: **до вызова `worker(...)`** немедленно поднять `ContextLengthExceededError(f"prompt_bytes={prompt_bytes} exceeds hard limit {self.hard_prompt_limit_bytes}")` (исключает тихое усечение и сразу запускает сплит чанка в Linguary).
   - В `_record_metric` записывать `prompt_bytes: int`, `truncation_risk: bool`, `repair_errors: list[str]`.
2. **Компактизация JSON Schema (`build_agent_prompt`, `L445-448`):**
   - `compact_schema_for_prompt(schema)` рекурсивно удаляет метаданные `"title"` и служебные `"description"` на узлах схемы, **никогда не удаляя** одноимённые поля внутри словаря `"properties"` (например, `GlossaryCandidateSchema.properties.description`). В `validate_json_schema` передаётся исходная `schema_obj`.
3. **Детекция отказа усечения на всех путях + отсутствие двойной записи метрик (`L952-1072, 1121-1131`):**
   - `class ContextLengthExceededError(BadRequestError): error_code = "context_length_exceeded"`.
   - Проверять `_is_truncated_prompt_refusal(raw)` на **каждом** ответе `worker(...)`: сразу после первого вызова (ДО `if not expect_json:`) и на каждом ремонтном ходе (`L1011`).
   - В `handle_completion`: добавить явный блок **до** `except Exception as exc:`:
     ```python
     except ContextLengthExceededError as exc:
         elapsed = round(time.monotonic() - t0, 2)
         self._record_metric(
             cid=cid, ..., prompt_bytes=prompt_bytes, truncation_risk=True,
             repair_errors=repair_errors, elapsed=elapsed, status="context_length_exceeded",
         )
         self.log_fn(f"[bridge #{cid}] <- 400 CONTEXT_LENGTH_EXCEEDED in {elapsed}s: {exc}")
         raise
     ```
   - В `handle_client`: при `ContextLengthExceededError` отправлять HTTP `400 Bad Request` с `{"error": {"message": str(exc), "type": "invalid_request_error", "code": "context_length_exceeded"}}`.
4. **Фоновый read-only прогресс-монитор (`run_cmd`, `L790-837`):**
   - Расширить сигнатуру: `run_cmd(..., db_path: Path | None = None, progress_interval_seconds: float = 90.0, bridge: SubagentBridgeServer | None = None)`.
   - При `db_path is not None`: запускать фоновую задачу, раз в `progress_interval_seconds` читающую `sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)` (`SELECT task_type, status, COUNT(*) FROM processing_tasks GROUP BY task_type, status`) + счётчики `bridge.metrics` (при `sqlite3.OperationalError` / отсутствии файла тихо пропускать тик) и печатающую 1 компактную строку в `log_fn`, с гарантированной отменой в `finally`.

### 8.2 Тестовая матрица WP8-local
- `.agents/skills/linguary-subagent-run/tests/test_subagent_bridge.py`:
  - `compact_schema_for_prompt` удаляет метаданные `"title"`/`"description"`, но сохраняет свойства `"title"` и `"description"` внутри `"properties"`;
  - превентивный потолок `prompt_bytes > hard_prompt_limit_bytes` немедленно возвращает HTTP `400` `context_length_exceeded` без вызова `worker` и записывает в `bridge_metrics.json` ровно **одну** запись со `status == "context_length_exceeded"`;
  - `_is_truncated_prompt_refusal` срабатывает на первом ответе, в `plain_text`-ветке (`expect_json=False`) и на ремонтном ходе, записывая ровно **одну** запись в `bridge_metrics.json`;
  - `prompt_bytes`, `truncation_risk` и `repair_errors` корректно записываются в `bridge_metrics.json`;
  - фоновый монитор прогресса безопасно читает SQLite в read-only режиме, не падает при отсутствии БД и чисто завершается при отмене.
