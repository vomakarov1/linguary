# SDD (Backend): Контракты, алгоритмы и тестовая матрица (Everlost pipeline & illustration hardening)

**Дата:** 2026-11-26  
**Статус:** Approved (актуализировано по итогам всех 3 независимых ревью)  
**Связанные документы:** `BRIEF-everlost-pipeline-and-illustration-hardening-2026-11-26.md`, `ADR-everlost-pipeline-and-illustration-hardening-2026-11-26.md`

---

## Глобальные ограничения

- **Границы слоёв:** `domain ← application ← adapters / web / cli` (`uv run pytest tests/architecture`). В `domain` и `application` нет сторонних библиотек и прямого I/O.
- **Схема БД:** 0 миграций Alembic. Новые поля хранятся в JSON (`payload_json`) с полной обратной совместимостью при чтении старых записей.
- **Замороженные шаблоны:** `_TRANSLATION_INSTRUCTIONS`, `_SEGMENT_TRANSLATION_RULE` и `snapshot_prompt_templates(..., reader_level="b2")` сохраняют точные SHA-256 (`test_baseline_b2_templates_sha256_unchanged`).
- **Универсальность:** никаких хардкодов под конкретные книги, персонажей, локации или модели LLM.
- **Неизменяемые контракты:** `temperature=policy.temperature` в `_clean_llm_settings()`, `_exported_sections()` (`title`/`dedication`), `.lg-notes p { display: inline; margin: 0; }`, покрытие `ChapterCorrectionV2Schema`, `test_ambiguous_identity_remains_separate` и `test_transitive_identity_conflict_triangle_order_independent_deterministic` сохраняются.

---

## WP1 — Анализ глав: `spoiler_name_leak`, фильтрация дескрипторов, кэш-дедлок и автосинтез имён

### 1.1 `src/linguary/application/context_builder.py`
В `_NARRATIVE_ANALYSIS_INSTRUCTIONS` (`L197-205`):
- Строка `L200`: заменить `Описательные прозвища (например, «the freckled boy» и т. п.) прозрачны (opaque_name=false) и при переводе по смыслу получают applied_method="meaning".` на пример с капитализированным устойчивым прозвищем (например, `«Iron Hand»`), добавив явное правило: *«Обычные строчные описательные обозначения неназванных персонажей (`the boy`, `the younger boy`, `the stranger` и т. п.) НЕ включай в `glossary_candidates`; используй их только как временные метки в `scenes[].participants` и `context_facts` до первого появления имени в тексте.»*
- Обновить тест `tests/unit/application/test_context_builder.py:3737` под новый пример (`"Iron Hand"` вместо `"the freckled boy"`).

### 1.2 `src/linguary/application/analysis_merge.py`
1. **Обобщение `_is_generic_person_candidate(source_text: str) -> bool` (`L2221-2238`):**
   - Сохранить первые два существующих шаблона `_GENERIC_PERSON_CANDIDATE_PATTERNS[0]` (`^(?:the|a|an)\s+[a-z]+$`) и `[1]` (`of ...`), чтобы не потерять фильтрацию `"the baker"`, `"the giant"` и т. п.
   - Ввести `_GENERIC_PERSON_HEADS = _GENERIC_ROLE_PARTICIPANTS | frozenset({"kid", "kids", "teen", "teenager", "youth", "figure", "voice", "passenger", "pedestrian", "owner", "clerk", "guard", "driver", "keeper", "stranger", "goblin", "wizard", "witch", "person", "child", "children"})`.
   - Для полностью строчной `source_text` (`not any(ch.isupper() for ch in stripped)`): если сработал один из сохранённых шаблонов `[0..1]`, либо `tokens = re.findall(r"[a-z]+(?:-[a-z]+)*", stripped)` имеет длину **`2 <= len(tokens) <= 4`** и последний токен (`tokens[-1]`) входит в `_GENERIC_PERSON_HEADS` → вернуть `True`.
   - Обновить тесты `tests/unit/application/test_analysis_merge_scenes.py:1465-1484` и `1778-1827`: перенести `'the freckled boy'` в отклоняемые строчные родовые кандидаты и заменить разрешённый пример на капитализированное описательное прозвище (`'Iron Hand'`).
2. **Монотонность `_first_name_block_index(...)` (`L2738-2816`):**
   - Вычислить `matches_capitalized_glossary: bool` по **той же логике сопоставления, что `_matches_glossary_name()`** (`analysis_merge.py:2610-2639`, включая `form_stems <= name_stems` / `name_stems <= form_stems`) — истинно, если среди совпавших с `name` `GlossaryCandidateResponse` есть кандидат, чей `source_text` содержит хотя бы одну заглавную букву (`any(ch.isupper() for ch in candidate.source_text)`). Это гарантирует сохранение `test_analysis_merge_scenes.py:1214` для `'the dark voldemort'` при кандидате `'Voldemort'`.
   - На `L2767-2771`: после вычитания `_ANONYMOUS_PARTICIPANT_MODIFIERS`, **только если `not all_lowercase`** и `len(stems) >= 2`, проверить `distinctive = stems - _GENERIC_ROLE_PARTICIPANTS`; если `distinctive` непусто — использовать `stems = distinctive`. (Для строчных фраз вроде `"younger boy"` `boy` остаётся в `stems`, чтобы не ломать раннее совпадение и не провоцировать ложный спойлер по `"younger"`.)
   - На `L2785`: заменить `prefer_capitalized = (all_lowercase and matches_glossary) or any(_prefers_capitalized_name_match(form) for form in forms)` на:
     ```python
     prefer_capitalized = (
         all_lowercase
         and matches_capitalized_glossary
         and not _is_generic_person_candidate(name)
     ) or any(_prefers_capitalized_name_match(form) for form in forms)
     ```
     Это сохраняет детекцию `'the dark voldemort'` (`matches_capitalized_glossary=True`, `test_analysis_merge_scenes.py:1214`) и полностью исключает скачок `first_pos` с блока 2 на блок 38 (`"Chocolate Boy"`) для строчных родовых фраз.

### 1.3 `src/linguary/application/use_cases/analyze_chapter.py` и `retry_task.py`
1. **Очистка фрагментного кэша на слиянии (`analyze_chapter.py:575-590` для `NARRATIVE`, `L688-706` для `PEDAGOGICAL`):**
   - Добавить приватный метод `_clear_fragment_passes(self, book_id: str, section_id: str, pass_kind: AnalysisPassKind) -> None`:
     ```python
     with self._unit_of_work.transaction() as tx:
         delete_fn = getattr(tx.analyses, "delete_fragment_passes_for_section", None)
         if callable(delete_fn):
             delete_fn(book_id, section_id, pass_kind)
     ```
   - Обернуть `merge_narrative_fragments` + `validate_narrative_response` (`L575-590`) в `try ... except (AnalysisMergeError, DomainError): self._clear_fragment_passes(command.book_id, command.section_id, AnalysisPassKind.NARRATIVE); raise`.
   - Обернуть `merge_chapter_analysis(...)` + `pedagogical_response(...)` + `select_staged_a2_summary(...)` + **`self._assemble_staged(...)`** (`L688-706`) в `try ... except (AnalysisMergeError, DomainError): self._clear_fragment_passes(command.book_id, command.section_id, AnalysisPassKind.PEDAGOGICAL); raise` (общеглавный кэш `NARRATIVE` со `scope_key == ""` при этом не трогается).
2. **Точечная очистка фрагментного кэша при `RetryTask` (`retry_task.py:54-75`):**
   - Внутри транзакции `RetryTask.__call__()` перед `transaction.tasks.save(restarted)`:
     ```python
     if (
         task.task_type is TaskType.ANALYZE_CHAPTER
         and task.scope_id
         and (task.last_error is None or task.last_error.type is not ErrorType.TRANSIENT)
     ):
         delete_fn = getattr(transaction.analyses, "delete_fragment_passes_for_section", None)
         if callable(delete_fn):
             for pass_kind in (AnalysisPassKind.NARRATIVE, AnalysisPassKind.PEDAGOGICAL):
                 delete_fn(task.book_id, task.scope_id, pass_kind)
     ```
     При `ErrorType.TRANSIENT` (сетевой/провайдерский обрыв) полезный фрагментный кэш сохраняется; при `VALIDATION` / `MODEL_OUTPUT` / `CONFIGURATION` дедлок гарантированно разрывается.
3. **Расширение `known_names` для автосинтеза подсказок `GLOSSARY_ONLY` (`analyze_chapter.py:2537-2580`):**
   - В `_a2_vocab_known_names_from_glossary` и `_a2_vocab_known_names_with_narrative` (которые передают `known_names` в `_materialize_a2_glossary_difficulties` в `analysis_merge.py:5653`):
     - Дополнительно извлекать капитализированные латинские слова-имена (`re.findall(r"\b[A-Z][a-z]{2,}\b", text)`) из `concept.note` и `candidate.note`/`candidate.description` **строго только для записей с типом в `_A2_VOCAB_NAME_TYPES` (`PERSON`, `PLACE`, `ORGANIZATION`)**, исключая английские стоп-слова начала предложений (`The`, `She`, `He`, `Her`, `His`, `They`, `This`, `That`, `From`, `Also`, `Only`, `Note`, `Name`, `Short`, `See`).
     - Из `narrative.context_facts` (`kind in {"speaker", "pronoun_referent"}` с непустым капитализированным `value`) добавлять имена в `known_names`.
     - Проверить сохранение точного результата в `tests/unit/application/test_staged_analyze_chapter.py:4714-4720`.

---

## WP2 — Реестр иллюстраций: рёберное правило мостов, контекстный блокинг, `_choose_canonical_name`, `_display_name_at_position`, алиасы и глоссарий

### 2.1 `src/linguary/application/illustration_reference.py`

1. **Рёберное правило мостов в `resolve_verdict_graph_clusters(node_ids, *, same_edges, diff_edges)` (`L330-425`):**
   - Заменить Step 1 (`L364-381`): вместо удаления всех `same_edges`, инцидентных `bridge_nodes`, собирать `bridge_edges: set[tuple[str, str]]`:
     ```python
     bridge_edges: set[tuple[str, str]] = set()
     for x in nodes:
         neighbors = sorted(adj[x])
         for i in range(len(neighbors)):
             a = neighbors[i]
             for j in range(i + 1, len(neighbors)):
                 b = neighbors[j]
                 pair = (a, b) if a < b else (b, a)
                 if pair in diff_set:
                     bridge_edges.add((x, a) if x < a else (a, x))
                     bridge_edges.add((x, b) if x < b else (b, x))
     if bridge_edges:
         active_same = active_same - bridge_edges
     ```
   - Step 2 (`L410-424`) оставить без изменений.
   - Инварианты: 0 пар `diff_set` в одном кластере, 100% порядок-независимость; `A-B-C, A-D, diff(A,C)` → `{("A","D"), ("B",), ("C",)}`; `K_9` с 35 `same` + 1 `diff(u,v)` → `{(w_1..w_7), (u,), (v,)}`.

2. **Контекстный блокинг анонимных дескрипторов (`build_alias_candidate_pairs`, `L264-327`):**
   - Сохранить существующий токенный/глоссарный блокинг.
   - Ввести чистый хелпер `_parse_section_block_order(position_key: Any) -> tuple[int, int] | None`: парсит `SSSSSS:BBBBBB:CCCCCCCC`, возвращает `(int(parts[0]), int(parts[1]))` или `None` (никогда не подставляет default `0`!).
   - Ввести `_is_anonymous_person_descriptor(name: str) -> bool`: истинно, если `_score_canonical_name(name)[0] == 0` и после первого слова нет слов, начинающихся с заглавной буквы (`not any(w[0].isupper() for w in words[1:])` — исключает `"the Smiths’ son"` и `"the Bakers’ son"`, сохраняя `test_illustration_reference.py:1207`).
   - В `build_alias_candidate_pairs(mentions, *, glossary_links=None)`:
     - Для каждого `person`-упоминания `m` с валидным `(sec_ord, blk_ord) = _parse_section_block_order(m.get("position_key"))` и `_is_anonymous_person_descriptor(str(m["name"]))`:
       - Собрать все прочие `person`-упоминания в том же `sec_ord` с валидным `position_key` и отличным `mkey`.
       - Отсортировать их по `(0 if _score_canonical_name(str(o["name"]))[0] >= 1 else 1, abs(o_blk - blk_ord), str(o["mention_key"]))`.
       - Взять до `MAX_CONTEXT_ALIAS_NEIGHBORS = 8` ближайших кандидатов и добавить пары `(min(mkey, okey), max(mkey, okey))` в `pairs`.

3. **Приоритеты `_choose_canonical_name` и защита `_display_name_at_position` (`L2781-2849`):**
   - В `_choose_canonical_name(members, *, glossary_source_keys: frozenset[str] = frozenset()) -> str`:
     - `in_glossary = 1 if glossary_key(rep_name) in glossary_source_keys else 0`
     - `sort_key = (-tier, -in_glossary, -total_count, -short_len, -min(proper_word_count, 3), group_min_pos, rep_name.casefold())`
   - В `_display_name_at_position(entity: Mapping[str, Any], event_position_key: str) -> str` (`L2829-2849`):
     - После формирования `eligible = {name: pos for name, pos in raw_positions.items() if pos <= event_position_key}`:
       - Пусть `canonical = str(entity.get("canonical_name") or "").strip()`.
       - **Если `canonical` непуст и `canonical in eligible` (или совпадает по `normalize_name_token` с ключом в `eligible`):** немедленно вернуть `canonical`!
       - Иначе, если `eligible` непуст — вернуть `_choose_canonical_name([{"name": n, "position_key": p} for n, p in eligible.items()])`.
   - При формировании `ENTITY` (`L2400-2445`):
     - Считать глоссарий книги защитно: `glossary_repo = getattr(tx, "glossary", None)` → построить `glossary_source_keys: frozenset[str]` и карту `glossary_ru_by_key: dict[str, str]` (из `TermConcept` с русским `TermVariant`).
     - Передать `glossary_source_keys=glossary_source_keys` в `_choose_canonical_name(...)`.
     - Сохранить в `ENTITY` payload опциональные поля:
       - `"glossary_translation_ru": glossary_ru_by_key.get(glossary_key(canonical_name))`
       - `"alias_translations_ru": {n: tr for n in alias_first_positions if (tr := glossary_ru_by_key.get(glossary_key(n)))}`

### 2.2 `src/linguary/application/illustration_selection.py`
- В `_known_entity_names_for_chapter(tx, generation_id, *, section_order, limit_per_kind=_MAX_KNOWN_ENTITIES_PER_KIND)` (`L103-153`):
  - **Сохранить `known_characters` и `known_locations` чистыми списками имён** (не добавляя `(aliases: ...)` внутрь имён — сохраняет `test_illustration_reliability_integration.py:1279`).
  - Дополнительно собирать `known_entity_aliases: dict[str, list[str]]` — для каждой сущности из `known_characters` / `known_locations` включить до 3 уникальных неспойлерных алиасов из `alias_first_positions` с `pos <= chapter_end_pos` и `normalize_name_token(alias) != normalize_name_token(display_name)`.
  - Возвращать `(known_characters, known_locations, known_entity_aliases)` (либо добавить отдельный хелпер `_known_entities_context_for_chapter(...)`, сохранив старую 2-кортежную сигнатуру `_known_entity_names_for_chapter` для обратной совместимости с unit-тестами!).
  - Включать `"known_entity_aliases"` (только если непусто) в `_build_req` внутри `pack_items_by_wire_budget()` и переносить в `_split_children_for_scope()`.
  - В `_resolve_entity_name()` (`illustration_chapter.py:299`): оставить защитное снятие возможного суффикса `re.sub(r"\s*\(aliases:.*\)$", "", name).strip()`.

### 2.3 `src/linguary/application/illustration_chapter.py` и `illustration_reference.py`
1. **Локализованное `name_ru` и fallback `description_ru` (`illustration_chapter.py:1000-1035, 1380-1410`):**
   - Добавить `_display_name_ru_at_position(entity: Mapping[str, Any], event_position_key: str, display_name_en: str) -> str`:
     - Проверить `alias_translations_ru.get(display_name_en)`;
     - Иначе если `display_name_en == entity.get("canonical_name")` и задано `glossary_translation_ru` — вернуть его;
     - Иначе вернуть `display_name_en`.
   - Применять `display_name_ru = _display_name_ru_at_position(...)` для `IllustrationAppearance.name_ru` и для fallback `description_ru` (`L1029`, `L1397`), оставляя `name_en = eng_name`.
2. **Неспойлерный глоссарий книги `glossary_terms_ru` с контролем wire budget:**
   - Разместить чистый хелпер `_collect_book_glossary_terms_ru(tx, book_id, *, max_section_order: int | None = None, allowed_keys: frozenset[str] | None = None, limit: int = 40) -> dict[str, str]` в `illustration_context.py` (избегая циклических импортов между `illustration_reference`, `illustration_selection` и `illustration_chapter`):
     - Использует защитный `glossary_repo = getattr(tx, "glossary", None)`.
     - Для главовых этапов (`CANDIDATES`, `FACT_SCOPE`, `PROMPT`) проверяет появление термина в разделах `<= max_section_order` через `tx.analyses.list_for_book(book_id)` и/или `tx.analyses.list_passes_for_book(book_id, pass_kind=AnalysisPassKind.NARRATIVE, scope_key="")` (сопоставляя `section_id -> section.order` по `tx.documents.get(book_id)` — работает даже до завершения `PEDAGOGICAL`!) плюс `ENTITY.first_position_key <= chapter_end_pos`; не включает `TermConcept` без подтверждения в `<= max_section_order` (0 спойлеров будущих глав).
     - Для глобального этапа `DESIGN` (`illustration_reference.py:620, 2503`) ограничивает словарь `allowed_keys` самой сущностью (`canonical_name` + `aliases`) и терминами, упомянутыми в текстах переданных фактов сущности (без спойлеров и лишних байт).
   - Обязательно включать `"glossary_terms_ru"` (только если непусто) внутри probe-запросов `pack_items_by_wire_budget` / `_build_bounded_design_input` / `_build_prompt_scope_spec`, переносить `"glossary_terms_ru"` в `_split_children_for_scope` (`illustration_selection.py:2179-2209`), и при дефиците wire budget усекать `glossary_terms_ru` (вплоть до `{}`), чтобы не провоцировать ложный `ATOMIC_INPUT_TOO_LARGE`.
   - Добавить в `_STAGE_INSTRUCTIONS` (`DESIGN`, `CANDIDATES`, `FACT_SCOPE`, `PROMPT`) строку: *«В русскоязычных полях используй переводы имён и терминов из `glossary_terms_ru` (если термин там присутствует).»*

---

## WP3 — Иллюстрации: потолки токенов, авторетрай `provider_truncated`, wire-алиасинг ID, экономия вызовов, плотность и гигиена

### 3.1 Per-stage потолки токенов и защита `reasoning_max_tokens` (`illustration_context.py`, `composition/illustration_runtime.py`)
1. **`application/illustration_context.py`:**
   - Добавить константу `MIN_ILLUSTRATION_STAGE_COMPLETION_TOKENS = 6144` и чистую функцию `_stage_max_completion_tokens(stage: IllustrationScopeStage, policy: IllustrationPolicy) -> int`:
     - `DESIGN`, `SELECTION`, `FACT_SCOPE`, `PROMPT` → `min(policy.max_completion_tokens, 6144)`
     - `EXTRACT`, `ALIAS`, `CANDIDATES` → `min(policy.max_completion_tokens, 8192)`
   - В `IllustrationContext.build_for_kind()` (`L623-635`): передать `max_completion_tokens=_stage_max_completion_tokens(stage, policy)` в `LLMRequest(...)`.
2. **`composition/illustration_runtime.py:43-85` (`_clean_llm_settings`):**
   - **Не менять** `effective_max_tokens` (сохраняет `test_illustration_runtime.py:260, 267, 316-321`) и **не менять** `temperature=policy.temperature` (сохраняет `test_illustration_runtime.py:136`).
   - На `L63-68` ужесточить проверку наследования `reasoning_max_tokens`:
     ```python
     min_stage_cap = min(
         effective_max_tokens if effective_max_tokens is not None else policy.max_completion_tokens,
         MIN_ILLUSTRATION_STAGE_COMPLETION_TOKENS,
     )
     ```
     Наследовать `base_llm.reasoning_max_tokens`, только если `base_llm.reasoning_max_tokens < min_stage_cap`; иначе `reasoning_max_tokens = None` (исключает `LLMConfigurationError` на `openai_compatible.py:566`).

### 3.2 Авторетрай при `provider_truncated` во всех 3 точках (`illustration_reference.py:1232`, `illustration_chapter.py:1906`, `illustration_selection.py:1148`)
- Во всех трёх модулях в ветке `if is_truncated_error(call_error):` при `not is_composite`:
  - Если `self._can_continue_activation(activation_calls_used, scope.id, book_id, policy)`:
    - Записать попытку как `FAILED` (`outcome={"reason_code": IllustrationReasonCode.PROVIDER_TRUNCATED.value}`);
    - Установить `repair_summary = "Предыдущий ответ был обрезан по длине. Верни компактный валидный JSON без повторов и лишних пробелов."`;
    - Продолжить цикл активации (`continue`).
  - Иначе вызвать `_finalize_scope_failure(..., reason_code=IllustrationReasonCode.PROVIDER_TRUNCATED.value)`.

### 3.3 Wire-алиасинг длинных ID (`len(raw_id) > 16`) в `EXTRACT` и `CANDIDATES` + `quote_anchors`
1. **Граница и правила алиасинга (`illustration_context.py`, `illustration_reference.py`, `illustration_selection.py`):**
   - `scope.summary["request_input"]`, `input_sha256`, `find_reusable_scope` и сплиты **всегда хранят и используют канонические ID**.
   - Добавить чистые функции в `illustration_context.py`:
     - `encode_wire_id_aliases(stage, input_payload) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]` (возвращает `(wire_payload, alias_to_canonical, canonical_to_alias)`):
       - Действует для `stage in {EXTRACT, CANDIDATES}`.
       - Заменяет ID **только если `len(raw_id) > 16`** (все 36-символьные UUID, 64-hex SHA-256 `scene_id` и длинные `source_unit_id`; короткие тестовые ID `b-0`, `b-atomic`, `sec0-b0` не переименовываются!):
         - `source_unit_id` → `u1, u2, ...`
         - `block_id` → `b1, b2, ...`
         - `scene_id` → `s1, s2, ...`
     - `decode_wire_id_aliases(response, alias_to_canonical, *, single_scene_id: str | None = None)`: детерминированно разворачивает короткие алиасы обратно в канонические ID через `dataclasses.replace` во всех ID-полях DTO (`IllustrationExtractResponse`: `mentions[].quote_block_id`, `facts[].quote_block_id`, `covered_source_unit_ids`; `IllustrationCandidatesResponse`: `scene_id`, `events[].quote_anchors[].block_id`, `events[].insert_after_block_id`, `covered_source_unit_ids`) перед валидацией (сырые ID проходят без изменений; для 1-сценного `CANDIDATES` разрешает `s1 -> single_scene_id`).
     - `rewrite_repair_ids_to_wire(repair_summary: str, canonical_to_alias: Mapping[str, str]) -> str`: вызывается внутри `IllustrationContext.build_for_kind` до `compact_repair_summary`, подменяя канонические ID в тексте `repair_summary` обратно на wire-алиасы.
   - Вызывать `encode_wire_id_aliases` внутри `IllustrationContext.build_for_kind` (что автоматически учитывается в `pack_items_by_wire_budget` и `prompt_hash`), а сразу после `client.generate(request)` (`illustration_reference.py:1195`, `illustration_selection.py:1112`) восстанавливать `alias_to_canonical = encode_wire_id_aliases(scope.stage, scope.summary["request_input"])[1]` и вызывать `decode_wire_id_aliases(...)` перед `_validate_response`.
2. **`adapters/llm/illustration_payloads.py:237`:**
   - В `IllustrationCandidateEventSchema`: добавить `Field(min_length=1, description="Обязательно непустой список (минимум 1 элемент) дословных цитат {block_id, quote} из переданных блоков.")`.

### 3.4 Снижение числа LLM-вызовов и масштабирование плотности иллюстраций
1. **Отбор значимых сущностей для `DESIGN` (`illustration_reference.py:2489-2496`):**
   - Заменить `int(e["mention_count"]) >= 1` на:
     ```python
     ent_facts = facts_by_entity.get(str(e["entity_id"]), ())
     entity_needs_design = (
         int(e["mention_count"]) >= 2
         or len(ent_facts) >= 2
         or any(str(f.get("temporal_scope", "")) == "permanent" for f in ent_facts)
         or bool(e.get("glossary_translation_ru"))
     )
     ```
2. **Детерминированный fast-path для 1 `permanent`-факта в `FACT_SCOPE` (`illustration_chapter.py:1038-1158`):**
   - Срабатывает, только когда `len(entity_facts) == 1`: пусть `f = entity_facts[0]`, `desc_en = str(f.get("description_en", "")).strip()`, `desc_ru = str(f.get("description_ru", "")).strip()`.
   - Условия fast-path (все должны выполняться одновременно):
     - `str(f.get("temporal_scope", "")) == "permanent"`;
     - `bool(desc_en)` и `not has_cyrillic(desc_en)` и `bool(desc_ru)`;
     - если `revision.settings.style == ILLUSTRATION_STYLE_MONOCHROME_INK`: `find_monochrome_color_word(desc_en, allowed_names=entity_names, strict_palette=False) is None` и `find_monochrome_color_word(desc_ru, allowed_names=entity_names, strict_palette=False) is None`.
   - При выполнении условий: пропустить создание LLM-скоупа `FACT_SCOPE` и сразу (по тому же пути, что и после `ACCEPTED` `FACT_SCOPE` на `L1365-1404`) добавить `f["fact_id"]` в `allowed_fids`, наполнить `available_design_traits` / `allowed_dkeys` (с проверкой покрытия `design_payloads`), и сформировать `IllustrationAppearance(entity_id=eid, name_ru=display_name_ru, description_ru=desc_ru, description_en=desc_en, fact_ids=(str(f["fact_id"]),), assumptions_ru=(), name_en=eng_name)`.
   - Обновить 3 unit-теста в `tests/unit/application/test_illustration_chapter.py:634, 638-700, 886` (проверять попадание 1 `permanent`-факта в `PROMPT.allowed_fact_ids` / `moment` вместо перехвата `ILLUSTRATION_FACT_SCOPE`).
3. **Масштабирование лимита главы и дистанция иллюстраций (`illustration_selection.py`):**
   - Добавить чистую функцию `effective_chapter_illustration_cap(max_per_chapter: int, content_block_count: int, scene_count: int) -> int`:
     - `content_block_count <= 2 and scene_count <= 1`: `min(max_per_chapter, 1)`
     - `3 <= content_block_count <= 8 and scene_count <= 2`: `min(max_per_chapter, 2)`
     - иначе: `max_per_chapter`
   - Применять `effective_chapter_illustration_cap(revision.settings.max_per_chapter, len(entry.block_digests), len(entry.scenes))` в `_advance_section` (`L1727`) и `_schedule_selection_round` (`L1966`), передавать `chapter_cap: int | None = None` в `_build_selection_scope_spec` (`L589`, использовать в `input_sha256` и сохранять `"effective_max_per_chapter": effective_cap` в `scope.summary`), а в `_split_children_for_scope` (`L2220`, где нет `section`/`entry`) читать `int(scope.summary.get("effective_max_per_chapter", revision.settings.max_per_chapter))`.
   - В `moments_conflict_diversity(a, b, *, block_order_by_id: Mapping[str, int] | None = None, min_block_gap: int = 0)` (`L272-299`): если `min_block_gap > 0` и `block_order_by_id` передан, и обе позиции (`insert_after_block_id` или `anchors[0].block_id`) есть в `block_order_by_id`, считать конфликтом `abs(pos_a - pos_b) < min_block_gap`. В `enforce_intra_chapter_diversity(...)` включать `min_block_gap = 2` при `content_block_count >= 12` (и `0` по умолчанию при отсутствии `block_order_by_id`).

### 3.5 Гигиена промптов и контекста иллюстраций (`illustration_context.py`, `illustration_reference.py`, `illustration_chapter.py`)
1. **`_STAGE_INSTRUCTIONS[EXTRACT]` (`illustration_context.py:473`):** ограничить `back_applicable=true` только врождёнными/долговременными физическими чертами (запретить одежду конкретного эпизода, загрязнения, временные предметы, посмертные изменения).
2. **`_build_bounded_design_input()` (`illustration_reference.py:654-673`):** при **`len(permanent_facts) == 1`** и наличии `episodic`-фактов добавлять отдельное поле `"episodic_context_only"` (до 2 самых ранних кратких фактов, включённых в `_make_payload` для учёта `target_bytes`, с отсечением при нехватке бюджета), **не изменяя** поля `facts`, `non_permanent_context_only`, `omitted_episodic_count` и `omitted_fact_count` (`test_illustration_reference.py:1158-1169`).
3. **`_STAGE_INSTRUCTIONS[FACT_SCOPE]` (`illustration_context.py:510-517`):** добавить запрет отбрасывать привязку персонажа к конкретному транспорту/локации и запрет буквально трактовать образные реплики об одежде как телесные деформации.
4. **`_build_prompt_scope_spec()` и `_validate_render_response()` (`illustration_chapter.py`), `filter_non_assumption_placeholders()` (`illustration_validation.py:119`) + `_STAGE_INSTRUCTIONS[PROMPT]` (`illustration_context.py:540-555`):**
   - Передавать `"anchor_quotes": [q[:240] for q in card_quotes[:3]]` в `PROMPT` (с проверкой wire budget и усечением при дефиците байт).
   - В `_STAGE_INSTRUCTIONS[PROMPT]`: сохранить упоминание `characters[].description_en` (закреплено в `tests/unit/application/test_illustration_reliability.py:1478`), добавив явный запрет писать внутренние технические ключи полей (`attire_default`, `description_en`, `characters[]`, `used_design_keys`, `facts`, `compact_description_en`) в пользовательском `assumptions_ru`.
   - В `filter_non_assumption_placeholders()` (`src/linguary/application/illustration_validation.py:119`, вызывается из `_validate_render_response`) детерминированно отфильтровывать строки `assumptions_ru`, содержащие технические маркеры (`\b(?:attire_default|palette_value|description_en|description_ru|used_design_keys|depicted_entity_ids|compact_description_en)\b` или `characters\[\]`), без траты repair-попытки.

---

## WP4 — Перевод: синхронизация правил в `reader_text.py` и вложенные цитаты в `validators/dialogue.py`

### 4.1 `src/linguary/application/reader_text.py`
Подтверждено по `context_builder.py:3701-3776`: `_A2_NOTE_DISCIPLINE` и `_A2_DIALOGUE_AND_CALQUE_RULES` **не входят** в замороженные SHA-256 `test_baseline_b2_templates_sha256_unchanged`. Обновить напрямую:
1. В `_A2_DIALOGUE_AND_CALQUE_RULES` (`L516-528`):
   - Добавить правило: *«Реплику прямой речи после слов автора с точкой начинай с новой строки и «— » либо оформляй после двоеточия в кавычках-ёлочках (`Он обернулся: «…»`), но не ставь «— » в той же строке после точки.»*
   - В правило регистра `L525` добавить: *«При обращении одного персонажа сразу к нескольким собеседникам (даже детям, с которыми поодиночке на «ты») используй множественное число «вы».»*
2. В `_A2_NOTE_DISCIPLINE` (`L512`): уточнить: *«В glossary_hints приводи к инфинитиву с to только глаголы (`to snigger — хихикать`); существительные (в т. ч. с артиклем `a/an/the`, например `a guffaw`) оставляй существительными без `to`.»*

### 4.2 `src/linguary/application/validators/dialogue.py`
- Добавить чистый хелпер `_is_allowed_embedded_quote_span(text: str, span_start: int, span_end: int, *, source_text: str | None) -> bool`:
  - Истинен **только** если:
    1. `span_start > 0` и `«` стоит **не в начале строки** (перед `span_start` на той же строке есть непустой авторский текст);
    2. `source_text` передан и содержит минимум 2 отдельные закавыченные реплики, разделённые авторским текстом (как в `“Speech…” Narrative. “Quote,” said X.`).
- Применить `_is_allowed_embedded_quote_span` **в обоих правилах**:
  1. `_has_author_speech_inside_guillemets(text, *, source_text=None)` (`L173`): пропускать совпадение внутри `«…»`, если для этого диапазона `_is_allowed_embedded_quote_span` истинно (обычная реплика `«Привет, — сказал он»` с начала строки по-прежнему отклоняется!).
  2. `_has_inline_dialogue_dash(text, *, source_text=None)` (`L187`, `L314-329`): пропускать совпадение `_STRAY_INLINE_DASH`, если оно целиком лежит внутри диапазона `«…»`, для которого `_is_allowed_embedded_quote_span` истинно (что позволяет принять как однострочный, так и двухстрочный корректный перевод смешанного абзаца гл. 2 B35).

---

## WP5 — Раскладка учебного слоя A2: единый поиск подсказок, «Говорит …» и тавтологии

### 5.1 Единая сегментная привязка `glossary_hints` (`layout/notes.py`, `layout/book.py`)
1. **`src/linguary/application/layout/notes.py`:**
   - Ввести публичный внутри-пакетный хелпер `find_hint_marker_end(text: str, original: str, *, anchor_spans: Sequence[tuple[int, int]] = (), preferred_spans: Sequence[tuple[int, int]] = ()) -> int | None`:
     1. **Приоритет 1 (безусловный — явная опора `Difficulty` в `block_id`):** для каждого `(a_start, a_end)` в `anchor_spans` искать `local = _find_end(text[a_start:a_end], original)`; если найдено — вернуть `a_start + local`; если не найдено, но `0 <= a_start < a_end <= len(text)` — вернуть `a_end`.
     2. **Приоритет 2 (сегментный приоритет при отсутствии явной опоры):** для каждого `(s_start, s_end)` в `preferred_spans` (сначала спаны `GLOSSARY_ONLY`, затем `READ_AS_IS`) искать `local = _find_end(text[s_start:s_end], original)`; если найдено — вернуть `s_start + local`.
     3. **Fallback:** вернуть `_find_end(text, original)`.
   - Использовать `find_hint_marker_end(...)` в `_hint_draft()` (`L745`), передавая `anchor_spans` из `known[diff_id].anchors` (для `anchor.block_id == block_id`) и `preferred_spans` из `block_notes(...)`.
2. **`src/linguary/application/layout/book.py` (`_hint_segment_index`, `L1035-1089, 1154`):**
   - Прокинуть `plan: ContentPlan | None = None` (и `dropped_difficulty_ids`) из `_deduplicate_glossary_hints()` (`L1154`) через обе обёртки `_hint_is_glossary_only()` (`L1066`) и `_glossary_only_bucket()` (`L1088`) в `_hint_segment_index()` (`L1035`), используя тот же `find_hint_marker_end(...)` для 100% согласованности с `_hint_draft()`.

### 5.2 Неспойлерное обиходное имя и самопредставление в «Говорит …» (`layout/book.py:374, 555-697`)
1. **Обиходное имя персонажа:**
   - Ввести чистый хелпер `_extract_short_name_pair(concept: TermConcept, analyses: Sequence[ChapterAnalysis], language: str) -> tuple[str, str] | None`:
     - Пропускать титульные/родственные канонические формы (начинающиеся на `Mr`, `Mrs`, `Miss`, `Ms`, `Dr`, `Professor`, `Uncle`, `Aunt`, `Sir`, `Lady`, `Lord`, `King`, `Queen` — сохраняет `test_a2_harry_potter_regressions.py:1028-1035`).
     - Искать явную пару `(short_en, short_ru)` в: (a) скобках `canonical_source_text` (если задано), (b) явных записях в `concept.note` вида `Allie (Элли)` / `Allie — Элли` / `Allie - Элли`, либо (c) латинском имени в `concept.note`, совпадающем с `GlossaryCandidate(type=PERSON)` в `analyses`.
   - Предвычислять накопительные частоты `short_en` vs `canonical_source_text` по `<= section.order` один раз на документ внутри `_SpeakerIndex` (`L374`, 0 спойлеров будущих глав!); в `_speaker_name(..., section_order=section.order)` возвращать `short_ru`, только если `count(short_en) > count(canonical_source_text)`.
2. **Фильтр самопредставления в `_speakers_for_block()` (`L678-697`):**
   - Добавить `_part_self_identifies_speaker(part_text: str, speaker_tokens: Sequence[str]) -> bool`, передавая токены из `fact.value`, `canonical_source_text` и `short_en` (чтобы при `fact.value == "Alexandra"` подавлялось и `“My name’s Allie.”`):
     - Отфильтровать из `speaker_tokens` служебные/родовые слова (`the`, `a`, `an`, `boy`, `girl`, `man`, `woman`, `kid`, `mr`, `mrs`, `miss`, `ms`, `dr`, `professor`, `uncle`, `aunt`).
     - Проверить regex `rf"\b(?:my\s+name(?:'s|’s|\s+is)|i(?:'m|’m|\s+am)|call\s+me)\s+{re.escape(tok)}\b"` (case-insensitive).
     - Если истинно — не добавлять сноску «Говорит …» для этого `part`.

### 5.3 Фильтр тавтологических пояснений (`layout/notes.py`, после `L779`)
- Добавить `_is_tautological_gloss(translation: str, note: str) -> bool`:
  - Нормализовать слова `translation` и `note` (`casefold()`, `ё -> е`, без пунктуации).
  - Если `1 <= len(note_words) <= 3`, все значимые слова `translation` совпадают (точно или по префиксу >= 4 символов) со словами `note`, а все оставшиеся слова `note` начинаются на одну из пустых основ-заглушек `("люд", "человек", "вещ", "предмет", "мест", "действ", "состоян", "явлен")` → вернуть `True`.
- Вызывать проверку **после `L779`** (после вычисления финального нормализованного `translation`): если `kept_note and (lexical or verbal) and not signaled and _is_tautological_gloss(translation, kept_note)`: обнулить `kept_note = ""`.

---

## Тестовая матрица

| WP | Реальный файл тестов | Проверяемые сценарии |
|---|---|---|
| WP1 | `tests/unit/application/test_context_builder.py` | Обновление `L3737` (пример `"Iron Hand"` вместо `"the freckled boy"`), сохранность `test_baseline_b2_templates_sha256_unchanged` |
| WP1 | `tests/unit/application/test_analysis_merge_scenes.py`, `tests/integration/processing/test_analysis_everlost_regressions.py` | `_is_generic_person_candidate` ловит `younger boy`, `the freckled boy`, `the freckled kid`, `the baker` и пропускает `Iron Hand`, `Chocolate Boy`; `_first_name_block_index` монотонен при `"Chocolate Boy"` в блоке 38 И сохраняет спойлер `'the dark voldemort'` (`L1214`); обновление `L1465-1484, 1778-1827` |
| WP1 | `tests/unit/application/test_staged_analyze_chapter.py`, `tests/integration/processing/test_retry_task_recovery_guards.py` | `AnalyzeChapter` очищает фрагменты текущего `pass_kind` через `tx.analyses.delete_fragment_passes_for_section` при `AnalysisMergeError` (вкл. `_assemble_staged`); `RetryTask` очищает фрагменты при `VALIDATION`, но сохраняет их при `TRANSIENT`; расширенный `_a2_vocab_known_names_*` распознаёт `Allie` при `Alexandra` и сохраняет `L4714-4720` |
| WP2 | `tests/unit/application/test_illustration_reference.py` | Рёберное правило `resolve_verdict_graph_clusters`: 9-клик (35 `same` + 1 `diff`) сохраняет 7 вершин, порядок-независимость на 20 перестановках, `test_ambiguous_identity_remains_separate` и `L987` зелёные; контекстный блокинг `tier == 0` при валидном `position_key` (сохраняя `L1207`); `_choose_canonical_name` (`Nick` > `Chocolate Boy` и `Professor McGonagall` > `McGonagall`); `_display_name_at_position` возвращает `canonical_name`, если он уже раскрыт |
| WP2 | `tests/unit/application/test_illustration_selection.py`, `tests/unit/application/test_illustration_chapter.py`, `tests/integration/processing/test_illustration_reliability_integration.py` | `known_characters` чистые (`L1279` зелёный), неспойлерные алиасы в `known_entity_aliases` (и сохраняются при сплите); локализованный `name_ru` и fallback `description_ru` с обратной совместимостью старых `ENTITY`; неспойлерный `glossary_terms_ru` в пределах wire budget |
| WP3 | `tests/unit/composition/test_illustration_runtime.py`, `tests/unit/application/test_illustration_reference.py`, `tests/unit/application/test_illustration_selection.py`, `tests/unit/application/test_illustration_chapter.py`, `tests/unit/application/test_illustration_evaluation.py`, `tests/unit/llm/test_illustration_payloads.py`, `tests/integration/processing/test_illustration_pipeline.py` | Per-stage `max_completion_tokens` (6144/8192) в `build_for_kind` + защита `reasoning_max_tokens < 6144` против `LLMConfigurationError` (без поломки `L136, L260, L267, L316-321`); ретрай `provider_truncated` во всех 3 модулях; wire-алиасинг только `len > 16` (без поломки 5 тестов) + обратный маппинг + rewrite `repair_summary`; `quote_anchors` `description`; отбор `DESIGN` (сохраняя `L1066`); fast-path `FACT_SCOPE` строго для 1 `permanent`-факта (+ обновление `test_illustration_chapter.py:634, 638-700, 886`); `effective_chapter_illustration_cap` и `min_block_gap`; `episodic_context_only` при `len(permanent)==1` (сохраняя `L1158-1169`); `anchor_quotes` и детерминированная фильтрация технических ключей в `assumptions_ru` |
| WP4 | `tests/unit/application/test_context_builder.py`, `tests/unit/application/test_dialogue_validator.py`, `tests/unit/application/test_result_validators.py` | Новые правила в `_A2_DIALOGUE_AND_CALQUE_RULES` / `_A2_NOTE_DISCIPLINE` при зелёном `test_baseline_b2_templates_sha256_unchanged`; `_is_allowed_embedded_quote_span` пропускает однострочный и двухстрочный B35 со `strict_v3=True` и по-прежнему отклоняет `«Привет, — сказал он»` с начала строки |
| WP5 | `tests/unit/application/layout/test_notes.py`, `tests/unit/application/layout/test_book_layout.py`, `tests/unit/application/layout/test_units.py`, `tests/integration/layout/test_a2_harry_potter_regressions.py` | `find_hint_marker_end`: явная опора ищет слово внутри `[a_start:a_end]`, без опоры — приоритет `GLOSSARY_ONLY`-сегмента (кейс `cliff` в B10), полная согласованность `_hint_draft` и `_hint_segment_index`; обиходное «Говорит Элли» накопительно по `<= section.order` без поломки титульных имён (`L1028-1035`); подавление «Говорит Ник» на `“My name’s Nick.”` без ложного срабатывания на `I'm the boy`; подавление тавтологии `взрослые — взрослые люди` после L779 |
