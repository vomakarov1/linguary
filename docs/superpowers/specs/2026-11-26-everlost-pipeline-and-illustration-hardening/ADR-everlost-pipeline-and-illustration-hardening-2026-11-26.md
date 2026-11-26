# ADR: Архитектурные решения по надёжности анализа, реестра/бюджета иллюстраций, диалогов и раскладки A2

**Дата:** 2026-11-26  
**Статус:** Accepted (подтверждено финальным аудитом)  
**Связанный BRIEF:** `BRIEF-everlost-pipeline-and-illustration-hardening-2026-11-26.md`

---

## ADR-1: Монотонная проверка `spoiler_name_leak`, фильтрация строчных дескрипторов и точечная инвалидация фрагментного кэша

- **Контекст:** В гл. 2 *Everlost* `_NARRATIVE_ANALYSIS_INSTRUCTIONS:200` приводила `«the freckled boy»` как пример для `glossary_candidates`, узкий регекс `_GENERIC_PERSON_CANDIDATE_PATTERNS` пропускал его, а `_first_name_block_index()` включал `prefer_capitalized=True` для строчных кандидатов глоссария — в результате на фрагменте 0 находилось строчное `boy` (блок 2), а при слиянии главы заглавное `Boy` из чужого прозвища `"Chocolate Boy"` (блок 38) вызывало `AnalysisMergeError`. Фрагментный кэш `analysis_pass_results` удалялся через `tx.analyses.delete_fragment_passes_for_section(book_id, section_id, pass_kind)` только после успешного слияния, а `RetryTask` его не очищал, обрекая повторный запуск на бесконечный 0-вызовный дедлок.
- **Решение:**
  1. Убрать `«the freckled boy»` из примера `glossary_candidates` в `_NARRATIVE_ANALYSIS_INSTRUCTIONS:200` (заменить на капитализированное описательное прозвище, например `«Iron Hand»`), явно указав: строчные родовые описания (`the boy`, `the younger boy`, `the stranger`) не добавляются в `glossary_candidates`. Обновить 3 существующих теста, закреплявших пример `'the freckled boy'` (`test_context_builder.py:3737`, `test_analysis_merge_scenes.py:1465-1484, 1778-1827`).
  2. Обобщить `_is_generic_person_candidate()` (`analysis_merge.py:2221-2238`), **сохранив** первые два существующих шаблона (`^(?:the|a|an)\s+[a-z]+$` и `of ...`), а третий расширив на любую полностью строчную фразу из **2–4 слов**, оканчивающуюся на токен из расширенного `_GENERIC_PERSON_HEADS` (`_GENERIC_ROLE_PARTICIPANTS` + `kid`, `kids`, `teen`, `teenager`, `youth`, `figure`, `voice`, `passenger`, `pedestrian`, `owner`, `clerk`, `guard`, `driver`, `keeper`, `stranger`, `goblin`, `wizard`, `witch`, `person`, `child`, `children`).
  3. Сделать `_first_name_block_index()` (`analysis_merge.py:2738-2816`) монотонным без регрессии для `'the dark voldemort'` (`test_analysis_merge_scenes.py:1214`):
     - Включить `prefer_capitalized = True` для `all_lowercase and matches_glossary` **только когда совпавший `glossary_candidate` сам содержит заглавные буквы** (`matches_capitalized_glossary`, например `'Voldemort'` для `'the dark voldemort'`), и `False` когда кандидат глоссария полностью строчный (`the freckled boy`) или `_is_generic_person_candidate(name)`.
     - Исключать `_GENERIC_ROLE_PARTICIPANTS` из `stems` при наличии отличительных стемов **только для не-строчных имён** (`not all_lowercase`), чтобы строчное `"younger boy"` не теряло совпадение по `"boy"` в ранних блоках, а капитализированное `"Chocolate Boy"` не матчилось по чужому `"boy"`.
  4. Инвалидация фрагментного кэша строго через **`tx.analyses.delete_fragment_passes_for_section(book_id, section_id, pass_kind)`** (с защитным `getattr(tx.analyses, "delete_fragment_passes_for_section", None)` для совместимости с `FakeChapterAnalysisRepository`):
     - В `analyze_chapter.py` обернуть слияние и общеглавную валидацию (`L575-590` для `NARRATIVE`; `L688-706`, **включая `self._assemble_staged`** `L701-706`, для `PEDAGOGICAL`) в `try ... except (AnalysisMergeError, DomainError):` и перед пробросом очищать фрагменты **только текущего `pass_kind`** (не затрагивая уже сохранённый общеглавный кэш другого прохода).
     - В `RetryTask.__call__()` (`retry_task.py:54-75`) для `TaskType.ANALYZE_CHAPTER` и непустого `task.scope_id` вызывать очистку для `pass_kind in (AnalysisPassKind.NARRATIVE, AnalysisPassKind.PEDAGOGICAL)` **только когда причина падения не транспортная** (`task.last_error is None or task.last_error.type is not ErrorType.TRANSIENT`), чтобы сохранять полезный фрагментный кэш при сетевых сбоях и гарантированно разрывать дедлок при ошибках валидации/слияния.
  5. В сборке `known_names` для автосинтеза подсказок `GLOSSARY_ONLY` (`analyze_chapter.py:2537-2580`, `_a2_vocab_known_names_from_glossary` и `_a2_vocab_known_names_with_narrative`) дополнительно извлекать латинские капитализированные имена из `note`/`description` **строго только для `_A2_VOCAB_NAME_TYPES` (`PERSON`/`PLACE`/`ORGANIZATION`)** и из `narrative.context_facts` (`speaker`/`pronoun_referent`), сохраняя точный контракт `test_staged_analyze_chapter.py:4714-4720`.
- **Альтернативы:**
  - *Отключить кэширование фрагментов полностью или сбрасывать кэш даже при `ErrorType.TRANSIENT`* — отвергнуто: теряется экономия при сетевых обрывах (`test_staged_analyze_chapter.py:971-999`).
  - *Отключить `prefer_capitalized` для всех `all_lowercase`* — отвергнуто: ломает детекцию спойлера `'the dark voldemort'` (`test_analysis_merge_scenes.py:1214`).
- **Последствия:** Локальная валидация фрагмента и общеглавная валидация согласованы; `retry` разрывает дедлок без потери кэша при сетевых сбоях; миграций нет.

---

## ADR-2: Рёберное правило мостов в кластеризации, контекстный блокинг и приоритет глоссария/частоты в иллюстрациях

- **Контекст:** В `resolve_verdict_graph_clusters()` Step 1 удалял **все** `same_edges`, инцидентные `bridge_nodes` (включая рёбра между самими `bridge_nodes`), поэтому одно ошибочное ребро `different` среди 36 пар леса (35 `same_entity`) развалило 9-вершинный клик на 9 одиночек. Одновременно `_blocking_keys_for_mention()` отрезал анонимные дескрипторы (`the boy`, `the girl`, `the other boy`) от именных героев (0 пар в `ALIAS`), `_choose_canonical_name()` поставил разовое прозвище `Chocolate Boy` выше имени `Nick` (8 упоминаний), а `_display_name_at_position()` перевызывал `_choose_canonical_name()` с `count=1` и без глоссария.
- **Решение:**
  1. **Рёберное правило мостов в `resolve_verdict_graph_clusters()` (`illustration_reference.py:364-381`):**
     - Вместо удаления всех рёбер, инцидентных `bridge_nodes`, собирать множество двусмысленных мостовых рёбер `bridge_edges: set[tuple[str, str]]`: для каждой вершины `x` и каждой пары её `same`-соседей `(a, b)` такой, что `(min(a, b), max(a, b)) in diff_set`, добавлять в `bridge_edges` **только рёбра `(x, a)` и `(x, b)`** (сохраняя все рёбра `(x, y)` между самими `bridge_nodes`!).
     - Удалить `bridge_edges` из `active_same`; Step 2 (страховка длинных конфликтных цепей) оставить прежним.
     - Это математически строго, 100% порядок-независимо, гарантирует 0 пар `diff_set` в одном кластере, полностью сохраняет `test_ambiguous_identity_remains_separate` и `test_transitive_identity_conflict_triangle_order_independent_deterministic` (`{("A","D"), ("B",), ("C",)}`), и на 9-клике леса Everlost (35 `same` + 1 `diff`) сохраняет 7 вершин в едином кластере вместо распада на 9 одиночек.
  2. **Контекстный блокинг анонимных дескрипторов (`build_alias_candidate_pairs`, `illustration_reference.py:264-327`):**
     - Добавить ограниченный контекстный блокинг **только** для `person`-упоминаний, которые (a) имеют валидный `position_key` (`section_order`, `block_order`, без подстановки default `0`) и (b) являются полностью безымянными родовыми дескрипторами (`_score_canonical_name(name)[0] == 0` и отсутствие заглавных неначальных слов — сохраняет `test_illustration_reference.py:1207` для `"the Smiths’ son"` vs `"the Bakers’ son"`).
     - Связывать такое упоминание с не более чем `MAX_CONTEXT_ALIAS_NEIGHBORS = 8` ближайшими по `|block_order - m_block_order|` `person`-упоминаниями в том же `section_order` (с валидным `position_key`; приоритет — именные `tier >= 1`, затем другие родовые с отличным `normalize_name_token`).
  3. **Приоритеты `_choose_canonical_name()` и защита `_display_name_at_position()` (`illustration_reference.py:2781-2849`):**
     - В `_choose_canonical_name(members, *, glossary_source_keys=frozenset())` использовать единый ключ сортировки:
       `(-tier, -in_glossary, -total_count, -short_len, -min(proper_word_count, 3), group_min_pos, rep_name.casefold())`.
       При равной частоте (`count=1` у `"Professor McGonagall"` и `"McGonagall"`) сохраняется победа более полного имени, а частое имя из глоссария (`Nick`, `count=8, in_glossary=1`) гарантированно побеждает разовое прозвище (`Chocolate Boy`, `count=1, in_glossary=0`).
     - В `_display_name_at_position(entity, event_position_key)`: если `canonical_name` непуст и уже встречался к `event_position_key` (`alias_first_positions[canonical_name] <= event_position_key`), **сразу возвращать `canonical_name`** (без повторного вызова `_choose_canonical_name(eligible)` с потерянными частотами и глоссарием!). Вызывать `_choose_canonical_name(eligible)` только когда `canonical_name` ещё не раскрыт к `event_position_key`.
  4. **Отдельное поле `known_entity_aliases`, локализованное `name_ru` и неспойлерный `glossary_terms_ru`:**
     - В `_known_entity_names_for_chapter()` (`illustration_selection.py:103-153`): оставлять `known_characters` и `known_locations` **чистыми списками имён** (сохраняет `test_illustration_reliability_integration.py:1279`), а неспойлерные на `chapter_end_pos` алиасы передавать в **отдельном опциональном поле `known_entity_aliases: dict[str, list[str]]`** (только если непусто, с учётом wire budget и переносом в `_split_children_for_scope()`).
     - В `ENTITY` payload (`illustration_reference.py:2400-2445`) опционально сохранять `glossary_translation_ru: str | None` и `alias_translations_ru: dict[str, str]` (читая `glossary_repo = getattr(tx, "glossary", None)` защитно для fake UoW); использовать локализованное неспойлерное имя в `IllustrationAppearance.name_ru` и fallback `description_ru` (`illustration_chapter.py:1028-1029, 1396-1397`), оставляя `name_en` английским.
     - Передавать компактный неспойлерный словарь `glossary_terms_ru` (внутри wire budget probe-запросов, с усечением при нехватке байт; для глобального `DESIGN` — только термины самой сущности и её фактов/`section_orders`) в `DESIGN`, `CANDIDATES`, `FACT_SCOPE` и `PROMPT`.
- **Последствия:** Ликвидирует распад кликов и подмену персонажей в гл. 4, синхронизирует терминологию иллюстраций с глоссарием книги, не ломает существующие тесты и старые записи БД.

---

## ADR-3: Per-stage потолки токенов (совместимые с `reasoning_max_tokens`), авторетрай `provider_truncated`, wire-алиасинг длинных ID и экономия вызовов

- **Контекст:** В прогоне 3,1 глав на иллюстрации ушло 163 вызова: `IllustrationContext.build_for_kind()` не передавал `max_completion_tokens`, 2 вызова `DESIGN` выжгли по 32 768 токенов и упали на 1-й попытке без ретрая; 6 сбоев вызваны опечатками в сырых 36-символьных UUID и 64-hex `scene_id`, 4 сбоя — пустым `quote_anchors: []`; 64% `DESIGN` не использовались, `FACT_SCOPE` вызывал LLM для 1 тривиального факта, а гл. 4 из 2 абзацев получила 2 иллюстрации.
- **Решение:**
  1. **Per-stage потолок токенов и защита `reasoning_max_tokens` (`illustration_context.py`, `composition/illustration_runtime.py`):**
     - В `IllustrationContext.build_for_kind()` передавать `max_completion_tokens=_stage_max_completion_tokens(stage, policy)`:
       - `DESIGN`, `SELECTION`, `FACT_SCOPE`, `PROMPT` → `min(policy.max_completion_tokens, 6144)` (`_MIN_STAGE_COMPLETION_TOKENS = 6144`);
       - `EXTRACT`, `ALIAS`, `CANDIDATES` → `min(policy.max_completion_tokens, 8192)`.
       Этот per-request лимит в `openai_compatible.py:545` автоматически перекрывает глобальный `settings.max_output_tokens`, не ломая пул клиентов и `test_illustration_runtime.py:260, 267, 316-321`.
     - В `_clean_llm_settings()` (`composition/illustration_runtime.py:59-70`) **оставить `temperature=policy.temperature` без изменений** (сохраняет `test_illustration_runtime.py:136`) и проверять `base_llm.reasoning_max_tokens < min(effective_max_tokens, _MIN_STAGE_COMPLETION_TOKENS)` (6144), иначе ставить `reasoning_max_tokens = None` — это полностью предотвращает `LLMConfigurationError` на `openai_compatible.py:566` при `reasoning_max_tokens >= 6144`.
  2. **Авторетрай при `provider_truncated` во всех 3 точках (`illustration_reference.py:1232`, `illustration_chapter.py:1906`, `illustration_selection.py:1148`):**
     - Для не-composite областей (`DESIGN`, `FACT_SCOPE`, `PROMPT`, 1-групповой `SELECTION`, атомарный `EXTRACT`/`CANDIDATES`) при `is_truncated_error(call_error)` проверять `_can_continue_activation(...)` и выполнять повторную попытку с короткой ремонтной подсказкой о компактном JSON без повторов в пределах `policy.max_calls_per_scope_activation` перед `_finalize_scope_failure`.
  3. **Wire-алиасинг длинных ID (`len(raw_id) > 16`) в `EXTRACT` и `CANDIDATES`:**
     - **Чистая граница:** `scope.summary["request_input"]`, `input_sha256`, `find_reusable_scope` и сплиты **всегда работают с каноническими ID** (без изменений).
     - На границе формирования `LLMRequest` (и в probe-сайзере `pack_items_by_wire_budget`) для ID с `len(raw_id) > 16` строится детерминированная двусторонняя карта (`b1..bN`, `u1..uN`, `s1..sM`); короткие ID (`<= 16`, например тестовые `b-0`, `b-atomic`, `sec0-b0`) остаются как есть.
     - Сразу после `llm.complete(...)` выполняется обратный маппинг алиасов в канонические ID перед валидацией (с поддержкой и сырых ID); при формировании `repair_summary` канонические ID из текста ошибки заменяются обратно на wire-алиасы.
     - В `IllustrationCandidateEventSchema` (`illustration_payloads.py:237`) добавить `Field(min_length=1, description=...)` для `quote_anchors`.
  4. **Экономия вызовов и масштабирование плотности без регрессий:**
     - **Отбор `DESIGN`** (`illustration_reference.py:2489-2496`): `mention_count >= 2 or len(facts) >= 2 or any(f.get("temporal_scope") == "permanent" for f in facts) or bool(e.get("glossary_translation_ru"))` (отсекает одноразовый фон без постоянных фактов, сохраняя `test_illustration_reference.py:1066`).
     - **Fast-path `FACT_SCOPE`** (`illustration_chapter.py:1038-1158`): строго только когда после хронологической фильтрации остаётся ровно **1 факт**, его `temporal_scope == "permanent"`, `description_en.strip()` непуст и не содержит кириллицы (`not has_cyrillic`), и в режиме `monochrome_ink` ни `description_en`, ни `description_ru` не содержат цветовых слов (`find_monochrome_color_word(..., allowed_names=entity_names, strict_palette=False) is None`). При срабатывании напрямую наполняет `allowed_fids`, `available_design_traits`, `allowed_dkeys` и `IllustrationAppearance` без LLM-скоупа `FACT_SCOPE`. (Обновляет 3 точечных unit-теста в `test_illustration_chapter.py`, ожидавших перехват `ILLUSTRATION_FACT_SCOPE` на 1 `permanent`-факте.)
     - **Масштабирование лимита главы и дистанция (`illustration_selection.py`):**
       - `effective_chapter_illustration_cap(max_per_chapter, content_block_count, scene_count)`: `content_block_count <= 2 and scene_count <= 1 -> min(max_per_chapter, 1)`, `3 <= content_block_count <= 8 and scene_count <= 2 -> min(max_per_chapter, 2)`, иначе `max_per_chapter` (ограничивает 2-абзацную гл. 4 Everlost до 1 иллюстрации и полностью сохраняет все многосценные и 3/10-блочные существующие тесты!).
       - `min_block_gap = 2` в `moments_conflict_diversity` / `enforce_intra_chapter_diversity` при `content_block_count >= 12` (и `0` по умолчанию при отсутствии `block_order_by_id`).
  5. **Гигиена промптов (`illustration_context.py`, `illustration_reference.py`, `illustration_chapter.py`):**
     - `EXTRACT`: запретить `back_applicable=true` для одежды конкретного эпизода, загрязнений и посмертных эффектов.
     - `DESIGN`: при `len(permanent_facts) == 1` и наличии `episodic`-фактов добавлять отдельное поле `"episodic_context_only"` (до 2 самых ранних, внутри wire budget), не меняя `facts`, `non_permanent_context_only`, `omitted_episodic_count`, `omitted_fact_count` (`test_illustration_reference.py:1158-1169`).
     - `FACT_SCOPE`: сохранять привязку к транспорту/локации и не трактовать образные реплики об одежде как телесные деформации.
     - `PROMPT`: передавать `"anchor_quotes"` (до 3 цитат по <= 240 символов, внутри wire budget); детерминированно отфильтровывать строки `assumptions_ru` с техническими ключами (`attire_default`, `description_en`, `characters[]`, `used_design_keys`, `depicted_entity_ids`, `compact_description_en`, `palette_value`) без траты repair-вызова.

---

## ADR-4: Синхронизация правил диалогов в `reader_text.py` с валидатором `validators/dialogue.py`

- **Контекст:** По коду `context_builder.py:3701-3776` подтверждено, что `_A2_DIALOGUE_AND_CALQUE_RULES` и `_A2_NOTE_DISCIPLINE` (`reader_text.py:508-528`) **не входят** ни в один из замороженных SHA-256 дайджестов (`test_baseline_b2_templates_sha256_unchanged`) и одновременно подключаются во все рабочие шаблоны V2/V3/A2. При этом `validators/dialogue.py:173` и `187` в паре отклоняли оба корректных варианта перевода смешанного абзаца гл. 2 B35 (однострочный и двухстрочный с встроенной цитатой `«Мёртвый лес, — сказал папа… — Так бывает.»`).
- **Решение:**
  1. Напрямую добавить в `_A2_DIALOGUE_AND_CALQUE_RULES` правила оформления реплики после авторской фразы с точкой (`\n— ` либо `: «…»`) и мн. ч. «вы» при обращении к группе собеседников; в `_A2_NOTE_DISCIPLINE:512` уточнить, что в инфинитив с `to` приводятся только глаголы (не существительные с `a/an/the`).
  2. В `validators/dialogue.py`: ввести общий чистый предикат `_is_allowed_embedded_quote_span(text, span_start, span_end, *, source_text)` — истинен для пары `«…»` **не в начале строки** (после уже закрытой прямой речи и авторского текста в том же абзаце), когда `source_text` содержит >= 2 отдельных закавыченных реплик, разделённых авторским текстом. Применять его **одновременно** в `_has_author_speech_inside_guillemets` (L173) и в `_has_inline_dialogue_dash` (L187, L314-329), чтобы пропускать оба корректных варианта перевода B35, по-прежнему отклоняя `«Привет, — сказал он»` с начала строки.

---

## ADR-5: Единый сегментный поиск `glossary_hints`, неспойлерное обиходное имя говорящего и фильтр тавтологий в раскладке A2

- **Контекст:** В смешанном блоке гл. 2 B10 `_hint_draft()` и `_hint_segment_index()` независимо вызывали `_find_end(block.text, hint.original)` от начала блока, прицепляя `cliff` к переведённому сегменту 0. При этом наивная замена на `anchor.end` сдвинула бы маркер в конец широкой фразы `Difficulty`, а наивный парсинг `note` в `_SpeakerIndex` мог бы перебить титульные имена (`дядя Вернон`, `мистер Дёрсли`) или проспойлерить имя из будущей главы.
- **Решение:**
  1. **Единый хелпер `find_hint_marker_end(...)` в `layout/notes.py` (используемый и в `_hint_draft`, и в `_hint_segment_index`):**
     - Приоритет 1 (безусловный): если у `hint` есть привязанная `Difficulty` в `known` с опорой `anchor` в текущем `block_id`, сначала искать `_find_end(text[anchor.start:anchor.end], hint.original)` внутри диапазона опоры (возвращая `anchor.start + local_end`, fallback `anchor.end`) — это сохраняет точное положение маркера на слове и никогда не уводит легитимную `TRANSLATE`-подсказку в другой сегмент.
     - Приоритет 2 (когда привязанной опоры нет): для сегментированного блока искать `_find_end` сначала внутри спанов `GLOSSARY_ONLY`-сегментов, затем `READ_AS_IS`-сегментов, затем fallback на весь `block.text`.
     - Передавать `plan` в `_hint_segment_index()` (`layout/book.py:408, 1035`), гарантируя 100% согласованность `_hint_is_glossary_only()` и `_hint_draft()`.
  2. **Неспойлерное обиходное имя и фильтр самопредставления в «Говорит …» (`layout/book.py:599-697`):**
     - Извлечение пары `(short_en, short_ru)` для `TermConcept(type=PERSON)` строго детерминировано: из скобок `canonical_source_text`, из явных пар сокращения в `concept.note` (вида `Allie (Элли)` / `Allie — Элли` / `сокращ.`) или совпадения латинского имени из `concept.note` с `GlossaryCandidate(type=PERSON)`.
     - Никогда не сокращать титульные/родственные формы (`Mr/Mrs/Miss/Ms/Dr/Professor/Uncle/Aunt + <name>` — сохраняет `test_a2_harry_potter_regressions.py:1028-1035`).
     - Частоту `short_en` vs `canonical_source_text` считать **накопительно до текущей секции (`<= section.order`)**, чтобы не спойлерить имя из будущей главы; переключаться на `short_ru`, только если `short_en` строго чаще `canonical_source_text`.
     - В `_speakers_for_block()` подавлять «Говорит …», если `part` содержит явное самопредставление (`My name's / My name is / I'm / I am / Call me <name_token>`) по не-служебному токену имени (исключая `the`, `a`, `an`, `boy`, `girl`, `man`, `woman`, `Mr`, `Mrs`, `Professor`).
  3. **Фильтр тавтологий (`layout/notes.py`, после L779):**
     - Проверять `_is_tautological_gloss(translation, kept_note)` **после** вычисления нормализованного `translation` (после `_strip_leading_translation` и инфинитивной нормализации), сопоставляя пустые слова-заглушки по основам (`люд`, `человек`, `вещ`, `предмет`, `мест`, `действ`, `состоян`, `явлен`).
