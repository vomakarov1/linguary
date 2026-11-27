# BRIEF: Устранение блокеров пайплайна, масштабирование иллюстраций, устойчивость SQLite и качество A2/стихов (Harry Potter audit)

**Дата:** 2026-11-27  
**Статус:** Approved (пройдены 3 независимых ревью: Spec Architect Reviewer, Regression & Affected Code Auditor, Final Independent Spec Auditor; все замечания устранены)  
**Контекст:** Пост-анализ полного сквозного прогона *Harry Potter and the Philosopher's Stone* (главы 1–8 + первые 2 абзаца главы 9; `hp-ch1-ch9p2-gpt61terra-20261127` на `main` `4e143d4`, модель `openai/gpt-6.1-terra`, уровень `a2`, профиль `content-plan-v3`, `staged_analysis = true`, `sequential_analysis = true`), два ревью результата (архитектор + лингвист), исследование первопричин и три независимых ревью спецификаций.

---

## 1. Проблема

Прогон 9 глав (1 379 блоков, 1 333 LLM-вызова, $12.07) показал, что на чистом `HEAD` книга **не может быть обработана автономно до конца**, а стоимость и надёжность нескольких подсистем не масштабируются на полные книги:

1. **Блокер анализа главы 7 (P0, WP1):**
   - В `analysis_merge.py:2825, 2854-2855` ветка `lowercase_non_glossary` вызывает `scan(require_capitalized=True)`, пропуская строчные вхождения `first-years` (блоки 19/21/28/32) и цепляя начало предложения `First-years` в блоке 145 → ложный `spoiler_name_leak`.
   - В `analysis_merge.py:3563-3595, 7199-7230, 7600-7644` `_opening_remark_names_speaker` пословно и без учёта регистра матчит строчное `"fat"` в `"a fat little monk"` на `"Fat Friar"` и повышает `inferred → explicit` **после** локального `_collect_spoiler_name_leaks`; общеглавная валидация (`analyze_chapter.py:582`), чтение кэша (`620`) и `illustration_source.py:930` падают на собственном повышенном факте, а слияние сразу сбрасывает оплаченный кэш всех фрагментов.
2. **Падение SQLite `database is locked` при `worker.concurrency=4` (P0, WP2):**
   - `SqliteTaskStore.save()` и `enqueue()` (`task_store.py:98, 160`) используют `DEFERRED`-транзакции без `BEGIN IMMEDIATE` и без retry → апгрейд `SHARED → RESERVED` мгновенно кидает `SQLITE_BUSY`, минуя `busy_timeout=5000`.
   - `TaskWorkerPool._drain_worker()` (`worker.py:167-171`, режим `cli worker --once`) не перехватывает `is_transient_lock_error`.
   - `dispatch_scope_task()` (`illustrations.py:102-106`) вызывает `tx.tasks.list_for_book()` в цикле 540 раз подряд под одним `BEGIN IMMEDIATE` ($O(N^2)$ под эксклюзивным write-локом).
3. **Квадратичный взрыв `alias`, шум в `get_registry`, лимиты схем и дефекты промптов иллюстраций (P0–P1, WP3):**
   - `build_alias_candidate_pairs()` (`illustration_reference.py:328-396`) строит клики $K_n$ внутри бакетов (`Harry` × 68 → 2 278 пар; всего 7 484 пары / **540 вызовов `alias` / $3.87 = 61.6% бюджета иллюстраций**, 1 035 из 1 333 вызовов всей книги).
   - `build_illustration_registry_view()` (`illustration_view.py:1063-1145`) при `kind=None` возвращает все 7 484 служебных `alias_decision` (8 440 записей при 188 сущностях и 672 фактах).
   - Конфликт лимитов `used_fact_ids`: `_MAX_APPLICABLE_FACTS = 32` на сущность (`illustration_chapter.py:105`) против `_MAX_LIST = 32` на весь момент (`illustration_payloads.py:35, 370`).
   - Английские `name_ru` во всех 25 моментах и дрейф русских имён (`Дурслей`, `Клык`, `Хогвартс`): иллюстрации читают только `term_concepts` (8 записей), игнорируя 177 `suggested`-терминов книги.
   - Дефекты промптов: загрузка всех `episodic`-фактов со всей книги в `FACT_SCOPE` (носки из чулана гл. 2 в зоопарке; `baggy old clothes` из гл. 2 вместо мантии в гл. 8); 10-летние факты Гарри у младенца в гл. 1 (`is_back_applicable`); комикс-баббл `"GRYFFINDOR!"` в гл. 7 из-за отсутствия запрета в `_STAGE_INSTRUCTIONS[PROMPT]`; рассинхрон `title_ru` («Хагрид») и `name_ru` (`"the man"`) в гл. 1 мом. 2; дробность `illustration_chapter` (307 вызовов на 25 моментов).
4. **39.2% ошибок `chapter_correction` (20 из 51 вызова) (P0, WP4):**
   - `CorrectChapter._candidates()` (`correct_chapter.py:933-956`) валидирует через одноблочный `TranslationUnit` → `build_repair_instruction()` требует вернуть ровно 1 блок → модель возвращает `ret_blocks=1` → `LLMPayloadError` в `validate_for()` (9 из 20 ошибок).
   - Catch-22 между `paragraphs.py:37` и `dialogue.py:305, 320` на однострочных абзацах вида «реплика + авторское предложение + реплика» (11 из 20 ошибок + 6 «слипшихся» диалогов без тире в EPUB).
5. **Заморозка операционных параметров и прокси в `book_conditions` (P1, WP5):**
   - `timeout_seconds=180.0`, `requests_per_minute`, `max_attempts` и адреса прокси из env заморожены в SQLite (`book_runtime.py:64-100, 473-493`; `correction_runtime.py:206-219`), из-за чего ретраи длинных глав (6, 7) не могут поднять таймаут, а смена прокси блокирует книгу ошибкой `route_unavailable`.
6. **Вёрстка сносок в EPUB и ложные `result_warnings` глоссария (P1, WP6):**
   - В `adapters/epub/layout_render.py:302` маркеры с одинаковым `offset` вставляются в обратном порядке `<sup>2</sup><sup>1</sup>` (визуально «21», 3 места в EPUB).
   - 9 из 10 `result_warnings` — ложные срабатывания (`validators/language.py:261-333`): `re.IGNORECASE` и `(?<!\w)` матчат `Moon` на `half-moon` / `full moon` и `Bill` на `a bill`, а `Sorting` срабатывает внутри `The Sorting Hat`.
7. **Педагогика A2, стихи, `GLOSSARY_ONLY`, `fragment_assistance` и дедупликация акцента (P1–P2, WP7):**
   - `legend.py:33` обещает подстрочный перевод стихов без рифмы, но `_POEM_TRANSLATION_RULE` подаётся только на `BlockType.POEM` (не на `PARAGRAPH`-стихи), а `_NARRATIVE_ANALYSIS_INSTRUCTIONS` не запрещает требовать рифму → провал песни Распределяющей шляпы в гл. 7.
   - `detect_verse_stanza_runs()` (`verse.py:136-158`) теряет хвостовую строфу `#62–#65` (3 end-stopped строки подряд) и 2-строчное заклинание `#215–#216` → разрыв стихов режимом `READ_AS_IS`.
   - 11 из 16 ошибок `translation_unit` — пустой `glossary_hints: []` на `GLOSSARY_ONLY`.
   - Оба `unresolved` в `fragment_assistance`: в `layout/notes.py:118` нет `"open"` для `"flicked it open"` ↔ `"to flick open"`, а в `a2_core_vocab.py:1021` `_matches_ing` требует `len(stem) >= 3`, отбрасывая `"going"` (`stem="go"`).
   - 7 повторяющихся заметок про акцент Хагрида: `_repeated_speech_style_ids()` (`layout/book.py:1794-1799`) индексирует только `fact.anchors` (`"said Hagrid"`), забыв `fact.subject_anchor` (саму реплику в кавычках).

---

## 2. Цели

1. Устранить детерминированные отказы анализа на строчных нарицательных участниках и ложном повышении `inferred → explicit`, не ослабляя детекцию реальных строчных спойлеров имён, и добавить точечное понижение `explicit → inferred` после кросс-фрагментного слияния (WP1).
2. Гарантировать устойчивость SQLite и `cli worker --once` при `concurrency >= 4` без `database is locked`, без $O(N^2)$ чтения `processing_tasks` и с сохранением существующих retry-контрактов `UnitOfWork`/`SqliteTaskStore` (WP2).
3. Сократить число LLM-вызовов `alias` на ~85–90% за счёт линейной топологии `звезда + цепочка` и представительского кросс-матчинга (сохраняя `same name != identity`, `alias_of_hint`, защиту от распада омонимов и порядок-независимость), очистить `get_registry(kind=None)`, согласовать `used_fact_ids = 128`, локализовать `name_ru` / `glossary_terms_ru` с учётом `suggested`-глоссария без взрыва вызовов `DESIGN`, устранить 5 классов дефектов промптов и сократить дробность `illustration_chapter` (WP3).
4. Устранить 1-блочный баг repair-промпта (вкл. `LLMPayloadError`) и Catch-22 `paragraphs.py`/`dialogue.py` в `chapter_correction`, снизив долю ошибок коррекции с ~39% до единичных случаев (WP4).
5. Развести контентные и транспортно-операционные настройки в `book_conditions` / `CorrectionSnapshot` с сохранением `public_url()` и 100% обратной совместимостью старых БД (WP5).
6. Починить порядок и визуальное разделение смежных сносок в EPUB (во всех позициях, включая конец блока) и убрать ложные `result_warnings` глоссария (WP6).
7. Согласовать политику подстрочного перевода стихов, закрыть разрывы строф/двустиший `READ_AS_IS`, снизить пустые `GLOSSARY_ONLY` без изменения заголовков блоков, устранить обе причины `unresolved` в `fragment_assistance` и починить дедупликацию `speech_style`-заметок (WP7).

---

## 3. Осознанные не-цели (Non-goals)

- **0 миграций БД / Alembic:** схема SQLite не меняется; новые поля в JSON обратно совместимы со старыми записями БД.
- **Замороженные SHA-256 не меняем:** `_TRANSLATION_INSTRUCTIONS`, `snapshot_prompt_templates(..., reader_level="b2", profile="legacy-v1")` (с/без `pronunciation_help_enabled`) остаются побайтово неизменными (`test_baseline_b2_templates_sha256_unchanged`).
- **Не ломаем OpenAI strict JSON Schema:** не добавляем `minItems` и `default_factory` там, где поля обязательны.
- **Не вводим хардкодов под книги, персонажей, локации или модели LLM.**
- **Не меняем семантику `before_you_read = []` для A2** (`test_normalize_a2_dialogue_response_clears_before_you_read`) и не ослабляем запрет `«Привет, — сказал он»` с начала строки в `validators/dialogue.py`.
- **Не меняем `query_registry_items(..., kind=None)` в репозитории** (используется для `registry_sha256`); фильтрацию `alias_decision` выполняем на уровне view через опциональный параметр `kinds`.
- **Не меняем `10` попыток и backoff `[0.005..0.02]`** в `UnitOfWork` и `SqliteTaskStore` (закреплены 3 unit/integration тестами).
- **Не делаем масштабный рефакторинг разбиения `analysis_merge.py` (7 711 LOC)** в рамках этого цикла.

---

## 4. Критерии приёмки

- `uv run pytest tests/architecture`, `uv run pytest -m "not slow"` и `make check` (Ruff + ty + architecture tests) полностью зелёные.
- Новые unit- и integration-тесты покрывают все 7 рабочих пакетов (WP1–WP7) и подтверждают обратную совместимость со старыми записями SQLite.
- Все замороженные и критические инварианты (`test_baseline_b2_templates_sha256_unchanged`, `test_ambiguous_identity_remains_separate`, `test_transitive_identity_conflict_triangle_order_independent_deterministic`, `test_dialogue_line_breaks_rejected_if_extra_line_lacks_dash`, `test_dialogue_validator.py`, `test_translation_context.py`, `test_book_layout.py:2047-2128`, `test_analysis_merge_scenes.py:1216-1248`, `test_work_state_indexes.py`, `test_recovery_partition.py`) полностью зелёные.
- Повторный контрольный прогон `harry-potter-ch1-ch9p2.epub` (на чистом `HEAD` без обходных скриптов, при `concurrency=4`, `openai/gpt-6.1-terra`, `a2`) проходит автономно от `import` до `export` и `illustrations ready` без блокера главы 7, без `database is locked`, с сокращением вызовов `alias` ≥ 85%, без `<sup>2</sup><sup>1</sup>` в EPUB и с чистым `EPUBCheck: passed`.
