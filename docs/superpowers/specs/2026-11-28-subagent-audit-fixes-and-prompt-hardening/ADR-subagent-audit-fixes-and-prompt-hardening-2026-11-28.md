# ADR: Архитектурные решения по атомарности старта, экспорту EPUB, CLI и усилению промптов/валидаторов A2

**Дата:** 2026-11-28  
**Статус:** Approved (одобрено по итогам 3 независимых ревью кода и архитектуры; все замечания устранены)  
**Связанный BRIEF:** `BRIEF-subagent-audit-fixes-and-prompt-hardening-2026-11-28.md`

---

## ADR-1: Предвалидация маршрута и единый источник `route` во всех точках `StartProcessing` (WP1)

- **Контекст:** В `container.py` `StartProcessing` инстанцируется в **двух** боевых точках входа: `Application.start_processing()` (`L211-230`, используется из CLI и Web UI) и `_start_book_processing(book_id)` (`L466-478`, используется при `auto_start_processing = True` из `analyze_chapter.py:2423-2425`). В обеих точках `StartProcessing.__call__()` (`approve_glossary.py:483-531`) не проверяет совместимость живого маршрута с `book_conditions.payload['route']` до коммита `ProcessingRun` и перехода в `BookStatus.PROCESSING`, а в `work_bindings['run'].correction.route` пишет текущий (или закешированный в `candidate`) маршрут. Кроме того, `web/routes/processing.py:172-186` не перехватывает `ConditionsError` через `input_error_response`, в отличие от `start_analysis` (`L157-158`).
- **Решение:**
  1. Добавить в `BookRuntimeResolver` (`src/linguary/composition/book_runtime.py`) метод `validate_route(self, pinned_route: Mapping[str, Any] | None) -> None`, сверяющий **живые** `self.runtime.current.settings.llm` (наличие `api_key` и `routes_compatible(route_snapshot(current.llm), thaw(pinned_route))`), и поднимающий `ConditionsError("route_unavailable")` при несовместимости.
  2. Добавить в `StartProcessing.__init__` опциональный keyword-only callback `validate_route: Callable[[Mapping[str, Any] | None], None] | None = None` (значение по умолчанию `None` сохраняет совместимость с существующими unit-тестами `test_approve_glossary.py` и `comparison/runner.py` и строго соблюдает границу слоёв `application ← composition`).
  3. В `StartProcessing.__call__()` **до** `transaction.runs.add(run)` и смены статуса книги:
     - если `conditions is not None` и `self._validate_route is not None` — вызвать `self._validate_route(conditions.payload.get("route"))` (при `ConditionsError` транзакция `UnitOfWork` откатывается целиком);
     - при сборке `payload_dict["correction"]`: если `conditions is not None` и `conditions.payload.get("route")` является `Mapping` со строгим набором ключей `{"provider", "base_url", "proxy", "route_id"}` (требование `CorrectionSnapshot.__post_init__`), записать в `corr_payload["route"]` значение `thaw(conditions.payload["route"])` (чтобы не ломать существующие unit-тесты с минимальными моками `{"provider": ...}`).
  4. В `container.py` использовать единую фабрику/сборку `StartProcessing` с `validate_route=book_runtime.validate_route` и живыми настройками `runtime.current.settings` в **обеих** точках: `Application.start_processing()` (`L211-230`) и `_start_book_processing()` (`L466-478`).
  5. В `web/routes/processing.py:172-186` (`start_processing`) обернуть запуск (`start_processing()` и `await application.processing.start(book_id)`) в `try ... except ConditionsError as error: return input_error_response(request, _conditions_http_error(error))` аналогично `start_analysis`, унифицировав JSON-контракт (`detail: {"reason_code": ...}`) с обновлением `tests/integration/web/test_glossary_and_processing.py:406`.
- **Альтернативы:**
  - *Откатывать `ProcessingRun` и статус книги в `except ConditionsError` внутри CLI и Web routes* — отвергнуто: дублирует компенсацию в трёх точках входа и оставляет окно неконсистентности при падении процесса между транзакциями.
  - *Перенести `processing.start()` внутрь `StartProcessing.__call__()`* — отвергнуто: `processing.start()` асинхронный, а `StartProcessing` — синхронный use case с границей `UnitOfWork`.
- **Последствия:** Невозможно создать `ProcessingRun` или перевести книгу в `PROCESSING` при недоступном маршруте ни из CLI/Web, ни через автостарт; маршрут `correction` в `work_bindings['run']` гарантированно совпадает с `book_conditions`.

---

## ADR-2: Классификация обложки, перенос `dc:rights`/`dc:description` без миграций БД, очистка ресурсов и CSS в EPUB/Web (WP2)

- **Контекст:**
  1. В `is_cover_section()` (`domain/document.py:703-734`) ранние правила `L715-725` распознают явные обложки (`cover*` в пути, `"Cover"` в `<title>` или заголовке), а fallback-цикл `L727-733` проверяет только `block.type in _TEXT_BLOCK_TYPES`, не учитывая `BlockType.HEADING`. Титульник с заголовками и декоративной картинкой попадает в fallback и ошибочно признаётся обложкой.
  2. `BookRow` (`adapters/db/models.py:45-61`) хранит метаданные книги в отдельных реляционных столбцах (`title`, `authors_json`, `identifier`, `publisher`), а не в JSON. При инварианте «0 миграций Alembic» поля `rights` и `description` нельзя просто положить в БД.
  3. `ExportBook._plan()` (`export_book.py:288`) всегда передаёт `layout` в `EpubWriteRequest` (и в `FULL`, и в `STRUCTURE_ONLY`), а в `EpubExportWriter.write()` (`writer.py:300-341`) `_resource_payloads()` вызывается **до** `_render_chapters()`, из-за чего неиспользуемые CSS, `.xpgt` и картинки отброшенных секций попадают в архив.
  4. В `linguary.css` отсутствуют стили для переносимых классов `.center` / `.pagebreak`, а `.lg-notes > li { text-indent: -1em }` ломает выравнивание ненумерованных заметок атрибуции говорящего и в EPUB (`layout_render.py:246`), и в Web UI (`web/templates/partials/result_block.html:70`).
- **Решение:**
  1. **Fallback `is_cover_section()`:** Изменять **только** цикл `L727-733` (`if (block.type in _TEXT_BLOCK_TYPES or block.type is BlockType.HEADING) and block.text.strip(): return False`), **не трогая** ранние правила `L715-725` (`cover*`, title/heading `"Cover"`). Таким образом, все явные обложки (включая все 6 кейсов в `test_epub_writer.py:135-259` и `test_epub_export_content.py:774`) по-прежнему возвращают `True`, а титульные страницы без имени `cover*` с заголовками + картинкой возвращают `False`.
  2. **`dc:rights` и `dc:description` при 0 миграций БД:**
     - Добавить опциональные поля `rights: str | None = None` и `description: str | None = None` в `BookMetadata` (`domain/document.py:359-370`) и заполнять их в `adapters/epub/reader.py` при чтении OPF.
     - Таблицу `books` (`models.py` / `mapping.py`) не менять (0 миграций Alembic).
     - В `adapters/epub/writer.py` добавить хелпер `_read_source_dc_metadata(source_epub_path)` (безопасно читает первый непустой `dc:rights` и `dc:description` из OPF архива `request.source_epub_path`, возвращая `(None, None)` при отсутствии файла/тегов). В `_append_metadata()` записывать `<dc:rights>` и `<dc:description>` из `document.metadata`, используя `_read_source_dc_metadata(request.source_epub_path)` как fallback.
  3. **Очистка неиспользуемых ресурсов после рендера в `EpubExportWriter.write()`:**
     - Перестроить порядок сборки в `writer.py:299-341`: сначала отрендерить `chapters` (`_render_chapters`, `_insert_book_pages`, `_pronunciation_notices_chapter`) и выполнить `_resolve_references` + сборку `nav` (при этом `used` для генерации уникальных имён служебных страниц по-прежнему резервирует все пути потенциальных ресурсов, чтобы избежать коллизий).
     - Когда `request.layout is not None` (то есть во всех боевых экспортах `ExportBook` — и `FULL`, и `STRUCTURE_ONLY`, сохраняя при этом совместимость с прямыми unit-тестами `writer.py`, передающими `layout=None`), собрать множество `referenced_paths` из всех отрендеренных XHTML-документов (`chapters` + `nav`):
       - атрибуты `src`, `href`, `{http://www.w3.org/1999/xlink}href`, `data` (`<object>`), `poster` (`<video>`);
       - разрешение относительных путей через существующий `_resolve_target(raw_ref, base=doc_href)` + `urllib.parse.unquote` (игнорируя внешние схемы `http:`, `https:`, `mailto:`, `data:`, `javascript:` и чистые `#anchor`);
       - транзитивный обход локальных `url(...)` и `@import` внутри CSS-файлов, попавших в `referenced_paths`.
     - В `_resource_payloads()` (вызываемом уже с `referenced_paths`) включать и проверять на `MISSING_RESOURCE` только ресурсы, входящие в `referenced_paths` (и синхронно фильтровать `media_types`). Картинка обложки автоматически входит в `referenced_paths` через `cover.xhtml`.
  4. **CSS и разметка ненумерованных заметок:**
     - В `_STYLESHEET` (`writer.py:111-285`) и `web/static/linguary.css`:
       - `.center { text-align: center; text-indent: 0; }`
       - `.pagebreak { page-break-before: always; break-before: page; }` и `p.pagebreak { margin: 0; padding: 0; line-height: 0; }` (сохраняет якоря `id`, не добавляя пустую строку);
       - `.lg-notes > li.lg-note-plain { text-indent: 0; }` (без `::before`, `content:`, цветов — строго в рамках инвариантов `test_layout_render.py:566-604` и `test_a2_harry_potter_regressions.py:1550`).
     - Проставлять `class="lg-note-plain"` при `number is None` **в обоих рендерерах**: `adapters/epub/layout_render.py:246` (`_append_notes`) и `web/templates/partials/result_block.html:70`.
- **Альтернативы:**
  - *Добавлять миграцию Alembic для `rights`/`description` в таблицу `books`* — отвергнуто ради глобального ограничения «0 миграций БД»; `request.source_epub_path` уже обязателен для экспорта и используется `_OpaqueSource`/`_resource_payloads`.
  - *Использовать CSS `:has(.lg-num)`* — отвергнуто: старые и e-ink EPUB-ридеры не поддерживают `:has()`.
- **Последствия:** Чистый EPUB без мёртвых ресурсов, сохранённые `dc:rights`/`dc:description` без миграций, корректная классификация титульников и ровные ненумерованные заметки в EPUB и Web UI.

---

## ADR-3: Операционная полнота CLI для автономного headless-пайплайна (WP3)

- **Контекст:** Для полностью автономного прогона книги через CLI без Web UI и ручных Python-скриптов не хватает трёх операций: подтверждения помеченных моделью терминов глоссария, разрешения кандидатов корректуры `awaiting_human` и строгого отказа при экспорте с неразрешёнными правками. При этом реальный доменный контракт корректуры (`domain/correction.py:100-107`, `ResolveCorrectionCommand` в `resolve_correction_candidate.py:22-30`) принимает действия строго `{'accept', 'keep'}`, непустой `reason` и CAS-хэши `proposal_sha256` / `input_sha256`.
- **Решение:**
  1. Добавить в `cli/app.py` группу команд `glossary` (все флаги `--json` регистрировать с `dest="json_output"`, как требует глобальный обработчик ошибок `main()` на `app.py:64-84`):
     - `glossary list <book_id> [--pending-only] [--json]` — вызывает `application.approve_glossary.review(book_id)`.
     - `glossary accept-all <book_id> [--json]` — вызывает `application.approve_glossary.accept_all_awaiting(book_id)`.
  2. Добавить в `cli/app.py` группу команд `corrections`:
     - `corrections list <book_id> [--awaiting-only] [--json]` — читает кандидатов корректуры напрямую из `tx.work` (чтобы выводить в т.ч. `section_id`, которого нет в `_correction_candidates_projection`, а также `candidate_id`, `block_id`, `state`, `proposal_sha256`, `input_sha256`, `review_reason` и diff до/после).
     - `corrections resolve <book_id> <candidate_id> --action {accept|keep} [--reason <text>] [--proposal-sha256 <sha>] [--input-sha256 <sha>] [--json]` — проверяет `application.resolve_correction_candidate is not None`, при неуказанных CAS-хэшах подбирает текущие `proposal_sha256`/`input_sha256` кандидата из `tx.work.get_correction_candidate(candidate_id)` (при отсутствии или другом `book_id` поднимая `CorrectionCandidateNotFound`), использует `--reason` (по умолчанию `"Решено через CLI"`, проверяя непустоту после `.strip()`) и вызывает `application.resolve_correction_candidate(ResolveCorrectionCommand(...))`.
  3. Добавить в `export` флаг `--strict-corrections`:
     - В `_export` (`cli/app.py:618-647`) проверять наличие кандидатов `awaiting_human` **ДО** вызова `application.export_book(...)` (и до снятия паузы) напрямую через `tx.work` (без подавления `WorkStateError`, которое делает `book_overview()`). При `pending > 0` поднимать `ApplicationError` (код возврата `1`), не создавая выходной файл. Без флага сохраняется текущее поведение (предупреждение в `stderr`).
- **Область действия:** Речь идёт именно об операционной полноте для автономного пайплайна (точечные операции `correct`/`lock`/`skip`/`set_note` по глоссарию остаются в Web UI).
- **Последствия:** Сквозной CLI-пайплайн от импорта до валидного EPUB выполняется без единого обходного скрипта и без дублирования бизнес-логики юзкейсов.

---

## ADR-4: Устранение ошибочного хардкода «Прайвет-драйв», фонетический контроль стратегии `sound` и дисциплина вложенных полей (WP4)

- **Контекст:** В `context_builder.py` фонетически ошибочный книгоспецифичный пример **«Прайвет-драйв»** (*privet* = `/ˈprɪvɪt/`, не *private*) зашит в двух местах: в `_NARRATIVE_ANALYSIS_INSTRUCTIONS` (`L207`) и в `_with_name_policy` (`L3262`, ветка `NameTranslationStrategy.SOUND`). Кроме того, 2 из 12 вызовов `chapter_analysis_narrative` на `flash` пропустили обязательные вложенные поля (`name_translation_assessment.rationale`, `visual_facts.{anchor, subject, detail}`).
- **Решение:**
  1. Убрать «Прайвет-драйв» из **обеих** точек `context_builder.py`:
     - в `_NARRATIVE_ANALYSIS_INSTRUCTIONS:207` заменить на нейтральные универсальные примеры (`«Бейкер-стрит», «Оквуд-лейн»`);
     - в `_with_name_policy:3262` (ветка `SOUND`) заменить часть `и «Тисовая улица» вместо «Прайвет-драйв»` на нейтральный универсальный пример смыслового перевода vs звучания (например, `и «Дубовая аллея» вместо «Оквуд-лейн»`), обновив ровно 2 теста, привязанных к этим строкам (`test_illustration_reliability.py:938` и `test_context_builder.py:76-78`).
  2. В общую ветку `NameTranslationStrategy.SOUND` (`_with_name_policy:3247-3262`, для анализа и перевода одинаково) добавить правило фонетической сверки: при передаче имён и составных топонимов по звучанию опираться на реальное английское произношение слова (не подменяя его созвучным, но другим словом с иной гласной или ударением), а при сомнении в произношении редкого топонима на этапе анализа ставить `model_flagged_review = true`.
  3. В `_NARRATIVE_ANALYSIS_INSTRUCTIONS` кратко зафиксировать обязательность всех вложенных полей у каждого элемента `glossary_candidates[].name_translation_assessment` (включая `rationale`) и каждого `scenes[].visual_facts[]` (`anchor`, `subject`, `detail`).
- **Защита замороженных инвариантов:** `_ANALYSIS_INSTRUCTIONS` (`L2427-2430`) и `_TRANSLATION_INSTRUCTIONS` не затрагиваются; `test_baseline_b2_templates_sha256_unchanged` вызывает `snapshot_prompt_templates` с `profile="legacy-v1"` и `name_translation_policy=None`, поэтому остаётся побайтово неизменным. Строки `L3250-3259` (правило Ермоловича, `McGonagall` → `Макгонагалл`, `маглы`) не трогаются во избежание избыточной переработки регрессионных тестов. `A2_CONTENT_PLAN_V8_VERSIONS` (`prompts: 8`, `context_builder: 6`) не меняется.

---

## ADR-5: Кореференция между сегментами, калибровка режимов A2 и снятие артиклей в дедупликации/A1-фильтре `glossary_hints` (WP5)

- **Контекст:** Аудит выявил три проблемы A2-адаптации:
  1. Разрыв антецедента на границе сегментов (`GLOSSARY_ONLY`/`READ_AS_IS` → `TRANSLATE`): в русском переводе повисает местоимение 3-го лица («открыл её», «Она могла бы») без существительного в русском потоке.
  2. Размытая калибровка режимов: промптовый потолок `GLOSSARY_ONLY` в `45 слов` указан в 4 местах `context_builder.py` (`L2154, 2196, 2198, 3686`) и пропускает длинные предложения с *Complex Object* и эллипсисами, а короткие реплики A1 (1–3 слова) выпадают из окна `4–8 слов` (`L2166-2173`).
  3. В `layout/book.py` **уже** работает книжная 3-ступенчатая дедупликация `_deduplicate_glossary_hints()` (`L1515-1655`) с окном `_HINT_LOCAL_WINDOW_BLOCKS = 20` и предфильтром `_a1_pre_suppressed_indices()` (`L1473-1512`). Причина просачивания дублей (`reptile house` / `the reptile house`) и A1-существительных (`a glass`, `the paper`) — в том, что `vocabulary_dedup_key()` (`layout/notes.py:609-619`) и `is_trivial_a1_hint()` (`a2_core_vocab.py:1124-1145`) снимают `to `, но не снимают ведущие артикли `a`/`an`/`the`.
- **Решение:**
  1. **Кореференция между сегментами:** дополнить правила в конце `_PEDAGOGICAL_ANALYSIS_INSTRUCTIONS` и `_CONTENT_PLAN_PROCESSING_V3` (`context_builder.py`): не оставлять вводящий антецедент сегмент без перевода, если следующий переводимый сегмент ссылается на него только местоимением, а при переводе `TRANSLATE`-сегмента после непереведённого сегмента восстанавливать существительное вместо «висячего» местоимения 3-го лица.
  2. **Калибровка A2 (`context_builder.py:2150-2199, 3686`):**
     - Единообразно снизить промптовый потолок простого нарратива в `GLOSSARY_ONLY` с `45` до **`30 слов`** во **всех 4 местах** (`L2154`, `L2196`, `L2198`, `L3686`), обновив 3 завязанных теста (`test_context_builder.py:1437, 2393, 3815`). Кодовый предохранитель `_NARRATIVE_GLOSSARY_WORD_LIMIT = 45` в `analysis_merge.py:4832` намеренно **не меняется** (остаётся мягким защитным потолком без ломки `test_staged_analyze_chapter.py` и `test_a2_harry_potter_regressions.py`).
     - Добавить в стоп-лист `GLOSSARY_ONLY` / `READ_AS_IS` без языковой заметки: *Complex Object* (`noticed/saw/heard/watched + объект + инфинитив/причастие`), вложенные придаточные, разрывающие глагол и дополнение, и оборванные эллиптические реплики.
     - Расширить существующее окно коротких элементарных реплик с `4–8 слов` до **`1–8 слов`** (`L2166-2173`), добавив требование согласованности режимов для соседних параллельных реплик одного диалога (без введения второго конкурирующего диапазона).
  3. **Точечное усиление дедупликации и A1-фильтра без дублирования `_deduplicate_glossary_hints`:**
     - В `vocabulary_dedup_key()` (`src/linguary/application/layout/notes.py:609-619`) снимать ведущий артикль (`a`, `an`, `the`) при `len(words) >= 2` (аналогично снятию `to `), чтобы `reptile house` и `the reptile house` давали один ключ дедупликации. Существующий механизм `_deduplicate_glossary_hints()` (`_HINT_LOCAL_WINDOW_BLOCKS = 20`, `glossary_only_rescue_by_bucket`) не дублируется и сохраняет все свои инварианты.
     - В `is_trivial_a1_hint()` (`src/linguary/application/a2_core_vocab.py:1124-1145`) снимать ведущий артикль `a`/`an`/`the` перед проверкой одного слова (проверяя его только против существительных/A1-словаря, без ложного срабатывания на глагольные конструкции с `to`), чтобы `a glass`, `the paper` отфильтровывались наравне с `glass`, `paper`.
     - В `_A2_NOTE_DISCIPLINE` (`reader_text.py:508-514`) усилить промптовый запрет на базовую лексику A1–A2 в прямом значении и повторы в пределах главы, а также добавить напоминание о формате цитат (английские слова в кавычках, без кириллицы внутри английской цитаты).
- **Альтернативы:**
  - *Добавлять второй механизм дедупликации с окном `N = 15`* — отвергнуто: дублирует уже работающий `_deduplicate_glossary_hints` (`N = 20`) и ломает его тесты в `test_book_layout.py`.
- **Последствия:** Отсутствие «висячих» местоимений, последовательная калибровка A2, закрытие обхода дедупликации и A1-фильтра через артикли без регрессий в `layout/book.py`.

---

## ADR-6: Автоматный валидатор в `dialogue.py`, поддержка британских реплик в `paragraphs.py` и единая норма в промптах (WP6)

- **Контекст:** В прогоне 6 раз встретилось лишнее тире перед самостоятельным авторским предложением с прописной на той же строке после закрытой реплики (`— Молодчина, Дадли! — Он взъерошил Дадли волосы.`, `…его может стошнить. — Потом он побежал…`). В `_A2_DIALOGUE_AND_CALQUE_RULES` (`reader_text.py:524`) и `CORRECTION_PROMPT_V2` (`correction_runtime.py:91`) сейчас противоречиво разрешена однострочная форма `«— Реплика? — Авторское предложение.»`, что порождает двусмысленность с репликой второго собеседника. При этом `validators/paragraphs.py:23` (`_POST_SPEECH_CLOSE_RE`) разрешает перенос авторского предложения на новую строку без тире только после `[.!?…]["”’']`, но не после британской реплики с запятой + ремаркой + точкой (`‘Hello,’ he said. Then he ran.`).
- **Решение:**
  1. **Единая норма в промптах** (`_A2_DIALOGUE_AND_CALQUE_RULES` в `reader_text.py:524` и `CORRECTION_PROMPT_V2` в `correction_runtime.py:91`):
     - Слова автора после реплики — через тире со **строчной** буквы (`— Реплика! — сказал он.`).
     - Самостоятельное авторское предложение с прописной буквы после закрытой реплики — **с новой строки без тире** (`— Реплика!\nАвторское предложение.`, а если речь продолжается — три строки: `— Реплика 1.\nАвторское предложение.\n— Реплика 2.`).
     - Явно запретить вводить самостоятельное авторское предложение через `— ` с прописной буквы на той же строке (`— Реплика? — Авторское предложение.` или `— Реплика 1. — Авторское предложение. — Реплика 2.`), а также по-прежнему запрещать 2 строки `Авторское предложение. — Реплика 2.` и кавычки-ёлочки с начала абзаца.
  2. **Снятие Catch-22 в `validators/paragraphs.py`:**
     - Расширить `_count_post_speech_narrative_sentences(source_text)` (`paragraphs.py:48-58`): помимо `[.!?…]["”’']\s+[A-Z]` учитывать также закрытие реплики запятой с последующей ремаркой, заканчивающейся точкой/восклицанием/вопросом, за которой вне кавычек стоит самостоятельное предложение с прописной буквы (например, `‘Hello,’ he said. Then he ran.` → `1`, при сохранении `0` для `'"Hello," he said. "How are you?"'`, где после точки открывается новая кавычка).
  3. **Автоматная проверка в `validators/dialogue.py` (`strict_v3=True`):**
     - Добавить `_has_block_post_quote_narrative_dash(text, *, block, item, source_text)` и `_has_dash_before_post_quote_narration(text, *, source_text)`, вызываемые из `_text_dialogue_violation(...)` при `strict_v3=True` (симметрично `_has_block_inline_dialogue_dash`; при наличии `block.segments` проверять каждый переводимый сегмент против его `source_span`).
     - Алгоритм опирается на конечный автомат состояний строки, начинающейся с `—`/`–` (аналогично `_count_embedded_post_speech_narratives` в `paragraphs.py:27-45`), с защитой от сокращений `_ABBREVIATION_BEFORE_DASH`:
       - `speech` + `[,!?.…]\s+[—–]\s+[а-яё]` → `attribution` (слова автора со строчной);
       - `attribution` + `[.!?…]\s+[—–]\s+[А-ЯЁ]` → `speech` (новая реплика после слов автора; увеличиваем счётчик `speech_segments`);
       - `speech` + `[.!?…]\s+[—–]\s+[А-ЯЁ]` → **нарушение** (`embedded_post_narrative`: самостоятельное авторское предложение введено через `— ` с прописной прямо после реплики, например `— Молодчина! — Он взъерошил…` или `one_line` в `ollivander_source`);
       - **дополнительный случай с ремаркой перед хвостовым авторским предложением:** если в `source_text` по спанам `_outer_unclosed_open` есть хвостовое авторское предложение после последней закрывающей кавычки без новой открывающей кавычки, и суммарное число `speech`-сегментов в переводе (по всем строкам блока/сегмента) **строго больше** числа верхнеуровневых реплик `len(quote_spans)` в `source_text` (например, `‘Hello,’ he said. Then he ran.` → `— Привет, — сказал он. — Потом он убежал.`: в оригинале 1 реплика, в переводе 2 `speech`-сегмента) → **нарушение**!
     - Легитимная русская речь (`— Посмотри на это! — крикнул он. — Посмотри! — ахнула тётя Петуния.`, `— Первая, — сказал он. — Вторая.`) даёт `embedded = 0` и `speech_segments == len(quote_spans)` → **0 ложных срабатываний**.
     - Детерминированный валидатор работает на этапе `translation_unit` (`strict_v3=True`), а обновлённый `CORRECTION_PROMPT_V2` исправляет редкие остатки/существующие переводы на этапе `chapter_correction`.
  4. **Обновление 3 существующих тестов:** синхронизировать `test_dialogue_validator.py:657-681`, `test_result_validators.py:3787-3836` и `test_correction_runtime.py:342-350` с отказом от однострочного `«— Реплика? — Авторское предложение.»` и сохранением 3-строчной формы.
- **Альтернативы:**
  - *Сохранять однострочное `«— Реплика? — Авторское предложение.»` и проверять только последний хвост* — отвергнуто: создаёт расхождение между промптом и валидатором и даёт ложные срабатывания на легитимных 2-репликовых конструкциях.
- **Последствия:** Все 6 реальных дефектов прогона и аналогичные случаи с запятой+ремаркой отлавливаются детерминированно без единого Catch-22 и без ложных срабатываний на нормативной русской прямой речи.
