# SDD (Backend): Контракты, алгоритмы и тестовая матрица (Subagent audit fixes & prompt hardening)

**Дата:** 2026-11-28  
**Статус:** Approved (одобрено по итогам 3 независимых ревью кода и архитектуры; все замечания устранены)  
**Связанные документы:** `BRIEF-subagent-audit-fixes-and-prompt-hardening-2026-11-28.md`, `ADR-subagent-audit-fixes-and-prompt-hardening-2026-11-28.md`

---

## Глобальные ограничения

- **Границы слоёв:** `domain ← application ← adapters / web / cli` (`uv run pytest tests/architecture`). В `domain` и `application` нет сторонних библиотек и прямого I/O.
- **Схема БД:** 0 миграций Alembic. Таблица `books` (`adapters/db/models.py:45-61`) и `adapters/db/mapping.py` не меняются.
- **Замороженные SHA-256 и версии:** `_TRANSLATION_INSTRUCTIONS`, `_ANALYSIS_INSTRUCTIONS` (`L2427-2430`) и `snapshot_prompt_templates(source_language="en", target_language="ru", reader_level="b2", profile="legacy-v1")` (с/без `pronunciation_help_enabled`, `name_translation_policy=None`) остаются побайтово неизменными (`test_baseline_b2_templates_sha256_unchanged`). `A2_CONTENT_PLAN_V8_VERSIONS` (`prompts: 8`, `context_builder: 6`) не меняется.
- **Универсальность:** 0 новых хардкодов под конкретные книги, персонажей, локации или модели LLM; ошибочный хардкод «Прайвет-драйв» удаляется из обеих точек `context_builder.py`.
- **Совместимость схем:** OpenAI strict JSON Schema не меняется.

---

## WP1 — Атомарность `StartProcessing` и защита от split-brain маршрутов

### 1.1 `src/linguary/composition/book_runtime.py`

1. **`BookRuntimeResolver.validate_route(self, pinned_route: Mapping[str, Any] | None) -> None`:**
   - Добавить публичный метод на `BookRuntimeResolver` (`L454-540`):
     ```python
     def validate_route(self, pinned_route: Mapping[str, Any] | None) -> None:
         current = self.runtime.current.settings
         if not current.llm.api_key or not isinstance(pinned_route, Mapping):
             raise ConditionsError("route_unavailable")
         if not routes_compatible(route_snapshot(current.llm), thaw(pinned_route)):
             raise ConditionsError("route_unavailable")
     ```
   - Всегда читает живые `self.runtime.current.settings` (а не закешированный снимок генерации).

### 1.2 `src/linguary/application/use_cases/approve_glossary.py`

1. **Импорты (`L1-25`):**
   - Добавить `Any` и `Mapping` (из `typing` / `collections.abc`) и вынести `thaw` из локального импорта `L548` на уровень модуля.
   - Определить константу `_CORRECTION_ROUTE_KEYS = frozenset({"provider", "base_url", "proxy", "route_id"})` (строгий набор ключей `CorrectionSnapshot.__post_init__` в `correction_dto.py:56-57`).
2. **`StartProcessing.__init__` (`L455-475`):**
   - Добавить опциональный keyword-only параметр:
     ```python
     validate_route: Callable[[Mapping[str, Any] | None], None] | None = None
     ```
   - Сохранить `self._validate_route = validate_route`. Значение `None` по умолчанию сохраняет совместимость с существующими unit-тестами и `comparison/runner.py`.
3. **`StartProcessing.__call__` (`L476-532`):**
   - Сразу после получения `conditions` (`L485-488`) и **до** `transaction.runs.add(run)` (`L515`):
     ```python
     if conditions is not None and self._validate_route is not None:
         self._validate_route(conditions.payload.get("route"))
     ```
   - При формировании `payload_dict["correction"]` (`L526-527`):
     ```python
     if self._correction_snapshot is not None:
         corr_payload = self._correction_snapshot.to_payload()
         pinned_route = conditions.payload.get("route") if conditions is not None else None
         if (
             isinstance(pinned_route, Mapping)
             and frozenset(pinned_route) == _CORRECTION_ROUTE_KEYS
         ):
             corr_payload = {**corr_payload, "route": thaw(pinned_route)}
         payload_dict["correction"] = corr_payload
     ```
   - *Инварианты:*
     - при `ConditionsError("route_unavailable")` транзакция `UnitOfWork` откатывается целиком (`ProcessingRun` не создаётся, `work_bindings["run"]` не пишется, статус книги остаётся `GLOSSARY_REVIEW`);
     - минимальные моки `conditions.payload["route"] = {"provider": ...}` в `test_approve_glossary.py:134-145` и `test_block_processing.py:165-175` не портят `corr_payload["route"]`.

### 1.3 `src/linguary/composition/container.py` и `src/linguary/web/routes/processing.py`

1. **`container.py` — обе точки создания `StartProcessing`:**
   - В `Application.start_processing()` (`L211-230`): передавать `validate_route=self.book_runtime.validate_route`.
   - В `_start_book_processing(book_id)` (`L466-478`, автостарт после анализа при `auto_start_processing = True`): использовать живые `live = runtime_settings.current.settings` (`L512`) для `live.llm.ensure_ready()`, `_llm_snapshot(live)`, `correction_snapshot(live)` и передавать `validate_route=book_runtime.validate_route` (доступен в замыкании `L521`).
2. **`src/linguary/web/routes/processing.py` (`L172-186`):**
   - В `start_processing` обернуть `application.start_processing()(StartProcessingCommand(book_id=book_id))` и `await application.processing.start(book_id)` в `try ... except ConditionsError as error: return input_error_response(request, _conditions_http_error(error))` (как в `start_analysis` на `L157-158`), чтобы HTML/HTMX и JSON-клиенты получали единый структурированный ответ `{"detail": {"reason_code": ...}}`.

### 1.4 Тестовая матрица WP1
- `tests/unit/application/test_approve_glossary.py`:
  - `validate_route` бросает `ConditionsError("route_unavailable")` → книга остаётся в `GLOSSARY_REVIEW`, `runs.active_for_book()` равно `None`, `work.get_binding("run", ...)` отсутствует;
  - при полном совместимом `pinned_route` `work_bindings["run"].payload["correction"]["route"]` совпадает с `conditions.payload["route"]`;
  - существующие тесты с минимальным `{"provider": ...}` в `conditions.payload["route"]` остаются зелёными.
- `tests/integration/processing/test_pinned_runtime.py`:
  - сценарий смены порта (`127.0.0.1:51247` → `127.0.0.1:51594`) перед `process`: отказ с `route_unavailable`, статус книги остаётся `GLOSSARY_REVIEW` (а при `auto_start_processing` — `READY_TO_PROCESS`, без `ProcessingRun` и перехода в `PROCESSING`), после возврата `51247` `process` + перевод + `chapter_correction` проходят без единой ошибки `route_unavailable`.
- `tests/integration/web/test_glossary_and_processing.py`:
  - обновить `L406` (`test_processing_start_rejects_explicit_null_name_strategy`) под единый структурированный контракт `response.json()["detail"] == {"reason_code": "conditions_unavailable"}`;
  - добавить тест: `POST /books/{id}/process` при несовместимом маршруте возвращает `409` со структурированным `{"reason_code": "route_unavailable"}` без перевода книги в `PROCESSING`.

---

## WP2 — Экспорт EPUB: `is_cover_section`, `dc:rights`/`dc:description`, очистка ресурсов и CSS

### 2.1 `src/linguary/domain/document.py`

1. **`BookMetadata` (`L358-370`):**
   - Добавить опциональные поля:
     ```python
     rights: str | None = None
     description: str | None = None
     ```
2. **`is_cover_section(section: DocumentSection) -> bool` (`L703-734`):**
   - **Не менять** ранние правила `L715-725` (`stem == "cover" or stem.startswith("cover")`, `title_norm == "cover"`, heading `"cover"` — благодаря этому все 6 явных cover-фикстур в `tests/unit/test_epub_writer.py:135-259` и `tests/integration/epub/test_epub_export_content.py:774` остаются `True`).
   - Расширить **только** fallback-цикл на `L729`:
     ```python
     if (block.type in _TEXT_BLOCK_TYPES or block.type is BlockType.HEADING) and block.text.strip():
         return False
     ```

### 2.2 `src/linguary/adapters/epub/reader.py`

- При чтении OPF читать первый непустой `dc:rights` и `dc:description` (нормализуя пробелы, без падения при их отсутствии) и сохранять в `BookMetadata.rights` / `BookMetadata.description`.
- Таблицу `books` (`adapters/db/models.py`) и `adapters/db/mapping.py` **не менять** (0 миграций Alembic).

### 2.3 `src/linguary/adapters/epub/writer.py`, `layout_render.py`, `web/static/linguary.css`, `web/templates/partials/result_block.html`

1. **Перенос `dc:rights` и `dc:description` (`writer.py:1315-1330`):**
   - Добавить компактный хелпер `_read_source_dc_metadata(source_epub_path: str | None) -> tuple[str | None, str | None]`, который безопасно открывает `source_epub_path` (если передан и существует), находит OPF через `META-INF/container.xml` и извлекает первый непустой `dc:rights` и `dc:description` (при любой ошибке ввода-вывода/разбора возвращает `(None, None)`).
   - В `_append_metadata(metadata, document, *, source_dc: tuple[str | None, str | None] = (None, None))`:
     - `rights = document.metadata.rights or source_dc[0]` → если непусто, добавить `<dc:rights>...</dc:rights>`;
     - `description = document.metadata.description or source_dc[1]` → если непусто, добавить `<dc:description>...</dc:description>`.
   - В `EpubExportWriter.write()` читать `source_dc` один раз, только если `not (request.document.metadata.rights and request.document.metadata.description)`.
2. **Очистка неиспользуемых ресурсов (`writer.py:299-341, 586-625`):**
   - Перестроить порядок этапов в `EpubExportWriter.write()`:
     1. Зарезервировать имена в `used` по множеству потенциальных путей ресурсов + экспортируемых секций + служебных файлов (как раньше, чтобы `pages` и `notices_href`/`provenance_href` не конфликтовали).
     2. Выполнить `_render_chapters`, `_insert_book_pages`, `_pronunciation_notices_chapter`.
     3. Сформировать `nav_root` (предварительно или через выделенный `_navigation_document(...)`) и собрать `referenced_paths: set[str] | None`:
        - если `request.layout is None`: `referenced_paths = None` (сохранить прежнее поведение для прямых unit-тестов без раскладки);
        - если `request.layout is not None` (все боевые экспорты `ExportBook` — и `FULL`, и `STRUCTURE_ONLY`): обойти все отрендеренные XHTML-документы (`chapters` + `nav_root`), извлекая локальные ссылки из атрибутов `src`, `href`, `{http://www.w3.org/1999/xlink}href`, `data`, `poster` через существующий `_resolve_target(raw_ref, base=doc_href)` (добавляя в `referenced_paths` оба варианта пути — нормализованный без `unquote` и после `urllib.parse.unquote`, пропуская внешние схемы `http:`, `https:`, `mailto:`, `data:`, `javascript:` и чистые `#anchor`); затем, если в `referenced_paths` попал перенесённый CSS-ресурс, транзитивно добавить локальные цели `url(...)` и `@import` из этого CSS относительно его пути.
     4. Вызвать `_resource_payloads(request, warnings, referenced_paths=referenced_paths)`: при `referenced_paths is not None` пропускать любой ресурс, чей `_contained_name(resource.source_path)` не входит в `referenced_paths` (не создавая при этом ложных предупреждений `MISSING_RESOURCE`), сохранить `payloads[provenance_href]` (если `notices is not None`) и синхронно отфильтровать `media_types`.
     5. Вызвать `_resolve_references(chapters, known_files=known_files, warnings=warnings)` и `_append_book_payloads(...)`.
   - Картинка обложки автоматически входит в `referenced_paths` через `cover.xhtml` (`<img>` или `<svg><image xlink:href>`).
3. **CSS-правила и разметка ненумерованных заметок:**
   - В `_STYLESHEET` (`writer.py:111-285`) и `src/linguary/web/static/linguary.css` добавить:
     ```css
     .center {
       text-align: center;
       text-indent: 0;
     }

     .pagebreak {
       page-break-before: always;
       break-before: page;
     }

     p.pagebreak {
       margin: 0;
       padding: 0;
       line-height: 0;
     }

     .lg-notes > li.lg-note-plain {
       text-indent: 0;
     }
     ```
     (Строго без `::before`, `content:`, цветов; инварианты `test_layout_render.py:566-604` и `test_a2_harry_potter_regressions.py:1550` сохраняются; `?v=5` в `base.html` не трогать, если не требуется cache-busting, чтобы не ломать `test_templates.py:965`, либо при бампе до `?v=6` синхронно обновить `test_templates.py:965`.)
   - В `src/linguary/adapters/epub/layout_render.py:246` (`_append_notes`):
     - при `numbered.number is None` создавать `<li id="..." class="lg-note-plain">`, при `numbered.number is not None` — `<li id="...">`.
   - В `src/linguary/web/templates/partials/result_block.html:70`:
     - `<li id="{{ item.id }}"{% if item.note.number is none %} class="lg-note-plain"{% endif %}>`.

### 2.4 Тестовая матрица WP2
- `tests/unit/domain/test_canonical_document.py`:
  - секция `title.html` с `BlockType.HEADING` («Title», «Author») + `BlockType.IMAGE` → `is_cover_section` возвращает `False`;
  - секция `cover.html` (в т.ч. с `BlockType.HEADING` / `BlockType.PARAGRAPH`) и чисто графическая секция `front.html` только с `BlockType.IMAGE` → `True`.
- `tests/unit/test_epub_writer.py` и `tests/integration/epub/test_epub_export_content.py`:
  - все существующие тесты обложки (`test_is_cover_section_distinguishes_cover_from_other_front_matter`, `test_learner_export_keeps_cover_and_drops_other_front_matter`) остаются зелёными;
  - `dc:rights` и `dc:description` переносятся в экспортированный `content.opf` как из `document.metadata`, так и через fallback из `source_epub_path` после чтения книги из SQLite;
  - при `layout is not None` неиспользуемые `hp_style.css`, `page-template.xpgt`, картинки из отброшенных front-matter секций отсутствуют в архиве, OPF-манифесте и не порождают ложных `MISSING_RESOURCE`;
  - используемые `<img src="../Images/fig%201.jpg">`, SVG `<image xlink:href>`, `<object data>` и ресурсы из CSS `url(...)` сохраняются и проходят `verify_exported_epub`.
- `tests/unit/adapters/epub/test_layout_render.py` и `tests/integration/web/test_templates.py`:
  - ненумерованная заметка атрибуции рендерится с `class="lg-note-plain"` и в EPUB, и в Web UI, нумерованная — без него;
  - `.center`, `.pagebreak`, `p.pagebreak` и `.lg-notes > li.lg-note-plain` присутствуют в `_STYLESHEET` и `web/static/linguary.css`.

---

## WP3 — Операционная полнота CLI (`glossary`, `corrections`, `export --strict-corrections`)

### 3.1 `src/linguary/cli/app.py`

Все новые парсеры (`glossary`, `corrections`) создаются с `parents=[reader_parent]` для единообразия CLI, а все флаги `--json` регистрируются с `dest="json_output"`, чтобы глобальный обработчик ошибок `main()` (`app.py:64-84`) корректно форматировал ошибки в JSON.

1. **Подкоманда `glossary`:**
   - `glossary list <book_id> [--pending-only] [--json]` — вызывает `application.approve_glossary.review(book_id)` и выводит элементы глоссария (`concept_id`, `source_text`, `decision`, `requires_review`, `suggested_translation`).
   - `glossary accept-all <book_id> [--json]` — вызывает `application.approve_glossary.accept_all_awaiting(book_id)` и выводит число утверждённых терминов (или JSON `{accepted_count, remaining_pending}`).
2. **Подкоманда `corrections`:**
   - `corrections list <book_id> [--awaiting-only] [--json]` — получает активный прогон (`run = tx.runs.active_for_book(book_id)`) и читает кандидатов через `tx.work.list_correction_candidates(run.id, states=("awaiting_human",) if awaiting_only else None)` (при `run is None` возвращает пустой список), выводя `candidate_id`, `section_id`, `block_id`, `state`, `proposal_sha256`, `input_sha256`, `review_reason` и фрагмент до/после.
   - `corrections resolve <book_id> <candidate_id> --action {accept|keep} [--reason <text>] [--proposal-sha256 <sha>] [--input-sha256 <sha>] [--json]`:
     - проверяет `application.resolve_correction_candidate is not None` (иначе поднимает `ApplicationError`);
     - `reason = (args.reason or "Решено через CLI").strip()` (если пусто — `ApplicationError`);
     - если `--proposal-sha256` или `--input-sha256` не переданы явно, загружает кандидата из `tx.work.get_correction_candidate(candidate_id)` (при `None` или `candidate.book_id != book_id` поднимает `CorrectionCandidateNotFound`) и берёт отсутствующий хэш из актуального кандидата;
     - вызывает `application.resolve_correction_candidate(ResolveCorrectionCommand(book_id=..., candidate_id=..., proposal_sha256=..., input_sha256=..., action=args.action, reason=reason))`.
3. **Флаг `export --strict-corrections` (`_export`, `app.py:618-647`):**
   - При переданном `--strict-corrections` **ДО** вызова `application.export_book(...)` (и до снятия паузы) открыть транзакцию, получить `run = tx.runs.active_for_book(args.book_id)` и при `run is not None` напрямую посчитать кандидатов через `tx.work.list_correction_candidates(run.id, states=("awaiting_human",))` (без использования `book_overview()`, который молча проглатывает `WorkStateError`!).
   - Если число кандидатов `> 0` (или `WorkStateError` при чтении кандидатов) — поднять `ApplicationError` (код возврата `1`), не записывая выходной файл.
   - Без флага сохраняется текущее поведение (предупреждение в `stderr`).

### 3.2 Тестовая матрица WP3
- `tests/integration/cli/test_cli.py`:
  - `glossary list --pending-only [--json]` и `glossary accept-all [--json]` разблокируют `cli process` на книге с `model_flagged_review = true`;
  - `corrections list --awaiting-only [--json]` выводит `section_id`, `candidate_id`, `proposal_sha256`, `input_sha256`;
  - `corrections resolve ... --action accept|keep [--reason ...] [--proposal-sha256 ...] [--input-sha256 ...]` применяет/отклоняет кандидата и отклоняет устаревший CAS-хэш;
  - `export --strict-corrections` завершается с кодом `1` **без создания выходного файла** при наличии `awaiting_human`, и с кодом `0` после `corrections resolve`.

---

## WP4 — Промпты анализа и глоссария: удаление хардкода «Прайвет-драйв», фонетика `sound` и обязательность вложенных полей

### 4.1 `src/linguary/application/context_builder.py`

1. **`_NARRATIVE_ANALYSIS_INSTRUCTIONS` (`L207`):**
   - Заменить:
     `Для составных иноязычных топонимов вроде «Прайвет-драйв» указывай грамматический род и склоняемость строго по нормам русского языка.`
     на:
     `Для составных иноязычных топонимов (например, «Бейкер-стрит», «Оквуд-лейн») указывай грамматический род и склоняемость строго по нормам русского языка.`
   - Добавить явное напоминание полноты вложенных полей (для стабильной работы моделей без constrained JSON decoding):
     `В каждом элементе glossary_candidates обязательно заполняй все поля name_translation_assessment (включая rationale), а в каждом элементе scenes[].visual_facts — все три поля: anchor, subject и detail.`
2. **`_with_name_policy` (`L3247-3262`, ветка `NameTranslationStrategy.SOUND` — общая для анализа и перевода):**
   - На `L3262` заменить часть `и «Тисовая улица» вместо «Прайвет-драйв»` на `и «Дубовая аллея» вместо «Оквуд-лейн»` (убирая ошибочный «Прайвет-драйв» из **всех** `SOUND`-шаблонов без затрагивания строк `L3250-3259` с правилом Ермоловича).
   - Добавить в `rule` ветки `SOUND` уточнение фонетической сверки:
     `При передаче имён и составных топонимов по звучанию сверяй реальное произношение исходного английского слова (не подменяй его созвучным, но другим словом с иной гласной или ударением); при сомнении в произношении на этапе анализа ставь model_flagged_review=true.`
3. **Защита инвариантов:**
   - `_ANALYSIS_INSTRUCTIONS` (`L2427-2430`) и `_TRANSLATION_INSTRUCTIONS` **не изменять** (`test_baseline_b2_templates_sha256_unchanged` вызывает `snapshot_prompt_templates` с `name_translation_policy=None` и остаётся побайтово неизменным).

### 4.2 Тестовая матрица WP4
- Обновить 2 существующих теста, закреплявших удалённый хардкод:
  - `tests/unit/application/test_illustration_reliability.py:938` (`"Прайвет-драйв"` → проверка нового нейтрального примера `"Оквуд-лейн"` / `"Бейкер-стрит"`);
  - `tests/unit/application/test_context_builder.py:78` (`'Тисовая улица'` → `'Дубовая аллея'`, с проверкой отсутствия `'Прайвет-драйв'`).
- Новые проверки в `tests/unit/application/test_context_builder.py`:
  - во всех шаблонах `snapshot_prompt_templates(..., profile="content-plan-v3", name_translation_policy=NameTranslationPolicy(strategy=SOUND))` отсутствует `"Прайвет-драйв"`, присутствуют правило фонетической сверки и обязательность `rationale` / `visual_facts.{anchor, subject, detail}`;
  - `test_baseline_b2_templates_sha256_unchanged`, `test_sound_policy_preserves_mc_prefix_in_mcgonagall` и `test_a2_ch1_ch7_reading_quality_fixes_end_to_end` остаются зелёными.

---

## WP5 — Педагогика A2, кореференция сегментов и снятие артиклей в дедупликации/A1-фильтре `glossary_hints`

### 5.1 `src/linguary/application/context_builder.py` и `src/linguary/application/reader_text.py`

1. **Калибровка A2 (`_reader_level_treatment_guidance("a2")` на `L2150-2199` и `_a2_progression_guidance` на `L3686`):**
   - Единообразно снизить промптовый потолок простого нарратива для `GLOSSARY_ONLY` с `45` до **`30 слов`** во **всех 4 вхождениях**: `L2154`, `L2196`, `L2198`, `L3686`.
   - *Инвариант:* кодовый предохранитель `_NARRATIVE_GLOSSARY_WORD_LIMIT = 45` в `src/linguary/application/analysis_merge.py:4832` **не меняется** (остаётся мягкой защитой без ломки `test_staged_analyze_chapter.py:2989-3109` и `test_a2_harry_potter_regressions.py:3437-3507`).
   - В стоп-лист `GLOSSARY_ONLY` / `READ_AS_IS` без отдельной грамматической заметки (`L2181-2187`) добавить:
     - конструкции сложного дополнения (*Complex Object*: `noticed/saw/heard/watched + объект + инфинитив/причастие`);
     - вложенные придаточные, разрывающие глагол и дополнение;
     - оборванные эллиптические реплики.
   - Расширить существующее окно элементарных реплик A1–A2 на `L2166-2173` с `4–8 слов` до **`1–8 слов`** (без введения конкурирующего диапазона) и добавить правило согласованности режимов для соседних параллельных реплик одного диалога.
2. **Запрет разрыва кореференции между сегментами:**
   - В конец блока сегментации `_PEDAGOGICAL_ANALYSIS_INSTRUCTIONS`: при дроблении абзаца на сегменты не оставлять короткий сегмент, впервые вводящий ключевой предмет или существо, без перевода, если следующий переводимый сегмент ссылается на него только местоимением 3-го лица.
   - В конец блока сегментированного перевода `_CONTENT_PLAN_PROCESSING_V3`: если сегмент в режиме `TRANSLATE` следует сразу после непереведённого сегмента (`READ_AS_IS` или `GLOSSARY_ONLY`), запрещено оставлять в русском переводе «висячее» местоимение 3-го лица без антецедента в русском потоке — явно восстанови существительное.
3. **Антишум и формат цитат в `_A2_NOTE_DISCIPLINE` (`reader_text.py:508-514`):**
   - Уточнить запрет на базовую лексику A1–A2 в прямом значении в `glossary_hints` и запрет на повтор уже подсказанного слова в той же главе.
   - Явно напомнить правило оформления цитат: цитируемые английские слова внутри русского пояснения всегда заключать в кавычки, а внутри английской цитаты не вставлять русские слова (пояснение и перевод писать отдельно по-русски).

### 5.2 Точечное усиление дедупликации и A1-фильтра (`layout/notes.py` и `a2_core_vocab.py`)

Существующая книжная 3-ступенчатая дедупликация `_deduplicate_glossary_hints()` в `src/linguary/application/layout/book.py:1515-1655` (`_HINT_LOCAL_WINDOW_BLOCKS = 20`, `glossary_only_rescue_by_bucket`, защита `content_evidence` / `plan.difficulties`) **сохраняется без дублирования**. Закрываем конкретный обход через ведущие артикли:

1. **`vocabulary_dedup_key(original: str)` (`src/linguary/application/layout/notes.py:609-619`):**
   - Добавить снятие ведущего артикля (`_LEADING_ARTICLES = frozenset({"a", "an", "the"})`) при `len(words) >= 2` (если не снят префикс `to `):
     ```python
     if len(words) >= 2 and words[0] == "to" and words[1] not in _NOT_A_VERB_AFTER_TO:
         words = words[1:]
     elif len(words) >= 2 and words[0] in _LEADING_ARTICLES:
         words = words[1:]
     ```
   - Результат: `the reptile house` и `reptile house` (а также `a wand` и `wand`) получают одинаковый ключ и попадают под существующее 20-блочное окно дедупликации.
2. **`is_trivial_a1_hint(original: str)` (`src/linguary/application/a2_core_vocab.py:1124-1145`):**
   - Добавить снятие ведущего артикля `a`/`an`/`the` перед проверкой одиночного слова (если не был снят `to `):
     - при снятом артикле проверять оставшееся одиночное слово против `A1_TRIVIAL_HINT_WORDS` / `_matches_plural_or_3sg(..., _A1_S_ROOTS)` (и не принимать глаголы с `to`).
   - Результат: `a glass`, `the paper`, `the glasses` отфильтровываются в `_a1_pre_suppressed_indices()` наравне с `glass`, `paper`, сохраняя fallback в `GLOSSARY_ONLY`-блоках и защиту подсказок с непустым `note` или non-vocabulary `plan.difficulties`.

### 5.3 Тестовая матрица WP5
- Обновить 5 существующих проверок в `tests/unit/application/test_context_builder.py`:
  - `L1437`, `L2393`, `L3815`: `"45 слов"` → `"30 слов"`;
  - `L398`, `L1422`: `"4–8 слов"` → `"1–8 слов"`.
- Новые проверки в `tests/unit/application/test_context_builder.py`: наличие порога `30 слов` (в т.ч. в `_a2_progression_guidance`), окна `1–8 слов`, *Complex Object*, кореференции сегментов и правила оформления цитат при зелёном `test_baseline_b2_templates_sha256_unchanged`.
- Новые проверки в `tests/unit/application/layout/test_notes.py`, `tests/unit/application/test_a2_core_vocab.py` и `tests/unit/application/layout/test_book_layout.py`:
  - `vocabulary_dedup_key("the reptile house") == vocabulary_dedup_key("reptile house")`;
  - `is_trivial_a1_hint("a glass") is True`, `is_trivial_a1_hint("the paper") is True`, `is_trivial_a1_hint("a wand") is False`;
  - в `build_book_layout` повтор `"the reptile house"` через 3 блока после `"reptile house"` подавляется, а контракт `glossary_only_rescue_by_bucket` и `_HINT_LOCAL_WINDOW_BLOCKS = 20` не нарушаются.

---

## WP6 — Пунктуация прямой речи: валидаторы `dialogue.py` / `paragraphs.py` и единая норма в промптах

### 6.1 `src/linguary/application/validators/paragraphs.py`

1. **`_count_post_speech_narrative_sentences(source_text: str) -> int` (`L48-58`):**
   - Сохранить существующий подсчёт для `[.!?…]["”’'](?=\s)` + прописная вне кавычек.
   - Дополнительно поддержать реплику с запятой, восклицанием или вопросом и последующей строчной ремаркой + самостоятельным авторским предложением:
     - искать закрытие реплики `[,!?…]["”’'](?=\s+[a-z])` вне кавычек (`_outer_unclosed_open(source_text, close_end) is None`);
     - сканировать вперёд от `close_end` до первой терминальной пунктуации (`.`, `!`, `?`, `…`) вне кавычек (останавливаясь, если встретилась новая открывающая кавычка), пропуская точки сокращений/титулов (`Mr.`, `Mrs.`, `Ms.`, `Dr.`, `St.`, `Prof.` или одиночный инициал `[A-Z].`);
     - если после найденной терминальной пунктуации через пробелы следует прописная буква `[A-Z]` (а **не** открывающая кавычка) — увеличить счётчик на `1`.
   - *Инварианты:*
     - `"‘Hello,’ he said. Then he ran."` → `1` и `"‘Hello!’ he cried. Then he ran."` → `1` (позволяет `— Привет, — сказал он.\nПотом он убежал.` без `extra_paragraphs`!);
     - `'"Hello," he said. "How are you?"'` → `0` и `"‘Hello,’ said Mr. Ollivander. ‘Wait.’"` → `0` (тест `test_result_validators.py:3799-3802` и сокращения титулов остаются зелёными!).

### 6.2 `src/linguary/application/validators/dialogue.py`

1. **Автоматная проверка `_has_dash_before_post_quote_narration(text: str, *, source_text: str | None) -> bool`:**
   - Активируется в `_text_dialogue_violation(...)` при `strict_v3=True` через обёртку `_has_block_post_quote_narrative_dash(text, *, block, item, source_text)` (аналогично `_has_block_inline_dialogue_dash` на `L350-362`: при наличии `block.segments` проверяет каждый переводимый сегмент против его `source_span = item.block.text[span.start:span.end]`).
   - Если `source_text is None` или в `source_text` нет кавычек прямой речи (`not has_direct_speech_quotes(source_text)`) → `False`.
   - Собрать верхнеуровневые спаны реплик `quote_spans` в `source_text` через `_outer_unclosed_open` (как на `L274-294`).
   - Пройти по каждой непустой строке `line` в `text`, начинающейся с `—` или `–`:
     - Вычислить допустимые встроенные цитаты `allowed_spans` через `_GUILLEMET_SPAN` и `_is_allowed_embedded_quote_span(line, span.start(), span.end(), source_text=source_text)` (симметрично `_has_inline_dialogue_dash` на `L373-387`) — совпадения внутри `allowed_spans` полностью пропускаются и не меняют `state` / `speech_segments` (защищает `b35_single_leading_dash` в `test_dialogue_validator.py:576-596` и `test_result_validators.py:3734-3744`).
     - Вести конечный автомат: `state = "speech"`, `speech_segments += 1` для начала строки.
     - Искать переходы `([,!?.…])\s+[—–]\s+([А-ЯЁа-яё])`, пропуская совпадения внутри `allowed_spans` и сокращения через `_ABBREVIATION_BEFORE_DASH.search(line[:match.start() + 1])` (когда `punct == "."`):
       - если `state == "speech"`:
         - при `next_char.islower()` → `state = "attribution"` (слова автора со строчной);
         - при `punct in ".!?…"` и `next_char.isupper()` → **вернуть `True`** (прямой переход реплики в самостоятельное авторское предложение через `— ` с прописной, например `— Молодчина, Дадли! — Он взъерошил…` или `one_line` в `ollivander_source`!);
       - если `state == "attribution"`:
         - при `punct in ".!?…"` и `next_char.isupper()` → `state = "speech"`, `speech_segments += 1` (новая реплика после слов автора);
         - при `punct == ","` и `next_char.islower()` → `state = "speech"` (продолжение реплики после ремарки).
   - **Дополнительная проверка хвоста после ремарки:** если все строки прошли без прямого `speech → [А-ЯЁ]`, проверить случай `‘Hello,’ he said. Then he ran.` → `— Привет, — сказал он. — Потом он убежал.`:
     - если после `quote_spans[-1][1]` в `source_text` есть непустой хвост с самостоятельным авторским предложением с прописной буквы (`_count_post_speech_narrative_sentences(source_text) > 0` и после последней кавычки нет новой открывающей кавычки), **и** `speech_segments > len(quote_spans)` → **вернуть `True`**!
   - **Ложные срабатывания исключены:**
     - `— Посмотри на это! — крикнул он. — Посмотри! — ахнула тётя Петуния.` → 2 `speech`-сегмента, 2 `quote_spans`, 0 прямых переходов `speech → [А-ЯЁ]` → `False`;
     - `— Первая, — сказал он. — Вторая.` → 2 `speech`-сегмента, 2 `quote_spans` → `False`;
     - `— Реплика!\nОн взъерошил волосы.` → 1 `speech`-сегмент, вторая строка без тире → `False` (и принимается `paragraph_violation`).
2. **Сообщение об ошибке в `_text_dialogue_violation(...)` (оборачивается `TranslationResultValidator` в `ResultValidationError` с `rule_code="dialogue_punctuation"`):**
   - `f"в переводе блока {block_id} самостоятельное авторское предложение после закрытой реплики отделено тире «— » на той же строке (читается как новая реплика): перенеси его на новую строку без тире либо встрой как слова автора со строчной буквы"`

### 6.3 `src/linguary/application/reader_text.py` и `src/linguary/composition/correction_runtime.py`

- Синхронно обновить `_A2_DIALOGUE_AND_CALQUE_RULES` (`reader_text.py:524`) и `CORRECTION_PROMPT_V2` (`correction_runtime.py:91`):
  - убрать разрешение однострочной формы `«— Реплика? — Авторское предложение.»`;
  - закрепить единое правило:
    - слова автора после реплики — через тире со строчной буквы (`— Реплика! — сказал он.`);
    - самостоятельное авторское предложение после закрытой реплики — **с новой строки без тире** (`— Реплика!\nАвторское предложение.`, а если после него речь продолжается — три строки `— Реплика 1.\nАвторское предложение.\n— Реплика 2.`), запрещены однострочная форма через тире с прописной буквы (`— Реплика! — Авторское предложение.`), две строки вида `Авторское предложение. — Реплика 2.` и кавычки-ёлочки с начала абзаца.
- *Область действия корректора:* `CorrectChapter._candidates()` (`correct_chapter.py:994-998`) остаётся на `validation_policy="v2"`; детерминированный гейт `strict_v3=True` отсекает ошибку на этапе `translation_unit`, а обновлённый `CORRECTION_PROMPT_V2` устраняет остатки в существующих переводах.

### 6.4 Тестовая матрица WP6
- Обновить 3 существующих теста, закреплявших прежнюю однострочную форму:
  1. `tests/unit/application/test_dialogue_validator.py:657-681` (`test_dialogue_validator_allows_ollivander_1_line_and_3_line_and_rejects_2_line_inline_dash` → `three_line` проходит `is None`, `one_line` и `two_line_bad` отклоняются `is not None`);
  2. `tests/unit/application/test_result_validators.py:3787-3836` (`test_post_speech_narrative_paragraph_allows_1_line_and_3_line_and_rejects_absorbed_neighbor` → `three_line` проходит `validate(..., validation_policy="v3")`, `one_line` отклоняется с `dialogue_punctuation`);
  3. `tests/unit/composition/test_correction_runtime.py:342-350` (`test_correction_prompt_v2_and_a2_dialogue_rules_sync_post_speech_narrative_guidance` → обновить ассерт под новую синхронизированную формулировку с запретом однострочного тире).
- Новые проверки в `tests/unit/application/test_dialogue_validator.py` и `tests/unit/application/test_result_validators.py`:
  - все 5 реальных негативных кейсов прогона (включая `— Молодчина, Дадли! — Он взъерошил Дадли волосы.` и `…его может стошнить. — Потом он побежал…`) + британский случай с запятой и ремаркой (`‘Hello,’ he said. Then he ran.` → `— Привет, — сказал он. — Потом он убежал.`) отклоняются `dialogue_violation(..., strict_v3=True)`;
  - перенос на новую строку без тире (`— Реплика!\nОн взъерошил волосы.` и `— Привет, — сказал он.\nПотом он убежал.`) принимается и `dialogue_violation(..., strict_v3=True)`, и `paragraph_violation(...)`;
  - легитимные 2-репликовые конструкции (`— Посмотри на это! — крикнул он. — Посмотри! — ахнула тётя Петуния.`, `— Первая, — сказал он. — Вторая.`), сокращения (`Mr.`, `г.`) и сегментированные блоки не дают ложных срабатываний;
  - `test_dialogue_line_breaks_rejected_if_extra_line_lacks_dash` и `test_result_validators.py:3799-3802` остаются зелёными.
