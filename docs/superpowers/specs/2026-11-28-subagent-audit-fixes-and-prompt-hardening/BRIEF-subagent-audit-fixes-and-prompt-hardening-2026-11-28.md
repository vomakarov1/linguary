# BRIEF: Атомарность старта обработки, исправление экспорта EPUB, полнота CLI и усиление промптов/валидаторов A2 (Subagent audit)

**Дата:** 2026-11-28  
**Статус:** Approved (одобрено по итогам 3 независимых ревью кода и архитектуры; все замечания устранены)  
**Контекст:** Пост-анализ сквозного прогона первых трёх глав *Harry Potter and the Philosopher's Stone* (`hp-ch1-3-a2`, уровень `a2`, профиль `content-plan-v3`, `staged_analysis = true`, `sequential_analysis = true`, 66 LLM-вызовов через локальный OpenAI-совместимый мост `flash`/`pro`, 373 текстовых блока, валидный EPUB 3.3 `EPUBCheck 0/0/0`) и двух независимых ревью результата (программист-архитектор + лингвист-переводчик).

---

## 1. Проблема

Аудит выявил 6 групп реальных дефектов в приложении и промптах:

1. **Неатомарность `StartProcessing` и split-brain маршрутов (P0, WP1):**
   - В `cli/app.py:551-566` (`_process`), `web/routes/processing.py:167-186` (`start_processing`) и автостарте `container.py:466-478` (`_start_book_processing`) старт обработки разбит на две независимые транзакции: сначала коммитится `StartProcessing.__call__()` (`approve_glossary.py:483-531`), и только затем `processing.start()` вызывает `BookRuntimeResolver.settings_for()` (`book_runtime.py:513-515`), проверяющий `routes_compatible`.
   - `Application.start_processing()` (`container.py:211-230`) и `_start_book_processing()` (`container.py:466-478`) передают `correction_snapshot` без сверки с замороженным маршрутом книги `book_conditions.payload['route']` (а автостарт к тому же использует закешированный `candidate` вместо живых настроек).
   - При смене `base_url` / локального порта между анализом и запуском обработки команда `process` падает с `route_unavailable`, но оставляет книгу в статусе `PROCESSING` с активным `ProcessingRun`, в котором `book_conditions` привязан к старому маршруту, а `work_bindings['run'].correction` — к новому. После возврата исходного маршрута перевод завершается успешно, а все задачи `chapter_correction` падают в `CorrectionRuntimeResolver.settings_for()` (`correction_runtime.py:214-217`) с `route_unavailable`. Кроме того, `web/routes/processing.py:176` не перехватывает `ConditionsError` через `input_error_response`.

2. **Ошибки экспорта EPUB: классификация обложки, потеря метаданных прав, мёртвые ресурсы и CSS (P1, WP2):**
   - В `domain/document.py:727-733` (запасная ветка `is_cover_section` для секций без имени `cover*` и заголовка `"Cover"`) проверка отсутствия текста опирается только на `_TEXT_BLOCK_TYPES`, куда не входит `BlockType.HEADING`. Титульные страницы с заголовками названия/автора и декоративным изображением ошибочно признаются обложкой и выживают в экспорте, тогда как `copyright` и `dedication` отбрасываются в `_exported_sections` (`writer.py:493-510`).
   - `BookMetadata` (`domain/document.py:359-370`) и `EpubExportWriter._append_metadata()` (`writer.py:1315-1330`) не переносят `dc:rights` и `dc:description` исходного EPUB. (В SQLite `BookRow` хранит метаданные в реляционных столбцах без JSON-поля, поэтому при ограничении «0 миграций Alembic» перенос при экспорте из БД требует резервного чтения из `request.source_epub_path`.)
   - `_resource_payloads()` (`writer.py:586-625`) отфильтровывает только пути пропущенных XHTML-секций и вызывается до `_render_chapters()`, оставляя в итоговом архиве неиспользуемые CSS, `.xpgt` и картинки отброшенных секций (например, `hp_style.css`, `page-template.xpgt`, `images/logo.jpg`).
   - В HTML сохраняются классы оригинала (`.center`, `.pagebreak`), но исходный CSS отключён, а в `linguary.css` (`writer.py:111-285` и `web/static/linguary.css`) для них нет правил — разделители сцен `*` прижимаются к левому краю, а пустой `<p class="pagebreak">&nbsp;</p>` может создавать лишнюю пустую строку/страницу.
   - В `linguary.css` правило `.lg-notes > li { text-indent: -1em; }` (`writer.py:175-177`) рассчитано на `<span class="lg-num">`, но у ненумерованных заметок атрибуции говорящего (20 штук в прогоне, вида «Говорит Петуния») номера нет, из-за чего строка съезжает влево в колонку номеров — и в EPUB (`layout_render.py:246`), и в Web UI (`web/templates/partials/result_block.html:70`).

3. **Операционная неполнота CLI для автономных прогонов (P1, WP3):**
   - В `cli/app.py` отсутствуют команды для просмотра и массового утверждения помеченных моделью терминов глоссария (даже при `auto_approve_glossary = true` термины с `model_flagged_review = true` останавливают `cli process`), а также команды для просмотра и разрешения кандидатов корректуры в статусе `awaiting_human` (`ResolveCorrectionCandidate` доступен только из Web UI).
   - `cli export` (`cli/app.py:622-640`) проверяет `pending_corrections` только после записи EPUB и лишь печатает предупреждение в `stderr` (причём `book_overview()` молча подавляет `WorkStateError`), без опции строгого отказа (`--strict-corrections`) до сборки файла.

4. **Книгоспецифичный хардкод с фонетической ошибкой в промптах + пропуски вложенных полей (P1, WP4):**
   - Ошибочная транскрипция **«Прайвет-драйв»** (*privet* произносится `/ˈprɪvɪt/` — «Привит-драйв», а не *private*) зашита сразу в двух местах `context_builder.py`: в `_NARRATIVE_ANALYSIS_INSTRUCTIONS` (`L207`) и в `_with_name_policy` (`L3262`, ветка `NameTranslationStrategy.SOUND`). Это нарушает принцип универсальности и напрямую подталкивает модель к ошибке при стратегии `sound`.
   - На лёгких моделях без аппаратного constrained JSON decoding (`flash`) 2 из 12 вызовов `chapter_analysis_narrative` упали на 1-й попытке с `ModelOutputError` из-за пропуска обязательных вложенных полей (`name_translation_assessment.rationale`, `visual_facts.{anchor, subject, detail}`).

5. **Дефекты педагогических и переводческих промптов A2: разрывы кореференции между сегментами, калибровка режимов и обход артиклями фильтра/дедупликации `glossary_hints` (P1, WP5):**
   - **Разрыв кореференции в смешанных блоках:** когда сегмент с существительным остаётся в `GLOSSARY_ONLY` / `READ_AS_IS`, а следующий сегмент переводится (`TRANSLATE`), в русском переводе повисает местоимение без антецедента (гл. 1 #47: «открыл её… щёлкнул ею» без слова «зажигалка»; гл. 2 #72: «Она могла бы…» без слова «змея»).
   - **Размытая калибровка режимов A2:** в `GLOSSARY_ONLY` попадают длинные синтаксически тяжёлые предложения (38–39 слов, *Complex Object*, эллипсисы) при промптовом потолке 45 слов (указан в 4 местах `context_builder.py`: `L2154, 2196, 2198, 3686`), а короткие односложные реплики A1 (`‘So?’`, `‘That’s mine!’`, `‘Let me see it!’`, `‘Make Harry get it.’` рядом с непереведённым `‘Make Dudley get it.’`) получают полный `TRANSLATE`.
   - **Шум и повторы в `glossary_hints`:** в `layout/notes.py:609-619` (`vocabulary_dedup_key`) и `a2_core_vocab.py:1124-1145` (`is_trivial_a1_hint`) снимается префикс `to `, но не снимается ведущий артикль (`a`/`an`/`the`), поэтому подсказки с артиклем (`the reptile house` vs `reptile house`, `a glass`, `the paper`) обходят существующую 20-блочную дедупликацию и A1-фильтр. Плюс в 3 вызовах перевода на 1-й попытке происходило языковое просачивание в заметках (незакавыченные английские слова / кириллица внутри английского примера).

6. **Лишнее тире перед авторским текстом после закрытой реплики: неполнота валидатора, противоречие в промптах и британская реплика в `paragraphs.py` (P1, WP6):**
   - В переводе 6 раз возникла однотипная пунктуационная ошибка: после реплики самостоятельное авторское предложение отделялось лишним тире с прописной на той же строке (`«— Молодчина, Дадли! — Он взъерошил Дадли волосы.»`, `«…его может стошнить. — Потом он побежал…»`).
   - Причина двоякая: в `_A2_DIALOGUE_AND_CALQUE_RULES` (`reader_text.py:524`) и `CORRECTION_PROMPT_V2` (`correction_runtime.py:91`) формулировка сама разрешает однострочное `«— Реплика? — Авторское предложение.»`, а `validators/dialogue.py` пропустил все 6 случаев; `chapter_correction` поймал лишь 1 из 6.
   - При этом `validators/paragraphs.py:23` (`_POST_SPEECH_CLOSE_RE`) допускает перенос авторского предложения на новую строку без тире только после `[.!?…]["”’']`, но не после британской реплики с запятой + ремаркой + точкой (`‘Hello,’ he said. Then he ran.`), что без согласованной правки создаст Catch-22 между `dialogue.py` и `paragraphs.py`.

---

## 2. Цели

1. Сделать `StartProcessing` полностью атомарным относительно проверки маршрута `book_conditions` и согласования `correction_snapshot` во всех точках входа (CLI, Web UI и автостарт `auto_start_processing`), исключив split-brain между `book_conditions` и `work_bindings['run']` (WP1).
2. Исправить запасную ветку `is_cover_section`, обеспечить перенос `dc:rights`/`dc:description` при экспорте EPUB (без миграций БД), очистить неиспользуемые ресурсы после рендера XHTML, добавить CSS-правила для `.center`, `.pagebreak` и ненумерованных заметок `.lg-note-plain` в EPUB и Web UI (WP2).
3. Добавить в CLI команды для автономного пайплайна: `glossary list|accept-all`, `corrections list|resolve` (в доменном контракте `accept|keep` с поддержкой `--reason` и CAS-хэшей) и флаг `export --strict-corrections` до сборки файла (WP3).
4. Удалить ошибочный хардкод «Прайвет-драйв» из `_NARRATIVE_ANALYSIS_INSTRUCTIONS` и `_with_name_policy`, усилить правило фонетической сверки в `NameTranslationStrategy.SOUND` и явно закрепить обязательность вложенных полей схемы анализа (WP4).
5. Устранить разрывы кореференции между сегментами, сузить промптовый потолок `GLOSSARY_ONLY` A2 до 30 слов во всех 4 точках `context_builder.py` (сохраняя кодовый предохранитель `45` в `analysis_merge.py`), расширить окно элементарных реплик до `1–8 слов` и добавить снятие ведущего артикля `a`/`an`/`the` в `vocabulary_dedup_key` и `is_trivial_a1_hint` (WP5).
6. Убрать разрешение однострочного `«— Реплика? — Авторское предложение.»` из `_A2_DIALOGUE_AND_CALQUE_RULES` и `CORRECTION_PROMPT_V2`, добавить автоматную проверку лишнего тире перед авторским текстом в `validators/dialogue.py` (`strict_v3=True`) и поддержать британскую реплику с запятой + ремаркой в `validators/paragraphs.py` (WP6).

---

## 3. Осознанные не-цели (Non-goals)

- **0 миграций БД / Alembic:** схема SQLite (`books` и прочие таблицы) не меняется. `dc:rights`/`dc:description` в `BookMetadata` опциональны (`None` по умолчанию) и при экспорте из БД добираются из `request.source_epub_path`.
- **Замороженные SHA-256 и версии условий не меняем:** `_TRANSLATION_INSTRUCTIONS`, `_ANALYSIS_INSTRUCTIONS` и `snapshot_prompt_templates(source_language="en", target_language="ru", reader_level="b2", profile="legacy-v1")` (с/без `pronunciation_help_enabled`, `name_translation_policy=None`) остаются побайтово неизменными (`test_baseline_b2_templates_sha256_unchanged`). Версия `A2_CONTENT_PLAN_V8_VERSIONS` (`prompts: 8`, `context_builder: 6`) не повышается: новые книги получают уточнённые инструкции, уже запущенные книги сохраняют замороженные `BookConditions`.
- **Никаких новых хардкодов под конкретные книги, персонажей, локации или модели LLM** (ошибочный хардкод «Прайвет-драйв» удаляется в WP4; незатронутые правила Ермоловича в `_with_name_policy:3250-3259` оставлены вне скоупа WP4 во избежание избыточной переработки регрессионных тестов).
- **Не меняем OpenAI strict JSON Schema** (без `minItems`, без `default_factory` на обязательных полях).
- **Не чиним баги сторонних читалок (например, потерю верхнего поля в Apple Books на macOS)** через ломку стандартного EPUB CSS.
- **Не включаем в репозиторий Linguary агентский мост субагентов** — он оформлен отдельным рабочим скиллом `.agents/skills/linguary-subagent-run/`.

---

## 4. Критерии приёмки

- `uv run pytest tests/architecture`, `uv run pytest -m "not slow"` и `uv run pytest --cov=linguary --cov-fail-under=90` полностью зелёные.
- `test_baseline_b2_templates_sha256_unchanged`, `test_dialogue_line_breaks_rejected_if_extra_line_lacks_dash`, `test_pinned_runtime.py`, `test_layout_render.py`, `test_book_layout.py` и `test_epub_export_content.py` полностью зелёные; перечисленные в SDD/Plan обновляемые тесты WP4–WP6 синхронизированы с новым контрактом.
- Новые unit- и integration-тесты покрывают каждый из 6 рабочих пакетов (WP1–WP6), включая сценарий смены порта в `StartProcessing` (и `auto_start_processing`), титульник с заголовками в `is_cover_section`, отсутствие неиспользуемых ресурсов в экспортированном EPUB, CLI-команды `glossary` / `corrections` / `export --strict-corrections`, снятие артиклей в `vocabulary_dedup_key` / `is_trivial_a1_hint` и все 6 реальных примеров лишнего тире из перевода (5 дошедших до итогового EPUB и 1 снятый корректором) в `dialogue_violation` (в связке с `paragraph_violation`).
- Повторный контрольный прогон тестового 3-главного EPUB на уровне `a2` проходит полностью через CLI без ручных Python-скриптов, с `EPUBCheck: passed (0/0/0)`, без «Прайвет-драйв», без лишних тире после реплик и без мёртвых ресурсов в архиве.
