# BRIEF: Устранение дефектов анализа, кластеризации иллюстраций, диалоговой валидации и раскладки A2 (Everlost audit)

**Дата:** 2026-11-26  
**Статус:** Approved (пройдены 3 независимых ревью: Spec & Plan Reviewer, Regression & Adjacent Code Auditor, Final Independent Spec Auditor; все замечания устранены)  
**Контекст:** Пост-анализ полного прогона книги Нила Шустермана *Everlost* (гл. 1–3 + 2 абзаца гл. 4, `everlost-ch1-ch4p2-20261126` на `main` `3e69d8b`) и forensic-атрибуция 33 выявленных дефектов (код vs промпт/схема vs модель).

---

## 1. Проблема

Прогон реальной книги с настройками по умолчанию и последующая forensic-проверка 245 вызовов LLM выявили системные дефекты кода и промптов/схем (~63% всех проблем), которые **не устраняются заменой модели** и проявляются в пяти зонах:

1. **Анализ глав: ложный `spoiler_name_leak`, вечный кэш-дедлок при `retry` и автосинтез имён (WP1):**
   - Системная инструкция `_NARRATIVE_ANALYSIS_INSTRUCTIONS` (`context_builder.py:200`) дословно приводит `«the freckled boy»` как пример записи для `glossary_candidates`, а `_GENERIC_PERSON_CANDIDATE_PATTERNS` (`analysis_merge.py:2221-2229`) проверяет закрытый список из 14 прилагательных и 16 существительных, пропуская `younger boy`, `the freckled boy` и `the freckled kid`.
   - `_first_name_block_index()` (`analysis_merge.py:2785-2815`) немонотонен: на фрагменте 0 строчная фраза `younger boy` / `the freckled boy` находит строчное `boy` в блоке 2 и кэшируется; при общеглавной проверке включается `prefer_capitalized=True`, находится заглавное `Boy` внутри чужого прозвища `"Chocolate Boy"` (блок 38), и `first_pos` скачет с 2 на 38 → `AnalysisMergeError`.
   - Фрагментный кэш `analysis_pass_results` очищается через `tx.analyses.delete_fragment_passes_for_section(...)` только после **успешного** слияния (`analyze_chapter.py:1288-1290`), а `RetryTask` (`retry_task.py:54-75`) не инвалидирует его → повторный запуск детерминистически падает с 0 LLM-вызовов.
   - Сборка `known_names` для автосинтеза словарных подсказок (`analyze_chapter.py:2537-2580`) не извлекает сокращённые/обиходные имена (`Allie` при каноническом `Alexandra`) из `note`/`description` и `context_facts`, создавая бесполезную подсказку на имя главного героя в середине книги.

2. **Реестр иллюстраций: коллапс кластеризации, потеря имени `Nick`, отсутствие глоссария и английский в `name_ru` (WP2):**
   - В `resolve_verdict_graph_clusters()` (`illustration_reference.py:364-381`) **одно** ошибочное ребро `different` из 36 пар (35 `same_entity` + 1 `different`) делает все остальные 7 вершин `bridge_nodes` и сносит **все** инцидентные им рёбра, включая 21 ребро между самими мостовыми вершинами → один лес распался на 10 сущностей (вкл. 3× `dead forest`, 2× `forest`, 2× `the forest`).
   - `_blocking_keys_for_mention()` (`illustration_reference.py:264-294`) не имеет контекстного блокинга для анонимных дескрипторов (`the boy`, `the girl`, `the other boy`) → `Lief / Freckle-boy / the boy`, `Allie / the girl`, `Nick / the other boy` ни разу (0 пар) не попали в `ALIAS`.
   - В `_choose_canonical_name()` (`illustration_reference.py:2816-2824`) ключ `-min(proper_word_count, 3)` выше частоты и глоссария → прозвище `Chocolate Boy` (1 упоминание) победило имя `Nick` (8 упоминаний); кроме того, `_display_name_at_position()` (`illustration_reference.py:2829-2849`) перевызывает `_choose_canonical_name()` с `count=1` и без глоссария. При этом `_known_entity_names_for_chapter()` (`illustration_selection.py:103-153`) не передаёт алиасы → в гл. 4 имени `Nick` нет в контексте `CANDIDATES`, и модель подменила 14-летнего Ника на 11-летнего Лифа (`the boy`).
   - В `illustration_chapter.py:1028, 1396` поле `name_ru` (и fallback `description_ru`) заполняется английским `display_name`, а русские термины глоссария книги вообще не передаются в промпты иллюстраций → разнобой терминов (*«Мэри Королева Соплей»*, *«Гринсоулами»*, *«Мёртвые зоны»*).

3. **Иллюстрации: бюджет токенов, отсутствие ретрая `provider_truncated`, сырые UUID/SHA-256, 163 вызова и гигиена промптов (WP3):**
   - `IllustrationContext.build_for_kind()` (`illustration_context.py:623-635`) не передаёт `max_completion_tokens` в `LLMRequest`, из-за чего используется глобальный лимит `32768`. Два зациклившихся вызова `DESIGN` выжгли по 32 768 токенов и сразу упали в `FAILED` на 1-й попытке без ретрая (`illustration_reference.py:1232-1254`; аналогично без ретрая для не-composite скоупов в `illustration_chapter.py:1906` и `illustration_selection.py:1148`).
   - В `EXTRACT` и `CANDIDATES` передаются сырые 36-символьные UUID и 64-hex SHA-256 `scene_id` (6 из 14 сбоев иллюстраций), а у `quote_anchors` в `illustration_payloads.py:237` нет `description` про `min >= 1` (4 из 14 сбоев).
   - На 3,1 главы ушло 163 вызова: `DESIGN` запускается для 100% сущностей (`mention_count >= 1`, 64% вызовов нигде не использованы), `FACT_SCOPE` делает отдельный LLM-вызов даже для 1 тривиального `permanent`-факта, а в `SELECTION` нет масштабирования лимита от длины главы и минимальной дистанции между иллюстрациями (2 иллюстрации на 2 абзаца гл. 4; 3 подряд на соседних абзацах гл. 1).
   - Пробелы в промптах и контексте иллюстраций: `back_applicable=true` переносит одежду и посмертные детали назад в ранние главы; `DESIGN` при 1 `permanent`-факте отбрасывает `episodic`-контекст для `atmosphere`; `PROMPT` не получает исходные цитаты якорей (`anchors[].quote`) и не отфильтровывает технические ключи (`attire_default`, `characters[].description_en`) в `assumptions_ru`.

4. **Перевод: рассинхрон промпта и валидатора диалогов + поломка смешанных абзацев (WP4):**
   - Валидатор `validators/dialogue.py:187-191` жёстко требует переноса строки или двоеточия для реплики после авторской фразы с точкой (`Action beat. "Speech"`), но в `_A2_DIALOGUE_AND_CALQUE_RULES` (`reader_text.py:516-528`) этого правила нет → 6 из 18 отказов `translation_unit`.
   - На абзаце вида `“Speech…” Narrative. “Embedded quote,” said X. “Continuation.”` (гл. 2 B35) сочетание правил `dialogue.py:173` и `dialogue.py:187` дважды отвергло корректный перевод в 3-м лице (включая двухстрочный вариант), вынудив модель сломать лицо абзаца.
   - В `_A2_DIALOGUE_AND_CALQUE_RULES` и `_A2_NOTE_DISCIPLINE` не оговорены мн. ч. «вы» при обращении к группе детей и запрет превращать существительные с артиклем (`a guffaw`) в глагольный инфинитив `to + V`.

5. **Раскладка учебного слоя A2: привязка подсказок в смешанных блоках, «Говорит …» и тавтологии (WP5):**
   - В смешанных блоках (`TRANSLATE` + `GLOSSARY_ONLY`) `_hint_draft()` (`layout/notes.py:745`) и `_hint_segment_index()` (`layout/book.py:1045`) ищут `hint.original` через `_find_end(block.text, ...)` — первое вхождение по всему блоку, игнорируя опору трудности из `ContentPlan` и режимы сегментов → подсказка `cliff` прилипла к переведённому сегменту 0, оставив `GLOSSARY_ONLY`-сегмент 1 пустым.
   - `_speaker_name()` / `_speakers_for_block()` (`layout/book.py:599-697`) всегда ставят полное каноническое имя (`Александра` вместо обиходного `Элли`) и добавляют «Говорит Ник» даже когда реплика начинается с самопредставления (`“My name’s Nick.”`).
   - В `layout/notes.py:751-779` фильтр `_headword_gloss` пропускает тривиальные синонимические повторы вида `взрослые — взрослые люди`.

---

## 2. Цели

1. Полностью исключить ложные `spoiler_name_leak` на строчных дескрипторах и кэш-дедлок при `AnalysisMergeError` / `retry` (сохраняя кэш при транспортных сбоях), а также устранить автосинтез подсказок на алиасы имён главных героев (WP1).
2. Обеспечить устойчивую кластеризацию реестра иллюстраций при одиночных шумных `diff`-рёбрах, контекстный блокинг анонимных дескрипторов, корректный приоритет глоссария и частоты в `_choose_canonical_name()` и `_display_name_at_position()`, передачу алиасов отдельным полем `known_entity_aliases` в `CANDIDATES`, а также проброс неспойлерного русского глоссария и локализованного `name_ru` (WP2).
3. Ввести per-stage потолок токенов (согласованный с `reasoning_max_tokens`) и авторетрай при `provider_truncated` во всех 3 модулях иллюстраций, wire-алиасинг длинных ID (`len > 16`) в `EXTRACT`/`CANDIDATES`, сократить лишние вызовы `DESIGN`/`FACT_SCOPE`, добавить масштабирование плотности и дистанцию иллюстраций, а также закрыть утечки контекста и технических ключей в промптах (WP3).
4. Синхронизировать правила оформления диалогов, мн. ч. «вы» и частей речи в `_A2_DIALOGUE_AND_CALQUE_RULES` / `_A2_NOTE_DISCIPLINE` с валидатором `validators/dialogue.py` без изменения замороженных SHA-256 шаблонов и без ложных отказов на вложенных цитатах в повествовании (однострочных и двухстрочных) (WP4).
5. Починить сегментную привязку `glossary_hints` в смешанных блоках через единый хелпер, улучшить сноски «Говорит …» (неспойлерное обиходное имя + подавление самопредставления) и отсечь тривиальные тавтологии в `glossary_hints.note` (WP5).

---

## 3. Осознанные не-цели (Non-goals)

- **0 миграций БД / Alembic:** схема SQLite не меняется; новые ключи в JSON-полезных нагрузках обратно совместимы со старыми записями.
- **Никаких привязок к конкретным книгам, персонажам, локациям или моделям LLM:** все правила и алгоритмы универсальны.
- **Замороженные шаблоны не трогаем:** `_TRANSLATION_INSTRUCTIONS`, `_SEGMENT_TRANSLATION_RULE` и `snapshot_prompt_templates(..., reader_level="b2")` сохраняют точные SHA-256 (`test_baseline_b2_templates_sha256_unchanged`).
- **Изоляция `temperature=policy.temperature` в `_clean_llm_settings()` не меняется:** текущая передача `temperature=policy.temperature` (`illustration_runtime.py:83`) защищает reasoning-модели от чужой температуры из `[llm]` (`test_illustration_runtime.py:136`).
- **Экспорт `title.xhtml` и `dedication.xhtml` в `learner layout` не меняем:** фильтрация front-matter в `_exported_sections()` (`writer.py:489-506`) — осознанное продуктовое поведение, закреплённое в `tests/integration/epub/test_epub_export_content.py:830` и `tests/unit/test_epub_writer.py`.
- **Разметку `<p>` внутри `<li>` в `.lg-notes` не меняем:** `.lg-notes p { display: inline; margin: 0; }` явно закреплена тестами `test_layout_render.py:456` и `test_a2_harry_potter_regressions.py:1550`.
- **Контракт покрытия `ChapterCorrectionV2Schema` (`unchanged` + `proposed`) не ломаем:** замедление `chapter_correction` на 95% вызвано скрытым reasoning конкретной модели (`deepseek-v4.1-flash-1017`), что решается конфигурацией модели/reasoning, а не ослаблением проверки полноты покрытия.
- **Строгие инварианты кластеризации сохраняем:** `test_ambiguous_identity_remains_separate`, `test_transitive_identity_conflict_triangle_order_independent_deterministic` и порядок-независимость `resolve_verdict_graph_clusters()` остаются безусловными гарантиями.
- **Чисто модельные ошибки** (`got where they were going -> ехали`, `had not been` у `claude-haiku-5.5`, самоподтверждение буквализмов одной моделью) не лечим хардкодами — после внедрения WP1–WP5 они адресуются выбором моделей в `config.toml`.

---

## 4. Критерии приёмки

- `uv run pytest tests/architecture` и `uv run pytest -m "not slow"` полностью зелёные.
- `make check` (Ruff + ty + architecture tests) проходит без ошибок.
- Все новые и обновлённые unit- и integration-тесты покрывают каждый пункт WP1–WP5 и подтверждают отсутствие регрессий (включая `test_baseline_b2_templates_sha256_unchanged`, `test_ambiguous_identity_remains_separate`, `test_transitive_identity_conflict_triangle_order_independent_deterministic`, `test_analysis_everlost_regressions.py`, `test_a2_harry_potter_regressions.py`, `test_epub_export_content.py` и обратную совместимость старых JSON-записей).
- Контрольный прогон `everlost-ch1-ch4p2.epub` в новом `book_id` проходит анализ, перевод, иллюстрации и экспорт (`EPUBCheck: passed`) без кэш-дедлока, без подмены персонажей в иллюстрациях, с существенным снижением числа вызовов иллюстраций и отказов `translation_unit`.
