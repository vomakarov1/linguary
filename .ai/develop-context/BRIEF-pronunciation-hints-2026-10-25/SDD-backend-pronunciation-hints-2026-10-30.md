---
type: sdd
status: accepted
platform: backend
date: 2026-10-30
adr: ADR-pronunciation-hints-2026-10-25.md
prd: BRIEF-pronunciation-hints-2026-10-25.md
ux: []
lbs: []
related_sdd:
  - file: SDD-frontend-pronunciation-hints-2026-10-30.md
    owner_of: [API-PH-01, DTO-PH-01]
    consumer_of: []
phases:
  - id: P1
    title: Условия книги и сохранение подтверждённых вхождений
  - id: P2
    title: Поиск кандидатов и проверка произношения
  - id: P3
    title: Контракты чтения и экспорт EPUB
  - id: P6
    title: Изоляция ошибок кандидатов и границы словоформ
  - id: P8
    title: Закреплённый снимок ipa-dict и лицензионные материалы
  - id: P9
    title: Типизированное основание точной формы и совместимость хранения
  - id: P10
    title: Локальный источник ipa-dict и консервативное принятие
  - id: P11
    title: Уведомления в приложении и EPUB и проверка реального набора
contracts:
  - id: API-PH-01
    kind: endpoint
    role: owner
    defined_in: P1.T1
  - id: DTO-PH-01
    kind: dto
    role: owner
    defined_in: P3.T1
  - id: LIC-PH-01
    kind: endpoint
    role: owner
    defined_in: P11.T1
open_questions:
  - id: Q-1
    title: Лицензированный источник и права
    gates: [P2.T1, P3.T2, P3.T3, P8.T2, P11.T1, P11.T2]
  - id: Q-2
    title: Контрольный набор и порог точности
    gates: [P3.T3, P11.T3]
  - id: Q-3
    title: Целевые программы чтения EPUB
    gates: [P3.T2, P11.T2]
---

# SDD-backend: Проверяемые подсказки произношения

> **Статус:** Accepted

- Платформа: backend
- Проверка всего: `uv run pytest` (настроенная команда платформы backend).

- ADR: [ADR-pronunciation-hints-2026-10-25.md](ADR-pronunciation-hints-2026-10-25.md)
- PRD: [BRIEF-pronunciation-hints-2026-10-25.md](BRIEF-pronunciation-hints-2026-10-25.md)
- UX: отдельные макеты не заданы; поведение описано в BRIEF.
- Связанные SDD: [SDD-frontend-pronunciation-hints-2026-10-30.md](SDD-frontend-pronunciation-hints-2026-10-30.md)

## 1. Обзор

Бэкенд сохраняет отдельную помощь произношения для точных вхождений, получает кандидатов из обработки книги и показывает только IPA с проверяемым основанием. Выбор UK/US входит в неизменяемые условия книги; уровень читателя берётся из уже существующего `reader_level`. В первой версии поддерживается английский, а другие языки и региональные варианты остаются вне объёма.

**Затронутые модули:**

- `src/linguary/domain` и `src/linguary/application` — снимок условий, модель произношения, проверка кандидатов и общий путь обработки главы/повторного запуска.
- `src/linguary/adapters/db` и `alembic/versions` — сохранение вхождений и совместимость прежних результатов.
- `src/linguary/web` — форма выбора условий на сервере, схема результата и HTML-ответ блока.
- `src/linguary/adapters/epub` — сноска, ссылка на первое появление и обратная навигация.

**Существующий flow:** `BookConditionsService.freeze_before_analysis` закрепляет снимок условий; `StartProcessing` и повторная обработка используют привязанный к нему runtime. `application/block_processing.py` собирает результат для группы блоков, а `SqlAlchemyBlockResultRepository.replace_results` атомарно заменяет результаты. Новая модель должна сохраняться вместе с результатом блока и переживать архивирование/неудачную повторную обработку. Сейчас `ResultBlockResponse` передаёт словарные подсказки и заметки, `ResultGroupResponse` также передаёт `fragment_assistance`; EPUB-экспорт получает сохранённые результаты через `ExportBook` и строит точные ссылки на основе якорей.

Оба SDD относятся к единственной настроенной платформе `backend`. Этот план владеет фазами P1–P3, веб-компонент — P4–P5; порядок исполнения: P1 → P2 → P3 → P4 → P5. Контракты API-PH-01 и DTO-PH-01 определены здесь, веб-план только потребляет их.

Условия задаются через `POST /books/{book_id}/conditions`; их форма и отображение остаются во frontend SDD. Проверка `source_sha256` запрещает менять оригинал внутри книги: другой файл импортируется как новая книга. Изменение условий применяет существующий путь новой ревизии и не пересчитывает готовые материалы автоматически.

Дополнение раунда 6: выполненные P1–P3 и их ID сохраняются. Новая P6 исправляет обработку ошибок LLM-PH-01 и проверку полной словоформы; при противоречии с прежней проверкой всего ответа действуют уточнения P6. Показ и навигацию после частичной перегенерации исправляет P7 соседнего веб-плана. API-PH-01/DTO-PH-01, условия и версии модельного канала не меняются.

Пересмотр 2026-10-31 по принятому ADR: P1–P7 и их выполненные результаты сохраняются. Новые P8–P11 подключают локальный `ipa-dict`; только для нового вида основания заменяют требование словарного значения/POS на одну непосредственно записанную IPA точной формы. Это уникальность записи в снимке, а не подтверждение контекстной точности. LLM-PH-01, инструкции и данные, передаваемые модели, не меняются. Прежние критерии P2/P3 описывают завершённое состояние с тестовым/отключённым источником; новые условия подключения и выпуска заданы ниже. Q-1/Q-2/Q-3 остаются условиями выпуска и не запрещают реализацию и локальную техническую проверку.

## 3. Контрактные зависимости

| ID | Роль | Тип | Кратко | Source of truth |
|---|---|---|---|---|
| API-PH-01 | owner | endpoint | Условия книги с включением помощи и вариантом английского | это SDD, [P1.T1] |
| DTO-PH-01 | owner | dto | Подтверждённое вхождение для просмотра и повторных ссылок | это SDD, [P3.T1] |
| LIC-PH-01 | owner | endpoint | Локальные уведомления и тексты лицензий используемого снимка | это SDD, [P11.T1] |

## 4. Реестр идентификаторов

| ID | Тип | Определён | NORMATIVE? |
|---|---|---|---|
| P1 | фаза | § 5, сводная таблица | — |
| P2 | фаза | § 5, сводная таблица | — |
| P3 | фаза | § 5, сводная таблица | — |
| P6 | фаза | § 5, сводная таблица | — |
| P8 | фаза | § 5, сводная таблица | — |
| P9 | фаза | § 5, сводная таблица | — |
| P10 | фаза | § 5, сводная таблица | — |
| P11 | фаза | § 5, сводная таблица | — |
| SRC-PH-02 | снимок источника | [P8.T1], канонический блок | да |
| EVID-PH-02 | свидетельство | [P9.T1], канонический блок | да |
| LIC-PH-01 | endpoint | [P11.T1], канонический блок | да |
| API-PH-01 | endpoint | [P1.T1], канонический блок | да |
| SQL-PH-01 | DDL | [P1.T2], канонический блок | да |
| LLM-PH-01 | кандидаты модели | [P2.T3], канонический блок | да |
| ALG-PH-01 | группировка вхождений | [P2.T2], канонический блок | да |
| DTO-PH-01 | dto | [P3.T1], канонический блок | да |
| Q-1 | открытый вопрос | [P2.T1], [P3.T2], [P3.T3], § 7 | — |
| Q-2 | открытый вопрос | [P3.T3], § 7 | — |
| Q-3 | открытый вопрос | [P3.T2], § 7 | — |

## 5. План реализации

### Сводная таблица

| Фаза | ID | Название | Результат |
|---|---|---|---|
| 1 | P1 | Условия и сохранение | Версионируемая настройка и архивируемые подтверждённые вхождения |
| 2 | P2 | Проверка источника и контекста | Безопасно принятые кандидаты с устойчивой группировкой повторов |
| 3 | P3 | Просмотр и EPUB | Единый контракт результата и автономные сноски |
| 4 | P6 | Изоляция ошибок кандидатов и границы словоформ | Корректный перевод сохраняется, подсказка принадлежит целому слову |
| 5 | P8 | Снимок ipa-dict | Исходные UK/US файлы закреплены по SHA и снабжены уведомлениями |
| 6 | P9 | Основание точной формы | Новый вид свидетельства не выдумывает смысл и сохраняет совместимость |
| 7 | P10 | Локальный поиск и принятие | Одна различимая IPA точной формы; неоднозначное пропускается |
| 8 | P11 | Уведомления и проверка | Лицензии доступны в приложении/EPUB, малая книга проходит реальный набор |

> Прогресс и статусы фаз живут в `PROGRESS.md`, не здесь.

## Фаза P1. Условия книги и сохранение подтверждённых вхождений (зависит от: —)

**ID:** P1  
**Цель:** новый вариант произношения закрепляется в условиях книги, а принятая помощь сохраняется вместе с результатами и архивом.  
**Зависимости:** нет.

### P1.T1 Добавить настройку произношения в условия книги

**Уровень:** Complex

- Файлы: ~ src/linguary/domain/book_conditions.py, ~ src/linguary/application/book_conditions.py, ~ src/linguary/application/use_cases/reset_processing.py, ~ src/linguary/web/routes/processing.py, ~ src/linguary/composition/book_runtime.py, ~ tests/integration/processing/test_book_conditions.py
- Интерфейс: API-PH-01: `pronunciation_help_enabled: bool = false`, `pronunciation_variant: Literal["uk", "us"] | None`; доменный BookConditions и неизменяемый снимок сериализуют значения и версии оснований.
- Тесты: test_pronunciation_conditions_uk_us; test_legacy_conditions_disable_pronunciation; test_pronunciation_rejects_non_english; test_pronunciation_conditions_revision_conflict; test_pronunciation_version_change_creates_revision
- Готово, когда: новые условия воспроизводимы, прежние снимки не меняются, серверные ошибки и полный набор `uv run pytest` проходят.
- Коммит: `feat(conditions): preserve pronunciation settings`

**Референсы:** `src/linguary/application/book_conditions.py`, `src/linguary/application/use_cases/reset_processing.py`, `src/linguary/composition/book_runtime.py`.

**Что сделать:** расширить существующий доменный снимок условий и его прикладную сериализацию двумя независимыми значениями: включена ли помощь произношения и выбран ли вариант `uk` или `us`. При включённой помощи вариант обязателен; при выключенной новые подсказки не создаются. Старые снимки без этих полей читаются как выключенная помощь. Переиспользовать `reader_level`, не добавляя настройку уровня; не переиспользовать `english_help`, так как это отдельная возможность. Изменение условий создаёт новую ревизию в существующем сценарии и не меняет готовые книги автоматически. В основании снимка фиксируются версии политики отбора, правил словоформ и словарного набора; их смена исключает использование прежнего результата как результата новой ревизии.

**Ключевые ограничения:**

- Сервер разрешает включение помощи только для английского оригинала книги (`BookRecord.source_language == 'en'`); для другого языка оригинала включение отклоняется. Язык перевода (`target_language`) на доступность помощи не влияет.
- Старый исходный файл остаётся неизменным; изменённый файл — новая книга, согласно `verify_source`.
- Вызовы словаря не выполняются внутри короткой транзакции SQLite.
- Настройки и ошибка ревизии проверяются на сервере, независимо от HTML-формы.

##### Канонический блок: API-PH-01 (NORMATIVE)

> NORMATIVE. Расширение существующей формы выбора условий и её серверного ответа.

- URL: `POST /books/{book_id}/conditions`
- Method: `POST`, принимает HTML-форму или JSON.
- Request:
  - `action`: `continue_legacy` или `reanalyze`;
  - `expected_revision`: целое число не меньше 0;
  - `settings_revision`: целое число не меньше 0;
  - `confirmed`: `true`;
  - `english_help`: логическое значение, по умолчанию `false`;
  - `reader_level`: `a2`, `b1`, `b2` или `null`;
  - `pronunciation_help_enabled`: логическое значение, по умолчанию `false`;
  - `pronunciation_variant`: `uk`, `us` или `null`; при включённой помощи обязательно `uk` или `us`, при выключенной сохраняется `null`.
- Response: для HTML — `303` на страницу книги; для JSON — `201` с `conditions_id`, `revision` и `reproducibility`.
- Errors: `404` для неизвестной книги; `409` для конфликта ревизии/условий; `422` для некорректного выбора или неподтверждённого запроса.

**Проверка:** тесты условий проверяют обе разновидности английского, отсутствие новой настройки уровня, выключенные старые снимки, неверный вариант и запрет для неанглийской книги; маршрут проверяет HTTP-ошибки и создание новой ревизии.

### P1.T2 Добавить типизированное сохранение и архивирование вхождений

**Уровень:** Complex

- Файлы: + src/linguary/domain/pronunciation.py, ~ src/linguary/domain/processing.py, ~ src/linguary/application/ports/persistence.py, ~ src/linguary/application/block_processing.py, ~ src/linguary/adapters/db/models.py, ~ src/linguary/adapters/db/mapping.py, ~ src/linguary/adapters/db/repositories.py, + alembic/versions/0010_pronunciation_assistance.py, ~ tests/integration/db/test_block_result_queries.py, ~ tests/integration/db/test_alembic_migrations.py
- Интерфейс: `PronunciationAssistance` хранит SourceAnchor, фактическую форму, IPA, UK/US, hint_id и свидетельство (идентификатор записи/значения, версии источника и при необходимости правила); SQL-PH-01 задаёт отдельную nullable JSON-колонку.
- Тесты: test_pronunciation_result_roundtrip; test_pronunciation_legacy_null; test_pronunciation_archive_preserves_evidence; test_pronunciation_atomic_replace; test_pronunciation_migration_refuses_data_loss
- Готово, когда: результаты и архив проходят круг записи/чтения без потери свидетельства и остальные данные сохранены; `uv run pytest` проходит.
- Коммит: `feat(storage): persist pronunciation evidence`

**Референсы:** `src/linguary/domain/content_plan.py` (`SourceAnchor`), `src/linguary/domain/processing.py` (`BlockProcessingResult`), `src/linguary/adapters/db/models.py` и `src/linguary/adapters/db/mapping.py` (`fragment_assistance_json`), `alembic/versions/0008_fragment_assistance.py`.

**Что сделать:** создать отдельный неизменяемый тип помощи произношения с точным якорем, фактической словоформой, IPA, вариантом, идентификатором общей подсказки и свидетельством для каждой привязки. Добавить его в результат блока и сериализацию БД отдельным полем; не расширять смысловой `FragmentAssistance`. Старое значение `null` читается как пустой список. Проверить, что архивирование условий/результатов сохраняет версию источника и правила, а атомарная замена блока не теряет уже принятую помощь при временном сбое. Проверка целостности запрещает два принятых вхождения с одним и тем же якорем в результате блока; поиск по книге и условиям сначала ограничивается текущим `run_id`, затем сверяет точные якоря после чтения результата.

**Ключевые ограничения:** запись связывается с текущей книгой, ревизией условий и попыткой; якорь повторно сверяется с неизменённым оригиналом. Для каждой словоформы сохраняется основание; лишний словарный текст хранится только в объёме, разрешённом лицензией. SQLAlchemy/Alembic остаются владельцами схемы.

##### Канонический блок: SQL-PH-01 (NORMATIVE)

> NORMATIVE. Сохранённые произношения принадлежат результату блока и версионируются вместе с ним.

Добавить nullable JSON-колонку `pronunciation_assistances_json` к `block_results`. Каждая запись содержит отдельное вхождение и его свидетельство; связи по `run_id` и `block_id` берутся из строки результата. `NULL` у прежней строки означает пустую помощь. Alembic-миграция имеет `down_revision = "0009"`; откат запрещён, если в колонке уже сохранены данные, по образцу `0008_fragment_assistance.py`. Приложение использует SQLite; эквивалентная схема для SQLite, PostgreSQL и MySQL:

```sql
-- SQLite
ALTER TABLE block_results ADD COLUMN pronunciation_assistances_json JSON;

-- PostgreSQL
ALTER TABLE block_results ADD COLUMN pronunciation_assistances_json JSON NULL;

-- MySQL
ALTER TABLE block_results ADD COLUMN pronunciation_assistances_json JSON NULL;
```

**Проверка:** миграция применяется к пустой и существующей БД; round-trip сохраняет IPA, вариант, якорь и свидетельство; повторная обработка и архив читают прежние строки с пустым значением без потери остальных результатов.

#### Проверка фазы

**Автотесты:** `test_book_conditions.py`, `test_block_result_queries.py`, `test_alembic_migrations.py`; покрыть старый снимок, новую ревизию, архив, атомарную замену результата и пустое значение миграции.

**Откат:** выключить помощь в условиях книги. Миграция удаляется только пока новая колонка пуста; после сохранения IPA откат схемы откажется удалять данные, а приложение откатывается с сохранением колонки и артефактов.

## Фаза P2. Поиск кандидатов и проверка произношения (зависит от: P1)

**ID:** P2  
**Цель:** обработка создаёт только те подсказки, которые источник подтверждает для выбранного варианта и контекста.  
**Зависимости:** P1. Внутри фазы выполнять P2.T1 → P2.T3 → P2.T2: источник, контракт ответа, затем сквозная локальная проверка. Это одна последовательная фаза; общие файлы контейнера и обработки не правятся параллельно. P1 и её коммит `e26c71c` остаются выполненными.

### P2.T1 Подключить разрешённый источник через отдельный адаптер

**Уровень:** Complex

- Файлы: + src/linguary/application/ports/pronunciation.py, + src/linguary/adapters/pronunciation/source.py, ~ src/linguary/composition/container.py, ~ src/linguary/composition/book_runtime.py, + tests/integration/processing/test_pronunciation_source.py
- Интерфейс: порт `lookup(form: str, variant: Literal["uk", "us"]) -> SourceLookupResult`: варианты found/missing/unavailable различимы; found содержит разрешённые записи с IPA, значением, частью речи, версией и атрибуцией. Без настроенного источника адаптер возвращает отсутствие подтверждения.
- Тесты: test_pronunciation_source_variant_and_sense; test_pronunciation_source_missing_vs_unavailable; test_pronunciation_source_rejects_untrusted_url; test_pronunciation_disabled_source_skips
- Готово, когда: технический порт и тестовый/отключённый адаптер работают без словарных данных; подключение реальных данных остаётся заблокировано Q-1, автоматические проверки и `uv run pytest` проходят.
- Коммит: `feat(pronunciation): add replaceable source port`

**Референсы:** `src/linguary/application/ports/llm.py`, `src/linguary/composition/container.py`, сетевые и файловые адаптеры приложения.

**Что сделать:** определить порт поиска и адаптер лицензированного локального/пакетного набора. До подключения проверить образцы записей: они должны связывать IPA с вариантом UK/US, значением, частью речи и словарной формой; получение должно возвращать атрибуцию и стабильный идентификатор записи. Сетевой адаптер разрешён только если договор покрывает те же хранение и экспорт. Выбор поставщика остаётся заменяемым, но любые данные без нужных прав отклоняются.

**Ключевые ограничения:**

- Словарные данные не передаются модели без явного разрешения договора.
- Текст книги и полный абзац не отправляются поставщику; адреса из текста/ответа модели не становятся адресами запросов.
- Запросы выполняются вне транзакции; временный отказ не кэшируется как отсутствие записи.
- Без разрешённого источника обработка завершается, а IPA пропускается.

> Q-1 — поставщик, образцы записей и полные права на обработку, хранение и EPUB должны быть согласованы до подключения данных.

**Проверка:** поддельный источник проверяет UK/US, различение значений и частей речи, временный отказ и запрет на неразрешённый адрес; выбранный набор принят только после документированной проверки прав.

### P2.T3 Передать кандидатов через явный контракт модели

**Уровень:** Complex

- Файлы: ~ src/linguary/application/llm_dto.py, ~ src/linguary/application/ports/llm.py, ~ src/linguary/adapters/llm/schemas.py, ~ src/linguary/adapters/llm/payloads.py, ~ src/linguary/adapters/llm/openai_compatible.py, ~ src/linguary/adapters/llm/fake.py, ~ tests/unit/llm/test_llm_dto_contract.py, ~ tests/unit/llm/test_llm_response_schemas.py, ~ tests/unit/llm/test_llm_payload_mapping.py, ~ tests/unit/llm/test_openai_compatible_client.py, ~ tests/unit/llm/test_fake_llm_client.py
- Интерфейс: LLM-PH-01: `TranslationResultResponse.pronunciation_candidates: tuple[PronunciationCandidateResponse, ...] = ()`; `LLMRequest.for_translation(..., pronunciation_candidates_version: int | None = None)`; тот же параметр у `schema_for` и `parse_payload`, передаваемый реальным и тестовым клиентами при построении схемы и чтении ответа.
- Тесты: test_pronunciation_candidate_schema_and_dto_roundtrip; test_pronunciation_candidate_schema_for_each_base_version; test_pronunciation_candidate_schema_rejects_ipa_source_and_url; test_pronunciation_candidate_clients_use_same_contract; test_pronunciation_candidate_mode_rejects_unknown_version; test_pronunciation_disabled_contract_unchanged
- Готово, когда: JSON → строгая схема → DTO сохраняет все поля кандидата без подмены точной формы; режим поддержан для базовых контрактов 1–4, выключенные/старые запросы сохраняют прежнюю схему, `uv run pytest` проходит. P2.T2 начинается только после этой проверки.
- Коммит: `feat(pronunciation): define model candidate contract`

**Референсы:** `TranslationResultResponse` в `application/llm_dto.py`, `schema_for`/`_translation_result` в `adapters/llm/payloads.py`, `TranslationResultSchema` и версии 2–4 в `adapters/llm/schemas.py`. Путь начинается с `src/linguary/`.

**Что сделать:** добавить отдельные DTO/строгие схемы кандидата и его контекстных признаков. Режим `pronunciation_candidates_version=1` расширяет только ответ обработки группы или отдельного блока; независимо от базового `contract_version` он сохраняет его поля и правила (`content_evidence`, `fragment_assistances` и пр.). `None` сохраняет исходную схему без нового поля. Иная версия, логическое значение вместо номера или применение режима к анализу главы/вычитке отклоняются. Не менять `ContentPlanResponse`, `Difficulty`, `ContextFact.kind` или сохранённый содержательный план: канал произношения самостоятельный. Включить явное преобразование списка в `_translation_result`; нельзя полагаться на дополнительные поля Pydantic или неявную сериализацию. Метаданные запроса сохраняются при журналировании и повторных попытках через существующий `replace`.

##### Канонический блок: LLM-PH-01 (NORMATIVE)

> NORMATIVE. Модель предлагает кандидатов, но не является источником транскрипции. Это внутренний контракт, не публичный API и не контракт подтверждённого результата DTO-PH-01.

- В активной схеме обязательное поле `pronunciation_candidates` — список, допустим `[]`; в DTO это кортеж. Отсутствие списка у активного запроса — ошибка формата по существующей политике повторов. В выключенном режиме поле не запрашивается; старые DTO создаются с пустым кортежем.
- Каждый кандидат имеет ровно `anchor`, `form`, `context`, `selection_reason`; кандидат и вложенные объекты запрещают лишние поля.
- `anchor = {block_id: str, start: int, end: int, quote: str}` переиспользует правила `SourceAnchor`: смещения в символах Python, диапазон `[start, end)`, `end - start == len(quote)`. В модельном ответе `block_id` — псевдоним `b1..bN` именно целевой группы, после преобразования — постоянный ID. `quote` и `form` сохраняются дословно, без удаления пробелов или нормализации; `form == quote` и точность по оригиналу проверяются локально до поиска.
- `context = {meaning_in_context: str | null, part_of_speech: str | null, grammatical_form: str | null, is_proper_name: bool}`: описание значения на языке оригинала, часть речи, грамматическая форма (включая время/число, когда применимо), признак собственного имени. Все ключи обязательны; неизвестные текстовые признаки обозначаются `null`, непустые значения — только признаки из оригинала. Это предположения для локального сопоставления, а не словарное свидетельство; неизвестность или двусмысленность, мешающая выбору, даёт пропуск.
- `selection_reason: Literal["unpredictable_spelling", "unpredictable_stress", "context_dependent"]` обозначает риск ошибки чтения. Редкость и длина могут сопровождать отбор, но сами по себе не дают кандидата. Уровень берётся из условий книги; при отсутствии уровня новая настройка не нужна.
- Ни запрос, ни ответ кандидатов не содержат словарных записей, IPA, транскрипции, фонетического ударения/окончаний, `hint_id`, идентификаторов значений источника, атрибуции или URL. UK/US берётся локально из условий, а не выбирается моделью. Схема запрещает дополнительные поля `ipa`, `source`, `url` и их аналоги; текстовые сигналы не интерпретируются как адреса запросов.

**Проверка:** каждый базовый контракт 1–4 сохраняет прежние поля в обоих режимах; обязательность списка, пустой список и `null` в неизвестных признаках проверены. Подмена реального клиента тестовым не меняет результат разбора; точные строки и смещения не теряются.

### P2.T2 Проверять кандидатов и группировать одинаковое произношение

**Уровень:** Complex

- Файлы: ~ src/linguary/application/context_builder.py, ~ src/linguary/application/book_conditions.py, + src/linguary/application/validators/pronunciation.py, ~ src/linguary/application/block_processing.py, ~ src/linguary/application/use_cases/process_chapter.py, ~ src/linguary/application/use_cases/regenerate_block.py, ~ src/linguary/composition/container.py, ~ src/linguary/composition/book_runtime.py, ~ tests/unit/application/test_translation_context.py, ~ tests/unit/application/test_context_builder.py, ~ tests/integration/processing/test_book_conditions.py, ~ tests/integration/processing/test_content_plan_v3_pipeline.py, + tests/integration/processing/test_pronunciation_selection.py, ~ tests/integration/processing/test_process_chapter.py, ~ tests/integration/processing/test_regenerate_block.py
- Интерфейс: после P2.T3 `TranslationContext.to_block_ids` переносит LLM-PH-01 и преобразует якоря кандидатов; `snapshot_prompt_templates(..., pronunciation_help_enabled=False)` закрепляет версию канала и инструкции; локальная проверка получает кандидата, исходные тексты, условия и порт источника, возвращает подтверждённое вхождение либо пропуск с причиной. ALG-PH-01 группирует одинаковые форму/IPA/UK-US; временные отказы отмечаются для конкретных якорей; до возврата `TranslationCallResult.results` к подтверждённым новым вхождениям добавляются прежние для отказавших якорей, затем применяется существующий `TranslationCall.save` с проверкой `expected_bindings`.
- Тесты: test_pronunciation_read_as_is_without_translation; test_pronunciation_homographs_and_inflections; test_pronunciation_no_level_uses_context; test_pronunciation_rejects_unproven_names; test_pronunciation_primary_ignores_completion_order; test_pronunciation_failure_preserves_previous; test_pronunciation_candidate_alias_and_exact_form; test_pronunciation_candidate_invalid_does_not_discard_translation; test_pronunciation_candidate_mode_pinned_to_conditions; test_pronunciation_old_snapshot_unchanged; test_pronunciation_dictionary_never_reaches_model; test_pronunciation_candidates_regeneration_and_recovery; test_pronunciation_partial_failure_keeps_confirmed_new_and_previous_failed_anchors
- Готово, когда: P2.T3 выполнена; кандидат из настоящего JSON-ответа проходит все проекции до локального источника и сохранения на искусственных данных; источник и проверяемое правило подтверждают каждую IPA; повторы устойчивы, исходный текст и смысловые материалы сохранены, `uv run pytest` проходит. Обязательна проверка отсутствия словарных данных в запросах и инструкциях повторных попыток; она не закрывает Q-1/Q-2/Q-3.
- Коммит: `feat(pronunciation): verify contextual candidates`

**Референсы:** `src/linguary/domain/content_plan.py` (`SourceAnchor`), `src/linguary/application/context_builder.py` (`_translation_context`, `TranslationContext.to_block_ids`, `snapshot_prompt_templates`), `src/linguary/application/book_conditions.py` (`create`), `src/linguary/application/block_processing.py` (`TranslationCall.request`/`save`), `src/linguary/composition/container.py` (`generation_factory`).

**Что сделать:** включать LLM-PH-01 только из неизменяемых условий английского оригинала с `pronunciation_help_enabled=true`; `english_help` и режим блока на него не влияют. При создании новой ревизии условий закрепить в `prompt_templates` ключи `pronunciation_candidates_version: 1` и `pronunciation_processing` с инструкциями кандидатов и пересчитать существующий `prompt_sha256`, даже если включено только произношение. ContextBuilder получает уже сохранённые шаблоны через существующую сборку runtime; не брать переключатель из текущих глобальных настроек. Не менять базовые `versions.schemas`/`contract_version` 1–4 и проверки содержательной помощи. Старые снимки и их хеши не дополнять: без маркера канала остаётся прежняя схема; при старом включённом снимке новые IPA пропускаются как `candidate_contract_unavailable` с сохранением ранее принятых данных, включение нового канала требует новой ревизии. Неизвестная версия маркера не заменяется текущей молча.

Дополнить инструкции после всех существующих правил режима: самостоятельные кандидаты разрешены в TRANSLATE/GLOSSARY_ONLY/READ_AS_IS, даже когда смысловая помощь не нужна; не запрашивать IPA или словарь. Общая `_translation_context` выбирает дополнительную версию для обычной группы, повторного блока и дочерней группы восстановления. `to_block_ids` явно переносит кортеж и заменяет `candidate.anchor.block_id`; кандидат на соседа/чужой псевдоним пропускается с причиной, а исходные проверки чужих блоков/заметок не ослабляются. До обращения к источнику локально сверить принадлежность целевой группе, диапазон, дословный `quote`/`form`, повторный якорь и признаки контекста.

Передать порт/проверяющий компонент из контейнера в общий `TranslationCall` через ProcessChapter и RegenerateBlock; оба сценария используют одну проверку и существующий путь атомарного сохранения. После успешной проверки переводных и смысловых материалов, вне короткой транзакции и вне цикла исправлений модели, локально сопоставить кандидатов с источником по фактической форме, однозначному значению, грамматике и варианту условий. Для производной формы требовать источник основы и проверяемое версионированное правило. Недостоверный кандидат, неоднозначность, неподтверждённое имя и отсутствие данных дают пропуск с причиной; они не повторяют генерацию перевода и не стирают другие материалы. Временный отказ сохраняет прежнюю помощь именно для отказавших якорей текущих книги/условий/run_id: объединить её с подтверждёнными новыми вхождениями до возврата результата запроса, затем сохранить через существующую проверку `expected_bindings`. Недоступный канал сохраняет весь прежний список для целевых блоков. Отказы держатся в локальных переменных результата обработки; новый ContextVar и общий флаг, заменяющий подтверждённые новые данные прежним списком всего блока, не нужны. Проверенное отсутствие для одного якоря не сохраняет его прежнюю IPA; проверенный пустой набор отличается от отказа. Ошибка источника, его записи и свидетельства не попадают в инструкции исправлений модели. Одинаковые якоря проверяются один раз, конфликтующие кандидаты одного якоря пропускаются, а не принимаются по порядку ответа. Принятые вхождения присоединяются к своим результатам блока до сохранения; ALG-PH-01 применяется к сохранённым подтверждённым данным.

**Ключевые ограничения:** не выбирать слово только по длине/редкости; не использовать перевод или `FragmentAssistance` как доказательство произношения; не терять переводы, словарные подсказки и уже принятую IPA при частичном сбое.

##### Канонический блок: ALG-PH-01 (NORMATIVE)

> NORMATIVE. Идентичность общей подсказки и её первое появление не зависят от порядка завершения фоновых задач.

```text
для каждого подтверждённого вхождения:
    identity = (нормализованная фактическая форма, нормализованная IPA, variant)
    сохранить вхождение и отдельное свидетельство для его контекста
    сгруппировать вхождения с одинаковой identity
для каждой группы:
    primary = вхождение с минимальным порядком книги и смещением
    остальные вхождения ссылаются на primary.id
если IPA, форма или вариант отличаются:
    создать отдельную группу
```

**Проверка:** положительные и отрицательные примеры, омографы, формы, UK/US и неподтверждённые имена; параллельная обработка не меняет первое появление; сбой источника не удаляет принятый результат.

#### Проверка фазы

**Автотесты:** тесты DTO/схем/проекций и обоих LLM-клиентов P2.T3, `test_translation_context.py`, `test_context_builder.py`, `test_book_conditions.py`, `test_pronunciation_source.py`, `test_pronunciation_selection.py`, `test_process_chapter.py`, `test_regenerate_block.py`. Сквозной случай использует JSON через настоящий парсер и тестовый источник, проверяет READ_AS_IS без перевода/FragmentAssistance, обычную обработку, повторный запуск и дочернее восстановление; выключенный/старый режим сохраняет прежнее поведение. Отдельный обязательный барьер P2: уникальные текстовые метки словарной записи/IPA/свидетельства отсутствуют во всех исходящих запросах и инструкциях исправлений модели, включая временный отказ. Это техническая проверка границы, а не согласование лицензии Q-1.

**Откат:** отключить источник и помощь для новых условий. Не удалять уже сохранённые свидетельства; новые обработки продолжаются без IPA.

## Фаза P3. Контракты чтения и экспорт EPUB (зависит от: P1, P2)

**ID:** P3  
**Цель:** одна прикладная проекция принятых вхождений задаёт порядок, идентичность и разрешённую атрибуцию для веба и EPUB без сетевого запроса при чтении.  
**Зависимости:** P1 и P2.

### P3.T1 Выдавать общий контракт подтверждённого вхождения

**Уровень:** Complex

- Файлы: + src/linguary/application/pronunciation_projection.py, ~ src/linguary/web/schemas.py, ~ src/linguary/web/routes/result.py, + tests/unit/application/test_pronunciation_projection.py, ~ tests/integration/web/test_result_and_export.py
- Интерфейс: `project_pronunciations(blocks: Sequence[DocumentBlock], results: Mapping[str, BlockProcessingResult]) -> tuple[PronunciationOccurrence, ...]`; неизменяемый `PronunciationOccurrence(hint_id, block_id, start, end, quote, ipa, variant, source_label, source_ref, is_primary)` — прикладной источник DTO-PH-01 для веба и EPUB. ResultBlockResponse и ResultGroupResponse передают отдельный список тех же полей; полный ответ и фрагмент получают одну проекцию по снимку результата книги.
- Тесты: test_pronunciation_projection_stable_original_order; test_pronunciation_projection_rejects_wrong_owner_or_anchor; test_pronunciation_projection_primary_and_repeat_identity; test_pronunciation_result_and_fragment_contract; test_pronunciation_fragment_uses_book_snapshot; test_pronunciation_projection_exposes_only_allowed_attribution
- Готово, когда: прикладная проекция единожды проверяет принадлежность записи блоку и `SourceAnchor` по исходному тексту, а полная страница и повторная обработка потребляют один DTO с устойчивыми первичным вхождением и ссылками; `uv run pytest` проходит.
- Коммит: `feat(web): expose verified pronunciation data`

**Референсы:** `ResultBlockResponse.of_block`, `ResultGroupResponse.of_group` в `src/linguary/web/schemas.py`; `src/linguary/web/routes/result.py`.

**Что сделать:** добавить прикладную проекцию без зависимостей от веб-схем. Она принимает блоки в порядке оригинала и результаты текущего запуска, проверяет, что каждая запись принадлежит результату и блоку её якоря, а `SourceAnchor` совпадает с исходным текстом. Неверную запись не показывать. Для проверенных вхождений вызвать `group_pronunciations(block_order=...)` из P2 и вернуть их в порядке оригинала: одна группа имеет общий `hint_id`, первое вхождение — `is_primary=True`, повторы — `False`. `source_label` берётся только из разрешённой атрибуции принятого свидетельства, `source_ref` — из `entry_id`; полное свидетельство, словарное содержимое и непроверенные URL в DTO не передаются. Веб-схемы преобразуют эту запись без повторного сопоставления. Полный результат и ответ блока читают согласованный снимок книги; ответ блока выбирает нужную группу после расчёта первичного вхождения по всей книге, сохраняя состояние задачи повторной обработки из того же снимка.

**Ключевые ограничения:** один и тот же контракт используется JSON и серверным HTML; частичный ответ блока ссылается на первичную запись в книге, оригинальная строка не меняется. EPUB импортирует прикладную проекцию, а не `linguary.web.schemas`.

##### Канонический блок: DTO-PH-01 (NORMATIVE)

> NORMATIVE. Это контракт между бэкендом, веб-шаблонами и проекцией EPUB; источником остаётся сохранённое подтверждённое вхождение.

| Поле | Тип | Описание |
|---|---|---|
| `hint_id` | `str` | Общий идентификатор одинаковой фактической формы, IPA и UK/US внутри результата книги. |
| `block_id` | `str` | Блок точного вхождения. |
| `start`, `end` | `int` | Смещения исходного текста, проверенные через `SourceAnchor`. |
| `quote` | `str` | Дословный фрагмент оригинала для привязки. |
| `ipa` | `str` | Подтверждённая транскрипция с ударением. |
| `variant` | `Literal["uk", "us"]` | Выбранный вариант книги. |
| `source_label` | `str` | Атрибуция, разрешённая условиями источника. |
| `source_ref` | `str \| None` | Идентификатор `entry_id` принятого свидетельства; произвольная ссылка не выводится. |
| `is_primary` | `bool` | Первое вхождение по порядку оригинала показывает подсказку; остальные ссылаются на него. |

**Проверка:** API-ответ и HTML-фрагмент совпадают по данным; повторный вызов и перестановка завершения задач не меняют `hint_id` и первичную привязку.

### P3.T2 Добавить сноски с обратными ссылками в EPUB

**Уровень:** Complex

- Файлы: ~ src/linguary/adapters/epub/writer.py, ~ tests/integration/epub/test_epub_export_content.py, ~ tests/integration/epub/test_epub_export_structure.py
- Интерфейс: EPUB передаёт в `project_pronunciations` из P3.T1 все `section.ordered_blocks` из `request.document.ordered_sections` и `request.block_results`; получает `PronunciationOccurrence` для уникальных сносок, ссылок повторов и возврата, текстовой памятки IPA и разрешённой атрибуции без запроса в сеть.
- Тесты: test_pronunciation_epub_primary_and_repeat_links; test_pronunciation_epub_matches_web_projection; test_pronunciation_epub_backlinks; test_pronunciation_epub_original_markup_unchanged; test_pronunciation_epub_guide_and_offline_reading
- Готово, когда: искусственные данные проходят структурные проверки и `uv run pytest`; выпуск EPUB ждёт Q-1 и документированной проверки согласованной матрицы Q-3.
- Коммит: `feat(epub): render pronunciation notes and guide`

**Референсы:** существующая обработка точных `SourceAnchor`, сносок и обратных ссылок в `src/linguary/adapters/epub/writer.py`.

**Что сделать:** использовать те же принятые записи и порядок книги, что у веба, через прикладную проекцию P3.T1. Сформировать её один раз для всего экспортируемого снимка до обхода разделов, чтобы первое появление и `hint_id` не менялись на границе XHTML-разделов. Первое вхождение ведёт к самостоятельной сноске, повторы — к этой сноске; обратная ссылка возвращает к нужному слову. Сноска содержит IPA, UK/US, краткую памятку и разрешённую атрибуцию; сети при чтении нет.

**Ключевые ограничения:** не менять ни один символ оригинала и авторскую разметку; не занимать существующие идентификаторы сносок; не встраивать источник или атрибуцию без письменных прав. EPUB не импортирует веб-схемы и не повторяет проверку якорей или выбор первичного вхождения.

> Q-1 — права должны включать сохранение свидетельств, показ и экспорт готовых книг.
> Q-3 — совместимость EPUB подтверждается на согласованных программах чтения и начертаниях IPA.

**Проверка:** структура EPUB содержит корректные ссылки и обратные ссылки; исходный текст и inline-разметка совпадают до/после экспорта; проверяется чтение без сети и матрица целевых устройств.

### P3.T3 Проверить качество отбора и частичные сбои

**Уровень:** Standard

- Файлы: + tests/fixtures/pronunciation/acceptance_cases.json, + tests/integration/processing/test_pronunciation_quality_gate.py
- Интерфейс: контрольный случай содержит текст, вариант, контекст, ожидаемое наличие/отсутствие IPA и проверяемое основание; числовой порог и независимая вычитка задаются согласованным решением Q-2, без предустановленного проходного значения.
- Тесты: test_pronunciation_acceptance_positive_and_negative; test_pronunciation_quality_threshold_requires_agreement; test_pronunciation_partial_failure_preserves_materials; test_pronunciation_repeat_and_export_regression
- Готово, когда: автоматические проверки на искусственных данных и `uv run pytest` проходят; готовность к выпуску не подтверждается без допустимых данных Q-1 и отчёта независимой вычитки по Q-2.
- Коммит: `test(pronunciation): define acceptance and failure cases`

**Референсы:** `tests/integration/processing/test_content_plan_v3_pipeline.py`, `tests/integration/processing/test_regenerate_block.py`, AC2/AC4/AC6/AC8 BRIEF.

**Что сделать:** зафиксировать согласованный контрольный набор положительных и отрицательных примеров и независимую оценку точности/уместности. Включить форму слова, значение, ударение, UK/US, повторы, неоднозначность и отсутствие источника; проверить, что частичный отказ оставляет книгу и прежние смысловые материалы пригодными.

**Ключевые ограничения:** тестовые записи источника включаются только в объёме, разрешённом лицензией; без согласованного порога результат не считается готовым к выпуску.

> Q-1 — контрольный набор не должен включать данные, не разрешённые для хранения в тестах.
> Q-2 — независимый языковой редактор и владелец продукта согласуют набор и числовой порог до оценки готовности.

**Проверка:** тестовый набор автоматически прогоняется и отчёт вычитки подтверждает заданный порог; отдельно проверены повторный запуск, экспорт и отказ источника.

#### Проверка фазы

**Автотесты:** `test_result_and_export.py`, `test_epub_export_content.py`, `test_epub_export_structure.py`, `test_pronunciation_quality_gate.py`; сквозная проверка сравнивает порядок и якоря в веб-результате и EPUB.

**Откат:** отключить выдачу новых IPA в просмотре и экспорте, сохранив исходный текст, переводы и принятые свидетельства. Возврат к предыдущему коду не удаляет уже созданные EPUB или записи.

## Фаза P6. Изоляция ошибок кандидатов и границы словоформ (зависит от: P2, P3)

**Цель:** исправить две находки ревью без повторного выполнения P1–P3. Выполнять P6.T1 → P6.T2 → P6.T3 последовательно; фаза завершается одним коммитом. P4–P5 соседнего SDD и коммиты `e26c71c`/`5288fbf`/`ce9b3b2`/`3a95448`/`d112211` сохраняются.

### P6.T1 Отделить проверку элементов кандидатов от проверки перевода

- Файлы: ~ src/linguary/adapters/llm/payloads.py, ~ src/linguary/adapters/llm/schemas.py, ~ tests/unit/llm/test_llm_payload_mapping.py, ~ tests/unit/llm/test_llm_response_schemas.py
- Интерфейс: сигнатуры `schema_for(...)` и `parse_payload(...) -> LLMPayload` сохраняются. `schema_for(..., pronunciation_candidates_version=1)` по-прежнему выдаёт поставщику строгую схему `PronunciationCandidateSchema` с `extra="forbid"` на всех уровнях. При чтении активного ответа `parse_payload` отдельно проверяет основной перевод по его базовой версии 1–4 и каждый элемент списка по той же строгой схеме кандидата; в `TranslationResultResponse.pronunciation_candidates` попадают только корректные DTO в исходном порядке.
- Тесты: test_pronunciation_mixed_malformed_candidates_keep_valid_translation_and_candidate; test_pronunciation_all_malformed_candidates_keep_translation; test_pronunciation_forbidden_fields_skip_whole_candidate; test_pronunciation_invalid_main_response_still_rejected; test_pronunciation_candidate_envelope_still_required; test_pronunciation_provider_schema_stays_strict
- Готово, когда: для каждой базовой версии 1–4 один некорректный элемент, включая ошибку создания DTO, не вызывает `LLMResponseFormatError` всего корректного перевода и не удаляет другие корректные кандидаты; исходные проверки перевода, заметок, `content_evidence` и `fragment_assistances` остаются строгими; целевые тесты и `uv run pytest` проходят.
- Коммит: `fix(pronunciation): isolate candidates and enforce word boundaries`

**Уточнение LLM-PH-01 (NORMATIVE):** синтаксически неверный/оборванный JSON, неверный корневой объект, отсутствующее обязательное поле `pronunciation_candidates` или его значение не-список остаются ошибкой формата ответа. При корректном списке некорректный элемент пропускается целиком: `null`, строка, неверные типы (в том числе `bool` вместо смещения), отсутствующий ключ, неверный `selection_reason` или лишнее поле в кандидате/`anchor`/`context`. Не исправлять, не приводить типы и не удалять запрещённые поля ради сохранения кандидата. Элемент с `ipa`, `source`, `url`, `hint_id`, `variant` либо иным лишним полем не проходит в DTO, локальный поиск, сохранение или инструкции исправления модели. Основной ответ проверяется без единственного заранее извлечённого поля кандидатов; остальные лишние корневые поля запрещены. Не заменять схему поставщика на `Any`/свободный объект и не ослаблять `_ResponseModel`/схемы других каналов. Выключенный режим, неизвестные версии и запрет для анализа/вычитки сохраняются; изменение политики локального разбора не переписывает старые снимки условий.

### P6.T2 Проверять целую словоформу перед поиском и показом

- Файлы: ~ src/linguary/application/validators/pronunciation.py, ~ src/linguary/application/pronunciation_projection.py, ~ tests/integration/processing/test_pronunciation_selection.py, ~ tests/unit/application/test_pronunciation_projection.py
- Интерфейс: общий чистый предикат `is_complete_pronunciation_form(text: str, start: int, end: int) -> bool` вызывается после `SourceAnchor.validate` и дословного `form == quote`, до `source.lookup`; отказ получает причину `partial_word`. `project_pronunciations` использует тот же предикат для сохранённых вхождений, а `preserve_unavailable` не возвращает ранее принятую часть слова. Сигнатуры `verify_candidates`, `project_pronunciations` и DTO-PH-01 не меняются; общий `SourceAnchor` смысловых материалов не меняется.
- Тесты: test_pronunciation_rejects_read_inside_bread_before_lookup; test_pronunciation_rejects_prefix_suffix_and_combining_mark_splits; test_pronunciation_rejects_apostrophe_and_hyphen_word_parts; test_pronunciation_accepts_complete_unicode_form_and_quoted_word; test_pronunciation_projection_omits_legacy_partial_word; test_pronunciation_failure_does_not_restore_partial_word
- Готово, когда: срез `[1,5)` с `quote=form="read"` внутри `bread`, префиксы/суффиксы и части составных форм отклоняются до обращения к источнику; целая точная словоформа проходит при остальных выполненных проверках; веб и EPUB через общую проекцию не показывают старые неверные привязки; целевые тесты и `uv run pytest` проходят.
- Коммит: `fix(pronunciation): isolate candidates and enforce word boundaries`

**Границы (NORMATIVE):** смещения — исходные символы Python, без NFC/NFD-нормализации текста, цитаты или координат. Буквы категорий Unicode `L*` и следующие диакритические знаки `M*` образуют непрерывную словоформу: нельзя начать/закончить срез внутри такой последовательности или перед отделённой диакритикой; одна диакритика без буквы не слово. Апострофы `'`/`’`/`ʼ` и дефисы `-`/`‐` (U+2010)/`‑` (U+2011) между буквенными частями соединяют их в одну форму, включая последовательности диакритики рядом со знаком. Проверять обе стороны среза: `can`/`t` внутри `can't`/`can’t`, `read`/`only` внутри `read-only`/`read‑only` — пропуск; полный `can't`/`read-only`, слово на начале/конце блока и `read` в кавычках `'read'` или `“read”` — не часть слова. Пробел, точка/запятая, тире U+2013/U+2014 и внешние кавычки разделяют слова. Проверка границы не расширяет `valid_lookup_form` или покрытие источника: неподдерживаемая целая форма по-прежнему пропускается. Включить составные NFC/NFD-примеры, начало с комбинирующего знака и срез перед ним; не использовать ASCII `\b` как определение слова.

### P6.T3 Подтвердить независимость перевода в общем пути обработки

- Файлы: ~ tests/integration/processing/test_pronunciation_selection.py, ~ tests/unit/llm/test_openai_compatible_client.py, ~ tests/unit/llm/test_fake_llm_client.py
- Интерфейс: настоящий JSON-парсер и общий `TranslationCall` в ProcessChapter/RegenerateBlock/дочернем восстановлении получают смесь корректного кандидата, структурно ошибочных элементов и точного среза части слова; только подтверждённая целая форма доходит до тестового источника и SQLite. DTO и API не расширяются.
- Тесты: test_pronunciation_raw_malformed_candidates_preserve_translation_and_meaning; test_pronunciation_mixed_candidates_process_regenerate_and_recovery; test_pronunciation_clients_share_item_isolation; test_pronunciation_candidate_rejections_do_not_retry_translation; test_pronunciation_rejected_content_never_reaches_model
- Готово, когда: перевод, словарная помощь и смысловые материалы сохраняются без дополнительных попыток модели из-за ошибок кандидатов; корректный соседний кандидат принимается, запрещённые поля и частичные слова не попадают в источник/БД, а версии 1–4 и выключенный канал сохраняют прежние проверки. Искусственные метки словаря/IPA/свидетельства отсутствуют во всех исходящих запросах, включая исправления; полный `uv run pytest` проходит. Q-1/Q-2/Q-3 не закрываются этими тестами.
- Коммит: `fix(pronunciation): isolate candidates and enforce word boundaries`

**Проверка фазы:** сначала тесты P6.T1/P6.T2, затем сквозные проверки и `uv run pytest`. Не ограничиваться прежним `test_pronunciation_candidate_invalid_does_not_discard_translation`: там кандидаты структурно корректны и ошибочны лишь по якорю/форме. Отдельно проверить некорректный перевод с корректными кандидатами: он по-прежнему отклоняется, ни один результат не сохраняется. Откат — отключить создание новых подсказок; не удалять переводы, свидетельства и архивы.

## Фаза P8. Закреплённый снимок ipa-dict и лицензионные материалы (зависит от: P6)

**ID:** P8  
**Цель:** поставить воспроизводимые исходные UK/US данные, не обращаясь к сети во время обработки.  
**Зависимости:** P6; может выполняться независимо от P9. Файлы P1–P7 не пересоздаются.

### P8.T1 Сохранить исходные файлы и манифест закреплённой ревизии

- Файлы: + src/linguary/adapters/pronunciation/data/972cd67dd5f96bc5660e28070cde26e3928abcd7/en_UK.txt, + src/linguary/adapters/pronunciation/data/972cd67dd5f96bc5660e28070cde26e3928abcd7/en_US.txt, + src/linguary/adapters/pronunciation/data/972cd67dd5f96bc5660e28070cde26e3928abcd7/manifest.json, + tests/integration/processing/test_ipa_dict_snapshot.py
- Интерфейс: SRC-PH-02 — исходные UTF-8 файлы без редактирования; манифест фиксирует репозиторий, commit SHA, пути, байты, строки, SHA-256, `parser_version=1`, UK/US и лицензии. Загружать при подготовке только по полному SHA, не по `master`.
- Тесты: test_ipa_dict_snapshot_hashes; test_ipa_dict_snapshot_utf8_and_tab_format; test_ipa_dict_snapshot_packaged_resources
- Готово, когда: файлы совпадают с SRC-PH-02 и входят в пакет через существующий Hatch `packages=["src/linguary"]`; корневые конфиги и зависимости не меняются; проверка не требует сети.
- Коммит: `feat(pronunciation): vendor pinned ipa-dict snapshot`

##### Канонический блок: SRC-PH-02 (NORMATIVE)

Репозиторий: `https://github.com/open-dict-data/ipa-dict`; commit `972cd67dd5f96bc5660e28070cde26e3928abcd7` (проверен 2026-10-31). Файлы получаются с `https://raw.githubusercontent.com/open-dict-data/ipa-dict/<SHA>/data/<имя>`.

| Файл | Байты | Строки | SHA-256 |
| --- | --- | --- | --- |
| `en_UK.txt` | 1684124 | 65119 | `221394caef0cf723b4f2df81a98ac33191293257b88aed5b1fb89466d3a0dc77` |
| `en_US.txt` | 3180267 | 125927 | `2af6f154a5c363275f052d1f85acedef38ed185ca9745aa4314be77f6b70de67` |

`source_version = "ipa-dict@972cd67dd5f96bc5660e28070cde26e3928abcd7:parser-1"`. Исходный формат: одна словоформа, табуляция, поле IPA; число альтернатив не равно числу строк. Сохранять знаки ударения, окончания и фонемы источника; никакой генерации или исправления IPA. Обновление SHA/парсера — новое основание и новая ревизия условий, прежний ресурс сохраняется для старых результатов.

### P8.T2 Подготовить комплект происхождения и лицензий снимка

- Файлы: + src/linguary/adapters/pronunciation/data/972cd67dd5f96bc5660e28070cde26e3928abcd7/NOTICE.md, + src/linguary/adapters/pronunciation/data/972cd67dd5f96bc5660e28070cde26e3928abcd7/LICENSES.txt, ~ tests/integration/processing/test_ipa_dict_snapshot.py
- Интерфейс: `NOTICE.md` связывает каждый файл со снимком, источниками и лицензией; `LICENSES.txt` содержит полные применимые тексты и авторские уведомления. en_US: MIT ipa-dict/CMUdict-IPA/слогового преобразования и сохранённые уведомления CMUdict; en_UK: GPL-3.0 ipacards. Общий MIT репозитория не заменяет условия компонентов.
- Тесты: test_ipa_dict_notice_covers_each_locale; test_ipa_dict_license_texts_and_sources_present
- Готово, когда: нет потерянных авторских уведомлений, записаны закреплённые ссылки и отсутствие наших правок исходных файлов; отмечены автоматическое происхождение UK через eSpeak, возможные ошибки и неполное исключение омографов. Комплект подготовлен для Q-1, но Q-1/Q-2 не объявлены пройденными.
- Коммит: `docs(pronunciation): preserve ipa-dict licenses and provenance`

## Фаза P9. Типизированное основание точной формы и совместимость хранения (зависит от: P6)

**ID:** P9  
**Цель:** новое свидетельство честно хранит отсутствие семантических данных, сохраняя строгий старый вид и прежние записи.  
**Зависимости:** P6; может выполняться независимо от P8.

### P9.T1 Различить семантическое свидетельство и точную форму лексикона

- Файлы: ~ src/linguary/domain/pronunciation.py, ~ src/linguary/application/ports/pronunciation.py, + tests/unit/domain/test_pronunciation_evidence.py
- Интерфейс: EVID-PH-02 — `PronunciationEvidence.kind: Literal["dictionary_sense", "lexicon_form"] = "dictionary_sense"`; `sense_id`, `part_of_speech` допускают `None` только для `lexicon_form`; новые `license_id: str | None = None` и `source_uri: str | None = None`. `SourceEntry.sense` и `grammatical_form` также отсутствуют только у этого вида. Старые позиционные вызовы сохраняются.
- Тесты: test_lexicon_form_evidence_has_no_semantic_claim; test_dictionary_sense_evidence_stays_strict; test_lexicon_form_requires_license_and_pinned_source; test_lexicon_form_rejects_inflection_and_proper_name_claim
- Готово, когда: вид лексикона требует источник/снимок/стабильный entry_id, точные dictionary_form/dictionary_ipa, атрибуцию, лицензию и закреплённый URI; запрещает sense/POS, правило вывода формы и подтверждение имени. Старый вид по-прежнему требует смысл, POS и согласованное правило; произвольный/неполный вид отклоняется.
- Коммит: `feat(pronunciation): distinguish exact-form lexicon evidence`

##### Канонический блок: EVID-PH-02 (NORMATIVE)

`dictionary_sense` сохраняет все прежние гарантии; отсутствие нового `kind` в сохранённом старом JSON означает именно этот вид. `lexicon_form` подтверждает только непосредственно записанную форму и IPA выбранного набора; `sense_id=None`, `part_of_speech=None`, `SourceEntry.sense=None`, `grammatical_form=None`. Нельзя заполнять их значениями модели, фиктивными строками или трактовать `None` как совпадение любого смысла. `dictionary_form` остаётся точным ключом набора и равен форме в якоре; произвольное изменение регистра, Unicode или окончания не создаёт новой подтверждённой формы. Источник не подтверждает собственное имя. Уникальность IPA не является флагом независимого качества или контекстной точности.

### P9.T2 Сохранить новый вид и читать старый JSON без изменения схемы БД

- Файлы: ~ src/linguary/adapters/db/mapping.py, ~ tests/integration/db/test_block_result_queries.py
- Интерфейс: `_decode_pronunciation` распознаёт прежний точный набор полей и новый набор EVID-PH-02, назначая старому только безопасный `kind="dictionary_sense"`; `_encode_pronunciation` сериализует новый вид и метаданные лицензии. Колонка SQL-PH-01 и миграция 0010 не меняются.
- Тесты: test_pronunciation_legacy_evidence_json_roundtrip; test_lexicon_form_evidence_roundtrip_and_archive; test_pronunciation_unknown_kind_and_fields_rejected
- Готово, когда: старые результаты/архивы читаются без переписывания; новые сохраняют источник, версию, лицензию и отсутствующие семантические поля; неизвестные поля, неверные типы и частично смешанные виды отклоняются. `uv run pytest` проходит.
- Коммит: `fix(storage): preserve typed pronunciation provenance`

## Фаза P10. Локальный источник ipa-dict и консервативное принятие (зависит от: P8, P9)

**ID:** P10  
**Цель:** новые включённые условия используют свой снимок, а неизвестное и неоднозначное не попадает в помощь.  
**Зависимости:** P8, P9. Все три задачи последовательны; модельные схемы и инструкции не меняются.

### P10.T1 Прочитать и проверить весь набор вариантов локального источника

- Файлы: ~ src/linguary/adapters/pronunciation/source.py, ~ tests/integration/processing/test_pronunciation_source.py
- Интерфейс: `IpaDictPronunciationSource` реализует прежний `configured`/`async lookup(form, variant)`; использует `importlib.resources`, SRC-PH-02 и неизменяемый индекс по точному ключу/UK-US. Успешный индекс кэшируется по снимку и парсеру, за пределами транзакции; неуспех не превращается в постоянный missing. `DisabledPronunciationSource` остаётся.
- Тесты: test_ipa_dict_uk_us_exact_lookup; test_ipa_dict_all_alternatives_in_one_or_many_wrappers; test_ipa_dict_duplicate_rows_union_before_selection; test_ipa_dict_invalid_alternative_never_becomes_unique; test_ipa_dict_corrupt_or_missing_resource_is_unavailable; test_ipa_dict_no_network
- Готово, когда: хеши и UTF-8/табличный формат проверены до использования; `the\t/ðə, ði/` даёт два варианта, `read\t/ˈɹɛd/, /ˈɹid/` — два, точные дубли не порождают ложную неоднозначность. Разбор не теряет неисправную альтернативу, не выбирает первую и не исправляет фонемы/ударение. Непонятная необрабатываемая нотация делает запись непригодной; повреждение/отсутствие снимка даёт unavailable, отсутствие формы — missing.
- Коммит: `feat(pronunciation): load local ipa-dict pronunciations`

Все альтернативы объединяются до решения о единственности. Допустимо снять внешние `/…/` и пробелы разделителей; сравнение сохраняет ударения, длину и фонемы. Одна строка, допустимый алфавит IPA, одна оставшаяся после отбрасывания ошибок альтернатива и отсутствие в файле других чтений не служат доказательством качества. Пустой или неизвестный вариант UK/US не подменяется другим. Каждой записи соответствует стабильный entry_id с именем исходного файла, номером строки/ключом и версией; словарные данные не уходят модели.

### P10.T2 Принять только однозначную непосредственно записанную форму

- Файлы: ~ src/linguary/application/validators/pronunciation.py, ~ tests/integration/processing/test_pronunciation_selection.py
- Интерфейс: `verify_candidates` выбирает отдельную строгую ветку по `evidence.kind`; для `lexicon_form` требует точные form/variant, полный якорь, все семантические поля отсутствующими, `is_proper_name=false` и ровно одну различимую IPA во всём ответе. Для `dictionary_sense` сохраняется прежнее сопоставление контекста и правила форм.
- Тесты: test_lexicon_unique_exact_form_accepted; test_lexicon_multiple_ipa_and_proper_name_skipped; test_lexicon_missing_inflected_form_not_invented; test_lexicon_fake_semantic_fields_rejected; test_dictionary_sense_matching_unchanged; test_ipa_dict_failure_preserves_previous
- Готово, когда: омограф/альтернативное ударение пропускается, слова без прямой записи и имена не угадываются; запись без смысла/POS не маскируется под семантически проверенную. Сохраняются P6, изоляция ошибок, различие missing/unavailable и защита прежних результатов. Уникальность не закрывает Q-2; `uv run pytest` проходит.
- Коммит: `feat(pronunciation): accept only unambiguous lexicon forms`

### P10.T3 Связать источник с новой ревизией условий и стандартным runtime

- Файлы: ~ src/linguary/application/book_conditions.py, ~ src/linguary/composition/container.py, ~ src/linguary/composition/book_runtime.py, ~ tests/integration/processing/test_book_conditions.py
- Интерфейс: новые условия закрепляют `pronunciation_versions={selection_policy: 2, form_rules: 1, source: SRC-PH-02.source_version}`; контейнер по умолчанию передаёт соответствующий локальный адаптер. Явно внедрённый источник для тестов сохраняется. Прежний `source="disabled"`, неизвестная версия или отсутствующий снимок не получают текущую базу молча.
- Тесты: test_new_conditions_pin_ipa_dict_snapshot; test_old_disabled_conditions_never_enable_ipa_dict; test_unknown_pronunciation_source_skips_without_losing_translation; test_ipa_dict_runtime_process_regenerate_recovery; test_ipa_dict_content_never_reaches_model
- Готово, когда: реальные новые условия включают источник без ручной настройки/API-ключа, сохранённые старые условия/хеши не меняются; неподдерживаемый канал пропускает только новые IPA и сохраняет прежние при техническом отказе. LLM-PH-01, `pronunciation_candidates_version=1`, `prompt_templates`, схемы базовых версий 1–4 и инструкции исправлений не меняются; полный набор проходит.
- Коммит: `feat(pronunciation): enable pinned local source for new conditions`

## Фаза P11. Уведомления в приложении и EPUB и проверка реального набора (зависит от: P10)

**ID:** P11  
**Цель:** передать проверяемую атрибуцию и лицензионные материалы вместе с подсказками, подтвердить технический путь на малой книге.  
**Зависимости:** P10; старый DTO-PH-01 и навигация P7 сохраняются.

### P11.T1 Дать доступ к локальным уведомлениям в приложении

- Файлы: ~ src/linguary/web/routes/result.py, ~ src/linguary/composition/container.py, ~ src/linguary/web/templates/partials/result_block.html, ~ tests/integration/web/test_result_and_export.py
- Интерфейс: LIC-PH-01 — `GET /pronunciation-sources`, именованный маршрут `pronunciation_sources`, без входного имени файла/URL; ответ `200 text/plain; charset=utf-8` содержит NOTICE/LICENSES и версии всех поставленных снимков. При повреждённом комплекте — `503` без локальных путей. Памятка IPA даёт ссылку через `url_for`; атрибуция в записи содержит источник, UK/US, снимок и лицензию.
- Тесты: test_pronunciation_notices_route_is_offline_and_allowlisted; test_pronunciation_notice_link_and_escaped_attribution; test_legacy_pronunciation_attribution_still_visible
- Готово, когда: полные тексты доступны из приложения без сети, оригинал и отображение повторов не меняются, произвольные файлы/разметка не выдаются; комплект можно проверить по Q-1, но его наличие не считается автоматическим правовым согласованием.
- Коммит: `feat(web): expose pronunciation source notices`

### P11.T2 Включить источник и лицензионные материалы в автономный EPUB

- Файлы: ~ src/linguary/adapters/epub/writer.py, + src/linguary/adapters/epub/pronunciation_notices.py, ~ tests/integration/epub/test_epub_export_content.py, ~ tests/integration/epub/test_epub_export_structure.py
- Интерфейс: локальный сборщик получает только использованные принятые записи снимка и соответствующий NOTICE/LICENSES; EPUB содержит атрибуцию у подсказок, страницу источников/лицензий с переходом и машиночитаемое приложение использованных form/IPA/variant/source_version/license_id/entry_id. Закреплённые ссылки на полные исходные данные сохраняются текстом; никакого запроса источника при экспорте/чтении.
- Тесты: test_ipa_dict_epub_contains_used_source_licenses_and_provenance; test_ipa_dict_epub_notice_links_are_internal_and_unique; test_ipa_dict_epub_original_and_backlinks_unchanged; test_legacy_semantic_epub_needs_no_new_license_fields
- Готово, когда: уведомления экранируются, идентификаторы не конфликтуют с книгой, источник и лицензия каждого использованного вида доступны без сети, старые результаты экспортируются совместимо. Схема приложения подготовлена для проверки Q-1: необходимость включать полный UK-исходник в EPUB или иной способ предоставления соответствующих исходных данных остаётся открытой; соответствие GPL и матрица Q-3 этим тестом не объявляются.
- Коммит: `feat(epub): bundle pronunciation attribution and licenses`

### P11.T3 Прогнать малую синтетическую книгу через реальный снимок

- Файлы: + tests/integration/processing/test_ipa_dict_small_book.py, + .ai/develop-context/BRIEF-pronunciation-hints-2026-10-25/reports/IPA-DICT-small-book-2026-10-31.md
- Интерфейс: два коротких раздела с `colonel`, повтором `queue`, `quay`, `read`, `the` и собственным `Xyzzy`; кандидаты идут настоящим JSON-парсером через управляемый клиент, источник — реальный SRC-PH-02, все сетевые обращения запрещены. Один запуск для UK, один для US.
- Тесты: test_ipa_dict_small_book_uk; test_ipa_dict_small_book_us; test_ipa_dict_small_book_reprocessing_and_source_failure
- Готово, когда: `colonel`/`queue` используют точные IPA снимка, `quay` даёт UK запись и пропуск US с несколькими IPA, `read` — UK missing/US ambiguous, `the` неоднозначен, `Xyzzy` пропущен. SQLite/архив, общий JSON и EPUB согласованы по вхождениям, повторным ссылкам, источнику/лицензиям; отказ ресурса не стирает принятую помощь и перевод. Отчёт содержит команду, SHA/хеши, ожидаемые/фактические записи и пределы. Запуск: `uv run pytest tests/integration/processing/test_ipa_dict_small_book.py -q`, затем `uv run pytest`; это техническая проверка, без языкового балла и без закрытия Q-1/Q-2/Q-3.
- Коммит: `test(pronunciation): exercise real ipa-dict on a small book`

## 6. Глоссарий путей

```text
src/linguary/domain/book_conditions.py                       (~)
src/linguary/domain/pronunciation.py                         (+)
src/linguary/domain/processing.py                            (~)
src/linguary/application/book_conditions.py                  (~)
src/linguary/application/ports/persistence.py                (~)
src/linguary/application/ports/pronunciation.py               (+)
src/linguary/application/use_cases/reset_processing.py        (~)
src/linguary/application/context_builder.py                   (~)
src/linguary/application/llm_dto.py                            (~)
src/linguary/application/ports/llm.py                          (~)
src/linguary/application/use_cases/process_chapter.py          (~)
src/linguary/application/use_cases/regenerate_block.py         (~)
src/linguary/adapters/llm/schemas.py                           (~)
src/linguary/adapters/llm/payloads.py                          (~)
src/linguary/adapters/llm/openai_compatible.py                 (~)
src/linguary/adapters/llm/fake.py                              (~)
src/linguary/application/validators/pronunciation.py          (+)
src/linguary/application/block_processing.py                  (~)
src/linguary/application/pronunciation_projection.py          (+)
src/linguary/composition/container.py                         (~)
src/linguary/composition/book_runtime.py                       (~)
src/linguary/adapters/pronunciation/source.py                 (+)
src/linguary/adapters/db/models.py                             (~)
src/linguary/adapters/db/mapping.py                            (~)
src/linguary/adapters/db/repositories.py                       (~)
alembic/versions/0010_pronunciation_assistance.py              (+)
src/linguary/web/routes/processing.py                           (~)
src/linguary/web/routes/result.py                               (~)
src/linguary/web/schemas.py                                     (~)
src/linguary/adapters/epub/writer.py                            (~)
tests/unit/llm/test_llm_dto_contract.py                         (~)
tests/unit/llm/test_llm_response_schemas.py                     (~)
tests/unit/llm/test_llm_payload_mapping.py                      (~)
tests/unit/llm/test_openai_compatible_client.py                 (~)
tests/unit/llm/test_fake_llm_client.py                          (~)
tests/unit/application/test_translation_context.py            (~)
tests/unit/application/test_context_builder.py                (~)
tests/unit/application/test_pronunciation_projection.py        (+)
tests/integration/processing/test_book_conditions.py            (~)
tests/integration/processing/test_content_plan_v3_pipeline.py    (~)
tests/integration/processing/test_pronunciation_source.py        (+)
tests/integration/processing/test_pronunciation_selection.py     (+)
tests/integration/processing/test_process_chapter.py             (~)
tests/integration/processing/test_regenerate_block.py             (~)
tests/integration/processing/test_pronunciation_quality_gate.py   (+)
tests/integration/web/test_result_and_export.py                  (~)
tests/integration/db/test_block_result_queries.py                (~)
tests/integration/db/test_alembic_migrations.py                  (~)
tests/integration/epub/test_epub_export_content.py                (~)
tests/integration/epub/test_epub_export_structure.py              (~)
tests/fixtures/pronunciation/acceptance_cases.json                (+)
```

## 7. Открытые вопросы

### Q-1. Лицензированный источник и права

- **Gates:** [P2.T1], [P3.T2], [P3.T3], [P8.T2], [P11.T1], [P11.T2] — условия выпуска, не запрет локальной реализации.
- **Контекст:** выбран SRC-PH-02 ipa-dict; en_US имеет MIT-происхождение, en_UK — GPL-3.0 через ipacards. Точный план соблюдения GPL-3.0 при поставке приложения и UK-материалов в EPUB ещё не согласован: исходные данные, уведомления, применимость к приложениям использованных записей и способ предоставления соответствующих исходников.
- **Текущее решение по умолчанию:** реализовать локальный набор и подготовить NOTICE/LICENSES, атрибуцию и приложение происхождения. Владелец продукта проверяет фактический комплект и план распространения; до этого Q-1 открыт, выпуск и соответствие GPL не объявляются. Переговоры с коммерческим словарём не нужны.
- **При смене решения:** уточнить состав поставки/EPUB и доступность исходников, сохраняя старые снимки и сведения об их правах; не заменять компоненты общей лицензией MIT.

### Q-2. Контрольный набор и порог точности

- **Gates:** [P3.T3], [P11.T3].
- **Контекст:** нужны согласованный числовой порог и независимая вычитка самих IPA и уместности отбора. Британская часть происходит из автоматически сгенерированных eSpeak транскрипций ipacards; автор предупреждает об ошибках и оставшихся омографах. Одна различимая IPA в файле не гарантирует правильность, полноту чтений или соответствие контексту.
- **Текущее решение по умолчанию:** Q-2 открыт; малый сквозной прогон проверяет только реализацию. Владелец продукта и независимый языковой редактор согласуют положительные/отрицательные примеры и порог; до вычитки выпуск и контекстная точность не подтверждаются.
- **При смене решения:** обновить набор и критерий выпуска; запрещено автоматически закрывать качество счётчиком уникальных IPA, разрешёнными символами или успешными автотестами.

### Q-3. Целевые программы чтения EPUB

- **Gates:** [P3.T2], [P11.T2]
- **Контекст:** разные программы могут по-разному отображать IPA, ударение и навигационные ссылки; обязательная матрица устройств не согласована.
- **Текущее решение по умолчанию:** сохранять IPA текстом Unicode и не объявлять совместимость до проверки согласованной матрицы.
- **При смене решения:** обновить тесты экспорта, памятку и допустимое оформление сносок.
