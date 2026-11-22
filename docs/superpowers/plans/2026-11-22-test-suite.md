# План реализации: надёжный прогон тестов Linguary

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Документированная команда `uv run pytest -v` выполняет набор, проверка прокси не зависит от уже заданного окружения, доля покрытых строк пакета не опускается ниже 90%.

**Architecture:** Корень проекта добавляется в путь импорта тестового запуска. Проверка прокси сама очищает переменные прокси. Порог покрытия — отдельная команда. Краевые ветки маркера условий, подготовки глоссария-списка и пустого ключа прогона закрепляются тестами; код меняется только там, где закрепление не сходится с текущим поведением.

**Tech Stack:** Python 3.13, pytest 9.1.1, pytest-cov, uv.

**Spec:** `docs/superpowers/specs/2026-11-22-test-suite/BRIEF-test-suite-2026-11-22.md`. Архитектура — `ADR-test-suite-2026-11-22.md`. Задачи и файлы — `SDD-backend-test-suite-2026-11-22.md`. Этот план повторяет задачи SDD шагами с командами.

## Global Constraints

- Обычный прогон остаётся `uv run pytest -v` из корня репозитория. Порог покрытия в его параметры не входит.
- Медленные тесты остаются в полном прогоне. Живая оценка имён без флага `LINGUARY_RUN_NAME_STRATEGY_EVAL=1` пропускается.
- Значения `ALL_PROXY`, `HTTP_PROXY`, `HTTPS_PROXY` и их строчных двойников не печатать и не записывать в документы.
- Ошибка недоступности условий сохраняет прежний тип и причину `conditions_unavailable`.
- Порог — 90% строк пакета `linguary`, ниже последнего замера. Покрытие ветвей в порог не входит.

## Review Focus

- Консольный `uv run pytest -v` на среде, где корень не лежит в `sys.path`: пакет `tests` импортируется.
- Проверка прокси при постороннем секрете, которого нет в списке ожиданий: восемь случаев проходят, этого секрета нет в выводе.
- Пустой `run_id` не читается как сущность. Проверка изоляции отмены сама считает предупреждение о пустом ключе ошибкой и при этом проходит.
- Неподдерживаемый маркер контракта отвергается сам по себе, той же ошибкой недоступности условий.
- Возврат подготовки глоссария по списку записей не содержит служебных ключей и пустой заметки. Запрос правки главы по-прежнему передаёт словарь.

---

### Task 1: Корень проекта в пути импорта

Соответствует SDD `[P1.T1]`.

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: секцию `[tool.pytest.ini_options]`
- Produces: корень репозитория в пути импорта pytest

- [ ] **Step 1: Добавить путь**

В `[tool.pytest.ini_options]` добавить `pythonpath = ["."]`.

- [ ] **Step 2: Проверить сбор**

Run: `uv run pytest --collect-only -q`
Expected: код 0, нет `ModuleNotFoundError: No module named 'tests'`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "test: add project root to pytest pythonpath"
```

### Task 2: Изолировать проверку прокси

Соответствует SDD `[P1.T2]`.

**Files:**
- Modify: `tests/unit/llm/test_openai_compatible_client.py`

**Interfaces:**
- Consumes: конструктор клиента, который запоминает секреты при создании
- Produces: проверка, которая сама очищает переменные прокси

- [ ] **Step 1: Убедиться, что в занятом окружении автоматический режим падает**

Run: `uv run pytest tests/unit/llm/test_openai_compatible_client.py::test_proxy_diagnostics_redact_encoded_decoded_credentials -q --tb=line`
Expected: если в окружении уже есть переменные прокси, четыре случая `auto` падают, четыре `custom` проходят. Если окружение чистое, все 8 проходят, и шаг 2 всё равно нужен.

- [ ] **Step 2: Очистить переменные в начале проверки**

Удалить каждый ключ окружения, в имени которого есть `proxy` без учёта регистра. Затем для режима `auto` задать только `HTTPS_PROXY` фиктивным адресом. Клиент не менять.

- [ ] **Step 3: Повторить проверку с посторонним секретом**

Перед запуском задать посторонний секрет прокси фиктивным значением, которого нет в списке ожиданий проверки. Адрес машины не использовать и не печатать.

Run: та же команда, что в шаге 1.
Expected: `8 passed`, постороннего секрета нет в выводе.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/llm/test_openai_compatible_client.py
git commit -m "test: isolate proxy redaction from the process environment"
```

### Task 3: Удалить повтор проверки ремонта

Соответствует SDD `[P2.T1]`.

**Files:**
- Delete: `tests/unit/test_openai_compatible_llm_client.py`
- Test: `tests/unit/llm/test_openai_compatible_client.py::test_repair_instructions_sent_as_separate_user_messages`

**Interfaces:**
- Consumes: уже существующую проверку четырёх сообщений
- Produces: одно определение этой проверки

- [ ] **Step 1: Удалить файл в корне модульных тестов**

- [ ] **Step 2: Проверить оставшуюся проверку**

Run: `uv run pytest tests/unit/llm/test_openai_compatible_client.py::test_repair_instructions_sent_as_separate_user_messages -q`
Expected: `1 passed`.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_openai_compatible_llm_client.py
git commit -m "test: drop duplicated repair-instructions case"
```

### Task 4: Порог покрытия отдельной командой

Соответствует SDD `[P2.T2]`.

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `README.md`

**Interfaces:**
- Consumes: группу `dev` и раздел Testing
- Produces: `uv run pytest --cov=linguary --cov-fail-under=90`

- [ ] **Step 1: Добавить pytest-cov в группу dev и обновить lock**

- [ ] **Step 2: Дописать команду в README, не заменяя обычный прогон**

- [ ] **Step 3: Прогнать порог**

Run: `uv run pytest --cov=linguary --cov-fail-under=90 -q`
Expected: код 0, покрытие не ниже 90%.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock README.md
git commit -m "test: keep line coverage from falling under 90 percent"
```

### Task 5: Закрепить отказ неподдерживаемого маркера

Соответствует SDD `[P3.T1]`.

**Files:**
- Modify: `tests/integration/processing/test_translation_legacy_compatibility.py`
- Modify: `src/linguary/application/translation_compatibility.py` только если проверка красная

**Interfaces:**
- Consumes: `RecoveryError` с причиной `conditions_unavailable`
- Produces: проверку маркера контракта, не равного 5, для профиля третьей версии плана

- [ ] **Step 1: Добавить проверку**

Хранимый снимок равен объекту под проверкой. `prompt_templates` — словарь. Профиль — `content-plan-v3`. Единственное отклонение — целочисленный `analysis_contract_version`, не равный 5. Ожидание — `RecoveryError` с причиной `conditions_unavailable`. Не копировать соседний приём `replace(...)` без записи снимка обратно: иначе отказ случится раньше проверки маркера, и проверка будет зелёной зря.

- [ ] **Step 2: Прогнать файл**

Run: `uv run pytest tests/integration/processing/test_translation_legacy_compatibility.py -q --tb=line`
Expected: все проверки файла проходят. Если новая красная из-за другого исключения, поправить совместимость до прежней ошибки и повторить.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/processing/test_translation_legacy_compatibility.py src/linguary/application/translation_compatibility.py
git commit -m "test: reject an unsupported contract marker"
```

### Task 6: Закрепить очистку глоссария-списка

Соответствует SDD `[P3.T2]`.

**Files:**
- Modify: `tests/unit/application/test_correct_chapter_v2.py`
- Modify: `src/linguary/application/use_cases/correct_chapter.py` только если проверка красная

**Interfaces:**
- Consumes: `_llm_run_glossary`
- Produces: проверку возврата по списку записей без служебных ключей и пустой заметки

- [ ] **Step 1: Добавить модульную проверку**

Вызвать `_llm_run_glossary` со списком записей. В записях есть ключи `id`, `book_id`, `created_at`, `concept_id` и пустая заметка (`note` is None или пробелы). В возврате этих полей нет. Запрос правки главы не трогать: он по-прежнему кладёт в контекст словарь.

- [ ] **Step 2: Прогнать модульный файл**

Run: `uv run pytest tests/unit/application/test_correct_chapter_v2.py -q --tb=line`
Expected: файл зелёный. `correct_chapter.py` менять только при красной новой проверке.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/application/test_correct_chapter_v2.py src/linguary/application/use_cases/correct_chapter.py
git commit -m "test: clean a glossary list in the correction helper"
```

### Task 7: Не читать прогон с пустым ключом

Соответствует SDD `[P3.T3]` и `ALG-01`.

**Files:**
- Modify: `src/linguary/adapters/db/work_state.py`
- Test: `tests/integration/processing/test_processing_run_isolation.py::test_result_of_a_cancelled_run_is_discarded`

**Interfaces:**
- Consumes: `begin_recovery_attempt`
- Produces: отказ `conditions_unavailable` до чтения прогона, если `run_id is None`, и постоянный фильтр предупреждения в самой проверке

- [ ] **Step 1: Увидеть предупреждение как ошибку**

Run: `uv run pytest tests/integration/processing/test_processing_run_isolation.py::test_result_of_a_cancelled_run_is_discarded -q -W error::sqlalchemy.exc.SAWarning`
Expected: падение на чтении прогона, когда `run_id is None`.

- [ ] **Step 2: Отказать до чтения**

После уже существующих проверок корня и области: если `run_id is None`, бросить `RecoveryError('conditions_unavailable', root_id)` и не вызывать `Session.get`.

- [ ] **Step 3: Закрепить предупреждение внутри проверки**

В `test_result_of_a_cancelled_run_is_discarded` постоянно считать `sqlalchemy.exc.SAWarning` ошибкой. Внешний `-W` этого не заменяет.

Run: `uv run pytest tests/integration/processing/test_processing_run_isolation.py::test_result_of_a_cancelled_run_is_discarded -q --tb=line`
Expected: `1 passed`, предупреждения о пустом ключе нет.

- [ ] **Step 4: Полный прогон**

Run: `uv run pytest -v`
Expected: падений нет, один пропуск живой оценки имён.

- [ ] **Step 5: Commit**

```bash
git add src/linguary/adapters/db/work_state.py tests/integration/processing/test_processing_run_isolation.py
git commit -m "fix: reject recovery when the processing run id is missing"
```
