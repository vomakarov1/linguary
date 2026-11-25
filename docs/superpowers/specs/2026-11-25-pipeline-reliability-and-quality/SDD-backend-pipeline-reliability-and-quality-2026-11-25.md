# SDD: Техническая спецификация исправлений конвейера иллюстраций, бюджета рассуждения и редакционного качества

> **Статус:** Accepted (проверка спецификаций пройдена, все замечания учтены)

- Дата: 2026-11-25
- BRIEF: [BRIEF-pipeline-reliability-and-quality-2026-11-25.md](BRIEF-pipeline-reliability-and-quality-2026-11-25.md)
- ADR: [ADR-pipeline-reliability-and-quality-2026-11-25.md](ADR-pipeline-reliability-and-quality-2026-11-25.md)

Документ разбивает изменения на 6 независимых пакетов работ (WP-1 … WP-6) с точными контрактами модулей, инвариантами, граничными случаями и обязательными регрессионными тестами.

---

## WP-1. Совместимость рантайма иллюстраций с моделями провайдера

### Затрагиваемые файлы
- `src/linguary/domain/illustrations.py`
- `src/linguary/composition/illustration_runtime.py`
- `src/linguary/application/illustration_context.py` (единый хелпер классификации ошибок провайдера)
- `src/linguary/application/illustration_reference.py`
- `src/linguary/application/illustration_selection.py`
- `src/linguary/application/illustration_chapter.py`
- `tests/unit/composition/test_illustration_runtime.py` (обновить устаревшие ожидания `0.2` / `4096` + добавить новые сценарии)

### Контракт

1. **`IllustrationPolicy` (`domain/illustrations.py`):**
   - Изменить значения по умолчанию и типы:
     - `max_completion_tokens: int = 16384` (было `4096`).
     - `temperature: float | None = None` (было `float = 0.2`).
   - В `__post_init__()`:
     - Если `self.temperature is not None`: проверить `type(self.temperature) in (int, float)` и `0.0 <= float(self.temperature) <= 2.0`, иначе `IllustrationValidationError(INVALID_REQUEST, ...)`.
   - В `to_payload()`: сохранить ключ `"temperature": self.temperature` (значение `None` или `float`).
   - В `from_payload(payload)`:
     - `"max_completion_tokens"` по умолчанию `16384`.
     - `"temperature"` по умолчанию `None` (`payload.get("temperature", None)`), при этом сохранённое `0.2` (или `0.0`) сохраняется без изменений.

2. **`_clean_llm_settings()` (`composition/illustration_runtime.py`):**
   - `temperature`: `policy.temperature` (по умолчанию `None`; для новых ревизий ключ `"temperature"` не попадает в JSON-запрос OpenRouter; явная `policy.temperature` в сохранённых/тестовых политиках уважается).
   - `max_output_tokens`:
     ```python
     effective_max_tokens = (
         base_llm.max_output_tokens
         if base_llm.max_output_tokens is not None
         else policy.max_completion_tokens
     )
     ```
   - `reasoning_effort` и `reasoning_max_tokens`:
     - `same_primary_model = (model.strip() == base_llm.model.strip())`
     - Если `same_primary_model`:
       - `reasoning_effort = base_llm.reasoning_effort`
       - `reasoning_max_tokens = base_llm.reasoning_max_tokens` **только если** одновременно:
         - `reasoning_effort is None`,
         - `base_llm.reasoning_max_tokens is not None`,
         - `_is_openrouter_url(base_llm.base_url)` (на не-OpenRouter URL `openai_compatible._body()` бросает `LLMConfigurationError`, что ломало бы даже `sizer_for()`),
         - `effective_max_tokens is None or base_llm.reasoning_max_tokens < effective_max_tokens`.
         Иначе `reasoning_max_tokens = None`.
     - Иначе (`model != base_llm.model`):
       - `reasoning_effort = None`
       - `reasoning_max_tokens = None`
   - Остальные поля (`provider`, `base_url`, `proxy_mode`, `proxy_url`, `api_key`, `model`, `timeout_seconds`, `max_attempts=1`, `requests_per_minute`, `debug_llm_payloads`, цены) — как прежде.

3. **Ключ пула и эвикция в `IllustrationRuntime._pool` (`composition/illustration_runtime.py`):**
   - Расширить `_transport_fingerprint(base_llm)` (или выделить `_runtime_fingerprint(base_llm)`), включив помимо транспортных полей глобальные параметры генерации `base_llm`:
     `(base_llm.provider, base_llm.base_url, base_llm.proxy_mode, base_llm.proxy_url, base_llm.api_key, base_llm.timeout_seconds, base_llm.model, base_llm.max_output_tokens, base_llm.reasoning_effort, base_llm.reasoning_max_tokens)`.
     Таким образом при смене глобальных параметров через `/system/config` старые `OpenAICompatibleLLMClient` корректно переводятся в `self._retired` и не текут до `aclose()`.
   - Выделить `_policy_generation_fingerprint(clean: LLMSettings)`:
     `(clean.temperature, clean.max_output_tokens, clean.reasoning_effort, clean.reasoning_max_tokens)`.
   - `pool_key = (rt_fp, model, policy_gen_fp)` — разные ревизии политик разных книг не вытесняют друг друга, но и не переиспользуют клиент с чужой температурой/лимитом токенов.

4. **Классификация постоянных ошибок провайдера (`application/illustration_context.py`):**
   - Вынести общие функции в `application/illustration_context.py` (и использовать их в `illustration_reference.py`, `illustration_selection.py`, `illustration_chapter.py` вместо 3 копий):
     - `is_context_limit_error(error: Exception) -> bool`:
       - `LLMRequestTooLargeError` или `LLMContextCapacityError` → `True`.
       - `LLMProviderError`: если `status == 413` → `True`; если `(status in (400, 422) or status is None)` и `_CONTEXT_LIMIT_RE.search(str(error))` → `True`.
     - `is_truncated_error(error: Exception) -> bool`: без изменений.
     - `is_transient_error(error: Exception) -> bool`: без изменений.
     - `is_incompatible_route_error(error: Exception) -> bool`:
       - `True` если `isinstance(error, LLMConfigurationError)`.
       - `True` если `isinstance(error, LLMProviderError)` и не `is_context_limit_error(error)` и не `is_transient_error(error)`, и выполняется одно из:
         - `getattr(error, "status_code", None) == 404`, либо
         - `_INCOMPATIBLE_ROUTE_RE.search(str(error))`, где:
           ```python
           _INCOMPATIBLE_ROUTE_RE = re.compile(
               r"(?::\s*404\b|\bhttp\s+404\b|\bstatus\s+404\b"
               r"|no endpoints found|unsupported parameter|unknown parameter"
               r"|unrecognized request argument|does not support"
               r"|model not found|not a valid model)",
               re.IGNORECASE,
           )
           ```
   - Во всех трёх обработчиках (`reference`, `selection`, `chapter`) в ветке `fallback_reason`:
     ```python
     fallback_reason = (
         IllustrationReasonCode.ROUTE_UNAVAILABLE.value
         if is_incompatible_route_error(call_error)
         else IllustrationReasonCode.PROVIDER_ERROR.value
     )
     ```

### Обязательные тесты WP-1
- `IllustrationPolicy()` имеет `temperature is None`, `max_completion_tokens == 16384`; `from_payload({"temperature": 0.2, "max_completion_tokens": 4096})` сохраняет старые значения; невалидная `temperature` (`-0.1`, `2.5`, `True`, `"0.2"`) отклоняется.
- Обновить `tests/unit/composition/test_illustration_runtime.py` (`temperature is None`, `max_output_tokens == 16384`) и добавить проверки:
  - при `base_llm.temperature=None` в сериализованном JSON отсутствует `"temperature"`;
  - при `model == base_llm.model` наследуются `reasoning_effort` и `max_output_tokens`;
  - при `model != base_llm.model` `reasoning_effort is None` и `reasoning_max_tokens is None`;
  - при не-OpenRouter URL `reasoning_max_tokens` не наследуется и `sizer_for().measure(...)` не падает;
  - при `base_llm.reasoning_max_tokens >= effective_max_tokens` `reasoning_max_tokens` сбрасывается в `None`;
  - смена глобального `max_output_tokens` отправляет старый клиент в `_retired`, а смена `policy.temperature` создаёт отдельный ключ пула;
  - HTTP 404 (`отверг запрос: 404 - No endpoints found...`) классифицируется как `ROUTE_UNAVAILABLE` без повторов, а `404 tokens` не ложносрабатывает.

---

## WP-2. Жизненный цикл виртуального агрегата `CHAPTER` и финализация поколений

### Затрагиваемые файлы
- `src/linguary/domain/illustrations.py` (размещение чистых функций `executable_scopes` и `derive_chapter_state` — без риска циклических импортов!)
- `src/linguary/application/illustrations.py`
- `src/linguary/application/illustration_coordinator.py`
- `src/linguary/application/illustration_chapter.py`
- `src/linguary/application/illustration_view.py`
- `src/linguary/adapters/db/illustration_repository.py`

### Инварианты
1. Область `stage == IllustrationScopeStage.CHAPTER` — виртуальный агрегат (`task_id is None`), она никогда не исполняется воркером и не переводится в `FAILED` при падении дочерних областей.
2. Исполнимой областью считается любая область с `scope.stage is not IllustrationScopeStage.CHAPTER` и `scope.state is not IllustrationScopeState.SPLIT`.
3. Чистые доменные функции в `src/linguary/domain/illustrations.py` (экспортировать в `__all__`):
   ```python
   def executable_scopes(scopes: Sequence[IllustrationScope]) -> list[IllustrationScope]:
       return [
           s
           for s in scopes
           if s.stage is not IllustrationScopeStage.CHAPTER
           and s.state is not IllustrationScopeState.SPLIT
       ]

   def derive_chapter_state(
       chapter_state: IllustrationScopeState,
       chapter_reason_code: str | None,
       child_states: Sequence[tuple[IllustrationScopeState, str | None]],
   ) -> tuple[str, str | None]:
   ```
   Таблица переходов `derive_chapter_state`:
   - `chapter_state is ACCEPTED` → `("ready", None)`
   - `chapter_state is SKIPPED` → `("skipped", chapter_reason_code)`
   - `chapter_state is FAILED` → `("failed", chapter_reason_code)`
   - `chapter_state in {RUNNING, SPLIT}` → `("running", None)`
   - `chapter_state is PENDING`:
     - если любой `(st, _)` из `child_states` имеет `st is RUNNING` → `("running", None)`
     - иначе если любой `st is PENDING` → `("queued", None)`
     - иначе если есть первый `(st, rc)` с `st is FAILED` → `("failed", rc)`
     - иначе → `("queued", None)`
   (Сигнатура принимает `Sequence[tuple[IllustrationScopeState, str | None]]`, поэтому её удобно вызывать как из `build_illustration_view` по `IllustrationScope`, так и из `get_chapter_header_page` по компактным строкам SQL без загрузки `payload_json`!)

### Точные изменения по местам

1. **`finalize_generation_state_in_tx()` (`application/illustrations.py`):**
   - Возвращает `bool` (`True`, если поколение перешло в `FAILED` или уже находилось в `FAILED`, иначе `False`).
   - Использовать `exec_scopes = executable_scopes(scopes)`.
   - `has_active = any(s.state in {PENDING, RUNNING} for s in exec_scopes)`.
   - Если `has_active` → `return False`.
   - Искать `failed_scope` сначала в `exec_scopes`, затем среди `scopes` с `state == FAILED`.
   - Если `failed_scope is not None` → обновить поколение до `IllustrationGenerationState.FAILED` с `reason_code = failed_scope.reason_code or fallback_reason` и вернуть `True`.
   - Иначе вернуть `gen.state is IllustrationGenerationState.FAILED`.

2. **`IllustrationChapter.plan_chapter_scopes()` (`application/illustration_chapter.py`):**
   - Сразу после `self._maybe_mark_generation_ready(tx, generation)` (строка 676) вызывать:
     ```python
     finalize_generation_state_in_tx(tx, generation.book_id, generation.id)
     ```
     Это закрывает случай, когда поколение в `FAILED` (глава 1 упала) дозапрашивает главу 2 через `prepare()`, глава 2 мгновенно переиспользуется из кэша (`dispatched=False`), и без `finalize_generation_state_in_tx` поколение застряло бы в `QUEUED`.

3. **`IllustrationCoordinator.recover_interrupted()` (`application/illustration_coordinator.py`):**
   - Вместо ручной проверки `active_scopes` вызвать:
     ```python
     if finalize_generation_state_in_tx(
         tx,
         gen.book_id,
         gen.id,
         fallback_reason=IllustrationReasonCode.INTERRUPTED.value,
     ):
         continue
     ```
     Если поколение перешло в `FAILED`, `reconcile_in_transaction` для него не запускается.

4. **`build_illustration_view()` (`application/illustration_view.py`):**
   - Топ-уровневый расчёт (`lines 918-931`):
     - `exec_scopes = executable_scopes(scopes)`
     - `failed_scope = next((s for s in exec_scopes if s.state is FAILED), None) or next((s for s in scopes if s.state is FAILED), None)`
     - `has_running = any(s.state is RUNNING for s in exec_scopes)`
     - `has_pending = any(s.state is PENDING for s in exec_scopes)`
     - В `allowed_actions`: `has_retryable = any(s.is_retryable_failure for s in exec_scopes)`.
   - Расчёт `selected_chapter` (`lines 802-825`):
     - Для `matched_scope` собрать `child_states = [(s.state, s.reason_code) for s in exec_scopes if s.section_id == section_id]`, вычислить `(sec_state, sec_reason) = derive_chapter_state(full_scope.state, full_scope.reason_code, child_states)`.

5. **`get_chapter_header_page()` (`adapters/db/illustration_repository.py`):**
   - После формирования итогового среза страницы `page = merged[offset : offset + limit]` (или `merged_live` / `ordered_scope_headers`) собрать `pending_sids = [h.section_id for h in page if h.scope_id is not None and h.state == "queued"]`.
   - Только если `pending_sids` непуст, выполнить один компактный запрос **без `payload_json`** (`select(section_id, state, reason_code).where(generation_id == gen.id, stage != 'chapter', state != 'split', section_id.in_(pending_sids)).order_by(stage, scope_key)`), сгруппировать по `section_id`, и для каждого заголовка из `pending_sids` пересчитать `(state, reason_code)` через `derive_chapter_state(IllustrationScopeState.PENDING, header.reason_code, child_states)`.

6. **`requeue_failed_scopes()` (`adapters/db/illustration_repository.py`):**
   - Добавить явное условие `IllustrationScopeRow.stage != IllustrationScopeStage.CHAPTER.value` в `where(...)`.

### Обязательные тесты WP-2
- Падение одного `CANDIDATES`-scope при принятой второй главе: поколение в `FAILED`, `chapter_headers` показывает `failed` + `reason_code` для главы 1 и `ready` для главы 2, `selected_chapter` для главы 1 показывает `state="failed"` и `reason_code`, `has_unfinished_illustration_work()` == `False`.
- Параллельный `RUNNING`/`PENDING` scope в главе 1 удерживает поколение и главу в `running`/`queued` до завершения последнего активного scope, после чего оба переходят в `failed`.
- `IllustrationService.retry()` перезапускает только упавший дочерний scope, НЕ создаёт задачу для `CHAPTER`, и после успеха глава и поколение переходят в `ready`.
- `IllustrationService.prepare(..., section_id=sec2)` при упавшей главе 1 и кэш-переиспользовании главы 2 не оставляет поколение в `QUEUED` (оно возвращается в `FAILED`).
- `IllustrationCoordinator.recover_interrupted()` переводит поколение с `CHAPTER (pending)` + `CANDIDATES (failed)` в `FAILED`.
- Проверка отсутствия `payload_json` в SQL-запросах `get_chapter_header_page` остаётся зелёной.

---

## WP-3. Устойчивость этапа `CANDIDATES`

### Затрагиваемые файлы
- `src/linguary/adapters/llm/illustration_payloads.py`
- `src/linguary/application/illustration_context.py`
- `src/linguary/application/illustration_selection.py`
- `src/linguary/application/illustration_reference.py`

### Контракт

1. **Лимит схемы (`adapters/llm/illustration_payloads.py`):**
   - В `IllustrationCandidatesSchema` изменить:
     `covered_source_unit_ids: list[str] = Field(max_length=_MAX_COVERED_IDS)` (256 вместо 32).

2. **Ограничение числа элементов в упаковщике (`application/illustration_context.py`):**
   - Сигнатура:
     ```python
     def pack_items_by_wire_budget(
         items: Sequence[_T],
         *,
         build_request: Callable[[Sequence[_T]], LLMRequest[Any]],
         sizer: RequestWireSizer,
         target_bytes: int,
         max_items: int | None = None,
     ) -> tuple[tuple[_T, ...], ...]:
     ```
   - Если `max_items is not None` и `(type(max_items) is not int or max_items <= 0)` → `ValueError("max_items должен быть положительным целым")`.
   - Если `max_items is not None and len(current) >= max_items`: закрыть текущую страницу (`pages.append(tuple(current))`, `current = []`) перед проверкой следующего `item`.
   - Применение:
     - `IllustrationSelection.plan_scopes_in_tx()`: `max_items=24` (`_MAX_CANDIDATE_BLOCKS_PER_CHUNK = 24`).
     - `IllustrationReference.plan_scopes()`: `max_items=64` (`_MAX_EXTRACT_UNITS_PER_PAGE = 64`).

3. **Лаконичный `source_unit_id` в `CANDIDATES` (`application/illustration_selection.py`):**
   - В `_scene_candidate_units()`: для каждого блока сцены задавать `"source_unit_id": b.id` (в обеих ветках: с `entry.scenes` и без них).

4. **Устойчивая проверка покрытия и информативный repair (`application/illustration_selection.py`):**
   - В `_ValidationOutcome` добавить флаг `split_eligible: bool = False`.
   - В `_validate_candidates()`:
     - `expected_units = set(str(u) for u in scope.summary.get("expected_source_unit_ids", ()))`
     - `covered = set(dto.covered_source_unit_ids)`
     - Обратная совместимость со старыми/составными ID: если `covered != expected_units`, проверить `ordered_blocks = [str(b) for b in scope.summary.get("ordered_block_ids", ())]`; если `len(ordered_blocks) == len(expected_units)` и `covered == set(ordered_blocks)`, считать покрытие валидным.
     - Иначе вернуть `_ValidationOutcome(ok=False, repair_reason=..., split_eligible=True)`, где `repair_reason` содержит перечень `пропущены={missing[:4]}` и/или `лишние={extra[:4]}` и явную подсказку перечислить **все** `source_unit_id` из `blocks` (включая блоки без событий).

5. **Дробление составного `CANDIDATES`-scope при стойком отказе формата/покрытия (`application/illustration_selection.py`):**
   - В `IllustrationSelection.run_task()`:
     - В ветке `LLMResponseFormatError`, когда раунды ремонта или активационный бюджет исчерпаны: если `scope.stage is IllustrationScopeStage.CANDIDATES` и `is_composite`: вызвать `self._handle_post_send_split(..., fail_reason=IllustrationReasonCode.MODEL_OUTPUT_INVALID.value, detail=str(call_error))` и `break`.
     - В ветке `not validation.ok`, когда раунды ремонта или активационный бюджет исчерпаны: если `scope.stage is IllustrationScopeStage.CANDIDATES` и `is_composite` и `validation.split_eligible`: вызвать `self._handle_post_send_split(..., fail_reason=validation.reason_code, detail=validation.repair_reason)` и `break`.
   - В `_handle_post_send_split()` добавить необязательный параметр `detail: str | None = None`: если дробление невозможно (`depth >= max_split_depth` или `not children`), передавать `outcome` с безопасным `detail` в `fail_scope(...)`.

### Обязательные тесты WP-3
- `IllustrationCandidatesSchema` принимает 41 и 256 `covered_source_unit_ids` и отклоняет 257.
- `pack_items_by_wire_budget(..., max_items=24)` разбивает 45 блоков на 24 и 21, даже если все 45 влезают в `target_bytes`.
- `_scene_candidate_units` присваивает `source_unit_id == block.id`.
- `_validate_candidates` принимает 1-к-1 совпадение по `ordered_block_ids` для legacy-формата; при неполном покрытии `repair_reason` содержит `пропущены=` / `лишние=`.
- Составной `CANDIDATES`-scope из 4 блоков, дважды вернувший неполный `covered_source_unit_ids` (исчерпав `max_repair_turns=1`), делится на 2 дочерних scope по 2 блока и успешно завершается.

---

## WP-4. Редакционное качество иллюстраций, антиспойлер, `monochrome_ink` и промпты анализа/перевода

### Затрагиваемые файлы
- `src/linguary/application/illustration_context.py`
- `src/linguary/application/illustration_reference.py`
- `src/linguary/application/illustration_validation.py`
- `src/linguary/application/illustration_chapter.py`
- `src/linguary/application/context_builder.py`

### Контракт

1. **Антиспойлер и `monochrome_ink` в `DESIGN`:**
   - В `_build_bounded_design_input()` (`illustration_reference.py`) добавить параметр `style: str = ILLUSTRATION_STYLE_MONOCHROME_INK` и включить `"style": style` в `request_input` (передавать `revision.settings.style` в `_advance_reference_pipeline()`).
   - В `_build_design_scope_spec()` добавить `"style": revision.settings.style` в `summary` (и в `input_sha256`).
   - Обновить `_STAGE_INSTRUCTIONS[IllustrationScopeStage.DESIGN]` (`illustration_context.py`):
     - Строгий запрет внешних знаний: «Опирайся СТРОГО на переданные `facts` и `canonical_name`. ЗАПРЕЩЕНО использовать любые внешние знания о книге, серии, фильмах, экранизациях или событиях будущих глав (не подставляй канонический возраст, школьную форму, предметы, цвет глаз/волос или детали из других частей произведения, если их нет в `facts`).»
     - Непротиворечивость: «Не противоречь ни одному переданному факту из `facts` (возраст, волосы, одежда, телосложение) и не повторяй одинаковую черту дважды.»
     - Монохромный профиль: «Если `style == "monochrome_ink"`, описывай `palette_value` ТОЛЬКО тональными характеристиками чёрно-белой туши (контраст, светлые/средние/тёмные тона, плотность штриховки, чёткие силуэты) и НЕ используй названия цветов/оттенков (зелёный, красный, синий, бежевый, коричневый, жёлтый и т. п.) ни в `palette_value`, ни в придуманных чертах дизайна.»
   - В `IllustrationReference._validate_design()`:
     - Если `scope.summary.get("style", ILLUSTRATION_STYLE_MONOCHROME_INK) == ILLUSTRATION_STYLE_MONOCHROME_INK`: проверить черту `trait == "palette_value"` на наличие цветовых слов в `description_en` и `description_ru`:
       - EN: `\b(red|crimson|scarlet|green|emerald|blue|azure|cobalt|yellow|amber|gold|golden|orange|purple|violet|pink|magenta|teal|turquoise|beige|brown|chestnut|auburn|maroon|olive|khaki|indigo|cyan|lavender|burgundy|sepia|bronze|copper|cream|ochre)\b`
       - RU: `\b(красн\w*|зелён\w*|зелен\w*|син\w*|голуб\w*|жёлт\w*|желт\w*|оранжев\w*|фиолетов\w*|розов\w*|бежев\w*|коричнев\w*|рыж\w*|изумруд\w*|багров\w*|бирюзов\w*|золот\w*|пурпур\w*|бордов\w*|лилов\w*|бронзов\w*|медн\w*|кремов\w*|охр\w*)\b`
     - При совпадении вернуть `_ValidationResult(ok=False, repair_reason="для стиля monochrome_ink черта palette_value должна описывать только тональность, контраст и штриховку без названий цветов")`.

2. **Точное подавление конфликтующих `DESIGN`-черт (`illustration_reference.py`):**
   - Заменить наивный подстрочный поиск `any(kw in explicit_blob ...)` в `resolve_appearance_with_design_fallback()` на скомпилированные регулярные выражения с границами слов (`\bword\b` для полных EN-слов, `\bstem` для RU-основ) — это исключает ложные срабатывания вроде `"tie"` в `"parties"`, `"aged"` в `"managed"`, `"bun"` в `"bundle"`, `"mane"` в `"permanent"`.
   - Покрыть категории без неоднозначных общих прилагательных (`big`, `huge`, `black`, `white`):
     - `"hair"`: EN `hair|blond|blonde|redhead|curly|braid|braids|beard|bearded|mustache|moustache|bald|shaggy|bun`, RU `волос|кудр|кос[аыуе]|бород|усы|усов|усами|лыс|пучок|лохмат`.
     - `"eyes"`: EN `eye|eyes|gaze`, RU `глаз|взгляд`.
     - `"build"`: EN `tall|short|slender|stout|thin|broad|beefy|giant|plump|skinny|stocky|bony`, RU `рост|худощ|крепк|высок|крупн|грузн|великан|худой|худая|тощ|полн|костляв`.
     - `"age_appearance"`: EN `young|old|elderly|child|boy|girl|teen|teenager|baby|infant|toddler|middle-aged`, RU `молод|стар|ребён|ребен|девоч|мальч|младен|малыш|грудн|пожил`.
     - `"attire_default"`: EN `coat|dress|cloak|hat|robe|robes|uniform|suit|boots|trousers|jacket|shirt|tie|scarf`, RU `плать|плащ|пальт|шляп|костюм|куртк|галстук|шарф|рубаш|манти|сапог`.
     - `"distinctive_mark"`: EN `scar|freckle|freckles|tattoo|limp|glasses|spectacles|lightning`, RU `шрам|веснуш|очки|очков|очках|родинк|молни`.
     - `"palette_value"`: EN `pale|dark|bright|shadow`, RU `светл|тёмн|темн|бледн`.
     - `"architecture"`: EN `stone|wood|wooden|brick|arch|tower|hall`, RU `камен|деревян|башн|зал`.
     - `"atmosphere"`: EN `dim|mist|fog|warm|gloomy`, RU `туман|мрачн|светл|уютн`.

3. **Англоязычные имена и отсутствие кириллицы в `prompt_en` (`illustration_validation.py`, `illustration_chapter.py`):**
   - В `IllustrationAppearance` добавить необязательное поле `name_en: str | None = None`:
     - В `to_dict()`: включать `"name_en": self.name_en` только если `self.name_en is not None` (полная обратная совместимость с существующими тестами).
     - В `from_dict()`: читать `raw_name_en = raw.get("name_en")` и нормализовать `str(raw_name_en).strip() or None if raw_name_en is not None else None`.
   - Хелпер `has_cyrillic(text: str) -> bool` (`bool(re.search(r"[\u0400-\u04FF]", text))`).
   - Хелпер `_select_english_name(entity: Mapping[str, Any]) -> str | None` в `illustration_chapter.py`:
     - Проверяет `canonical_name`, затем отсортированные `aliases`; возвращает первое непустое имя без кириллицы; иначе `None`.
   - В **обоих** местах сборки `IllustrationAppearance` в `illustration_chapter.py` (ветка без фактов и ветка с `FACT_SCOPE`+`DESIGN`):
     - Передавать `name_en=_select_english_name(ent)`.
     - Для fallback `description_en` (когда нет фактов/дизайна или `desc_en_parts` пуст) использовать `_select_english_name(ent) or ("character" if kind_bucket == "person" else "location")` вместо `str(ent["canonical_name"])`, чтобы русское `canonical_name` не утекало в `description_en` и `prompt_en`!
   - В `compose_monochrome_prompt()` (`illustration_validation.py`):
     - Для `IllustrationAppearance`: `raw_name = item.name_en if item.name_en is not None else item.name_ru`; для `Mapping`: `raw_name = item.get("name_en") or item.get("name_ru", "")`.
     - Подставлять префикс `f"{name}: {desc_clean}"` **только если** `name` непустое и `not has_cyrillic(name)`; иначе `desc_clean` без префикса.
   - Обновить `_STAGE_INSTRUCTIONS[PROMPT]`:
     - Добавить: «Пиши `scene_composition_en` строго на английском языке (без кириллицы). Точно соблюдай позу, действие и физически возможную точку зрения камеры из `event_ru`: не показывай снаружи фасад здания в том же кадре, если действие происходит внутри комнаты без выхода наружу, и не меняй положение тела персонажа вопреки описанию события.»
   - В `_validate_render_response()` для `PROMPT`: отклонять `scene_composition_en`, если `has_cyrillic(payload.scene_composition_en)` (через repair-раунд).

4. **Усиление связки алиасов (`illustration_reference.py`, `illustration_context.py`):**
   - В `_advance_reference_pipeline()` при построении `pair_entries` для `ALIAS`:
     - Добавить в `left` и `right`:
       - `"alias_of_hint": m.get("alias_of_hint")` (если не `None`)
       - `"glossary_concept_id": glossary_links.get(glossary_key(str(m["name"])))` (если не `None`)
   - Обновить `_STAGE_INSTRUCTIONS[ALIAS]`:
     - «Учитывай совпадение `glossary_concept_id`, подсказки `alias_of_hint`, цитаты и контекст сцены. Если два упоминания явно обозначают одного и того же человека (например, имя и обращение с фамилией/титулом, такие как Petunia и Mrs Dursley, при подтверждении текстом/цитатой/глоссарием), ставь `same_entity`. При этом общая фамилия с разными титулами или полом (Mr Dursley и Mrs Dursley) — `different`. `unknown` ставь только когда доказательств действительно недостаточно.»

5. **Дедупликация `assumptions_ru` и описаний (`illustration_chapter.py`):**
   - Добавить чистую функцию `_dedupe_texts(items: Iterable[str]) -> tuple[str, ...]` (нормализованный ключ `" ".join(s.split()).rstrip(".;, ").casefold()`, сохраняет порядок и первую исходную формулировку) и `_dedupe_semicolon_parts(items: Iterable[str]) -> list[str]` (дополнительно разбивает части по `";"` перед дедупликацией).
   - Применять `_dedupe_texts` / `_dedupe_semicolon_parts` в **обеих** ветках `illustration_chapter.py` (`if not entity_facts:` и после `FACT_SCOPE` + `DESIGN`), а также при финальной сборке `assumptions` в `ValidatedMoment`.

6. **Точечное улучшение промптов анализа и перевода (`application/context_builder.py`):**
   - **Инвариант:** существующие формулировки в `_NARRATIVE_ANALYSIS_INSTRUCTIONS`, `_TRANSLATION_CONTEXT_V2`, `_CONTENT_PLAN_PROCESSING_V3` только **дополнять**, не удаляя проверяемые тестами подстроки; не добавлять неэкранированных фигурных скобок `{`/`}` в `_NARRATIVE_ANALYSIS_INSTRUCTIONS` (т. к. оно форматируется через `.format(...)`); не трогать замороженные legacy-шаблоны `_TRANSLATION_INSTRUCTIONS`, `_CONTENT_PLAN_ANALYSIS`, `_CONTENT_PLAN_PROCESSING`.
   - В `_NARRATIVE_ANALYSIS_INSTRUCTIONS` дописать:
     - личные имена персонажей (в т. ч. упомянутые без фамилии, например Petunia, Lily, James) всегда имеют `type="person"`, никогда не `type="term"`;
     - для составных иноязычных топонимов вроде «Прайвет-драйв» указывать грамматический род и склоняемость строго по нормам русского языка;
     - ориентир гранулярности сцен: разбивать длинные эпизоды на отдельные сцены при смене времени суток, места действия, появлении ключевых новых участников или переходе к новому сюжетному событию (ориентир: 5–25 блоков на сцену; не объединять 40+ блоков с разными событиями и участниками в одну сцену).
   - В `_TRANSLATION_CONTEXT_V2` и `_CONTENT_PLAN_PROCESSING_V3` дописать:
     - сохранять разговорный, просторечный и диалектный регистр речи персонажей в русском переводе (включая реплики с `speech_style`), не сглаживая их до сухого книжного стиля;
     - трудности с `help_kind="explanation"` (идиомы, устойчивые обороты, грамматические и культурные пояснения) оформлять через `language_notes`, а не переносить в `glossary_hints`; в `glossary_hints.original` придерживаться единообразной формы записи в пределах блока.

### Обязательные тесты WP-4
- `_validate_design()` отклоняет `palette_value` с цветовым словом (`"muted beige and green"`, `"золотые и зелёные тона"`) для `monochrome_ink` и принимает тональное описание (`"high-contrast black ink with crisp white highlights"`).
- `DESIGN` `request_input` содержит `"style": "monochrome_ink"`.
- `resolve_appearance_with_design_fallback`:
  - подавляет `hair` при `"shaggy beard"` / `"густая борода"`, `age_appearance` при `"infant"` / `"младенец"`, `build` при `"giant"` / `"великан"`;
  - **НЕ** даёт ложных срабатываний на `"managed"`, `"bundle"`, `"parties"`, `"permanent"`, `"big eyes"`.
- `compose_monochrome_prompt` использует `name_en` вместо `name_ru`; для сущности с чисто кириллическим именем (как с фактами, так и без фактов/дизайна) в `prompt_en` нет ни одной кириллической буквы.
- `_validate_render_response` для `PROMPT` отклоняет `scene_composition_en` с кириллицей.
- `_dedupe_texts` и `_dedupe_semicolon_parts` убирают дубли в `assumptions_ru` и склеенных описаниях в обеих ветках.
- `ALIAS` pair payload включает `alias_of_hint` и `glossary_concept_id` (по `name`), если они есть.

---

## WP-5. Адаптивный бюджет токенов рассуждения в переводе

### Затрагиваемые файлы
- `src/linguary/application/token_budget.py`
- `src/linguary/application/block_processing.py`
- `tests/unit/application/test_token_budget.py` (обновить тест приоритета `reasoning_effort` + добавить новые тесты)

### Контракт

1. **Минимальный резерв рассуждения (`application/token_budget.py`):**
   - Добавить и экспортировать в `__all__`:
     ```python
     REASONING_EFFORT_MIN_TOKENS: dict[str, int] = {
         "none": 0,
         "minimal": 1024,
         "low": 4096,
         "medium": 8192,
         "high": 12288,
     }
     ```
   - В `recommended_unit_max_completion_tokens(...)`:
     ```python
     visible = max(2048, int(source_chars * 1.35) + 1024)
     effort = reasoning_effort.strip().lower() if isinstance(reasoning_effort, str) else None
     if effort is not None:
         reserve = REASONING_EFFORT_RESERVE_PCT.get(effort, 0)
         min_reasoning = REASONING_EFFORT_MIN_TOKENS.get(effort, 0)
         pct_total = (visible * 100 + (99 - reserve)) // (100 - reserve) if reserve > 0 else visible
         total = max(pct_total, visible + min_reasoning)
     elif reasoning_max_tokens is not None:
         total = visible + reasoning_max_tokens
     else:
         total = visible
     ```
   - Логика ограничения сверху `max_output_tokens` остаётся прежней (`min(max_output_tokens, total)`).

2. **Эскалация `max_completion_tokens` при `provider_truncated` (`application/block_processing.py`):**
   - Внутри цикла `while True:`, после загрузки `loaded_conditions`, безопасно извлекать `configured_max_output_tokens`:
     ```python
     configured_max_output_tokens: int | None = None
     if loaded_conditions is not None and isinstance(loaded_conditions.payload, Mapping):
         raw_params = loaded_conditions.payload.get("llm_parameters")
         if isinstance(raw_params, Mapping):
             raw_max = raw_params.get("max_output_tokens")
             if type(raw_max) is int and raw_max > 0:
                 configured_max_output_tokens = raw_max
     ```
   - В обработчике `except Exception as error:`:
     - Определить `is_truncated` **вне** ветки `if recovery is not None` (избегая `UnboundLocalError: violation`!):
       ```python
       err_violation = getattr(error, "violation", None)
       is_truncated = (
           (isinstance(error, LLMResponseFormatError) and error.truncated)
           or (err_violation is not None and getattr(err_violation, "rule_code", None) == "provider_truncated")
       )
       ```
     - Определить:
       ```python
       can_escalate_tokens = (
           is_truncated
           and configured_max_output_tokens is not None
           and request.max_completion_tokens is not None
           and request.max_completion_tokens < configured_max_output_tokens
       )
       ```
     - В **recovery-режиме** (`recovery is not None and physical_context is not None`), после `if outcome == 'unknown': ...` и **до** `if split ... raise RecoverySplit`:
       - Если `can_escalate_tokens`:
         - В транзакции проверить актуальность корня и оставшийся бюджет (по образцу `LLMContextCapacityError`, строки 523-544, с защитой для тестовых фейков без `scope_attempt_count`):
           - `stale_reason` (`superseded`, `interrupted`, `stopped` с `unknown_outcome`/`interrupted`) → немедленно `raise RecoveryError(stale_reason, recovery[0], original_error=error) from error`!
           - Если корень активен, проверить `policy_map = (root.manifest or {}).get("policy") if isinstance(getattr(root, "manifest", None), Mapping) else None`, `count_fn = getattr(transaction.work, "scope_attempt_count", None)`:
             - `within_budget`: `root.calls_used < policy_map["max_calls"] + root.extra_calls` и `(not callable(count_fn) or count_fn(recovery[1]) < policy_map["max_scope_calls"])`.
         - Если `within_budget`: НЕ делать `RecoverySplit` и НЕ останавливать корень; обновить `request = replace(request, max_completion_tokens=configured_max_output_tokens, attempt=request.attempt + 1)`, `last_error = error`, и `continue`!
         - Важно: на следующей итерации `request.max_completion_tokens` уже не `None` (`== configured_max_output_tokens`), поэтому строка 286 (`and request.max_completion_tokens is None`) не перезапишет его обратно!
     - В **не-recovery режиме** (`recovery is None`):
       - Если `can_escalate_tokens` и `attempts < self._retry_policy.max_attempts`:
         - Обновить `request = replace(request, max_completion_tokens=configured_max_output_tokens, attempt=request.attempt + 1)`, `last_error = error`, и `continue`!
     - Если после эскалации до `configured_max_output_tokens` ответ снова усечён (`request.max_completion_tokens == configured_max_output_tokens`), `can_escalate_tokens` становится `False`, и управление штатно переходит к `RecoverySplit` или остановке корня с `output_truncated`.

### Обязательные тесты WP-5
- Обновить `tests/unit/application/test_token_budget.py`:
  - проверить `REASONING_EFFORT_MIN_TOKENS` для всех уровней;
  - в `test_recommended_unit_max_completion_tokens_reasoning_effort_precedes_reasoning_max_tokens` использовать `reasoning_max_tokens=12000` (для `low`) и `20000` (для `high`), чтобы проверить как новый floor (`6144`, `14336`), так и приоритет `reasoning_effort` над `reasoning_max_tokens`.
- Добавить тесты в `test_token_budget.py` / `test_block_processing.py`:
  - recovery-вызов (1 блок) с `provider_truncated` при `6144 < 32768` эскалирует до `32768` и завершается успешно;
  - recovery-вызов (2 блока) с `provider_truncated` при `6144 < 32768` сначала эскалирует до `32768` без split, а при повторном усечении на `32768` уходит в `RecoverySplit`;
  - non-recovery вызов с `provider_truncated` при `6144 < 32768` эскалирует до `32768` без `UnboundLocalError`;
  - при исчерпании `max_scope_calls` / `max_calls` или при `stale_reason` (`superseded`/`interrupted`) эскалация не выполняется.

---

## WP-6. Диагностика и телеметрия иллюстраций

### Затрагиваемые файлы
- `src/linguary/application/illustration_context.py` (`compact_safe_detail`)
- `src/linguary/application/illustration_reference.py`
- `src/linguary/application/illustration_selection.py`
- `src/linguary/application/illustration_chapter.py`

*(Порт `ports/illustrations.py` и репозиторий `adapters/db/illustration_repository.py` менять НЕ нужно — у `fail_scope` уже есть параметр `outcome: Mapping[str, Any] | None = None`!)*

### Контракт

1. **Чистый обрезчик безопасного `detail` (`application/illustration_context.py`):**
   - Добавить `compact_safe_detail(text: str, *, max_bytes: int = 256) -> str`:
     - схлопывает пробелы (`" ".join(text.split()).strip()`),
     - обрезает по границе UTF-8 codepoint до `max_bytes` с `"..."`, **без** добавления префикса `"Исправь ответ: "`.

2. **Финализация записей в `llm_call_logs` через `llm_rejections.py`:**
   - В `run_task()` каждого из трёх модулей (`IllustrationReference`, `IllustrationSelection`, `IllustrationChapter._run_render_scope_task`):
     - Инициализировать `response: LLMResponse[Any] | None = None` **внутри** `while True:` (рядом с `validation = None` и `call_error = None`), чтобы на повторной итерации не переиспользовать объект `response` предыдущей попытки!
     - **Инвариант однократной финализации:** для каждого `response.call_id` решение (`accepted` / `rejected` / `stale`) фиксируется ровно один раз — после вызова `record_rejection(self._uow, response, ...)` нельзя вызывать `finalize_response(self._uow, response, ...)`.
     - Точки вызова:
       1. Если `call_error is not None` и `response is not None` (например, `response.truncated` породил `LLMResponseFormatError` уже после успешного `client.generate`): вызвать `record_rejection(self._uow, response, call_error)` **вне** открытых транзакций до перехода на split/repair/failure.
       2. В `IllustrationReference` и `IllustrationSelection`, если `validation.request_split`: вызвать `record_rejection(self._uow, response, ResultValidationError("модель запросила дробление (complete=false)"))` до `_handle_post_send_split` / `_finalize_failure`.
       3. Если `not validation.ok`: сформировать чистый `detail_text` (отфильтровав пустые строки из `validation.repair_details` в `IllustrationReference`!) и вызвать `record_rejection(self._uow, response, ResultValidationError(detail_text))` до repair/split/failure.
       4. В TX3 (когда `validation.ok` и `record_rejection` НЕ вызывался):
          - при `stale_reason is not None` или `IllustrationOwnershipError`: `finalize_response(self._uow, response, "stale", transaction=tx)`;
          - при успешном `tx.illustrations.accept_scope(...)`: `finalize_response(self._uow, response, "accepted", transaction=tx)`.

3. **Безопасное поле `detail` в `illustration_attempts.outcome_json`:**
   - При записи `IllustrationAttemptState.REJECTED` записывать:
     ```python
     outcome={
         "reason_code": reason_code,
         "detail": compact_safe_detail(detail_text, max_bytes=256),
     }
     ```
   - В `_finalize_scope_failure` / `_finalize_failure` (и `_handle_post_send_split` при невозможности split) добавить необязательный параметр `detail: str | None = None` и передавать в существующий параметр `outcome` метода `tx.illustrations.fail_scope(...)`:
     ```python
     outcome = (
         {"reason_code": reason_code, "detail": compact_safe_detail(detail, max_bytes=256)}
         if detail and detail.strip()
         else None
     )
     tx.illustrations.fail_scope(scope_id, attempt_id=attempt_id, reason_code=reason_code, outcome=outcome)
     ```
   - Инвариант безопасности: `detail` никогда не содержит сырого текста ответа модели или секретов провайдера — только локальное сообщение валидатора, усечённое до 256 байт UTF-8.

### Обязательные тесты WP-6
- При семантическом отклонении ответа иллюстраций (`LoggingLLMClient` + `SqlAlchemyUnitOfWork`) запись в `llm_call_logs` получает `success=False`, `diagnostic_meta["decision"] == "rejected"`, а тело ответа сохраняется даже при `debug_llm_payloads=False`; `illustration_attempts.outcome_json` содержит `reason_code` и усечённый `detail` (без префикса `"Исправь ответ: "`).
- При успешном принятии области запись в `llm_call_logs` получает `diagnostic_meta["decision"] == "accepted"`.
- При устаревании попытки (`stale_reason` / `IllustrationOwnershipError`) запись в `llm_call_logs` получает `diagnostic_meta["decision"] == "stale"`.
- При двух итерациях цикла (1-я попытка отклонена, 2-я упала с `LLMTransportError` до получения ответа) первая запись остаётся `rejected`, а не переписывается повторно.
- Все существующие unit-тесты с `FakeLLMClient` (`response.call_id is None`) проходят без изменений.

---

## Матрица проверок перед слиянием

1. `uv run pytest tests/architecture` — проверка соблюдения границ Clean Architecture.
2. `uv run pytest -m "not slow"` — полный прогон unit- и integration-тестов.
3. Проверка отсутствия изменений схемы БД (`git diff alembic/` пуст) и чистоты `git status`.
