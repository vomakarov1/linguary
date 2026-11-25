# SDD: Техническая спецификация исправлений восстановления перевода, хронологии и монохромной гигиены иллюстраций

> **Статус:** Accepted (глубокая перепроверка пройдена, все замечания учтены)

- Дата: 2026-11-25
- BRIEF: [BRIEF-translation-recovery-and-illustration-fidelity-2026-11-25.md](BRIEF-translation-recovery-and-illustration-fidelity-2026-11-25.md)
- ADR: [ADR-translation-recovery-and-illustration-fidelity-2026-11-25.md](ADR-translation-recovery-and-illustration-fidelity-2026-11-25.md)
- План реализации: [2026-11-25-translation-recovery-and-illustration-fidelity.md](../../plans/2026-11-25-translation-recovery-and-illustration-fidelity.md)

Документ разбивает изменения на 6 независимых пакетов работ (WP-1 … WP-6) с точными контрактами модулей, инвариантами, граничными случаями и обязательными регрессионными тестами. Миграции Alembic не требуются (**0 миграций**). Зависимости слоёв строго соблюдают `domain ← application ← adapters / web / cli`.

---

## WP-1. Валидаторы перевода: баланс кавычек (`truncation.py`) и цитирование оригинала в заметках (`mixed_script.py`)

### Затрагиваемые файлы
- `src/linguary/application/validators/truncation.py`
- `src/linguary/application/validators/mixed_script.py`
- `src/linguary/application/validators/result.py`
- `tests/unit/application/test_result_validators.py`

### Контракт

#### 1.1. Баланс кавычек в `validators/truncation.py`
1. Импортировать `british_quote_counts` из `linguary.application.validators.quotes` (наряду с `has_unclosed_british_quote` либо заменив вызов специального костыля на общий расчёт).
2. В `_balances(text: str) -> Mapping[str, int]`:
   - Добавить учёт британских одиночных кавычек `‘`/`’`:
     ```python
     brit_open, brit_close = british_quote_counts(text)
     openings = text.count("«") + n_low + (n_high_open - paired_low) + brit_open
     closings = text.count("»") + n_high_close + paired_low + brit_close
     balances["кавычки"] = (openings - closings) + text.count(_SYMMETRIC_QUOTE) % 2
     ```
   - Заметим: `british_quote_counts` уже отличает настоящие открывающие/закрывающие `‘`/`’` от внутрисловных и концевых апострофов, диалектных усечений (`killin’ ’em`, `An’ ’em`, `don’`) и притяжательных форм (`boys’`), а также обрабатывает открытие `‘` в середине строки (`: ‘...`) и одиночное закрытие `’` (`...!’`). *(Известное ограничение `quotes.py:163-166`: притяжательное мн. ч. перед точкой/заглавной вроде `at the Dursleys’.` даёт `(0, 1)`; на корректном переводе с `tr_balance == 0` ложного отказа не возникает.)*
3. Выделить чистый хелпер `_quote_balance_matches(tr_balance: int, *, source: str, source_balance: int) -> bool`:
   - Возвращает `True`, если:
     - `tr_balance == 0` (сбалансированный перевод или русское тире в диалоге), **или**
     - `tr_balance == source_balance` (точное совпадение открытого `+1` или закрывающего `-1` дисбаланса с оригиналом), **или**
     - `abs(tr_balance) == 1 and source.count(_SYMMETRIC_QUOTE) % 2 == 1` (нечётное число симметричных прямых кавычек `"` в исходном сегменте может быть как открывающей, так и закрывающей кавычкой), **или**
     - `tr_balance == 1 and has_unclosed_british_quote(source)` (сохраняется как страховка совместимости).
4. В `_unbalanced(text: str, *, source: str, check_quotes: bool = True) -> str | None`:
   - Для всех скобок `() [] {}` поведение прежнее: `balance != 0 and balance != original[name]` → возврат `name`.
   - Для `"кавычки"` (когда `check_quotes=True`): если `not _quote_balance_matches(balance, source=source, source_balance=original["кавычки"])` → возврат `"кавычки"`.
5. **Двухуровневая проверка многосегментных блоков** (единая функция-хелпер `_block_truncation_issue(block, item) -> tuple[str, str, str] | None`, используемая как в `truncation_violation`, так и в `truncation_diagnostic` — устраняется дублирование логики):
   - Собрать список срезов `slices = list(_checked_translations(block, item))`.
   - Если блок **не** сегментированный (`len(slices) == 1` и `slices[0][0] is None`):
     - Проверить терминальную пунктуацию, скобки и кавычки как прежде (`_unbalanced(text, source=source, check_quotes=True)`).
   - Если блок **сегментированный** (`planned_segment_modes(item)` не пуст):
     - Шаг A (по каждому `TRANSLATE`-сегменту): проверить терминальную пунктуацию (`terminal_punctuation`) и баланс скобок (`_unbalanced(text, source=source, check_quotes=False)` → `bracket_balance`).
     - Шаг B (баланс кавычек по `TRANSLATE`-сегментам блока):
       - Найти все `TRANSLATE`-сегменты с `not _quote_balance_matches(tr_bal, source=src, source_balance=src_bal)`.
       - Если таких нет → нарушений нет.
       - Если есть: вычислить агрегированный баланс **на склеенных строках** `TRANSLATE`-срезов (а НЕ арифметической суммой `_balances(...)`, чтобы корректно считать парность симметричных `"` и вложенных `„...“` через границы сегментов!):
         - `joined_tr = " ".join(tr for _, _, tr in slices)`
         - `joined_src = " ".join(src for _, src, _ in slices)`
         - `total_tr = _balances(joined_tr)["кавычки"]`
         - `total_src = _balances(joined_src)["кавычки"]`
         - Если `total_tr == 0` или `_quote_balance_matches(total_tr, source=joined_src, source_balance=total_src)`: дисбаланс между соседними `TRANSLATE`-сегментами взаимно скомпенсирован внутри блока → **нарушения нет**.
         - Иначе: вернуть `quote_balance` с указанием первого проблемного сегмента (`_subject(block.source_block_id, first_bad_index)`).

#### 1.2. Дословные многословные цитаты оригинала в `validators/mixed_script.py` и `validators/result.py`
1. В `validators/mixed_script.py`:
   - Расширить сигнатуру `note_language_leak_violation`:
     ```python
     def note_language_leak_violation(
         payload: TranslationResultResponse,
         *,
         target_language: str,
         source_texts: Mapping[str, str] | None = None,
     ) -> str | None:
     ```
     (Значение по умолчанию `source_texts=None` гарантирует 100% обратную совместимость со всеми существующими вызовами и тестами.)
   - В `_note_texts_of(payload, source_texts=source_texts)` пробросить `source_texts` в **обе** ветки (как при непустых `payload.blocks`, так и в `else`-ветке для `payload.language_notes` и `payload.fragment_assistances`) и возвращать кортеж `(subject, text, allowed_sources)`, где `allowed_sources: tuple[str, ...]` включает:
     - для `hint.note`: `(hint.original, source_texts.get(block_id, ""))`;
     - для `note.note` и `note.translated_excerpt`: `(note.original_excerpt, source_texts.get(note.source_block_id, ""))`;
     - для `assistance.text`: `tuple(source_texts.values())` если `source_texts` передан, иначе `()`.
     (Пустые строки отфильтровываются.)
   - Выделить чистую функцию `_cited_source_spans(text: str, allowed_sources: Sequence[str]) -> list[tuple[int, int]]`:
     - Вспомогательная нормализация строки `_normalize_for_source_match(s: str) -> str`: типографские апострофы `’‘ʼ` → `'`, дефисы `\u2010\u2011` → `-`, `" ".join(s.split())` (схлопывает пробелы, переносы строк и `\xa0`), `.casefold()`.
     - **Важно (без смещения спанов):** искать латинские отрезки прямо в **исходном** `text` (сохраняя точные `(start, end)` индексы относительно `_LETTERS.finditer(text)`!), а нормализовать только проверяемую подстроку `text[start:end]` и `allowed_sources`.
     - Найти в исходном `text` все максимальные непрерывные **многословные латинские отрезки**: последовательности из **≥ 2** латинских слов (`[A-Za-zÀ-ɏ]+(?:['’-][A-Za-zÀ-ɏ]+)*`), разделённых пробелами или запятой с пробелом (для цитат вида `Well, very soon` / `very, very strange`, без кириллицы и без переносов строки).
     - Для каждого отрезка (и его непрерывных подпоследовательностей из ≥ 2 смежных слов): проверить, встречается ли `_normalize_for_source_match(subtext)` в нормализованной строке `src` из `allowed_sources` **по границам слов** (`(?<![A-Za-zÀ-ɏ])<re.escape(phrase)>(?![A-Za-zÀ-ɏ])`).
     - Возвратить исходные спаны `(start, end)` совпавших многословных цитат оригинала.
   - В `_note_text_leak(text: str, *, allowed_sources: Sequence[str] = ()) -> str | None`:
     - Вычислить `cited_spans = _cited_source_spans(text, allowed_sources)`.
     - При проверке немаркированных латинских токенов между первым и последним кириллическими словами пропускать токен, если `_overlaps_any(start, end, cited_spans)`.
     - **Гарантии сохранения строгости:**
       - Одиночное слово из `_LEAK_EN_WORDS` между русскими словами (`«это very важно»`) имеет длину 1 слово → никогда не попадает в `cited_spans` → по-прежнему отклоняется!
       - Многословная английская фраза, которой **нет** в исходном тексте/цитате (`«он был very happy»` при оригинале `"He smiled."`), не попадает в `cited_spans` → по-прежнему отклоняется!
       - Кириллица внутри английских кавычек (`«was strongly reminded of Дадли»`) проверяется на шаге `_MARKED_SPAN` раньше и по-прежнему строго отклоняется!
2. В `validators/result.py`:
   - В обоих вызовах `note_language_leak_violation(...)` (строки 298 и 306) передавать `source_texts={item.block.id: item.block.text for item in unit.items}`.

### Обязательные тесты WP-1
- `truncation.py`:
  - Многосегментный блок с британским открытием `‘` в середине сегмента после `: ` (`b18`-паттерн: сег. 4 открывает `«`, сег. 5 продолжает, сег. 6 закрывает `»`) → принят.
  - Многосегментный блок с несколькими британскими репликами (`‘...` / `...’` в разных сегментах, `b25`-паттерн) → принят.
  - Многосегментный блок с американскими прямыми кавычками `"` в сег. 1 и закрывающей `"` в сег. 2 (`«...` / `...»`) → принят.
  - Многосегментный блок, где сег. 1 открыл `«`, а сег. 2 забыл закрыть `»` (при закрытом оригинале) → отклонён с `rule_code == "quote_balance"`.
  - Сохранение всех существующих тестов `test_truncation_validator_checks_only_translated_segments`, `test_truncation_allows_british_unclosed_opening_quote_and_rejects_broken_closed_quote`, `test_truncation_allows_british_dialect_apostrophes_and_unclosed_quotes`.
- `mixed_script.py`:
  - Заметка на русском с дословной многословной цитатой `How very wrong` при `original_excerpt="How very wrong he was."` (или исходнике блока) → принята.
  - Одиночное немаркированное слово `very` между русскими словами (`«это very важный оборот»`), даже если `very` есть в исходнике блока → отклонена (`note_language_leak`).
  - Немаркированная фраза `very strong man`, которой нет в исходнике блока → отклонена (`note_language_leak`).
  - Кириллица внутри английских кавычек → по-прежнему отклонена.

---

## WP-2. Схема `segments[].index`, диагностика `_schema_hints()` и правила `source_block_id` / `translated_excerpt`

### Затрагиваемые файлы
- `src/linguary/adapters/llm/schemas.py`
- `src/linguary/adapters/llm/payloads.py`
- `src/linguary/application/context_builder.py` (только `_TRANSLATION_CONTEXT_V2` и `_CONTENT_PLAN_PROCESSING_V3`!)
- `src/linguary/application/validators/content_plan.py`
- `tests/unit/llm/test_segment_translation_contract.py`
- `tests/unit/llm/test_llm_payload_mapping.py`
- `tests/unit/llm/test_openai_compatible_client.py`
- `tests/integration/processing/test_llm_logging.py`
- `tests/unit/application/test_context_builder.py`

### Контракт

1. **`SegmentTranslationSchema.index` (`adapters/llm/schemas.py:179`):**
   ```python
   index: int = Field(
       ge=1,
       strict=True,
       description=(
           "1-based номер сегмента M из метки [bN.M] в исходном блоке "
           "(только для сегментов TRANSLATE; не с 0 и без перенумерации пропущенных сегментов)"
       ),
   )
   ```
   - `strict_json_schema()` сохраняет `"description"` (проверено в `json_schema.py:35`) и вырезает `"minimum"`, поэтому существующий контракт `test_segment_translation_contract.py:128` (`"minimum" not in repr(segment)`) остаётся зелёным.

2. **`_schema_hints()` (`adapters/llm/payloads.py:590-620`):**
   - Добавить ветки для:
     - `err_type == "greater_than_equal"`:
       - если `path.endswith(".index")` и `".segments[" in path`:
         `hint = f"{prefix}номер сегмента начинается с 1 и равен M из метки [bN.M]; не используй 0 и не перенумеровывай пропущенные сегменты"`
       - иначе:
         `ge = ctx.get("ge", 1)` → `hint = f"{prefix}значение должно быть не меньше {ge}"`
     - `err_type == "greater_than"`:
       - `gt = ctx.get("gt", 0)` → `hint = f"{prefix}значение должно быть больше {gt}"`
     - `err_type == "value_error"`:
       - **ЗАПРЕЩЕНО** копировать сырой `item["msg"]` напрямую (в `BlockResultSchema._check_treatment_composition` в `ValueError` интерполируются `self.source_block_id` и `{treatment}`, попадание которых в `str(error)`/`LLMCallLog` нарушает контракт приватности и ломает `test_openai_compatible_client.py:134-137` и `test_llm_logging.py:357-362`!).
       - Использовать строгий статический белый список / сопоставление суффиксов `BlockResultSchema._check_treatment_composition` (и статических `ValueError` схем) на безопасные константные подсказки **без** `source_block_id`, **без** названия режима (`TRANSLATE`/`GLOSSARY_ONLY`/`READ_AS_IS`) и **без** круглых скобок:
         - суффикс `"с переводами сегментов полный перевод блока недопустим"` → `"для блока с переводами сегментов полный перевод блока недопустим"`;
         - суффикс `"обязателен полный перевод"` → `"для выбранного режима блока обязателен полный перевод"`;
         - суффикс `"полный перевод недопустим"` → `"для выбранного режима блока полный перевод недопустим"`;
         - суффикс `"обязательны словарные подсказки"` → `"для выбранного режима блока обязательны словарные подсказки"`;
         - суффикс `"словарные подсказки недопустимы"` → `"для выбранного режима блока словарные подсказки недопустимы"`;
         - статическое `"пустой перевод сегмента"` → `"перевод сегмента не может быть пустым"`.
       - Все неизвестные `value_error` (не входящие в allowlist) пропускаются — утечка сырого ввода модели исключена конструктивно.

3. **Промпты перевода (`application/context_builder.py`):**
   - **НЕ ТРОГАТЬ** `_TRANSLATION_INSTRUCTIONS`, `_SEGMENT_TRANSLATION_RULE`, `_CONTENT_PLAN_ANALYSIS`, `_CONTENT_PLAN_PROCESSING` и B2/`legacy-v1`!
   - В `_TRANSLATION_CONTEXT_V2` и `_CONTENT_PLAN_PROCESSING_V3` добавить лаконичное правило:
     - «В `blocks[].source_block_id` и `language_notes[].source_block_id` всегда указывай идентификатор целого блока `bN`, никогда не метку сегмента `bN.M`. В `segments[].index` всегда указывай 1-based номер `M` из исходной метки `[bN.M]` только для сегментов с режимом `TRANSLATE`, не с нуля и без перенумерации пропущенных сегментов.»

4. **Подсказка ремонта `translated_excerpt` (`application/validators/content_plan.py`):**
   - Сохранить строгую проверку `_translated_excerpt_matches` без ослабления.
   - В тексте сообщения об отсутствии `translated_excerpt` в `translated_text` добавить конкретное указание для ремонта: копировать `translated_excerpt` дословно из получившегося `translated_text` (а при разрыве фраз в русском порядке слов разделять части через `...`).

### Обязательные тесты WP-2
- Wire JSON Schema `strict_json_schema(schema_for(LLMResponseKind.TRANSLATION_RESULT, contract_version=4))` содержит `"description"` у `SegmentTranslationSchema.index` с упоминанием `1-based` / `[bN.M]` и по-прежнему не содержит `"minimum"`.
- `_schema_hints()` для `segments[0].index: greater_than_equal` выдаёт подсказку про 1-based номер `M` из `[bN.M]` без круглых скобок.
- `_schema_hints()` для `BlockResultSchema` `value_error` (например, `READ_AS_IS` с `translated_text` или `GLOSSARY_ONLY` без `glossary_hints`) возвращает статическую причину нарушения композиции без круглых скобок и **без** `source_block_id` и значения `treatment`.
- Существующие тесты приватности `test_openai_compatible_client.py` и `test_llm_logging.py` и тест `test_baseline_b2_templates_sha256_unchanged()` проходят без изменений.

---

## WP-3. Адаптивный бюджет скрытых рассуждений, лимит блоков на юнит и BFS-приоритет скоупов `Recovery`

### Затрагиваемые файлы
- `src/linguary/application/token_budget.py`
- `src/linguary/application/block_processing.py`
- `src/linguary/application/translation_units.py`
- `src/linguary/application/recovery.py`
- `tests/unit/application/test_translation_units.py`
- `tests/unit/application/test_recovery_partition.py`
- `tests/unit/application/test_token_budget.py`

### Контракт

#### 3.1. Адаптивный бюджет скрытых рассуждений (`token_budget.py`, `block_processing.py`)
1. В `token_budget.py`:
   - Добавить константу `IMPLICIT_REASONING_MIN_TOKENS = 8192`.
   - Расширить `recommended_unit_max_completion_tokens(...)` опциональным параметром `implicit_reasoning_tokens: int = 0`:
     ```python
     def recommended_unit_max_completion_tokens(
         source_chars: int,
         *,
         max_output_tokens: int | None,
         reasoning_effort: str | None,
         reasoning_max_tokens: int | None,
         implicit_reasoning_tokens: int = 0,
     ) -> int:
     ```
   - Если `effort is None and reasoning_max_tokens is None and implicit_reasoning_tokens > 0`:
     `total = visible + max(IMPLICIT_REASONING_MIN_TOKENS, implicit_reasoning_tokens * 2)`
     (затем ограничивается сверху `max_output_tokens`, если он задан).
   - При `implicit_reasoning_tokens == 0` (значение по умолчанию) формула остаётся побитово идентичной текущей — полная обратная совместимость.
2. В `block_processing.py` (`TranslationCall`):
   - Добавить на экземпляр `TranslationCall` потокобезопасное (`threading.Lock`) хранилище `self._implicit_reasoning_floor: int = 0` (привязанное к текущему `TranslationCall`, т. е. к конкретной задаче/главе с одной настроенной моделью).
   - Рядом с вычиткой `configured_max_output_tokens` (строки `284-290`) вычитывать также `configured_reasoning_effort` и `configured_reasoning_max_tokens` из `loaded_conditions.payload["llm_parameters"]`, чтобы они были доступны и в ветке `except`.
   - После успешного ответа `response = await self._llm_client.generate(request)`:
     - Извлечь `reasoning_tokens` из `TokenUsage` (`domain/llm_log.py:35-41`): `getattr(getattr(response, "usage", None), "reasoning_tokens", None)`.
     - Если `type(reasoning_tokens) is int and reasoning_tokens > 0`: потокобезопасно обновить `self._implicit_reasoning_floor = max(self._implicit_reasoning_floor, reasoning_tokens)`.
   - В ветке `except` при любом усечении ответа провайдером (`is_truncated`, т. е. `LLMResponseFormatError.truncated` или `violation.rule_code == "provider_truncated"`):
     - Извлечь `err_reasoning = getattr(getattr(error, "usage", None), "reasoning_tokens", None)` (при обрыве JSON `client.generate()` бросает `LLMResponseFormatError` с `error.usage`, тогда как `response is None`).
     - Если `configured_reasoning_effort is None and configured_reasoning_max_tokens is None`: обновить `self._implicit_reasoning_floor` до `max(self._implicit_reasoning_floor, err_reasoning if type(err_reasoning) is int and err_reasoning > 0 else 0, request.max_completion_tokens or IMPLICIT_REASONING_MIN_TOKENS)` — независимо от того, задан ли `configured_max_output_tokens` (если `max_output_tokens is None`, следующие юниты и дочерние скоупы после сплита всё равно получат повышенный потолок вместо застревания на `2048`!).
   - При расчёте `limit = recommended_unit_max_completion_tokens(...)` на строках `297-302` передавать `implicit_reasoning_tokens=self._implicit_reasoning_floor`.

#### 3.2. Лимит блоков на `TranslationUnit` (`translation_units.py`)
1. Добавить константу `DEFAULT_MAX_UNIT_BLOCKS = 16` и экспортировать её в `__all__`.
2. Расширить сигнатуру `plan_translation_units(...)`:
   ```python
   def plan_translation_units(
       blocks: Iterable[DocumentBlock],
       plan: TreatmentPlan,
       *,
       id_factory: Callable[[], str],
       max_source_chars: int = DEFAULT_MAX_UNIT_SOURCE_CHARS,
       max_blocks: int | None = DEFAULT_MAX_UNIT_BLOCKS,
       scenes: Sequence[Scene] = (),
   ) -> tuple[TranslationUnit, ...]:
   ```
   - Если `max_blocks is not None and max_blocks < 1`: выбросить `TranslationUnitError("предел числа блоков в группе должен быть положительным")`.
3. В `_grouped()`, `_closes_group()`, `_cohesive_runs()`, `_split_oversized()`:
   - `_split_oversized(run, *, max_source_chars, max_blocks)`: связная серия (список/стих/таблица) остаётся целой, только если `sum(len(b.text) for b in run) <= max_source_chars` **и** `(max_blocks is None or len(run) <= max_blocks)`; иначе распадается на отдельные блоки.
   - В `_grouped()`:
     - `count_by_chars = max(1, -(-total // max_source_chars))`
     - `count_by_blocks = max(1, -(-len(blocks) // max_blocks)) if max_blocks is not None else 1`
     - `count = max(count_by_chars, count_by_blocks)`
     - Поддерживать `current_blocks = len(current)` и `placed_blocks`.
   - В `_closes_group()`:
     - Строгое ограничение: `if current_size + size > max_source_chars: return True` и `if max_blocks is not None and current_blocks + run_blocks > max_blocks: return True`.
     - Мягкая граница сцены (40% заполнения по символам ИЛИ по блокам):
       `if scene_boundary and (current_size * 5 >= max_source_chars * 2 or (max_blocks is not None and current_blocks * 5 >= max_blocks * 2)): return True`.
     - Балансировка по целевому `count`: когда `count == count_by_blocks > count_by_chars`, ориентироваться на равномерную границу по блокам (`len(blocks) * (filled + 1) / count`), иначе — по символам (`total * (filled + 1) / count`), сохраняя 100% прежнее поведение для всех коротких глав и тестов!

#### 3.3. BFS-приоритет скоупов (`recovery.py`)
1. В `Recovery.plan_unresolved(self, root_id)` (`src/linguary/application/recovery.py:331-343`):
   - Построить карту `by_id = {scope.id: scope for scope in scopes}`.
   - Вычислить глубину каждого скоупа по цепочке `parent_id`:
     ```python
     def _depth(scope: RecoveryScope) -> int:
         depth = 0
         cur = scope
         seen: set[str] = {cur.id}
         while cur.parent_id is not None and cur.parent_id in by_id:
             if cur.parent_id in seen:
                 break
             seen.add(cur.parent_id)
             depth += 1
             cur = by_id[cur.parent_id]
         return depth
     ```
   - Если у `transaction.work` есть вызываемый метод `scope_attempt_count`, использовать `attempts(scope) = int(count_fn(scope.id))`, иначе `0`.
   - Сортировать `pending`-скоупы по ключу:
     ```python
     key=lambda s: (_depth(s), attempts(s), order[s.ordered_blocks[0]])
     ```
   - **Инвариант:** при равной глубине и равном числе попыток порядок полностью совпадает с прежним (документным порядком `order[s.ordered_blocks[0]]`). При расщеплении левой ветви на глубину `d+1` правая ветвь на глубине `d` (ещё не получавшая попыток) гарантированно идёт раньше более глубокого дробления проблемного блока слева!

### Обязательные тесты WP-3
- `token_budget.py` / `block_processing.py`:
  - `recommended_unit_max_completion_tokens` без `implicit_reasoning_tokens` сохраняет прежние значения; при `implicit_reasoning_tokens=5000` и `effort=None` добавляет запас `max(8192, 10000)` в пределах `max_output_tokens`.
  - В `TranslationCall`: после первого ответа с `reasoning_tokens > 0` (или первого `provider_truncated` с эскалацией) следующий юнит сразу получает повышенный `max_completion_tokens` без повторного усечения.
- `translation_units.py`:
  - Диалоговая глава из 32 коротких блоков по 100 символов (`total=3200 < 8000`) с дефолтным `max_blocks=16` делится на 2 сбалансированных юнита по 16 блоков (а не 1 юнит на 32 блока).
  - При `max_blocks=None` все 32 блока попадают в 1 юнит (обратная совместимость).
  - Неположительный `max_blocks <= 0` вызывает `TranslationUnitError`.
  - Связный список длиннее `max_blocks` корректно разбивается, не превышая лимит.
- `recovery.py`:
  - Корень делится на `L` и `R` (глубина 1); `L` падает и делится на `L1`, `L2` (глубина 2). `plan_unresolved(root_id)` возвращает `R` **первым** (перед `L1`, `L2`), чтобы здоровая правая ветвь обработалась до исчерпания бюджета.
  - На одном уровне глубины порядок скоупов остаётся строго по порядку блоков в документе.

---

## WP-4. Иллюстрации: хронологическая целостность `DESIGN` / `FACT_SCOPE`, `used_design_keys` и очистка `assumptions_ru`

### Затрагиваемые файлы
- `src/linguary/adapters/llm/illustration_payloads.py`
- `src/linguary/application/illustration_context.py`
- `src/linguary/application/illustration_source.py` (`compute_chapter_fingerprint` → `illustration_chapter_v2`)
- `src/linguary/application/illustration_reference.py`
- `src/linguary/application/illustration_chapter.py`
- `src/linguary/application/illustration_validation.py`
- `tests/unit/llm/test_illustration_payloads.py`
- `tests/unit/application/test_illustration_reference.py`
- `tests/unit/application/test_illustration_chapter.py`
- `tests/unit/application/test_illustration_reliability.py`

### Контракт

#### 4.1. Чистота этапа `DESIGN` (`illustration_payloads.py`, `illustration_context.py`, `illustration_reference.py`)
1. **`IllustrationDesignTraitSchema` (`adapters/llm/illustration_payloads.py:167`):**
   - Изменить `assumption_ru: str = Field(default="", max_length=_MAX_STR)` (убрать `min_length=1`; в `strict_json_schema()` поле остаётся в `required`, а `default`/`maxLength` вырезаются, что безопасно для wire-схемы и `test_illustration_payloads.py`).
2. **`_build_bounded_design_input()` и `_build_design_scope_spec()` (`application/illustration_reference.py:517-645`):**
   - Разделять факты на `permanent_facts` (`temporal_scope == "permanent"`) и `non_permanent_facts` (`state_change`, `episodic`).
   - Всегда включать в `request_input` булево поле `"non_permanent_context_only": bool(not permanent_facts and non_permanent_facts)` (это также автоматически инвалидирует старый кэш `DESIGN` в `find_reusable_scope` на существующих БД!).
   - Если `permanent_facts` не пуст: передавать в `facts` **только** `permanent_facts` (а число пропущенных временных фактов учитывать в `omitted_episodic_count` / `omitted_fact_count`).
   - Если `permanent_facts` пуст (у сущности вообще нет постоянных фактов в книге): передавать до `initial_target_bytes` факты с их реальным `temporal_scope` при `"non_permanent_context_only": True`.
   - В `_build_design_scope_spec()` дополнительно сохранять `"entity_names"` (список `[canonical_name, *aliases]` без пустых строк) в `scope.summary` для защиты собственных имён в `_validate_design()` (с fallback на `summary.get("canonical_name")` для старых скоупов и юнит-тестов).
3. **`_STAGE_INSTRUCTIONS[IllustrationScopeStage.DESIGN]` (`application/illustration_context.py:345-355`):**
   - Усилить общее правило:
     - `DESIGN` задаёт **только вневременные базовые визуальные черты**, отсутствующие в `facts` (нейтральный силуэт, базовые пропорции/архитектура, тонально-графический характер для `monochrome_ink`).
     - **Категорически запрещено** закреплять в `DESIGN` эпизодические и изменяемые состояния (`temporal_scope="episodic"` или `"state_change"`): сон/закрытые глаза, временную позу или выражение лица, разовую одежду/обёртку конкретного эпизода, переменное освещение, случайных животных/предметы в локации, а также преходящий возрастной снимок, если его постоянство не подтверждено `permanent`-фактом.
     - Не дублировать черты, уже описанные в `facts`; если в `facts` достаточно деталей или переданы только временные факты, допустимо вернуть только `palette_value` или `traits: []`.
     - В `assumption_ru` записывать только реальное допущение дизайнера; если допущения нет, оставлять `assumption_ru: ""` (никогда не писать заглушки вроде «Допущений нет»).
4. **`_validate_design()` (`application/illustration_reference.py:1599-1615`):**
   - При `style == ILLUSTRATION_STYLE_MONOCHROME_INK` проверять хроматические цвета на **всех** чертах `DESIGN` (а не только на `trait == "palette_value"`), используя общий хелпер `find_monochrome_color_word(text, allowed_names=entity_names, strict_palette=(item.trait == "palette_value"))` (см. WP-5.1).
   - В `repair_reason` обязательно сохранять подстроку `"monochrome_ink"` (контракт `test_illustration_reliability.py:696`).

#### 4.2. Расширение `resolve_appearance_with_design_fallback()` (`illustration_reference.py`)
1. Сигнатура с полной обратной совместимостью:
   ```python
   def resolve_appearance_with_design_fallback(
       explicit_facts: Sequence[Mapping[str, Any]],
       design_traits: Sequence[Mapping[str, Any]],
       *,
       excluded_facts: Sequence[Mapping[str, Any]] = (),
       compact_description_en: str = "",
       compact_description_ru: str = "",
   ) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
   ```
2. Расширить `_TRAIT_PATTERNS` универсальными визуальными словами (без привязки к конкретной книге):
   - `eyes`: добавить `sleep|asleep|sleeping|awake|woke|waking|closed|opened|open\s+eyes|спящ|во\s+сне|закрыт\w*\s+глаз|проснул`.
   - `attire_default`: добавить `blanket|blankets|swaddl\w*|wrapped|shawl|pajamas|pyjamas|nightgown|одеял|пелёнк|пеленк|свёртк|свертк|пижам` (строго **без** `bundle`, чтобы не сломать `test_illustration_reliability.py:782`!).
   - `atmosphere`: добавить `lamp|lamps|lantern|lanterns|streetlamp|streetlamps|lit|unlit|extinguished|darkness|night|dusk|dawn|фонар|темнот|ночн|сумерк|освещен`.
3. Правила подавления `trait_obj` в `resolve_appearance_with_design_fallback()`:
   - Собрать `active_blob = " ".join([explicit_blob, compact_description_en, compact_description_ru])`.
   - Собрать `excluded_blob = " ".join(f"{f.get('description_en', '')} {f.get('description_ru', '')}" for f in excluded_facts)`.
   - Для каждого `trait_obj` из `design_traits`:
     1. Если `pattern is not None and pattern.search(active_blob)` → **подавить** (уже покрыто актуальным описанием).
     2. Если `trait_name != "palette_value"` и `excluded_blob.strip()`:
        - если `pattern is not None and pattern.search(excluded_blob)` → **подавить** (категория была явно исключена `FACT_SCOPE` как устаревшая/противоречащая текущему моменту).
        - **лексическое эхо исключённых фактов (строгое, без ложных срабатываний):** извлечь предметные токены длиной **≥ 5** символов из `trait_obj["description_en"]` и `trait_obj["description_ru"]`, исключив явный стоп-лист общих прилагательных/размеров/тональности/родовых существительных (`_ECHO_STOPWORDS`: `short`, `long`, `dark`, `light`, `pale`, `tall`, `small`, `large`, `heavy`, `simple`, `plain`, `worn`, `thick`, `thin`, `average`, `neutral`, `figure`, `person`, `house`, `room`, `street`, `wearing`, `dressed`, `looking`, `коротк*`, `длинн*`, `тёмн*`, `темн*`, `светл*`, `высок*`, `прост*`, `обычн*`, `нейтральн*`); подавлять черту **только** если **≥ 2** предметных токена одновременно присутствуют в `excluded_blob` и **отсутствуют** в `active_blob`.

#### 4.3. Унифицированное разделение `FACT_SCOPE` и `DESIGN` до `PROMPT`, фильтрация по `used_design_keys` и чистка `assumptions_ru` (`illustration_chapter.py`)
1. Выделить чистую функцию `filter_non_assumption_placeholders(items: Sequence[str]) -> tuple[str, ...]` в `illustration_validation.py`:
   - Отбрасывает пустые строки и мета-заглушки вроде `«Допущений нет.»`, `«Нет допущений»`, `«Нет: ...»`, `«Без допущений»`, `«Не требуется»`, `«No assumptions»`, `«None»` (универсальный регистронезависимый шаблон на начало/всю фразу).
   - Устраняет дубликаты через `_dedupe_texts`.
2. **Унификация обеих веток `illustration_chapter.py` (`if not entity_facts:` строки `838-870` И ветки с фактами строки `1170-1207`):**
   - В ветке с фактами собрать `excluded_fact_objs = [all_registry_facts[fid] for fid in accumulated_excluded if fid in all_registry_facts]` и передать в `resolve_appearance_with_design_fallback(..., excluded_facts=excluded_fact_objs, compact_description_en=running_en, compact_description_ru=running_ru)`.
   - В **обеих** ветках (`if not entity_facts:` и с фактами) **НЕ склеивать** `kept_design` в `IllustrationAppearance.description_en` / `description_ru` / `assumptions_ru` до этапа `PROMPT`!
   - Формировать:
     - `base_app = IllustrationAppearance(...)` — на базе `running_ru`/`running_en` (или `fallback_en`, если `running_en` пуст) и очищенных `FACT_SCOPE`-допущений; если использован пустой `fallback_en`, добавлять `eid` в `base_fallback_entity_ids: list[str]`.
     - `available_design_traits`: список записей `{"key": f"{eid}:{d['trait']}", "entity_id": eid, "entity_kind": kind_bucket, "trait": d["trait"], "description_ru": d["description_ru"], "description_en": d["description_en"], "assumption_ru": d.get("assumption_ru", "")}`.
     - `entity_names_all`: все `canonical_name` и `aliases` участвующих сущностей момента (для `find_monochrome_color_word` на этапе `PROMPT`).
   - В `_build_prompt_scope_spec()`:
     - В `request_input` передавать `"available_design_traits": available_design_traits` (наряду с `characters`, `locations`, `allowed_fact_ids`, `allowed_design_keys`).
     - В `summary` сохранять `"design_traits_by_key"`, `"base_fallback_entity_ids"` и `"entity_names"`.
3. Материализация `PROMPT` (общая чистая функция `_materialize_appearances_for_prompt(base_chars, base_locs, design_traits_by_key, used_design_keys, base_fallback_entity_ids=())`):
   - Используется в `_validate_render_response(PROMPT)` (для сборки `prompt_en`) и при построении `ValidatedMoment` в `_advance_chapter_rendering()` (`1300-1325`).
   - Если `"design_traits_by_key"` отсутствует в `scope.summary` (старые сохранённые скоупы в БД или старые фикстуры тестов) — обратная совместимость: использовать `base_chars`/`base_locs` как есть.
   - Иначе для каждой сущности к `base_app` добавляются **только** черты из `design_traits_by_key`, чей ключ присутствует в `used_design_keys`; если `eid in base_fallback_entity_ids` и для `eid` выбраны черты из `used_design_keys`, заглушка `fallback_en` заменяется реальными выбранными чертами (если ни одной черты не выбрано — сохраняется `fallback_en`, чтобы `description_en` никогда не был пустым).
   - В `ValidatedMoment.assumptions_ru` попадают только прошедшие `filter_non_assumption_placeholders(...)` строки из `PROMPT.assumptions_ru`, `FACT_SCOPE.assumptions_ru` и `assumption_ru` **исключительно использованных** `used_design_keys`.
4. **Инвалидация кэша `CHAPTER` (`illustration_source.py`):**
   - В `compute_chapter_fingerprint()` обновить маркер `"kind": "illustration_chapter_v2"`, чтобы на существующих БД не переиспользовались старые кэшированные `CHAPTER`-скоупы со старой логикой `DESIGN`/`Visual details:`.

### Обязательные тесты WP-4
- `_build_bounded_design_input()`: при наличии `permanent`-фактов `episodic` и `state_change` не попадают в `facts`; при отсутствии `permanent` передаётся `non_permanent_context_only: True`.
- `_validate_design()`: хроматический цвет в `hair`/`attire_default` (не только `palette_value`) отклоняется при `monochrome_ink`, но имя персонажа (например, *Mr Brown* / *Violet*) не вызывает ложного отказа.
- `IllustrationDesignTraitSchema`: `assumption_ru=""` валидно парсится.
- `resolve_appearance_with_design_fallback()`:
  - Если `FACT_SCOPE` исключил младенческие факты (`excluded_facts`), а `compact_description_en` описывает 10-летнего мальчика, `DESIGN`-черты `age_appearance` (младенец), `attire_default` (одеяла), `eyes` (закрыты во сне) **подавляются**, а постоянная черта `palette_value` сохраняется.
  - Если `FACT_SCOPE` исключил потушенные фонари/кошку в темноте, `DESIGN.atmosphere` с потушенными фонарями подавляется.
  - Сохраняются все существующие проверки `test_illustration_reliability.py:779-784` (`bundle`/`managed`/`parties`/`big eyes`).
- `IllustrationChapter` (`PROMPT` + `ValidatedMoment`):
  - Если модель `PROMPT` не включила ключ `ent-1:eyes` в `used_design_keys`, текст этой черты и её `assumption_ru` **не** попадают ни в `prompt_en`, ни в `character_descriptions`, ни в `assumptions_ru`.
  - Заглушки «Допущений нет.», «Нет: указано в фактах.» удаляются из итоговых `assumptions_ru`.

---

## WP-5. Ахроматическая проекция `monochrome_ink` в `FACT_SCOPE`/`PROMPT` и дедупликация `Visual details:`

### Затрагиваемые файлы
- `src/linguary/application/illustration_validation.py`
- `src/linguary/application/illustration_context.py`
- `src/linguary/application/illustration_reference.py` (переиспользование общего хелпера проверки цветов)
- `src/linguary/application/illustration_chapter.py`
- `tests/unit/application/test_illustration_chapter.py`
- `tests/unit/application/test_illustration_reliability.py`

### Контракт

#### 5.1. Общий хелпер проверки монохромности с защитой собственных имён (`illustration_validation.py`)
1. Перенести `_MONOCHROME_COLOR_EN_RE` и `_MONOCHROME_COLOR_RU_RE` из `illustration_reference.py` в `illustration_validation.py` (импортируя их в `illustration_reference.py` и `illustration_chapter.py` — единый источник истины).
   - Разделить два профиля проверки:
     - **`strict_palette=True`** (для `DESIGN.palette_value`): полный текущий палитровый набор (включая `bronze`, `copper`, `gold`, `golden`, `amber`, `cream`, `olive`, `chestnut`, `sepia`, `ochre` и русские эквиваленты).
     - **`strict_palette=False`** (для `FACT_SCOPE`, `PROMPT` и не-палитровых черт `DESIGN`): чистый хроматический набор (без `bronze`/`copper`, и с предварительным маскированием устойчивых предметных/природных сочетаний вроде `ice cream`, `whipped cream`, `olive tree/branch/oil`, `chestnut tree`, `orange tree`, чтобы не было ложных отказов на физических предметах и металлах в сценах!).
2. Добавить чистую функцию:
   ```python
   def find_monochrome_color_word(
       text: str,
       *,
       allowed_names: Sequence[str] = (),
       strict_palette: bool = False,
   ) -> str | None:
   ```
   - Безопасное маскирование собственных имён (без дыр для строчных прилагательных!):
     1. Полные строки из `allowed_names` (отсортированные по убыванию длины) маскируются в копии `text` по границам слов **регистронезависимо** (`"Mr Brown"`, `"Green Gables"`, `"Violet"`).
     2. Отдельные слова длиной ≥ 3 символов извлекаются **только из слов, начинающихся с заглавной буквы** в `allowed_names`, и маскируются в `text` **регистрозависимо** (только если в `text` это слово тоже написано с заглавной/прописными: `Brown`, `Violet`, но **НЕ** строчные `"brown coat"`, `"green eyes"`, `"red scarf"`!).
   - Ищет совпадение по соответствующему профилю EN/RU regex в очищенной строке; возвращает первое найденное слово или `None`.

#### 5.2. Стиль `monochrome_ink` в `FACT_SCOPE` и `PROMPT` (`illustration_chapter.py`, `illustration_context.py`)
1. В `_build_fact_scope_spec()`, `_build_fs_probe()` и `_build_consol_probe()` добавить `"style": revision.settings.style` в `request_input` (и сохранить `"entity_names"` — `[canonical_name, *aliases]` — в `summary` для `allowed_names`).
2. В `_STAGE_INSTRUCTIONS[IllustrationScopeStage.FACT_SCOPE]` (`illustration_context.py:356-360`):
   - Уточнить: `compact_description_ru` и `compact_description_en` описывают **только визуальный облик** сущности на момент события (без пересказа диалога и сюжетных действий).
   - Если `style == "monochrome_ink"`, переводить цветовые признаки исходных фактов в **светотень, тональность и фактуру** (например, светлый/тёмный/насыщенно-тёмный тон, контраст, металлический блеск, яркое свечение) и **не** использовать названия хроматических цветов (`blue`, `purple`, `emerald`, `orange`, `red`, `green` и т. п.) в `compact_description_en`/`compact_description_ru`.
   - Не писать заглушки в `assumptions_ru` (оставлять `[]`, если допущений нет).
3. В `_STAGE_INSTRUCTIONS[IllustrationScopeStage.PROMPT]` (`illustration_context.py:373-380`):
   - Добавить 1 фразу: при `style == "monochrome_ink"` не использовать названия хроматических цветов в `scene_composition_en`, выражая визуальные различия через светлоту, контраст и фактуру.
4. В `_validate_render_response()` (`illustration_chapter.py`):
   - Для `FACT_SCOPE` при `revision.settings.style == ILLUSTRATION_STYLE_MONOCHROME_INK`:
     - Проверить **`payload.compact_description_en`** через `find_monochrome_color_word(..., allowed_names=entity_names, strict_palette=False)`. *(Русское `compact_description_ru` не блокируется, т. к. не попадает в `prompt_en` и может описывать противоречивые исходные факты для карточки UI — сохраняется контракт `test_contradictory_facts_preserved_separately` в `test_illustration_chapter.py:749-756`!)* При обнаружении цвета — вернуть `ok=False` с понятным `repair_reason`: перевести хроматический цвет в тональность/светлоту/фактуру для стиля `monochrome_ink`.
   - Для `PROMPT` при `revision.settings.style == ILLUSTRATION_STYLE_MONOCHROME_INK`:
     - Проверить `payload.scene_composition_en` через `find_monochrome_color_word(..., allowed_names=all_moment_entity_names, strict_palette=False)` и при обнаружении цвета вернуть `ok=False` с тем же `repair_reason`.
   - *(Исходные факты в `EXTRACT` и `illustration_registry_items` остаются цветными и точными — никаких потерь информации для будущих стилей!)*

#### 5.3. Клаузная дедупликация `Visual details:` и обновление `validate_moment()` (`illustration_validation.py`)
1. В `compose_monochrome_prompt(scene_composition_en, appearances, style_profile=...)`:
   - Выделить чистый детерминированный хелпер `_clause_covered_in_text(clause: str, reference_text: str) -> bool`:
     - Извлекать значимые токены строго через `re.findall(r"[a-z]{4,}", text.casefold())`, исключая минимальный стоп-лист служебных слов (`with`, `from`, `that`, `this`, `have`, `been`, `into`, `over`, `near`, `very`, `wear`, `wearing`, `made`, `like`).
     - Токен `tok` считается присутствующим в множестве `ref_tokens` из `reference_text` при **точном совпадении** `tok in ref_tokens`, либо (для слов длиной ≥ 5) при простом единственном/множественном соответствии (`tok.endswith("s") and tok[:-1] in ref_tokens`, `f"{tok}s" in ref_tokens`, `tok.endswith("es") and tok[:-2] in ref_tokens`, `f"{tok}es" in ref_tokens`) — **никакого** префиксного сравнения (чтобы `scar` никогда не совпадало со `scarf`/`scared`).
     - Клауза считается уже покрытой в `reference_text`, если **либо** нормализованная клауза входит в `reference_text` целиком, **либо** в клаузе ≥ 2 значимых токена и **все** (100%) её значимые токены присутствуют в `ref_tokens`.
   - Для каждого образа `item` в `appearances`:
     - Разбить `desc_clean` по `;` на непустые клаузы.
     - Оставить в `remaining_clauses` только те клаузы, которые **не** покрыты в `comp` (`scene_composition_en`).
     - Если `remaining_clauses` не пуст: добавить `f"{name}: {'; '.join(remaining_clauses)}"` (или `'; '.join(remaining_clauses)` если имя пустое/кириллическое) в `appearance_lines`.
     - Если все клаузы всех образов уже естественно встроены в `scene_composition_en`, секция `Visual details:` не дублируется вообще!
2. В `validate_moment()` (`illustration_validation.py:377-391`):
   - Выделить чистый хелпер `_appearance_reflected_in_prompt(desc_clean: str, prompt_lower: str) -> bool`:
     - Если `desc_clean.casefold() in prompt_lower` → `True` (быстрый путь, полная обратная совместимость).
     - Иначе разбить `desc_clean` по `;` на непустые клаузы: **каждая** клауза должна быть покрыта в `prompt_lower` через тот же `_clause_covered_in_text(clause, prompt_lower)`.
   - Заменить строгую проверку `if desc_clean.casefold() not in prompt_lower:` на `if not _appearance_reflected_in_prompt(desc_clean, prompt_lower):`.
   - **Проверка безопасности:**
     - `test_id_only_prompt_without_description_en_rejected` (`"ent-alice raises a lantern."` при `desc="Tall figure in a grey cloak"`): ни `tall`, ни `figure`, ни `grey`, ни `cloak` не входят в промпт → **отклонён** как и прежде!
     - `test_prompt_contains_actual_scoped_descriptions_and_monochrome_directive` и `test_compose_monochrome_prompt_uses_name_en_and_avoids_cyrillic`: детали отсутствуют в `comp`, поэтому попадают в `Visual details:` → **проходят без изменений**!

### Обязательные тесты WP-5
- `find_monochrome_color_word()`:
  - ловит `blue eyes`, `purple cloak`, `glow orange`, `изумрудный плащ`;
  - пропускает `Mr Brown stood by the door`, `Violet looked up`, `Green Gables farmhouse` при передаче соответствующих `allowed_names`, но **НЕ** пропускает строчные `brown coat` / `green dress` при `allowed_names=("Mr Brown", "The Woman in Green")`;
  - пропускает ахроматические `black`, `white`, `grey`, `silver`, `pale`, `dark` и физические предметы (`copper kettle`, `bronze statue`, `ice cream`) при `strict_palette=False`.
- `FACT_SCOPE` validation при `monochrome_ink`: отклоняет `compact_description_en` с `purple cloak` (возвращает `repair_reason`), принимает `deep-toned cloak`, и сохраняет `test_contradictory_facts_preserved_separately` зелёным.
- `compose_monochrome_prompt()` + `validate_moment()`:
  - Если `scene_composition_en` уже содержит первую клаузу `description_en` (`lightning-shaped scar on forehead`), а вторая клауза (`dark ink crosshatching`) отсутствует, в `Visual details:` попадает **только** вторая клауза, и `validate_moment()` принимает такой момент.
  - Если все клаузы уже в `scene_composition_en`, `Visual details:` не дублируется, и `validate_moment()` принимает момент.
  - Если `prompt_en` ссылается только на `ent-alice` без визуальных деталей — `validate_moment()` отклоняет момент.

---

## WP-6. Blocking в `ALIAS`, `alias_of_hint` в `EXTRACT` и лемматизация `glossary_hints` в A2

### Затрагиваемые файлы
- `src/linguary/application/illustration_reference.py`
- `src/linguary/application/illustration_context.py`
- `src/linguary/application/reader_text.py`
- `tests/unit/application/test_illustration_reference.py`
- `tests/unit/application/test_context_builder.py`

### Контракт

#### 6.1. Blocking в `ALIAS` и подсказка `alias_of_hint` в `EXTRACT`
1. В `illustration_reference.py`:
   - Расширить `_BLOCKING_STOPWORDS` универсальными родовыми и родственными словами (EN/RU), чтобы исключить ложные пары по словам `son`, `house` и т. д.:
     ```python
     _BLOCKING_STOPWORDS = frozenset(
         {
             "the", "of", "and", "in", "at", "on", "to", "a", "an", "de", "la", "le",
             "son", "daughter", "mother", "father", "brother", "sister", "wife", "husband",
             "man", "woman", "boy", "girl", "child", "baby", "person", "people",
             "house", "room", "street", "road", "door", "building", "place",
             "сын", "дочь", "мать", "отец", "брат", "сестра", "жена", "муж",
             "мужчина", "женщина", "мальчик", "девочка", "ребёнок", "ребенок", "человек",
             "дом", "комната", "улица", "дорога", "дверь", "здание", "место",
         }
     )
     ```
   - В `_blocking_keys_for_mention()` для каждого значимого токена `tok` (как из `name`, так и из `alias_of_hint`) с `len(tok) >= 3 and tok not in _BLOCKING_STOPWORDS`:
     - добавлять `f"{kind}:tok:{tok}"`;
     - если `len(tok) >= 4 and tok.endswith("s") and not tok.endswith(("ss", "us", "is"))`:
       вычислить `singular = tok[:-1]`; если `len(singular) >= 3 and singular not in _BLOCKING_STOPWORDS`:
       дополнительно добавить `f"{kind}:tok:{singular}"`.
   - **Оценка сложности и безопасности:** число ключей на упоминание растёт не более чем на +1 на множественный токен (O(1)), никаких O(N²) пар не создаётся; все члены одной семьи (`Mr Smith`, `Mrs Smith`, `John Smith`, `the Smiths' son`) попадают в один бакет по фамилии `smith` и надёжно разделяются вердиктом LLM (`same_entity` / `different`) на этапе `ALIAS`.
2. В `_STAGE_INSTRUCTIONS[IllustrationScopeStage.EXTRACT]` (`illustration_context.py:327-334`):
   - Добавить 1 предложение: «Если в ближайшем тексте описательное или родственное обозначение прямо соотносится с названным персонажем или местом (например, „сын X по имени Y“), укажи это имя в `alias_of_hint`.»

#### 6.2. Снятие противоречия лемматизации `glossary_hints` в A2 (`reader_text.py`)
1. В `learner_instructions(reader_level)` (`src/linguary/application/reader_text.py:531-560`):
   - **НЕ МЕНЯТЬ** текст, возвращаемый для `reader_level == "b2"` и `reader_level == "c1"` (гарантируем сохранность всех эталонных шаблонов B2!).
   - В ветке `a2`: заменить строку `Цитаты, expression, original и original_excerpt сохраняй дословно.` на:
     `Цитаты, expression и original_excerpt сохраняй дословно; в glossary_hints.original глаголы приводи к инфинитиву с to, остальные слова и выражения сохраняй как в тексте.`
2. В `_A2_NOTE_DISCIPLINE` (`reader_text.py:512`):
   - Дополнить правило для `glossary_hints`: «Выноси в `glossary_hints` отдельные трудные слова и устойчивые словосочетания, а не перевод целого предложения целиком.»

### Обязательные тесты WP-6
- `build_alias_candidate_pairs()`:
  - `("Dudley Dursley", "the Dursleys’ son")` (или нейтральный пример `("John Smith", "the Smiths’ son")`) **образует** кандидатную пару через сингуляризованный ключ `smith`!
  - Две несвязанные описательные фразы `("the Smiths' son", "the Bakers' son")` **НЕ** образуют пару (токен `"son"` в `_BLOCKING_STOPWORDS`).
  - Слова на `-ss`/`-us`/`-is` (`"Ross"`, `"Marcus"`, `"Iris"`) не усекаются до обрубков.
- `learner_instructions("a2")` не содержит противоречащей фразы «original ... сохраняй дословно» рядом с правилом инфинитива `to`; `test_baseline_b2_templates_sha256_unchanged()` проходит без изменений.

---

## Матрица рисков и проверка совместимости

| Область риска | Потенциальная проблема | Защита в спецификации |
|---|---|---|
| Замороженные SHA-256 промптов | Изменение `_TRANSLATION_INSTRUCTIONS` или B2/`legacy-v1` сломает `test_baseline_b2_templates_sha256_unchanged` | Править только `_TRANSLATION_CONTEXT_V2`, `_CONTENT_PLAN_PROCESSING_V3`, ветку `a2` в `learner_instructions`, `_A2_NOTE_DISCIPLINE` и `description` в `SegmentTranslationSchema` |
| Литеральные проверки `_SEGMENT_TRANSLATION_RULE` | `test_translation_context.py:2271,2447,2663` сравнивают строку целиком | `_SEGMENT_TRANSLATION_RULE` не изменяется |
| Ложные отказы на «цветных» именах в `monochrome_ink` | Персонажи *Mr Brown*, *Violet*, *Amber*, топонимы *Green Gables* | `find_monochrome_color_word()` маскирует `allowed_names` (каноническое имя + алиасы) перед проверкой |
| Ослабление `validate_moment()` | ID-only промпты могут проскочить | `_appearance_reflected_in_prompt()` требует покрытия **каждой** клаузы значимыми токенами; тест `test_id_only_prompt_without_description_en_rejected` обязан оставаться зелёным |
| Старые сохранённые `illustration_scopes` в SQLite | В старых `scope.summary` нет `design_traits_by_key` или `entity_aliases` | Все новые ключи `summary` читаются через `.get(...)` с безопасным fallback на прежнее поведение; **0 миграций Alembic** |
| Чистые слои Clean Architecture | Запрещены внешние импорты и I/O в `domain`/`application` | Все изменения в `application` используют только stdlib (`re`, `threading`, `collections.abc`, `typing`); проверка `uv run pytest tests/architecture` обязательна |
