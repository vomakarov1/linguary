# BRIEF: Смягчение `reason` при подтверждении коррекции, устранение цикла `GLOSSARY_ONLY` и устойчивость этапа `EXTRACT`

**Дата:** 2026-11-26  
**Статус:** Approved (после глубокого ревью спецификаций и устранения всех замечаний)  
**Контекст:** Пост-анализ полного прогона Round 4 (`hp-ch1-ch2p5-20261126-r4` на `main` `3496d8c`).

---

## 1. Проблема

Прогон Round 4 подтвердил работоспособность основного конвейера от импорта до EPUBCheck и связных иллюстраций, но выявил три неблокирующих дефекта надёжности, которые вызвали лишние повторы, ручной перезапуск перевода и ручной повтор иллюстраций:

1. **Противоречие промпта и схемы в `chapter_correction_review` (`reason: ""` при `verdict: "confirmed"`):**
   - `CORRECTION_REVIEW_PROMPT` (`src/linguary/composition/correction_runtime.py:96-98`) просит указывать причину только при `needs_human`, а `strict_json_schema()` вырезает `minLength` из JSON-схемы для провайдера.
   - Модель корректно следует инструкции и при подтверждённом кандидате возвращает `verdict: "confirmed"` с `reason: ""` (3 из 7 вызовов: `#29`, `#48`, `#50`), но `CorrectionVerdictSchema.reason: str = Field(min_length=1)` и `CorrectionVerdict.__post_init__` отклоняют пустую строку, тратя 3 лишних повтора.
2. **Зацикливание перевода на элементарном сегменте `GLOSSARY_ONLY` (абзац `b#5`, 9 из 11 отказов перевода):**
   - На сегментированном смешанном блоке `b#5` (`7070b3ea`) педагогический анализ пометил элементарное предложение уровня A1–A2 (*«The Dursleys knew that the Potters had a small son, too, but they had never even seen him.»*) режимом `GLOSSARY_ONLY` без единой привязанной трудности и без лексики выше A2.
   - `normalize_pedagogical_response()` (`analysis_merge.py`) проверяет повышение `READ_AS_IS` → `GLOSSARY_ONLY`/`TRANSLATE`, но никогда не понижает пустой элементарный сегмент `GLOSSARY_ONLY` обратно в `READ_AS_IS`.
   - При этом в промптах перевода (`_CONTENT_PLAN_PROCESSING_V3` и `_SEGMENT_TRANSLATION_RULE`) не сказано, что для смешанного блока с режимом `TRANSLATE` и сегментом `GLOSSARY_ONLY` обязательно вернуть 1–3 `glossary_hints` именно из сегмента `GLOSSARY_ONLY` (что также вызвало первичный отказ на сложном `b#106`), а диагностика в `validators/treatment.py` не указывает индекс сегмента `GLOSSARY_ONLY`. В итоге переводчик исчерпал `scope_limit`, потребовал ручной `restart-translation` и был вынужден добавить тривиальные подсказки `even`/`too`.
3. **Хрупкость этапа `EXTRACT` в `illustration_reference` (`extract:p0002` упал 4 раза из 5, `extract:p0001` — 1 раз):**
   - `plan_scopes()` (`illustration_reference.py:815-858`) склеивает сцены всех глав в единый список и упаковывает до 64 юнитов в 23.4 КБ. В Round 4 это дало 3 огромных пакета; `extract:p0002` объединил 4 сцены (31 блок) через границу глав 1 и 2 при лимите схемы `_MAX_LIST = 32` на списки упоминаний и фактов, провоцируя деградацию рассуждений (12–19К токенов) и ошибки `subject_mention_id`, `covered_source_unit_ids` и сдвиг UUID блока.
   - В отличие от `CANDIDATES` (`illustration_selection.py:1213-1226, 1303-1320`), `IllustrationReference.run_task()` не запускает автодробление `_handle_post_send_split()` для составного `EXTRACT`-скоупа после исчерпания `max_repair_turns` на ошибках формата или валидации, сразу переводя скоуп в `failed`.
   - Инструкция `_STAGE_INSTRUCTIONS[EXTRACT]` не формулирует инвариант `facts[].subject_mention_id ∈ mentions[].mention_id`, а `resolve_quote_anchor()` не умеет детерминированно сопоставлять типографские кавычки/тире/сжатые пробелы.

---

## 2. Цели

1. **WP1 (`chapter_correction_review`):** разрешить пустой `reason` (`""`) при фактически подтверждённом вердикте (`confirmed` и все 4 булевых флага `True`), сохранив строгое требование непустого `reason` во всех случаях `not confirmed` (`needs_human` или ложный булев флаг) и понятную безопасную подсказку при ошибке схемы.
2. **WP2 (элементарные `GLOSSARY_ONLY`-сегменты и смешанные блоки):**
   - Детерминированно понижать пустые элементарные сегменты `GLOSSARY_ONLY` (без привязанных трудностей, без небазовой лексики A2, фразовых глаголов и стоп-листа) в смешанных сегментированных блоках до `READ_AS_IS` в `normalize_pedagogical_response()`.
   - Уточнить в промптах анализа (`_reader_level_treatment_guidance("a2")`) и перевода (`_CONTENT_PLAN_PROCESSING_V3`, `_SEGMENT_TRANSLATION_RULE`) правила выбора `GLOSSARY_ONLY` и обязательность `glossary_hints` для сегментов `GLOSSARY_ONLY` в смешанных блоках.
   - Указывать точный индекс сегмента `GLOSSARY_ONLY` в сообщениях валидатора `validators/treatment.py`.
3. **WP3 (устойчивость этапа `EXTRACT`):**
   - Разделять пакеты `EXTRACT` строго по границам секций/глав с умеренным бюджетом (не более 12 КБ, не более 16 блоков и не более 2 сцен в пакете; до 16 одноблочных юнитов для глав без сцен).
   - Подключить автодробление `_handle_post_send_split()` для составных `EXTRACT`-скоупов при исчерпании `max_repair_turns` (или лимита активации) на `LLMResponseFormatError` и ошибках семантической валидации — по образцу `CANDIDATES`.
   - Явно зафиксировать контракт `subject_mention_id` в `_STAGE_INSTRUCTIONS[EXTRACT]` и уточнить текст ремонтной диагностики в `_validate_extract()`.
   - Добавить детерминированный типографско-пробельный запасной поиск в `resolve_quote_anchor()` с сохранением точной дословной цитаты `block.text[s_start:s_end]` и проверкой однозначности (`ambiguous`).

---

## 3. Осознанные не-цели (Non-goals)

- **0 миграций БД / Alembic:** схема SQLite остаётся без изменений; все механизмы дробления скоупов `EXTRACT`, хранения `CorrectionVerdict` и `TreatmentPlan` уже поддерживаются текущей схемой.
- **Замороженные SHA-256 шаблоны не трогаем:** `_TRANSLATION_INSTRUCTIONS` и `snapshot_prompt_templates(..., reader_level="b2", profile="legacy-v1")` сохраняют точные SHA-256 дайджесты.
- **Инвариант `GLOSSARY_ONLY` в домене не ослабляем:** `BlockTreatment.GLOSSARY_ONLY` по-прежнему требует `>= 1` словарной подсказки в `domain/treatment.py`, `domain/processing.py` и `validators/treatment.py` — пустые элементарные сегменты устраняются на этапе планирования (`analysis_merge.py`), а не молчаливым принятием `glossary_hints: []`.
- **Одиночные несегментированные блоки `GLOSSARY_ONLY` не перекраиваем:** чтобы не нарушать существующие регрессионные контракты (например, короткие реплики), понижение пустых `GLOSSARY_ONLY` ограничиваем смешанными сегментированными блоками (где есть и `TRANSLATE`, и `GLOSSARY_ONLY`).
- **`covered_source_unit_ids` не отключаем:** проверка полноты покрытия юнитов в `EXTRACT` остаётся строгой, так как она отсекает галлюцинации и пропуски сцен.
- **Формат `block_id` в `EXTRACT` не меняем:** не вводим алиасы `b1..bN` в проволочный JSON `EXTRACT`, чтобы сохранить полную обратную совместимость с существующими тестами и сериализацией.

---

## 4. Критерии приёмки

- `uv run pytest tests/architecture` — 100% зелёный (границы Clean Architecture не нарушены).
- `uv run pytest -m "not slow"` — 100% зелёный, включая `test_baseline_b2_templates_sha256_unchanged`.
- Новые модульные тесты покрывают все три пакета (WP1, WP2, WP3) и их граничные случаи (отклонение пустого `reason` при `needs_human` или `False`-флаге, сохранение `GLOSSARY_ONLY` при наличии небазовой лексики или трудности, автодробление `EXTRACT` и защита `max_split_depth`, типографский и неоднозначный поиск в `resolve_quote_anchor`).
