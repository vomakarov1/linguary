# P7: проверка обновления области чтения в браузере

Дата: 2026-10-30T09:54:13+02:00. Результат: **PASS, 21 проверка**.

## Среда и способ проверки

- Рабочее дерево: `/Users/vladislavmakarov/www/linguary/.worktrees/pronunciation-hints`.
- ОС: `macOS-26.6.2-arm64-arm-64bit-Mach-O`. Python `3.13.7`.
- Настоящий Chromium: `Chrome/156.0.8078.4`, CDP `1.3`, V8 `15.6.75`.
- Исполняемый файл: `/Users/vladislavmakarov/Library/Caches/ms-playwright/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing`.
- Режим: `--headless=new --window-size=1200,900`, отдельный временный профиль. Управление через имеющийся `websockets` и Chrome DevTools Protocol. Playwright и новые зависимости не устанавливались; конфигурация проекта не менялась.
- Страница открывалась по HTTP на `http://127.0.0.1:59316/books/book-web-1/result`. Локальный `ThreadingHTTPServer` передавал запросы настоящему FastAPI-приложению через `TestClient`; CSS, JavaScript, полная страница и HTML-фрагменты — из рабочего дерева. Посредник позволял задержать заголовки или тело ответа, однократно вернуть 503 и неверный HTML.
- Данные: отдельная временная SQLite, три главы с вымышленным словом `zorp`, синтетическими IPA и атрибуцией `Invented test data`. Результаты и статусы задач менялись в тестовом хранилище, POST выполнял настоящий маршрут перегенерации. `FakeLLMClient`, фоновый обработчик выключен; словарь и модель не вызывались, производственный источник произношений не включался. Docker не запускался.
- Команда: `uv run python /tmp/p7-browser-check.py`. Это временный диагностический сценарий, постоянные зависимости или конфигурация для него не добавлялись. Структурированный результат текущего запуска: `/tmp/p7-browser-results.json`.

## Сценарии и результаты

| Проверка | Что выполнено | Итог |
| --- | --- | --- |
| `initial_later_primary` | Исходный снимок: первичная IPA во второй главе, повтор в третьей | PASS |
| `browser_pronunciation_add_early_demotes_later` | Добавление IPA в первую главу: поздняя первичная подсказка становится повтором | PASS |
| `browser_pronunciation_remove_early_promotes_later` | Удаление ранней IPA: первичная подсказка переносится во вторую главу | PASS |
| `browser_pronunciation_change_ipa_splits_primary` | Смена ранней IPA: две первичные подсказки для разных IPA, повтор ведёт ко второй | PASS |
| `repeat_anchor_navigation` | Переход по ссылке повтора: URL и фокус указывают на единственную цель | PASS |
| `browser_refresh_preserves_focus_scroll_and_open_guide` | Замена сохраняет исходный текст, открытую памятку, фокус summary и прокрутку | PASS |
| `focus_form_action_survives` | Фокус кнопки восстанавливается по action формы участника | PASS |
| `focus_repeat_target_survives` | Фокус ссылки повтора восстанавливается по целевому hint_id | PASS |
| `keyboard_tab_space_and_focus_restoration` | Настоящие Tab и Space: фокус памятки, раскрытие и сохранение после замены | PASS |
| `browser_parallel_regeneration_ignores_stale_response` | Два POST: задержанный ответ до заголовков отклоняется; принимается только свежий снимок | PASS |
| `browser_parallel_regeneration_ignores_stale_body` | Два POST: устаревание во время response.text() не откатывает DOM | PASS |
| `browser_groups_refresh_fail_keeps_previous_dom` | HTTP 503: прежний узел и его outerHTML сохранены, повторная попытка успешна | PASS |
| `browser_groups_refresh_invalid_keeps_previous_dom` | Неверный корень HTML: прежний узел и его outerHTML сохранены, повтор успешен | PASS |
| `detached_root_is_never_replaced` | Отсоединённый корень: поздний ответ не меняет старый узел и новое дерево | PASS |
| `continued_polling_after_no_ipa_completion` | Завершение одной задачи без IPA: опрос второй активной задачи продолжается | PASS |
| `continued_polling_until_other_task_fails` | Ошибка оставшейся задачи: получен общий terminal-снимок, дальнейшие GET прекращаются | PASS |
| `export_header_polling_continues` | Независимый опрос экспорта продолжается во время замены групп | PASS |
| `completion_after_snapshot_keeps_next_poll` | Завершение после чтения снимка: активный снимок приводит к следующему опросу | PASS |
| `instant_post_completion_uses_terminal_book_snapshot` | Завершение до первого GET после POST: получен terminal-снимок одним запросом | PASS |
| `domcontentloaded_resumes_pending_tasks` | Загрузка страницы с уже активной задачей: общий опрос возобновляется | PASS |
| `single_groups_request_and_timer` | Максимум один запрос области одновременно; нет дополнительного опроса после terminal-состояния | PASS |

Обязательные сценарии SDD `browser_pronunciation_add_remove_change_primary_updates_all_groups`, `browser_parallel_regeneration_ignores_stale_response`, `browser_groups_refresh_failure_keeps_previous_dom` и `browser_refresh_preserves_focus_and_scroll` покрыты строками выше. Это исполнение JavaScript в Chromium, а не вывод из проверок HTML.

## Наблюдаемые свидетельства

- За успешный запуск: **11 POST**, **31 GET result-groups**, **28 GET result-header**; максимум одновременных GET result-groups — **1**.
- До добавления ранней IPA первичная подсказка находится во второй главе. После добавления — одна первичная в первой и два повтора. После удаления — одна первичная во второй и один повтор. После смены IPA — две различные первичные подсказки и один повтор к соответствующей IPA. Для каждого повтора проверено ровно одно совпадение целевого `id`.
- Для обоих порядков задержки ответа `MutationObserver` фиксировал **ровно одну замену области**, уже со свежим `ph-latest-response` либо `ph-latest-body`; промежуточный устаревший снимок не вставлялся.
- При 503 и неверном HTML одновременно проверены идентичность узла старой области и точное совпадение её `outerHTML`. После сбоя автоматическая попытка получила свежую область.
- Открытая памятка, её фокус и `scrollY=410` сохранились после замены; отдельно проверены форма по `action`, ссылка по `hint_id` и реальные события клавиатуры Tab/Space. Массив исходных текстов абзацев до/после совпал дословно.
- Пока второй участник оставался RUNNING, первый уже завершился без IPA, но общий опрос продолжался. После FAILED у последнего участника следующая область стала terminal; за последующие две секунды дополнительных GET области не было. Опрос заголовка при этом продолжался отдельно.

## Проверки Python и синтаксиса

- Подход с тестами сначала: новые серверные проверки выявили отсутствие корня области; новая проверка версии JS падала на `v=2` до правки.
- `uv run pytest tests/integration/web/test_result_and_export.py -q` — **41 passed**.
- `uv run pytest tests/integration/web/test_templates.py tests/integration/web/test_result_and_export.py -q` — **91 passed**, одно существующее предупреждение об устаревшем использовании `httpx` в Starlette TestClient.
- `uv run pytest` — **3004 passed, 1 skipped, 1 warning**, 153.97 секунды.
- `node --check src/linguary/web/static/linguary.js` и `git diff --check` — успешно.

## Границы проверки и остаток

Проверен Chromium на macOS; другие браузеры и настоящий скринридер в этом запуске не использовались. Состояния обработки задавались синтетически в тестовом хранилище: это проверка HTTP/DOM/JavaScript, а не работы производственного словаря или модели. Предел 120 попыток не истощался в браузере; проверены продолжение, повтор после ошибки и прекращение опроса при завершении задач.

Q-1/Q-2/Q-3 и ранее незавершённая проверка скринридера **не закрываются** этим отчётом. Невыполненных обязательных сценариев P7 не осталось.
