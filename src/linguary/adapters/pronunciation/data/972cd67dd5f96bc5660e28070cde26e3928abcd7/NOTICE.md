# Данные произношения ipa-dict

Исходные файлы `en_UK.txt` и `en_US.txt` поставлены без изменений из
[ipa-dict, ревизия 972cd67dd5f96bc5660e28070cde26e3928abcd7](https://github.com/open-dict-data/ipa-dict/tree/972cd67dd5f96bc5660e28070cde26e3928abcd7).
Linguary добавляет манифест контрольных сумм и эти уведомления. Сами словоформы,
фонемы и ударения в исходных файлах не исправлены и не дополнены.

- en_UK.txt — GPL-3.0. Британский набор основан на `ipacards` Леонардо Бойко
  (Leonardo Boiko; в исходном проекте — `leoboiko`, репозиторий — `melissaboiko/ipacards`).
- en_US.txt — MIT. Американский набор основан на `CMUdict-IPA` Линляна Чжана
  (Lingliang Zhang), данные которого происходят из CMUdict Университета Карнеги — Меллона.
  Ударения добавлены с использованием `syllabify` Кайла Гормана (Kyle Gorman, MIT).
  Уведомление и условия исходного CMUdict также сохранены.
- Общие материалы ipa-dict: Copyright (c) 2016 dohliam, MIT. Общая лицензия
  репозитория не заменяет указанные условия отдельных наборов.

Полные применимые тексты и авторские уведомления находятся в `LICENSES.txt` рядом
с этим файлом. Оригинальные словарные исходники поставляются в том же каталоге;
для проверки и повторного получения доступны закреплённые ссылки:

- UK: https://raw.githubusercontent.com/open-dict-data/ipa-dict/972cd67dd5f96bc5660e28070cde26e3928abcd7/data/en_UK.txt
- US: https://raw.githubusercontent.com/open-dict-data/ipa-dict/972cd67dd5f96bc5660e28070cde26e3928abcd7/data/en_US.txt
- Манифест версии: `ipa-dict@972cd67dd5f96bc5660e28070cde26e3928abcd7:parser-1`.

## Происхождение и ограничения

Наборы содержат написание и IPA; в них нет значений слов, частей речи и
грамматических меток. Несколько чтений могут находиться в одном поле. Наличие
единственной IPA в записи не доказывает полноту вариантов или правильность
чтения в конкретном предложении.

Описание ipacards сообщает, что IPA автоматически получена через eSpeak и может
содержать ошибки. Автор пытался исключить омографы с несколькими чтениями, но
предупреждает, что некоторые могли остаться. Эти ограничения качества сохраняются
и при использовании производного набора; размер исходной учебной колоды не
описывает размер файла ipa-dict.

Исходный `CREDITS` ipacards указывает eSpeak, Google Translate, Awesome TTS,
Sublex-UK, wordlist.sourceforge.org, The Briticiser и списки омографов John Higgins,
Jon Vahsholtz и Wiktionary. Аудио и программы этих проектов в этот снимок не входят.

README ipa-dict не фиксирует ревизии исходных преобразующих проектов. Следующие
закреплённые ссылки документируют тексты уведомлений, сохранённые 2026-10-31;
они не утверждают точные версии, которыми когда-то преобразовали словарь:

- CMUdict-IPA: https://github.com/lingz/cmudict-ipa/blob/629483e9a7fa5b9ff11bb83e3fda6dcc25c5c511/LICENSE
- syllabify: https://github.com/kylebgorman/syllabify/blob/d816db784436e9de87ec1ef9bd11b8e229853710/syllabify.py
- CMUdict: https://github.com/cmusphinx/cmudict/blob/74790861f652b15e4ac49015a90074ad62a27690/LICENSE
- ipacards: https://github.com/melissaboiko/ipacards/blob/ac25f24df0712562b3c4762e21880fced2ae0b92/LICENSE
- Оговорки ipacards: https://github.com/melissaboiko/ipacards/blob/ac25f24df0712562b3c4762e21880fced2ae0b92/deck_description.md

Для распространения UK-данных и их производных материалов действуют условия
GPL-3.0, приведённые в `LICENSES.txt`. Эти уведомления описывают добавленные
словарные данные и сами по себе не определяют правовой режим целого произведения,
в которое включены подсказки.
