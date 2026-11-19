"""A conservative policy for final learner text; it makes no semantic guarantees."""
from dataclasses import replace
import re
import unicodedata

from linguary.application.ports.language_detection import LanguageDetector
from linguary.application.retry_policy import ResultValidationError
from linguary.domain import BlockProcessingResult
from linguary.domain.llm_log import Violation

_SERVICE_TOKEN = re.compile(r'\b(?:b\d+|source_block_id|difficulty_id|artifact_index)\b', re.IGNORECASE)
_EXPLANATION_LABEL = re.compile(r'\[b\d+\]\s*(?:[:—–-]\s*)?', re.IGNORECASE)
_LEADING_EN_DASH = re.compile(r'^– ', re.MULTILINE)
_INLINE_DIALOGUE_BREAK = re.compile(r'([.:,!?…])\n— ')
_EXPLANATIONS = frozenset({'before_you_read.explanation', 'glossary_hint.note', 'language_note.note'})
# Keep short but informative paragraphs; lower scores mean the expected language is unlikely.
_A2_SUMMARY_LANGUAGE_MIN_PROBABILITY = 0.05
A2_SUMMARY_MAX_SENTENCES = 6
_TITLE_PERIOD = re.compile(r'\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St)\.', re.IGNORECASE)
_INITIAL_PERIOD = re.compile(r'\b[A-Z]\.')


class ReaderTextError(ResultValidationError):
    def __init__(
        self,
        reason_code: str,
        *,
        kind: str | None = None,
        block_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.kind = kind
        self.block_id = block_id
        self.detail = detail
        details = tuple(item for item in (kind, block_id, detail) if item)
        suffix = f" ({', '.join(details)})" if details else ''
        super().__init__(
            'Учебный текст требует повторной генерации: ' + reason_code + suffix,
            violation=Violation(
                reason_code,
                'reader_text',
                scope='local' if block_id else 'unknown',
                block_ids=(block_id,) if block_id else (),
                artifact_kinds=(kind,) if kind else (),
                safe_message='Учебный текст требует повторной генерации',
            ),
        )


def _protected_spans(body, protected_quotes):
    spans = []
    for quote in protected_quotes:
        if not quote:
            continue
        start = 0
        while (start := body.find(quote, start)) >= 0:
            end = start + len(quote)
            # A plain source token does not protect a separately added bracket label.
            bracket_label = (re.fullmatch(r'b\d+', quote, re.I) is not None
                             and start > 0 and body[start - 1] == '['
                             and end < len(body) and body[end] == ']')
            if not bracket_label:
                spans.append((start, end))
            start = end
    return spans


def _outside_quotes(body, protected_quotes):
    chars = list(body)
    for start, end in _protected_spans(body, protected_quotes):
        chars[start:end] = ' ' * (end - start)
    return ''.join(chars)


def _dominant_script(text: str) -> str | None:
    counts = {}
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, '')
        if name.startswith('CJK UNIFIED IDEOGRAPH'):
            script = 'CJK'
        elif name:
            script = name.split(' ', 1)[0]
        else:
            continue
        counts[script] = counts.get(script, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _sentence_count(paragraph: str) -> int:
    """Count sentences without treating titles or initials as boundaries."""
    protected = _TITLE_PERIOD.sub(lambda match: match.group(0).replace('.', '\0'), paragraph)
    protected = _INITIAL_PERIOD.sub(lambda match: match.group(0).replace('.', '\0'), protected)
    return len([part for part in re.split(r'[.!?]+', protected) if part.strip()])


def validate_a2_summary(
    body: str,
    *,
    source_texts: tuple[str, ...],
    source_language: str,
    language_detector: LanguageDetector,
    require_parallel_russian: bool = False,
) -> None:
    """Validate an A2 chapter summary.

    By default requires a single source-language paragraph (no line breaks and
    no Russian translation mixed in). Legacy frozen conditions with the old
    two-paragraph prompt pass `require_parallel_russian=True`.
    """
    if require_parallel_russian:
        _validate_legacy_parallel_a2_summary(
            body,
            source_texts=source_texts,
            source_language=source_language,
            language_detector=language_detector,
        )
        return
    stripped = body.strip()
    paragraphs = [part.strip() for part in re.split(r'\r?\n[ \t]*\r?\n+', stripped)]
    non_empty = [part for part in paragraphs if part]
    if len(non_empty) != 1 or '\n' in non_empty[0] or '\r' in non_empty[0]:
        line_count = len([line for line in stripped.splitlines() if line.strip()])
        raise ReaderTextError(
            'a2_summary_requires_single_source_paragraph',
            detail=(
                f'получено абзацев — {max(len(non_empty), line_count)}, '
                'требуется ровно один абзац на языке оригинала без перевода на русский'
            ),
        )
    paragraph = non_empty[0]
    source_script = _dominant_script(' '.join(source_texts))
    summary_script = _dominant_script(paragraph)
    if source_script is not None and summary_script != source_script:
        raise ReaderTextError(
            'a2_summary_requires_single_source_paragraph',
            detail=(
                'алфавит не совпадает с языком оригинала; '
                'требуется ровно один абзац на языке оригинала без перевода на русский'
            ),
        )
    if source_script != 'CYRILLIC' and any(
        unicodedata.name(char, '').startswith('CYRILLIC ')
        for char in paragraph
        if char.isalpha()
    ):
        raise ReaderTextError(
            'a2_summary_requires_single_source_paragraph',
            detail=(
                'обнаружена кириллица в одноязычном пересказе; '
                'требуется ровно один абзац на языке оригинала без перевода на русский'
            ),
        )
    source_language_probability = language_detector.probability_for_language(
        paragraph, source_language
    )
    if (
        source_language_probability is not None
        and source_language_probability < _A2_SUMMARY_LANGUAGE_MIN_PROBABILITY
    ):
        raise ReaderTextError(
            'a2_summary_requires_single_source_paragraph',
            detail=(
                f'текст не распознан как язык оригинала ({source_language}); '
                'требуется ровно один абзац на языке оригинала без перевода на русский'
            ),
        )
    sentence_count = _sentence_count(paragraph)
    if sentence_count > A2_SUMMARY_MAX_SENTENCES:
        raise ReaderTextError(
            'a2_summary_too_many_sentences',
            detail=(
                f'абзац 1: {sentence_count} предложений, '
                f'не больше {A2_SUMMARY_MAX_SENTENCES}'
            ),
        )


def _validate_legacy_parallel_a2_summary(
    body: str,
    *,
    source_texts: tuple[str, ...],
    source_language: str,
    language_detector: LanguageDetector,
) -> None:
    """Legacy two-paragraph EN + RU validation for frozen pre-migration BookConditions."""
    paragraphs = [part.strip() for part in re.split(r'\r?\n[ \t]*\r?\n+', body.strip())]
    if len(paragraphs) != 2 or not all(paragraphs):
        bad_idx = (
            1
            if not paragraphs or not paragraphs[0]
            else (2 if len(paragraphs) < 2 or not paragraphs[1] else 3)
        )
        non_empty = len([part for part in paragraphs if part])
        raise ReaderTextError(
            'a2_summary_requires_parallel_russian',
            detail=(
                f'абзац {bad_idx}: получено абзацев — {non_empty}, '
                r'требуется двухабзацный формат EN\nRU через пустую строку '
                '(абзац 1 — язык оригинала, абзац 2 — перевод на русский)'
            ),
        )
    source_script = _dominant_script(' '.join(source_texts))
    summary_script = _dominant_script(paragraphs[0])
    if source_script is not None and summary_script != source_script:
        raise ReaderTextError(
            'a2_summary_requires_parallel_russian',
            detail=(
                'абзац 1: алфавит не совпадает с языком оригинала; '
                r'требуется формат EN\nRU (абзац 1 — язык оригинала, абзац 2 — перевод на русский)'
            ),
        )
    source_language_probability = language_detector.probability_for_language(
        paragraphs[0], source_language
    )
    if (
        source_language_probability is not None
        and source_language_probability < _A2_SUMMARY_LANGUAGE_MIN_PROBABILITY
    ):
        raise ReaderTextError(
            'a2_summary_requires_parallel_russian',
            detail=(
                f'абзац 1: текст не распознан как язык оригинала ({source_language}); '
                r'требуется формат EN\nRU (абзац 1 — язык оригинала, абзац 2 — перевод на русский)'
            ),
        )
    letters = [char for char in paragraphs[1] if char.isalpha()]
    cyrillic = [char for char in letters if unicodedata.name(char, '').startswith('CYRILLIC ')]
    if not letters or len(cyrillic) / len(letters) < 0.5:
        raise ReaderTextError(
            'a2_summary_requires_parallel_russian',
            detail=(
                'абзац 2: ожидается перевод на русский язык; '
                r'требуется формат EN\nRU (абзац 1 — язык оригинала, абзац 2 — перевод на русский)'
            ),
        )
    for index, paragraph in enumerate(paragraphs, start=1):
        sentence_count = _sentence_count(paragraph)
        if sentence_count > A2_SUMMARY_MAX_SENTENCES:
            raise ReaderTextError(
                'a2_summary_too_many_sentences',
                detail=(
                    f'абзац {index}: {sentence_count} предложений, '
                    f'не больше {A2_SUMMARY_MAX_SENTENCES}'
                ),
            )


class ReaderTextPolicy:
    semantic_correctness = 'not_evaluated'

    def prepare(self, kind, body, protected_quotes=(), *, block_id=None):
        if body is None:
            return None
        original = body
        if kind == 'translation':
            body = _normalize_dialogue_dashes(body, protected_quotes)
        if kind in _EXPLANATIONS:
            spans = _protected_spans(body, protected_quotes)
            matches = [match for match in _EXPLANATION_LABEL.finditer(body)
                       if not any(start < match.end() and end > match.start() for start, end in spans)]
            for match in reversed(matches):
                body = body[:match.start()] + body[match.end():]
            body = body.strip()
        self.validate_final(kind, body, protected_quotes, block_id=block_id)
        for quote in protected_quotes:
            if quote and original.count(quote) != body.count(quote):
                # Removed service labels may happen to contain a short quoted source token.
                if re.fullmatch(r'b\d+', quote, re.I) is None:
                    raise ReaderTextError('protected_quote_changed', kind=kind, block_id=block_id)
        return body

    def validate_final(self, kind, body, protected_quotes=(), *, block_id=None):
        if not body or not body.strip():
            raise ReaderTextError('required_help_erased', kind=kind, block_id=block_id)
        outside = _outside_quotes(body, protected_quotes)
        if _SERVICE_TOKEN.search(outside):
            raise ReaderTextError('ambiguous_service_marker', kind=kind, block_id=block_id)
        if kind in _EXPLANATIONS and not re.search(r'\w', outside, re.UNICODE):
            raise ReaderTextError('required_help_erased', kind=kind, block_id=block_id)
        return body

    def warnings(self, kind, body, *, reader_level):
        if reader_level.strip().lower() != 'a2':
            return ()
        sentences = [part for part in re.split(r'[.!?]+', body) if part.strip()]
        if any(len(re.findall(r'\w+', sentence)) > 15 for sentence in sentences):
            return ('complexity_requires_review',)
        return ()

    def prepare_results(self, results, protected_quotes=()):
        checked = []
        for result in results:
            segment_translations = result.segment_translations
            if segment_translations:
                segment_translations = tuple(
                    self.prepare('translation', slot, protected_quotes, block_id=result.block_id)
                    if slot and slot.strip() else slot
                    for slot in segment_translations
                )
                translated = _rebuilt_translation(segment_translations)
            else:
                translated = self.prepare('translation', result.translated_text, protected_quotes, block_id=result.block_id) if result.translated_text is not None else None
            hints = tuple(replace(hint,
                translation=self.prepare('glossary_hint.translation', hint.translation, (hint.original,), block_id=result.block_id),
                note=self.prepare('glossary_hint.note', hint.note, (hint.original, hint.translation), block_id=result.block_id) if hint.note else hint.note)
                for hint in result.glossary_hints)
            notes = tuple(replace(note, note=self.prepare('language_note.note', note.note,
                          (note.original_excerpt, note.translated_excerpt or ''), block_id=result.block_id)) for note in result.language_notes)
            checked.append(replace(
                result,
                translated_text=translated,
                segment_translations=segment_translations,
                glossary_hints=hints,
                language_notes=notes,
            ))
        return tuple(checked)


def _normalize_dialogue_dashes(body, protected_quotes):
    """Use an em dash for a reply and join one newline after punctuation.

    A match overlapping a protected quote stays as written. Both replacements
    keep the same length, so the quote text is unchanged.
    """
    body = _replace_outside_quotes(body, _LEADING_EN_DASH, '— ', protected_quotes)
    return _replace_outside_quotes(
        body, _INLINE_DIALOGUE_BREAK, r'\1 — ', protected_quotes,
    )


def _replace_outside_quotes(body, pattern, replacement, protected_quotes):
    spans = _protected_spans(body, protected_quotes)

    def replace(match):
        if any(start < match.end() and end > match.start() for start, end in spans):
            return match.group(0)
        return match.expand(replacement)

    return pattern.sub(replace, body)


def _rebuilt_translation(slots):
    """Join via the domain method. A half-built result would reject the new slots."""
    carrier = object.__new__(BlockProcessingResult)
    object.__setattr__(carrier, 'segment_translations', slots)
    return BlockProcessingResult.rebuild_translation(carrier)


_A2_NOTE_DISCIPLINE = """
В блоках TRANSLATE ставь не больше 1–2 заметок на абзац; запрещены очевидные сноски о прозрачных местоимениях, базовых предлогах и словах уровня A1 или прозрачных интернационализмах. Не склеивай разные концы абзаца в одну сноску. Не копируй в original или original_excerpt фразы длиннее 5 слов. Не повторяй объяснение конструкции или слова, уже разобранных ранее в главе.
В glossary_hints приводи глаголы к инфинитиву с to («to snigger — хихикать», «to nod — кивать»), не повторяй английское слово в начале поля translation и оставляй note пустым для обычных слов без игры слов и идиоматики. Никогда не переноси в заметки читателю внутренние переводческие пометы («не склонять», род существительных).
Запрещены «помета к глаголу речи», «реалия», «страдательный оборот», «превосходная степень», «причастие прошедшего времени» и кодовые обозначения вроде V2/V3 (пиши по-русски: «третья форма глагола»). Цитируй слова из оригинала точно в той форме, как они стоят в тексте (например, did become, а не did became). Не дублируй в сноске целое предложение и его перевод.
"""

_A2_DIALOGUE_AND_CALQUE_RULES = """
Русские реплики оформляй длинным тире «— » в начале реплики и кавычками-ёлочками «…», а не средним тире «– » и не немецкими кавычками „…“.
Сверяй род адресата во 2-м лице и местоимений 3-го лица с родом русского перевода (включая животных и неодушевлённые существительные).
Соблюдай единый регистр обращения «ты/вы» по ролям в сцене: ребёнок к незнакомому взрослому или учителю, а также вежливые обращения с титулом (Mr/Mrs/Professor) требуют «вы», если в сцене не установлен переход на «ты».
Если у персонажа в оригинале ярко выраженная разговорная или просторечная речь (диалект, усечения), сохраняй живую разговорность в русском переводе на протяжении всей сцены и избегай книжных оборотов.
Передавай идиомы, британские реалии и разговорные обороты естественным русским эквивалентом по контексту, без буквальных калек и вульгаризмов.
"""


def learner_instructions(reader_level):
    common = '''
Профиль content-plan-v3; инструкция learner-quality-v3.1.
Уровень читателя применяется к summary, before_you_read.explanation,
glossary_hint.translation, glossary_hint.note и language_note.note, а также режимам.
Сюжетные и предметные выводы подтверждай точными исходными фрагментами.
Не подменяй неизвестную атрибуцию выдуманным фактом; глоссарий выше предположений.
Не выводи служебные метки блоков, идентификаторы и технические поля в учебных телах.
Цитаты, expression, original и original_excerpt сохраняй дословно.
Смысловая точность остаётся непроверенной; длина фраз не гарантирует уровень CEFR.
Запрещены буквальные кальки и несочетаемые союзы («хотя зато», «больше того»), пропуск обстоятельств времени и места, термины «окказиональный», «окказиональное прилагательное/сложение», «дериват», «инверсия», «эллипсис», «V2/V3», пометы «вариант стоит проверить» и «в переводе передано как...», переводческие инструкции («не склонять») в заметках читателю, шаблон «не X, а Y» (кроме false_friend) и пустые пересказы в glossary_hints.note.
'''
    if reader_level.strip().lower() == 'a2':
        return common + '''
Для A2 обеспечь необходимую лексическую опору: не считай короткую реплику автоматически простой — если в ней есть редкое междометие, выразительный глагол речи или фразовый глагол выше A2, дай подсказку или перевод. В блоках без перевода проверяй каждое знаменательное слово выше A2, включая редкие значения знакомых слов и официальные титулы.
Пересказ A2 верни ровно одним абзацем на языке оригинала (без перевода на русский): простая лексика A2, ясные связи причин и событий, строго по фактам главы без выдуманных деталей.
Если встречаются модальный перфект вроде might have been, придаточное с what в значении
«то, что» или отрыв предлога в конце придаточного или вопроса, не ограничивай помощь
словарной подсказкой: переведи конструкцию полностью либо прямо объясни её форму и смысл
в отдельной языковой заметке.
Для двойного отрицания вида «not saying ... isn’t ...» сохраняй утвердительный смысл —
например, признание качества персонажа или события. Не переворачивай смысл при переводе;
объяснение BYR формулируй утвердительно, называя признаваемое качество.
Например, «not saying the spell isn’t brilliant» передавай как «не отрицаю, что
заклинание блестящее»; не превращай это в «заклинание не блестящее».
BYR, подсказки и заметки объясняют трудность понятным русским языком, без неподтверждённых деталей.
Длина предложений и редкая лексика — повод для редакторской проверки, не сертификат CEFR.
Для A2 без english help верни before_you_read: [] и before_you_read_evidence: []. Трудные места объясняй в language_notes под абзацем.
''' + _A2_NOTE_DISCIPLINE + _A2_DIALOGUE_AND_CALQUE_RULES
    if reader_level.strip().lower() == 'c1':
        return common + '\nДля C1 сохраняй естественный ритм и давай помощь только к действительно редким, архаичным или неочевидным местам.\n'
    return common + '\nСохраняй прежнюю калибровку B1–B2; давай достаточную помощь для выбранного уровня.\n'


def prepare_analysis_payload(payload):
    """Prepare every generated analysis body before deriving plan evidence hashes."""
    policy = ReaderTextPolicy()
    protected = tuple(entry.expression for entry in payload.before_you_read)
    summary = policy.prepare('summary', payload.chapter_summary, protected)
    entries = tuple(replace(entry, explanation=policy.prepare('before_you_read.explanation',
        entry.explanation, (entry.expression,))) for entry in payload.before_you_read)
    candidates = tuple(replace(candidate,
        suggested_translation=policy.prepare('glossary_hint.translation', candidate.suggested_translation, (candidate.source_text,)),
        note=policy.prepare('glossary_hint.note', candidate.note, (candidate.source_text,)) if candidate.note else candidate.note)
        for candidate in payload.glossary_candidates)
    plan = payload.content_plan
    if plan is not None:
        evidence = []
        for item in plan.before_you_read_evidence:
            if item.artifact_index >= len(entries):
                raise ReaderTextError('required_help_reference_missing')
            excerpt = policy.prepare('before_you_read.explanation', item.body_excerpt,
                                     (entries[item.artifact_index].expression,))
            evidence.append(replace(item, body_excerpt=excerpt))
        plan = replace(plan, before_you_read_evidence=tuple(evidence))
    return replace(payload, chapter_summary=summary, before_you_read=entries,
                   glossary_candidates=candidates, content_plan=plan)


def record_reader_warnings(unit_of_work, response, channels, *, reader_level):
    """Warnings retain uncertainty without copying learner text into metadata."""
    policy = ReaderTextPolicy()
    warnings = [dict(kind=kind, reason=reason) for kind, body in channels
                for reason in policy.warnings(kind, body, reader_level=reader_level)]
    if not warnings or response.call_id is None:
        return
    from linguary.application.diagnostics import diagnostics_for
    try:
        with unit_of_work.transaction() as transaction:
            record = transaction.llm_calls.diagnostic_record(call_id=response.call_id)
            if record is None:
                raise ReaderTextError('diagnostic_metadata_missing')
            metadata = dict(record['metadata'])
            metadata['reader_warnings'] = warnings
            transaction.llm_calls.update_diagnostics(response.call_id, metadata=metadata)
    except Exception:
        diagnostics_for(unit_of_work).fail()



def source_quote_fragments(texts):
    """Protect exact quoted source strings without interpreting their meaning."""
    result = []
    for text in texts:
        for match in re.finditer(r'«([^»]+)»|“([^”]+)”|"([^"]+)"', text):
            result.extend((match.group(0), next(part for part in match.groups() if part is not None)))
    return tuple(result)
