"""A conservative policy for final learner text; it makes no semantic guarantees."""
from dataclasses import replace
import re
import unicodedata

from linguary.application.analysis_merge import _significant_summary_stems, dominant_script
from linguary.application.ports.language_detection import LanguageDetector
from linguary.application.retry_policy import ResultValidationError
from linguary.domain import BlockProcessingResult
from linguary.domain.llm_log import Violation

_SERVICE_TOKEN = re.compile(r'\b(?:b\d+|source_block_id|difficulty_id|artifact_index)\b', re.IGNORECASE)
_EXPLANATION_LABEL = re.compile(r'\[b\d+\]\s*(?:[:—–-]\s*)?', re.IGNORECASE)
_LEADING_EN_DASH = re.compile(r'^– ', re.MULTILINE)
_EXPLANATIONS = frozenset({'before_you_read.explanation', 'glossary_hint.note', 'language_note.note'})
# Keep short but informative paragraphs; lower scores mean the expected language is unlikely.
_A2_SUMMARY_LANGUAGE_MIN_PROBABILITY = 0.05
_A2_SUMMARY_HEADING_PREFIX_RE = re.compile(
    r"^(?:previously\s+in\s+(?:the\s+)?chapter(?:\s+[\w-]+)?|chapter(?:\s+[\w-]+)?\s+summary|summary)\s*[:—–-]\s*",
    re.IGNORECASE,
)
A2_SUMMARY_MAX_SENTENCES = 6
A2_SUMMARY_WARNING_MAX_WORDS = 22
A2_SUMMARY_HARD_MAX_SENTENCE_WORDS = 26
_A2_SUMMARY_MIN_ANCHOR_STEMS = 4
_DEFAULT_A2_WARNING_MAX_WORDS = 15
_TITLE_PERIOD = re.compile(r'\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St)\.', re.IGNORECASE)
_INITIAL_PERIOD = re.compile(r'\b[A-Z]\.')
_APOSTROPHES = str.maketrans({'\u2019': "'", '\u2018': "'", '\u02bc': "'", '`': "'"})
_A2_PRESENT_SIMPLE_DURATION_VERB = re.compile(
    r'\b(live|lives|stay|stays|work|works|know|knows)\b',
    re.IGNORECASE,
)
_A2_DURATION_FOR = re.compile(
    r'\bfor\s+(?:a\s+long\s+time|many\s+years|years|centuries|months)\b',
    re.IGNORECASE,
)
_A2_DETERMINER_BEFORE = re.compile(
    r'\b(?:a|an|the|my|your|his|her|its|our|their|this|that|these|those)\s*$',
    re.IGNORECASE,
)
_A2_NOMINAL_VERB_FORMS = frozenset({'live', 'lives', 'stay', 'stays', 'work', 'works'})
_A2_PREPOSITION_BEFORE = re.compile(
    r'\b(?:at|in|on|for|from|of|with|without|into|after|before|during|about)\s*$',
    re.IGNORECASE,
)
_A2_BASE_VERB_AUX_BEFORE = re.compile(
    r"(?:"
    r"(?:\b(?:have|has|had|will|would|did|haven't|hasn't|hadn't|won't|wouldn't|didn't"
    r"|can|could|may|might|must|should|shall|cannot|can't|couldn't|mightn't|mustn't|shouldn't)"
    r"|(?:'ve|'s|'ll|'d))"
    r"(?:\s+(?:not|never|always|still|just|already|only|also|really|probably|ever|rather|even|been|got|going|to)){0,3}"
    r"|\bto"
    r")\s*$",
    re.IGNORECASE,
)
_A2_QUESTION_AUX_BEFORE = re.compile(
    r"(?:^|[,;:]\s*|\b(?:and|but|or)\s+)[“\"']?"
    r"(?:haven't|hasn't|hadn't|won't|wouldn't|didn't|don't|doesn't"
    r"|can't|couldn't|mightn't|mustn't|shouldn't|shan't"
    r"|have|has|had|will|would|did|can|could|may|might|must|should|shall)"
    r"(?:\s+not)?"
    r"(?:\s+[A-Za-z']+){1,4}\s*$",
    re.IGNORECASE,
)
_A2_COORDINATED_MODAL_BEFORE = re.compile(
    r"(?:^|[,;:]\s*)"
    r"[^,;:]*\b(?:will|would|did|can|could|may|might|must|should|shall|'ll|'d)\b"
    r"[^,;:]*\band\s*$",
    re.IGNORECASE,
)
_A2_CLAUSE_BREAK_BETWEEN_VERB_AND_FOR = re.compile(
    r"(?:[;:]"
    r"|,\s*(?:and|but|or|so|yet|while|when|where|because|although|though)\b"
    r"|\bthat\s+(?:i|you|he|she|it|we|they|there)\b"
    r"|\b(?:which|who|whom|whose|where|when|while|because|although|though|if|unless"
    r"|have|has|had|will|would|did|wait|waits|waited|waiting|lived|stayed|worked|known)\b"
    r"|(?:'ve|'d|'ll)\b)",
    re.IGNORECASE,
)
_A2_IRREGULAR_PAST = (
    r"was|were|went|came|saw|had|did|made|took|gave|found|left|kept|met|felt|"
    r"told|said|got|became|began|held|ran|sat|stood|slept|spoke|wrote|built|"
    r"bought|brought|caught|taught|thought|knew|heard|lost|sent|spent|grew|"
    r"flew|drew|drove|rode|rose|fell|broke|chose|ate|drank|sang|swam|won|hid"
)
_A2_THAT_SUBJECT_PAST_VERB = re.compile(
    r"\b[Tt]hat\s+"
    r"(?:"
    r"[A-Z][A-Za-z'.]*(?:\s+(?:and\s+)?[A-Za-z'.]+){0,3}"
    r"|(?:[Tt]he|[Aa]n?|[Mm]y|[Yy]our|[Hh]is|[Hh]er|[Ii]ts|[Oo]ur|[Tt]heir"
    r"|[Tt]his|[Tt]hese|[Tt]hose|[Ss]ome|[Mm]any|[Aa]ll|[Bb]oth|[Ee]very|[Ee]ach"
    r"|[Oo]ne|[Tt]wo|[Tt]hree)\s+[A-Za-z']+(?:\s+[A-Za-z']+){0,2}"
    r")"
    rf"\s+(?:[a-z]{{2,}}ed|{_A2_IRREGULAR_PAST})\b"
)


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


def _split_sentences(paragraph: str) -> list[str]:
    """Split text into sentences without treating titles or initials as boundaries."""
    protected = _TITLE_PERIOD.sub(lambda match: match.group(0).replace('.', '\0'), paragraph)
    protected = _INITIAL_PERIOD.sub(lambda match: match.group(0).replace('.', '\0'), protected)
    return [part.replace('\0', '.') for part in re.split(r'[.!?]+', protected) if part.strip()]


def _sentence_count(paragraph: str) -> int:
    """Count sentences without treating titles or initials as boundaries."""
    return len(_split_sentences(paragraph))


def _has_ungrammatical_present_for(sentence: str) -> bool:
    """Detect explicit Present Simple + `for <duration>` errors within one sentence."""
    normalized = sentence.translate(_APOSTROPHES)
    duration_matches = tuple(_A2_DURATION_FOR.finditer(normalized))
    if not duration_matches:
        return False
    for verb_match in _A2_PRESENT_SIMPLE_DURATION_VERB.finditer(normalized):
        verb = verb_match.group(1).casefold()
        prefix = normalized[:verb_match.start()]
        if _A2_DETERMINER_BEFORE.search(prefix) or (
            verb in _A2_NOMINAL_VERB_FORMS and _A2_PREPOSITION_BEFORE.search(prefix)
        ):
            continue
        is_base_verb = not verb.endswith('s')
        if is_base_verb and (
            _A2_BASE_VERB_AUX_BEFORE.search(prefix)
            or _A2_QUESTION_AUX_BEFORE.search(prefix)
            or _A2_COORDINATED_MODAL_BEFORE.search(prefix)
        ):
            continue
        for duration_match in duration_matches:
            if duration_match.start() < verb_match.end():
                continue
            between = normalized[verb_match.end():duration_match.start()]
            if not (
                _A2_CLAUSE_BREAK_BETWEEN_VERB_AND_FOR.search(between)
                or _A2_THAT_SUBJECT_PAST_VERB.search(between)
            ):
                return True
    return False


def validate_a2_summary(
    body: str,
    *,
    source_texts: tuple[str, ...],
    source_language: str,
    language_detector: LanguageDetector,
    require_parallel_russian: bool = False,
    narrative_summary: str | None = None,
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
    source_script = dominant_script(' '.join(source_texts))
    summary_script = dominant_script(paragraph)
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
    if _A2_SUMMARY_HEADING_PREFIX_RE.match(paragraph):
        raise ReaderTextError(
            'a2_summary_disallowed_heading_prefix',
            detail=(
                'уберите служебный заголовок вроде «Previously in Chapter:» — '
                'начинайте сразу с первого предложения пересказа'
            ),
        )
    if ';' in paragraph:
        raise ReaderTextError(
            'a2_summary_disallowed_semicolon',
            detail=(
                'точка с запятой «;» запрещена в пересказе A2: '
                'разделите мысль на два отдельных коротких предложения'
            ),
        )
    sentences = _split_sentences(paragraph)
    sentence_count = len(sentences)
    if sentence_count > A2_SUMMARY_MAX_SENTENCES:
        raise ReaderTextError(
            'a2_summary_too_many_sentences',
            detail=(
                f'абзац 1: {sentence_count} предложений, '
                f'не больше {A2_SUMMARY_MAX_SENTENCES}'
            ),
        )
    for index, sentence in enumerate(sentences, start=1):
        word_count = len(re.findall(r'\w+', sentence))
        if word_count > A2_SUMMARY_HARD_MAX_SENTENCE_WORDS:
            raise ReaderTextError(
                'a2_summary_sentence_too_long',
                detail=(
                    f'предложение {index}: {word_count} слов, '
                    f'не больше {A2_SUMMARY_HARD_MAX_SENTENCE_WORDS}'
                ),
            )
    for index, sentence in enumerate(sentences, start=1):
        if _has_ungrammatical_present_for(sentence):
            raise ReaderTextError(
                'a2_summary_ungrammatical_present_for',
                detail=(
                    f'предложение {index}: Present Simple с for + длительность; '
                    'используйте Present Perfect или Past Simple'
                ),
            )
    if narrative_summary and narrative_summary.strip():
        narrative_script = dominant_script(narrative_summary)
        if (
            narrative_script is None
            or summary_script is None
            or narrative_script == summary_script
        ):
            narrative_stems = _significant_summary_stems(narrative_summary)
            summary_stems = _significant_summary_stems(paragraph)
            if (
                len(narrative_stems) >= _A2_SUMMARY_MIN_ANCHOR_STEMS
                and len(summary_stems) >= _A2_SUMMARY_MIN_ANCHOR_STEMS
                and not (narrative_stems & summary_stems)
            ):
                raise ReaderTextError(
                    'a2_summary_off_topic',
                    detail=(
                        'пересказ A2 не имеет общих значимых основ с narrative_summary; '
                        'требуется пересказ строго по фактам главы'
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
    source_script = dominant_script(' '.join(source_texts))
    summary_script = dominant_script(paragraphs[0])
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
        max_words = (
            A2_SUMMARY_WARNING_MAX_WORDS
            if kind == 'summary'
            else _DEFAULT_A2_WARNING_MAX_WORDS
        )
        if any(len(re.findall(r'\w+', sentence)) > max_words for sentence in _split_sentences(body)):
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
    """Replace a leading en dash in a reply with an em dash.

    A match overlapping a protected quote stays as written. Newlines before a
    dialogue dash stay intact so replies are not joined into mid-line text.
    """
    return _replace_outside_quotes(body, _LEADING_EN_DASH, '— ', protected_quotes)


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
Различай роль заметок в GLOSSARY_ONLY и TRANSLATE: в GLOSSARY_ONLY заметка помогает читателю самостоятельно понять трудную фразу или конструкцию без готового перевода; в TRANSLATE перевод уже дан, поэтому ставь не больше 1–2 заметок на абзац и объясняй только неочевидный из перевода смысл, идиому, игру слов, культурную отсылку или грамматический нюанс. В блоках TRANSLATE не трать заметки на прозрачные идиомы с прямым русским аналогом и на редкую книжную грамматику уровня C1.
Запрещены очевидные сноски о прозрачных местоимениях, базовых предлогах и словах базовой лексики A1–A2 в прямом значении или прозрачных интернационализмах. Не склеивай разные концы абзаца в одну сноску. Не копируй в original или original_excerpt фразы длиннее 5 слов. Не повторяй объяснение конструкции или слова (в том числе в glossary_hints), уже разобранных ранее в главе.
Не домысливай сюжетные детали, причины и факты, которых нет в исходном тексте. Обращаясь к читателю в заметках, используй только вежливое «вы» (не «ты»). Сверяй географический и языковой контекст заметок с текстом сцены (например, португальский, а не испанский язык для Бразилии).
В glossary_hints приводи к инфинитиву с to только глаголы («to snigger — хихикать», «to nod — кивать»); существительные (в т. ч. с артиклем a/an/the, например a guffaw) оставляй существительными без to; не повторяй английское слово в начале поля translation и оставляй note пустым для обычных слов без игры слов и идиоматики. В glossary_hints.translation указывай точное контекстное значение сцены без лишних синонимов из других контекстов. Выноси в glossary_hints отдельные трудные слова и устойчивые словосочетания, а не перевод целого предложения целиком. Никогда не переноси в заметки читателю внутренние переводческие пометы («не склонять», род существительных).
Запрещены «помета к глаголу речи», «реалия», «страдательный оборот», «превосходная степень», «причастие прошедшего времени» и кодовые обозначения вроде V2/V3 (пиши по-русски: «третья форма глагола»). Цитируй слова из оригинала точно в той форме, как они стоят в тексте (например, did become, а не did became); цитируемые английские слова и переводы внутри русского пояснения всегда заключай в кавычки-ёлочки «…» (не прямые "...") и не вставляй русские слова внутрь английской цитаты. Не дублируй в сноске целое предложение и его перевод.
В заметках speech_style («Речь персонажа.») расшифровывай только нестандартные формы английского оригинала («flyin’» → «flying», «we was» → «we were»), не разбирая приёмы русского перевода (вроде русских разговорных частиц), и охватывай якорем original_excerpt разбираемую форму либо ставь якорь в конце реплики, если объясняешь несколько её элементов. Для уровня A2 пиши грамматические заметки лаконично: формула конструкции и прямой перевод фрагмента, ориентир 15–30 слов.
"""

_A2_DIALOGUE_AND_CALQUE_RULES = """
Соблюдай 4 правила оформления русской прямой речи и никогда не смешивай их:
1) Реплика диалога с начала абзаца — только длинным тире «— » (не средним тире «– ») без обрамляющих кавычек.
2) Прямая речь внутри абзаца после двоеточия — кавычками-ёлочками: Он сказал: «…».
3) Внутренняя речь (мысли) — кавычками-ёлочками: «…», — подумал он.
4) Названия, вывески и цитаты внутри реплики — кавычками-ёлочками «…», а вложенные цитаты внутри «…» — лапками „…“.
Запрещены гибриды — « и «—, слова автора внутри «…» (вида «… — сказал он…») и лишние тире между предложениями одной реплики без слов автора.
Реплику прямой речи после слов автора с точкой начинай с новой строки и «— » либо оформляй после двоеточия в кавычках-ёлочках (Он обернулся: «…»), но не ставь «— » в той же строке после точки.
Если в одном абзаце после закрытой реплики идёт самостоятельное авторское предложение с прописной буквы, переноси его на новую строку без тире («— Реплика!\\nАвторское предложение.»; если после авторского предложения речь продолжается — три строки «— Реплика 1.\\nАвторское предложение.\\n— Реплика 2.»); запрещено вводить самостоятельное авторское предложение через «— » с прописной буквы на той же строке («— Реплика? — Авторское предложение.»), а также использовать две строки вида «Авторское предложение. — Реплика 2.» или кавычки-ёлочки с начала абзаца.
Последовательно ставь букву «ё» там, где она предусмотрена русской орфографией (особенно когда от неё зависит смысл: «всё/все», «узнаёт/узнает», «ещё», «принёс»), но не допускай гиперкоррекции в словах с нормативным «е» (например, «починенный», а не «починённый»).
Сверяй род адресата во 2-м лице и местоимений 3-го лица с родом русского перевода (включая животных и неодушевлённые существительные). Сохраняй единство грамматического рода животных, перевоплощающихся персонажей и сквозных предметов (например, «мантия» или «плащ») в пределах всей сцены и главы, не переключая род между соседними фразами.
Соблюдай единый регистр обращения «ты/вы» по ролям в сцене: ребёнок к незнакомому взрослому или учителю, а также вежливые обращения с титулом (Mr/Mrs/Professor) требуют «вы», если в сцене не установлен переход на «ты». При обращении одного персонажа сразу к нескольким собеседникам (даже детям, с которыми поодиночке на «ты») используй множественное число «вы».
Если у персонажа в оригинале ярко выраженная разговорная или просторечная речь (диалект, усечения), сохраняй умеренную живую разговорность в русском переводе сквозь все реплики этого персонажа на протяжении сцены и главы (а не только в блоке с заметкой speech_style): избегай книжных оборотов, но не впадай в фольклорный лубок и современный сленг.
Отказывайся от избыточных притяжательных местоимений («его», «её», «их», «свой» для английских his/her/their) там, где в естественной русской фразе принадлежность очевидна из контекста (части тела, одежда, личные вещи).
Передавай идиомы, британские реалии и разговорные обороты естественным русским эквивалентом по контексту, без буквальных калек и вульгаризмов; если культурная или бытовая реалия уже поясняется в сноске, не вставляй её словарное толкование внутрь художественной фразы.
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
        a2_common = common.replace(
            'Цитаты, expression, original и original_excerpt сохраняй дословно.',
            'Цитаты, expression и original_excerpt сохраняй дословно; '
            'в glossary_hints.original глаголы приводи к инфинитиву с to, '
            'остальные слова и выражения сохраняй как в тексте.',
        )
        return a2_common + '''
Для A2 обеспечь необходимую лексическую опору: не считай короткую реплику автоматически простой — если в ней есть редкое междометие, выразительный глагол речи или фразовый глагол выше A2, дай подсказку или перевод. В блоках без перевода проверяй каждое знаменательное слово выше A2, включая редкие значения знакомых слов и официальные титулы.
Пересказ A2 верни ровно одним абзацем на языке оригинала (без перевода на русский): простая лексика A2, ясные связи причин и событий, строго по фактам главы без выдуманных деталей. Соблюдай согласование времён в пересказе («Previously in…»): события, произошедшие до описываемой сцены, передавай в прошедшем времени (не смешивай настоящее и прошедшее время без основания).
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
    from linguary.application.diagnostics import diagnostics_for, is_transient_lock_error
    try:
        with unit_of_work.transaction() as transaction:
            record = transaction.llm_calls.diagnostic_record(call_id=response.call_id)
            if record is None:
                raise ReaderTextError('diagnostic_metadata_missing')
            metadata = dict(record['metadata'])
            metadata['reader_warnings'] = warnings
            transaction.llm_calls.update_diagnostics(response.call_id, metadata=metadata)
    except Exception as error:
        if is_transient_lock_error(error):
            return
        diagnostics_for(unit_of_work).fail()



def source_quote_fragments(texts):
    """Protect exact quoted source strings without interpreting their meaning."""
    result = []
    for text in texts:
        for match in re.finditer(r'«([^»]+)»|“([^”]+)”|"([^"]+)"', text):
            result.extend((match.group(0), next(part for part in match.groups() if part is not None)))
    return tuple(result)
