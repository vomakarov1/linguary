"""A conservative policy for final learner text; it makes no semantic guarantees."""
from dataclasses import replace
import re

from linguary.application.retry_policy import ResultValidationError
from linguary.domain.llm_log import Violation

_SERVICE_TOKEN = re.compile(r'\b(?:b\d+|source_block_id|difficulty_id|artifact_index)\b', re.IGNORECASE)
_EXPLANATION_LABEL = re.compile(r'\[b\d+\]\s*(?:[:—–-]\s*)?', re.IGNORECASE)
_EXPLANATIONS = frozenset({'before_you_read.explanation', 'glossary_hint.note', 'language_note.note'})


class ReaderTextError(ResultValidationError):
    def __init__(self, reason_code):
        self.reason_code = reason_code
        super().__init__('Учебный текст требует повторной генерации: ' + reason_code,
            violation=Violation(reason_code, 'reader_text', safe_message='Учебный текст требует повторной генерации'))


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


class ReaderTextPolicy:
    semantic_correctness = 'not_evaluated'

    def prepare(self, kind, body, protected_quotes=()):
        if body is None:
            return None
        original = body
        if kind in _EXPLANATIONS:
            spans = _protected_spans(body, protected_quotes)
            matches = [match for match in _EXPLANATION_LABEL.finditer(body)
                       if not any(start < match.end() and end > match.start() for start, end in spans)]
            for match in reversed(matches):
                body = body[:match.start()] + body[match.end():]
            body = body.strip()
        self.validate_final(kind, body, protected_quotes)
        for quote in protected_quotes:
            if quote and original.count(quote) != body.count(quote):
                # Removed service labels may happen to contain a short quoted source token.
                if re.fullmatch(r'b\d+', quote, re.I) is None:
                    raise ReaderTextError('protected_quote_changed')
        return body

    def validate_final(self, kind, body, protected_quotes=()):
        if not body or not body.strip():
            raise ReaderTextError('required_help_erased')
        outside = _outside_quotes(body, protected_quotes)
        if _SERVICE_TOKEN.search(outside):
            raise ReaderTextError('ambiguous_service_marker')
        if kind in _EXPLANATIONS and not re.search(r'\w', outside, re.UNICODE):
            raise ReaderTextError('required_help_erased')
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
            translated = self.prepare('translation', result.translated_text, protected_quotes) if result.translated_text is not None else None
            hints = tuple(replace(hint,
                translation=self.prepare('glossary_hint.translation', hint.translation, (hint.original,)),
                note=self.prepare('glossary_hint.note', hint.note, (hint.original, hint.translation)) if hint.note else hint.note)
                for hint in result.glossary_hints)
            notes = tuple(replace(note, note=self.prepare('language_note.note', note.note,
                          (note.original_excerpt, note.translated_excerpt or ''))) for note in result.language_notes)
            checked.append(replace(result, translated_text=translated, glossary_hints=hints, language_notes=notes))
        return tuple(checked)


def learner_instructions(reader_level):
    common = '''
Профиль content-plan-v3; инструкция learner-quality-v3.1.
Уровень читателя применяется к summary, before_you_read.explanation,
glossary_hint.translation, glossary_hint.note и language_note.note, а также режимам.
Сюжетные и предметные выводы подтверждай точными исходными фрагментами.
Не подменяй неизвестную атрибуцию выдуманным фактом; глоссарий выше предположений.
Общую языковую справку явно отделяй словами «Языковая справка» от фактов сцены.
Не выводи служебные метки блоков, идентификаторы и технические поля в учебных телах.
Цитаты, expression, original и original_excerpt сохраняй дословно.
Смысловая точность остаётся непроверенной; длина фраз не гарантирует уровень CEFR.
'''
    if reader_level.strip().lower() == 'a2':
        return common + '''
Для A2 обеспечь необходимую лексическую опору: объясни по контексту Shoo,
snapped и went out, если они встречаются; не считай короткую реплику автоматически простой.
Английский summary составляй доступными словами, с ясными связями причин и событий.
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
'''
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
