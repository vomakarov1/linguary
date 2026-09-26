"""Проверка результата обработки одной группы блоков (ALG-04, PRD § 15, § 17).

Единственный вход пакета: сценарий обработки отдаёт сюда разобранный ответ модели и
получает либо готовые к сохранению результаты блоков, либо `ResultValidationError`.
Промежуточного состояния нет — блокирующая проверка отбраковывает всю группу целиком,
и частично сохранённого `TranslationUnit` не существует (PRD § 17).

Отбраковка всегда несёт `ResultExpectation`: политика повторов строит из неё уточняющую
инструкцию с перечнем ожидаемых блоков, их режимов и целевого языка (ALG-05). Ожидание
можно передать снаружи — сценарий адресует блоки модели так, как решил сам, и в
уточняющей инструкции должны стоять те же имена блоков, что были в запросе.

Порядок проверок значим: состав ответа проверяется первым, потому что все следующие
проверки обращаются к блокам ответа по идентификаторам группы.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from linguary.application.llm_dto import (
    BlockResultResponse,
    LanguageNoteResponse,
    TranslationResultResponse,
)
from linguary.application.retry_policy import (
    ExpectedBlock,
    ResultExpectation,
    ResultValidationError,
)
from linguary.application.validators.coverage import coverage_violation
from linguary.application.validators.language import language_violation
from linguary.application.validators.repetition import repetition_warning
from linguary.application.validators.service_messages import service_message_violation
from linguary.application.validators.treatment import (
    content_violation,
    treatment_violation,
)
from linguary.application.validators.truncation import truncation_violation
from linguary.domain import (
    BlockProcessingResult,
    DomainError,
    GlossaryHint,
    LanguageNote,
    ResultWarning,
    ResultWarningKind,
    TranslationUnit,
    known_markers_only,
    without_footnote_markers,
)


class TranslationResultValidator:
    """Блокирующие проверки и предупреждения результата обработки (PRD § 15).

    Пороговые значения приходят из `CheckSettings`, где они уже проверены на допустимость,
    поэтому здесь не перепроверяются: вторых умолчаний у проверок нет.
    """

    def __init__(
        self,
        *,
        target_language_min_share: float,
        language_check_min_length: int,
        repetition_ngram_size: int,
        repetition_min_repeats: int,
    ) -> None:
        self._target_language_min_share = target_language_min_share
        self._language_check_min_length = language_check_min_length
        self._repetition_ngram_size = repetition_ngram_size
        self._repetition_min_repeats = repetition_min_repeats

    def validate(
        self,
        payload: TranslationResultResponse,
        *,
        unit: TranslationUnit,
        target_language: str,
        note_id_factory: Callable[[], str],
        truncated: bool = False,
        expectation: ResultExpectation | None = None,
    ) -> tuple[BlockProcessingResult, ...]:
        """Проверяет ответ модели и собирает результаты блоков группы.

        Ответ должен быть выражен идентификаторами блоков группы: если сценарий адресовал
        блоки модели псевдонимами, он возвращает их к идентификаторам до проверки и
        передаёт своё `expectation` — то, которое перечисляет блоки так, как их видела
        модель.

        `note_id_factory` выдаёт идентификаторы учебных заметок (`ln-ch03-001`): их
        нумерация зависит от главы и от уже сохранённых заметок, а это знание сценария.
        """
        expected = expectation or expectation_for_unit(
            unit, target_language=target_language
        )
        self._ensure(coverage_violation(payload, unit=unit), expected)
        self._ensure(treatment_violation(payload, unit=unit), expected)
        self._ensure(content_violation(payload, unit=unit), expected)
        self._ensure(
            truncation_violation(payload, unit=unit, truncated=truncated), expected
        )
        self._ensure(
            language_violation(
                payload,
                target_language=target_language,
                min_share=self._target_language_min_share,
                min_length=self._language_check_min_length,
            ),
            expected,
        )
        self._ensure(service_message_violation(payload), expected)
        results = tuple(
            self._result(
                block,
                unit=unit,
                notes=payload.notes_for(block.source_block_id),
                note_id_factory=note_id_factory,
            )
            for block in payload.blocks
        )
        self._ensure_domain_rules(results, unit=unit, expectation=expected)
        return results

    def _result(
        self,
        block: BlockResultResponse,
        *,
        unit: TranslationUnit,
        notes: Sequence[LanguageNoteResponse],
        note_id_factory: Callable[[], str],
    ) -> BlockProcessingResult:
        """Результат одного блока: режим из плана, тексты модели, предупреждения.

        Метки ссылок на сноски (`[^1]`) остаются только в переводе и только те, которым
        у исходного блока есть ссылка: в подсказке и заметке ссылка не нужна, а
        выдуманная моделью метка вела бы в никуда.
        """
        source = next(
            item.block for item in unit.items if item.block.id == block.source_block_id
        )
        return BlockProcessingResult(
            block_id=block.source_block_id,
            treatment=unit.treatment_for(block.source_block_id),
            translated_text=(
                known_markers_only(block.translated_text, source)
                if block.translated_text is not None
                else None
            ),
            glossary_hints=tuple(
                GlossaryHint(
                    original=_unmarked(hint.original),
                    translation=_unmarked(hint.translation),
                    note=_unmarked(hint.note) if hint.note is not None else None,
                )
                for hint in block.glossary_hints
            ),
            language_notes=tuple(
                LanguageNote(
                    id=note_id_factory(),
                    block_id=note.source_block_id,
                    original_excerpt=_unmarked(note.original_excerpt),
                    note=_unmarked(note.note),
                    translated_excerpt=(
                        _unmarked(note.translated_excerpt)
                        if note.translated_excerpt is not None
                        else None
                    ),
                )
                for note in notes
            ),
            warnings=self._warnings(block),
        )

    def _warnings(self, block: BlockResultResponse) -> tuple[ResultWarning, ...]:
        """Непреградные замечания к результату блока (PRD § 15)."""
        if block.translated_text is None:
            return ()
        repetition = repetition_warning(
            block.translated_text,
            ngram_size=self._repetition_ngram_size,
            min_repeats=self._repetition_min_repeats,
        )
        if repetition is None:
            return ()
        return (ResultWarning(kind=ResultWarningKind.REPETITION, message=repetition),)

    def _ensure(self, violation: str | None, expectation: ResultExpectation) -> None:
        """Превращает найденное нарушение в отбраковку всей группы."""
        if violation is not None:
            raise ResultValidationError(violation, expectation=expectation)

    def _ensure_domain_rules(
        self,
        results: Sequence[BlockProcessingResult],
        *,
        unit: TranslationUnit,
        expectation: ResultExpectation,
    ) -> None:
        """Доменная перепроверка собранной группы.

        Правила состава и порядка знает сам `TranslationUnit`, и последнее слово остаётся
        за ним. Сюда управление доходит только если прикладная проверка чего-то не
        заметила, поэтому доменная ошибка переводится в ту же отбраковку с ожиданием:
        повтор должен получить перечень блоков, а не голое сообщение.
        """
        try:
            unit.validate_results(results)
        except DomainError as error:
            raise ResultValidationError(str(error), expectation=expectation) from error


def _unmarked(text: str) -> str:
    """Текст без меток ссылок на сноски; поле из одной метки остаётся как было."""
    unmarked = without_footnote_markers(text)
    if unmarked == text:
        return text
    return unmarked.strip() or text


def expectation_for_unit(
    unit: TranslationUnit, *, target_language: str
) -> ResultExpectation:
    """Что обязан вернуть повторный вызов по этой группе (ALG-05).

    Блоки перечислены их собственными идентификаторами: подойдёт сценарию, который
    адресует блоки модели ими же.
    """
    return ResultExpectation(
        target_language=target_language,
        blocks=tuple(
            ExpectedBlock(block_id=item.block.id, treatment=item.treatment)
            for item in unit.items
        ),
    )


__all__ = ["TranslationResultValidator", "expectation_for_unit"]
