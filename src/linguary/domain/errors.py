"""Базовые ошибки доменного слоя."""


class DomainError(Exception):
    """Нарушение доменного инварианта.

    Базовый класс для всех ошибок предметной модели: от него наследуются конкретные
    ошибки сущностей, чтобы прикладной слой мог отличать нарушения инвариантов домена
    от ошибок инфраструктуры.
    """


class IdentifierError(DomainError):
    """Некорректные составляющие стабильного идентификатора."""


class DocumentStructureError(DomainError):
    """Нарушение структуры канонического документа, раздела или блока."""


class UnknownBlockTypeError(DocumentStructureError):
    """Тип блока не входит в поддерживаемый набор."""


class UnknownBlockError(DomainError):
    """Обращение к `block_id`, которого нет в известном наборе блоков."""


class GlossaryError(DomainError):
    """Нарушение инвариантов глоссария: понятия, варианта или кандидата."""


class TreatmentPlanError(DomainError):
    """Нарушение инвариантов плана обработки блоков главы."""


class AnalysisIncompleteError(DomainError):
    """Анализ главы неполон: обработку этой главы запускать нельзя."""


class NonTextualBlockError(DomainError):
    """Нетекстовый блок использован там, где допустимы только текстовые.

    Блоки с режимами `PRESERVE_ONLY` и `DROP_WITH_WARNING` не попадают в `TranslationUnit`,
    не передаются модели как контекст и не имеют результата обработки.
    """


class BlockResultError(DomainError):
    """Результат обработки блока не соответствует своему режиму обработки."""


class TranslationUnitError(DomainError):
    """Нарушение инвариантов вычислительной группы блоков."""


class MissingBlockResultError(TranslationUnitError):
    """В ответе модели нет результата для одного из блоков группы."""


class BlockOrderError(TranslationUnitError):
    """Порядок блоков в результате не совпадает с порядком исходных блоков."""


class TreatmentMismatchError(TranslationUnitError):
    """Режим обработки в результате не совпадает с сохранённым планом главы."""


class ProcessingRunError(DomainError):
    """Нарушение инвариантов запуска обработки книги."""


class ActiveRunConflictError(ProcessingRunError):
    """У книги оказалось больше одного активного `ProcessingRun`."""


class TaskTransitionError(DomainError):
    """Переход состояния локальной задачи не разрешён."""


class BookStatusTransitionError(DomainError):
    """Переход состояния книги отсутствует в таблице разрешённых переходов."""


class LLMCallLogError(DomainError):
    """Нарушение инвариантов записи журнала вызовов LLM."""
