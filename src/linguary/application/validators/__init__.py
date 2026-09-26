"""Проверки результата обработки (PRD § 15, ALG-04).

Каждая проверка живёт в своём модуле и отвечает на один вопрос, возвращая описание
нарушения или `None`. Последствие нарушения — отбраковка всей группы блоков — решает
единственный публичный вход пакета `TranslationResultValidator`.
"""

from linguary.application.validators.result import (
    TranslationResultValidator,
    expectation_for_unit,
)

__all__ = ["TranslationResultValidator", "expectation_for_unit"]
