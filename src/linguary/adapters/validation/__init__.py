"""Проверка данных на внешних границах приложения.

Публичный вход адаптера — `EpubCheckValidator` и его настройка `EpubCheckSettings`:
реализация порта `ExportValidator` из `linguary.application.ports.export_validation`
поверх внешнего EPUBCheck. Прикладной слой видит только порт и о Java, jar-файле
и запуске процесса ничего не знает (ADR § 3).
"""

from linguary.adapters.validation.epubcheck import (
    EpubCheckSettings,
    EpubCheckValidator,
)

__all__ = ["EpubCheckSettings", "EpubCheckValidator"]
