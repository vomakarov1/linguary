"""Data and language validation adapters.

`EpubCheckValidator` implements the optional `ExportValidator` port on top of EPUBCheck.
`Py3LangidDetector` implements the `LanguageDetector` port using a bundled local model.
The application layer sees only these ports and no provider implementation details.
"""

from linguary.adapters.validation.epubcheck import (
    EpubCheckAvailability,
    EpubCheckSettings,
    EpubCheckValidator,
)
from linguary.adapters.validation.language_identifier import Py3LangidDetector

__all__ = [
    "EpubCheckAvailability",
    "EpubCheckSettings",
    "EpubCheckValidator",
    "Py3LangidDetector",
]
