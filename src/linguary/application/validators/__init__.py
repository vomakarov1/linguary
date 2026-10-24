"""Processing result validators (PRD § 15, ALG-04).

Each validator lives in its own module and answers a single question, returning a violation
description or `None`. The consequence of a violation — rejecting the entire unit — is decided
by the package's single public entry point `TranslationResultValidator`.
"""

from linguary.application.validators.result import (
    TranslationResultValidator,
    expectation_for_unit,
)

__all__ = ["TranslationResultValidator", "expectation_for_unit"]
