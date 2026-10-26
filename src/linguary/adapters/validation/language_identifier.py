"""Language identification adapter based on the local py3langid model."""

from __future__ import annotations

from functools import lru_cache

from linguary.application.ports.language_detection import LanguageDetector

# Below this length, language probabilities mostly reflect word form rather than context.
_MIN_LETTERS = 6


@lru_cache(maxsize=1)
def _identifier():
    from py3langid.langid import LanguageIdentifier, MODEL_FILE

    return LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)


class Py3LangidDetector(LanguageDetector):
    """Uses the package's bundled model; no text is sent to an external service."""

    def probability_for_language(self, text: str, language_code: str) -> float | None:
        if sum(character.isalpha() for character in text) < _MIN_LETTERS:
            return None
        code = language_code.strip().lower().replace("_", "-").split("-", 1)[0]
        if not code:
            return None
        distribution = dict(_identifier().rank(text))
        return distribution.get(code)
