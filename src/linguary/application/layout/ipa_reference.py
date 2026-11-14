"""Static IPA key and the signs used by one set of transcriptions.

`ipa_key` does not read a book. It walks each transcription left to right,
takes the longest reference sign at the current position, and skips one
character when nothing matches. The result is those signs in reference
order, without duplicates. A longer sign does not also list its parts
unless those parts occur on their own.

Multi-character signs come first, before the single characters they contain.
The mockup rows keep that table's example and Russian description. Further
rows are the single characters those signs contain, and the diphthongs that
would otherwise be reported as those characters.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IpaSymbol:
    """One IPA sign with a fixed example and a Russian description."""

    symbol: str
    example: str
    description: str


IPA_REFERENCE: tuple[IpaSymbol, ...] = (
    IpaSymbol("tʃ", "Cheshire /ˈtʃeʃə/", "твёрдое «ч»"),
    IpaSymbol("eɪ", "tale /teɪl/", "«эй»"),
    IpaSymbol("ɜː", "bird /bɜːd/", "похоже на «ё» без «й», губы не округлять"),
    IpaSymbol("əʊ", "go /ɡəʊ/ (UK)", "от неясного звука к «у»; в US — /oʊ/"),
    IpaSymbol("aɪ", "my /maɪ/", "«ай»"),
    IpaSymbol("ɔɪ", "boy /bɔɪ/", "«ой»"),
    IpaSymbol("aʊ", "now /naʊ/", "«ау»"),
    IpaSymbol("oʊ", "go /oʊ/ (US)", "от «о» к «у»; в UK — /əʊ/"),
    IpaSymbol("ɪə", "near /nɪə/", "от краткого «и» к неясному звуку"),
    IpaSymbol("eə", "hair /heə/", "от «э» к неясному звуку"),
    IpaSymbol("ʊə", "tour /tʊə/", "от краткого «у» к неясному звуку"),
    IpaSymbol("ˈ", "/ˈweɪskəʊt/", "ударение на следующем слоге"),
    IpaSymbol("ː", "see /siː/", "звук долгий"),
    IpaSymbol("ɪ", "sit /sɪt/", "краткий, между «и» и «ы»"),
    IpaSymbol("æ", "cat /kæt/", "между «э» и «а», рот открыт шире"),
    IpaSymbol("ʌ", "cup /kʌp/", "краткое «а»"),
    IpaSymbol("ə", "about /əˈbaʊt/", "безударный неясный звук"),
    IpaSymbol("θ", "think /θɪŋk/", "глухой: кончик языка между зубами"),
    IpaSymbol("ð", "this /ðɪs/", "то же, но звонкий"),
    IpaSymbol("ŋ", "sing /sɪŋ/", "«н» задней частью языка, без «г»"),
    IpaSymbol("w", "well /wel/", "губы трубочкой, краткое «у»"),
    IpaSymbol("e", "bed /bed/", "краткое «э»"),
    IpaSymbol("t", "sit /sɪt/", "«т»"),
    IpaSymbol("ʃ", "ship /ʃɪp/", "«ш»"),
    IpaSymbol("ɜ", "nurse /nɜs/", "как /ɜː/, но короткий"),
    IpaSymbol("ʊ", "foot /fʊt/", "краткое «у»"),
)

_BY_LENGTH: tuple[IpaSymbol, ...] = tuple(
    sorted(IPA_REFERENCE, key=lambda item: len(item.symbol), reverse=True)
)


def ipa_key(ipas: Iterable[str]) -> tuple[IpaSymbol, ...]:
    """Signs from `IPA_REFERENCE` that occur in `ipas`, in reference order.

    Matching is greedy: at each position the longest reference sign wins.
    A character that starts no sign is skipped. One sign is returned once
    even when several transcriptions use it. Parts of a longer sign are
    included only when the same characters also occur outside that sign.
    """
    found: set[str] = set()
    for ipa in ipas:
        index = 0
        while index < len(ipa):
            matched = _match_at(ipa, index)
            if matched is None:
                index += 1
                continue
            found.add(matched.symbol)
            index += len(matched.symbol)
    return tuple(item for item in IPA_REFERENCE if item.symbol in found)


def _match_at(ipa: str, index: int) -> IpaSymbol | None:
    """Longest reference sign starting at `index`, if any."""
    for item in _BY_LENGTH:
        if ipa.startswith(item.symbol, index):
            return item
    return None
