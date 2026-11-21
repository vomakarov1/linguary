"""Service block labels are excluded from analysis text."""

from linguary.application.block_labels import without_block_labels


def test_labels_removed_with_parenthesised_lists() -> None:
    assert without_block_labels("Сохранить паузу в блоке b3 (b3, b5–b7) и тон блока b9.") == "Сохранить паузу в и тон."


def test_text_without_labels_unchanged() -> None:
    text = "Сохранить паузу и тон рассказчика."
    assert without_block_labels(text) == text


def test_bare_ranges_and_lists_stripped_cleanly() -> None:
    assert without_block_labels("в b11–b13 сохранять паузу") == "в сохранять паузу"
    assert without_block_labels("b11–b13") == ""
    assert without_block_labels("b120–b121") == ""
    assert without_block_labels("b1, b2") == ""
    assert without_block_labels("См. b1, b2, b5–b7 в сцене.") == "См. в сцене."
    assert without_block_labels("b1–b2, b5; b9—b10") == ""


def test_russian_plural_and_case_forms_with_lists_stripped() -> None:
    assert without_block_labels("В блоках b11–b13 сохранять шёпот.") == "В сохранять шёпот."
    assert without_block_labels("Для блоков b1, b2 использовать короткие реплики.") == "Для использовать короткие реплики."
    assert without_block_labels("К блокам b3–b5 относится правило.") == "К относится правило."
    assert without_block_labels("Передать тон блоками b7, b8.") == "Передать тон."
    assert without_block_labels("Описать блоком b10 паузу.") == "Описать паузу."
    assert without_block_labels("Блоки b11–b12 содержат диалог.") == "содержат диалог."
    assert without_block_labels("Блок b4, b5 задаёт темп.") == "задаёт темп."


def test_dash_artifacts_next_to_em_dash_collapsed() -> None:
    assert without_block_labels("Финальный фрагмент b120–b121 — цитата") == "Финальный фрагмент — цитата"
    assert without_block_labels("Финальный фрагмент (b120)–(b121) — цитата") == "Финальный фрагмент — цитата"
    assert without_block_labels("Финальный фрагмент b120 — (b121) — цитата") == "Финальный фрагмент — цитата"
    assert without_block_labels("Финальный фрагмент — (b120)–(b121) цитата") == "Финальный фрагмент — цитата"
    assert without_block_labels("Финальный фрагмент b120– — цитата") == "Финальный фрагмент — цитата"


def test_hyphenated_words_and_non_label_dashes_untouched() -> None:
    samples = (
        "Сохранять по-русски и тёмно-синий оттенок — как в сцене с кем-то.",
        "Диапазон 11–13 и уровень A2–B1 — для читателя.",
        "— Кто-то пришёл, — сказал он.",
    )
    for text in samples:
        assert without_block_labels(text) == text
