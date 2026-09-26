"""Служебные метки блоков не входят в текст анализа."""

from linguary.application.block_labels import without_block_labels


def test_labels_removed_with_parenthesised_lists() -> None:
    assert without_block_labels("Сохранить паузу в блоке b3 (b3, b5–b7) и тон блока b9.") == "Сохранить паузу в и тон."


def test_text_without_labels_unchanged() -> None:
    text = "Сохранить паузу и тон рассказчика."
    assert without_block_labels(text) == text
