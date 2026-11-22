"""Regeneration saves note ids inside the shared translation transaction."""

from linguary.application.block_processing import TranslationCall
from linguary.application.retry_policy import RetryPolicy
from linguary.application.use_cases.regenerate_block import RegenerateBlock


def test_regenerate_block_save_is_the_shared_translation_save():
    block = RegenerateBlock(
        unit_of_work=object(),
        llm_client=object(),
        retry_policy=RetryPolicy(max_attempts=1),
        validator=object(),
    )
    assert getattr(block._call.save, "__func__", None) is TranslationCall.save
