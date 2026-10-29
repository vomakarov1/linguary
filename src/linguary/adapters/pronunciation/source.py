"""No production data is connected until source rights have been approved."""
from linguary.application.ports.pronunciation import SourceLookupResult


class DisabledPronunciationSource:
    configured = False

    async def lookup(self, form, variant):
        return SourceLookupResult.unavailable('source_disabled')
