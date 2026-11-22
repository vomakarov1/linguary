"""An explicit unchanged correction stage for existing scripted scenarios."""

from linguary.adapters.llm import FakeLLMClient, FakeReply
from linguary.application.ports.llm import LLMResponseKind


class CorrectionAwareFakeLLMClient(FakeLLMClient):
    """Keep analysis/translation replies in order while recording correction calls."""

    @property
    def scripted_requests(self):
        return [r for r in self.requests if r.kind is not LLMResponseKind.CHAPTER_CORRECTION]

    async def generate(self, request):
        if request.kind is LLMResponseKind.CHAPTER_CORRECTION:
            reply = FakeReply.of_payload(dict(blocks=[dict(block_id=b, status='unchanged')
                for b in request.ordered_block_ids]))
            response = await FakeLLMClient.always(reply).generate(request)
            self.requests.append(request)
            return response
        return await super().generate(request)

    def _next_step(self):
        index = len(self.scripted_requests)
        if index < len(self.replies):
            return self.replies[index]
        if self.default is not None:
            return self.default
        return super()._next_step()
