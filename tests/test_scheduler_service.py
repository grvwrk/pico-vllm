import asyncio
from types import SimpleNamespace

from pico_vllm.server.scheduler_service import SchedulerService


class FakeScheduler:
    """Small scheduler double for exercising the async service lifecycle."""

    def __init__(self):
        self.waiting = []
        self.running = {}
        self.finished = {}
        self.step_calls = 0

    def add_request(self, seq_id, prompt, max_new_tokens):
        self.waiting.append((seq_id, prompt, max_new_tokens))

    def step(self):
        self.step_calls += 1
        if self.waiting:
            seq_id, prompt, _ = self.waiting.pop(0)
            self.finished[seq_id] = SimpleNamespace(
                seq_id=seq_id,
                generated_ids=[len(prompt)],
            )
        return bool(self.waiting)


def test_scheduler_service_completes_multiple_queued_requests():
    async def scenario():
        scheduler = FakeScheduler()
        service = SchedulerService(scheduler, idle_sleep_seconds=0)
        await service.start()
        try:
            first, second = await asyncio.gather(
                service.submit("one", 1),
                service.submit("two", 1),
            )
        finally:
            await service.stop()

        assert {first.generated_ids[0], second.generated_ids[0]} == {3}
        assert scheduler.step_calls >= 2
        assert not scheduler.finished

    asyncio.run(scenario())


class TokenScheduler:
    def __init__(self):
        self.waiting = []
        self.running = {}
        self.finished = {}

    def add_request(self, seq_id, prompt, max_new_tokens):
        self.waiting.append((seq_id, prompt, max_new_tokens))

    def step(self):
        if self.waiting:
            seq_id, _, _ = self.waiting.pop(0)
            self.running[seq_id] = SimpleNamespace(seq_id=seq_id, generated_ids=[10])
        elif self.running:
            seq_id, managed = self.running.popitem()
            managed.generated_ids.append(20)
            self.finished[seq_id] = managed
        return bool(self.waiting or self.running)


def test_scheduler_service_streams_tokens_before_completion():
    async def scenario():
        service = SchedulerService(TokenScheduler(), idle_sleep_seconds=0)
        await service.start()
        try:
            stream = await service.submit_stream("prompt", 2)
            token_ids = [token_id async for token_id in stream.tokens()]
            completed = await stream.completion
        finally:
            await service.stop()

        assert token_ids == [10, 20]
        assert completed.generated_ids == [10, 20]

    asyncio.run(scenario())
