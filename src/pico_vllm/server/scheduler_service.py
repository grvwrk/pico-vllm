"""Async bridge between HTTP requests and the synchronous scheduler."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from pico_vllm.engine.scheduler import ManagedSequence, Scheduler


@dataclass
class GenerationStream:
    """Token stream and eventual result for one queued generation request."""

    request_id: str
    completion: asyncio.Future[ManagedSequence]
    _tokens: asyncio.Queue[int | None]

    async def tokens(self) -> AsyncIterator[int]:
        while True:
            token_id = await self._tokens.get()
            if token_id is None:
                return
            yield token_id


@dataclass
class _PendingGeneration:
    request_id: str
    completion: asyncio.Future[ManagedSequence]
    tokens: asyncio.Queue[int | None] | None = None
    emitted_token_count: int = 0


class SchedulerService:
    """Own one scheduler and continuously advance all queued requests.

    Model execution remains synchronous inside :class:`Scheduler`, which is
    intentional: a single service task serializes access to its shared model
    and KV cache. ``await asyncio.sleep(0)`` between iterations lets request
    handlers enqueue more work without creating a second scheduler.
    """

    def __init__(self, scheduler: Scheduler, *, idle_sleep_seconds: float = 0.01):
        self.scheduler = scheduler
        self.idle_sleep_seconds = idle_sleep_seconds
        self._pending: dict[str, _PendingGeneration] = {}
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

    async def start(self) -> None:
        if self._task is None:
            self._stopping = False
            self._task = asyncio.create_task(self._run(), name="pico-vllm-scheduler")

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

        for pending in self._pending.values():
            if pending.tokens is not None:
                pending.tokens.put_nowait(None)
            if not pending.completion.done():
                pending.completion.cancel()
        self._pending.clear()

    async def submit(self, prompt: str, max_new_tokens: int) -> ManagedSequence:
        """Queue a request and wait for its own completed sequence."""
        pending = self._enqueue(prompt, max_new_tokens)
        return await asyncio.shield(pending.completion)

    async def submit_stream(self, prompt: str, max_new_tokens: int) -> GenerationStream:
        """Queue a request and return an iterator of its generated token IDs."""
        pending = self._enqueue(prompt, max_new_tokens, stream_tokens=True)
        assert pending.tokens is not None
        return GenerationStream(
            request_id=pending.request_id,
            completion=pending.completion,
            _tokens=pending.tokens,
        )

    def _enqueue(
        self, prompt: str, max_new_tokens: int, *, stream_tokens: bool = False
    ) -> _PendingGeneration:
        if self._task is None:
            raise RuntimeError("SchedulerService has not been started.")

        seq_id = str(uuid.uuid4())
        future: asyncio.Future[ManagedSequence] = asyncio.get_running_loop().create_future()
        pending = _PendingGeneration(
            request_id=seq_id,
            completion=future,
            tokens=asyncio.Queue() if stream_tokens else None,
        )
        self._pending[seq_id] = pending
        self.scheduler.add_request(seq_id, prompt, max_new_tokens)
        return pending

    def _publish_generated_tokens(self) -> None:
        active = list(self.scheduler.running.values()) + list(self.scheduler.finished.values())
        for managed in active:
            pending = self._pending.get(managed.seq_id)
            if pending is None or pending.tokens is None:
                continue
            for token_id in managed.generated_ids[pending.emitted_token_count :]:
                pending.tokens.put_nowait(token_id)
            pending.emitted_token_count = len(managed.generated_ids)

    def _resolve_finished(self) -> None:
        for seq_id, managed in list(self.scheduler.finished.items()):
            pending = self._pending.pop(seq_id, None)
            if pending is not None:
                if pending.tokens is not None:
                    pending.tokens.put_nowait(None)
                if not pending.completion.done():
                    pending.completion.set_result(managed)
            # The result is handed to its waiting future; retaining it in the
            # scheduler would grow memory for a long-running server.
            del self.scheduler.finished[seq_id]

    async def _run(self) -> None:
        while not self._stopping:
            has_work = self.scheduler.step()
            self._publish_generated_tokens()
            self._resolve_finished()
            await asyncio.sleep(0 if has_work else self.idle_sleep_seconds)
