"""FastAPI application backed by one shared continuous-batching scheduler."""

from contextlib import asynccontextmanager
import json
import time
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pico_vllm.config import RuntimeSettings
from pico_vllm.core.block_manager import BlockManager
from pico_vllm.core.kv_cache import KVCache
from pico_vllm.engine.scheduler import Scheduler
from pico_vllm.models.registry import ModelRegistry
from pico_vllm.server.scheduler_service import GenerationStream, SchedulerService

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = getattr(app.state, "runtime_settings", RuntimeSettings.from_env())
    settings.validate()
    device = settings.resolved_device()
    dtype = settings.torch_dtype()
    runner = ModelRegistry.create(settings.model_id, device=device, dtype=dtype)
    tokenizer = runner.tokenizer

    block_manager = BlockManager(num_block=settings.kv_cache_blocks)
    kv_cache = KVCache(
        num_blocks=settings.kv_cache_blocks,
        block_size=settings.kv_cache_block_size,
        device=device,
        dtype=dtype,
        architecture=runner.architecture,
    )
    service = SchedulerService(Scheduler(
        runner, block_manager=block_manager, kv_cache=kv_cache,
        max_batched_tokens=settings.max_batched_tokens,
    ))
    await service.start()
    app.state.settings = settings
    app.state.tokenizer = tokenizer
    app.state.scheduler_service = service
    try:
        yield
    finally:
        await service.stop()


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=50, ge=1, le=512)


class GenerateResponse(BaseModel):
    request_id: str
    generated_text: str
    num_tokens_generated: int


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=50, ge=1, le=512)
    stream: bool = False


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, request: Request):
    service: SchedulerService = request.app.state.scheduler_service
    tokenizer = request.app.state.tokenizer
    managed = await service.submit(req.prompt, req.max_new_tokens)

    generated_text = tokenizer.decode(managed.generated_ids, skip_special_tokens=True)

    return GenerateResponse(
        request_id=managed.seq_id,
        generated_text=generated_text,
        num_tokens_generated=len(managed.generated_ids),
    )


@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest, request: Request):
    """Stream generation events as Server-Sent Events (SSE)."""
    service: SchedulerService = request.app.state.scheduler_service
    tokenizer = request.app.state.tokenizer
    stream = await service.submit_stream(req.prompt, req.max_new_tokens)

    async def events():
        async for token_id in stream.tokens():
            yield _sse_event(
                "token",
                {
                    "request_id": stream.request_id,
                    "token_id": token_id,
                    "text": tokenizer.decode([token_id], skip_special_tokens=True),
                },
            )

        managed = await stream.completion
        yield _sse_event(
            "done",
            {
                "request_id": stream.request_id,
                "generated_text": tokenizer.decode(managed.generated_ids, skip_special_tokens=True),
                "num_tokens_generated": len(managed.generated_ids),
            },
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": app.state.settings.model_id,
        "device": app.state.settings.resolved_device(),
    }


@app.get("/v1/models")
def list_models():
    settings = app.state.settings
    return {"object": "list", "data": [{"id": settings.model_id, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    settings = request.app.state.settings
    if req.model is not None and req.model != settings.model_id:
        raise HTTPException(status_code=404, detail=f"Model {req.model!r} is not loaded.")

    tokenizer = request.app.state.tokenizer
    service: SchedulerService = request.app.state.scheduler_service
    prompt = _chat_prompt(tokenizer, req.messages)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if not req.stream:
        managed = await service.submit(prompt, req.max_tokens)
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": settings.model_id,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": tokenizer.decode(managed.generated_ids, skip_special_tokens=True),
                },
                "finish_reason": "stop",
            }],
        }

    stream = await service.submit_stream(prompt, req.max_tokens)

    async def events():
        yield _openai_sse(completion_id, created, settings.model_id, {"role": "assistant"})
        async for token_id in stream.tokens():
            yield _openai_sse(completion_id, created, settings.model_id, {
                "content": tokenizer.decode([token_id], skip_special_tokens=True),
            })
        await stream.completion
        yield _openai_sse(completion_id, created, settings.model_id, {}, finish_reason="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _chat_prompt(tokenizer, messages: list[ChatMessage]) -> str:
    conversation = [message.model_dump() for message in messages]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    formatted = "\n".join(f"{message.role}: {message.content}" for message in messages)
    return f"{formatted}\nassistant:"


def _openai_sse(request_id: str, created: int, model: str, delta: dict, finish_reason=None) -> str:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def main() -> None:
    """Run the local development server."""
    import uvicorn

    settings = RuntimeSettings.from_cli()
    app.state.runtime_settings = settings
    uvicorn.run(app, host=settings.host, port=settings.port)
