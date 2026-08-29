"""
server/api.py

Thin FastAPI layer over the Scheduler. Deliberately minimal — the
interesting engineering (paging, scheduling) already lives in
core/ and scheduler/, this file is just plumbing to accept HTTP
requests and return generated text.

Note: this version runs requests through the scheduler synchronously
per HTTP call, which is fine for local testing/benchmarking but isn't
truly continuous batching across concurrent HTTP requests yet — for
that, requests would need to be added to a shared Scheduler instance
that's stepped by a background loop, with each HTTP handler just
waiting on its own request's completion. Noted as a known simplification,
not something this file pretends to solve.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import uuid

from transformers import AutoModelForCausalLM, AutoTokenizer

from pico_vllm.core.block_manager import BlockManager
from pico_vllm.core.kv_cache import KVCache
from pico_vllm.engine.scheduler import Scheduler

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

app = FastAPI()

# Loaded once at startup, shared across all requests
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

bm = BlockManager(num_block=64)
kv_cache = KVCache()


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50


class GenerateResponse(BaseModel):
    request_id: str
    generated_text: str
    num_tokens_generated: int


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    # Each call gets a fresh scheduler for now, sharing the global bm/kv_cache.
    # A single /generate call currently only ever runs one request through
    # them at a time. This keeps the endpoint simple; batching multiple
    # concurrent /generate calls together is exactly the "known
    # simplification" noted above.
    sched = Scheduler(model, tokenizer, bm, kv_cache)

    request_id = str(uuid.uuid4())
    sched.add_request(request_id, req.prompt, max_new_tokens=req.max_new_tokens)

    finished = sched.run()
    managed = finished[request_id]

    generated_text = tokenizer.decode(managed.generated_ids, skip_special_tokens=True)

    return GenerateResponse(
        request_id=request_id,
        generated_text=generated_text,
        num_tokens_generated=len(managed.generated_ids),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


def main() -> None:
    """Run the local development server."""
    import uvicorn

    uvicorn.run("pico_vllm.server.api:app", host="127.0.0.1", port=8000)
