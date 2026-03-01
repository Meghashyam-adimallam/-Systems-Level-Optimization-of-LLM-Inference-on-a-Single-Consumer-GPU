from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print("Starting... loading model (this may take 1-2 min)...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cuda":
        model = model.to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("\n  >>> Open in browser: http://127.0.0.1:8000\n")
    yield


app = FastAPI(title="Systems-Level Optimization of LLM Inference on a Single Consumer GPU — Batched", lifespan=lifespan)


@app.get("/")
async def custom_docs():
    path = Path(__file__).parent / "static" / "custom_docs.html"
    return FileResponse(path)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    temperature: Optional[float] = 0.7


class GenerateResponse(BaseModel):
    text: str
    prompt: str
    num_tokens: int


class GenerateBatchRequest(BaseModel):
    prompts: List[str]
    max_new_tokens: int = 64
    temperature: Optional[float] = 0.7


class GenerateBatchResponse(BaseModel):
    responses: List[GenerateResponse]


def _run_batch(prompts: List[str], max_new_tokens: int, temperature: float = 0.7):
    if not prompts:
        return []
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
    input_lengths = attention_mask.sum(dim=1)
    results = []
    for i in range(len(prompts)):
        start = input_lengths[i].item()
        generated = outputs[i, start:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        results.append(GenerateResponse(text=text.strip(), prompt=prompts[i], num_tokens=len(generated)))
    return results


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    results = _run_batch([req.prompt], req.max_new_tokens, req.temperature or 0.7)
    return results[0]


@app.post("/generate_batch", response_model=GenerateBatchResponse)
async def generate_batch(req: GenerateBatchRequest):
    results = _run_batch(req.prompts, req.max_new_tokens, req.temperature or 0.7)
    return GenerateBatchResponse(responses=results)


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(next(model.parameters()).device) if model else "not loaded"}
