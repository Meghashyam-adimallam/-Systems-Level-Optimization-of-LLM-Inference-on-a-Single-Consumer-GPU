from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
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
    print("\n  >>> Open in browser: http://127.0.0.1:8000\n")
    yield


app = FastAPI(title="Systems-Level Optimization of LLM Inference on a Single Consumer GPU — Baseline", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
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


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    device = next(model.parameters()).device
    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=req.temperature and req.temperature > 0,
            temperature=req.temperature or 0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[:, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    num_tokens = generated.shape[1]
    return GenerateResponse(text=text.strip(), prompt=req.prompt, num_tokens=num_tokens)


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(next(model.parameters()).device) if model else "not loaded"}
