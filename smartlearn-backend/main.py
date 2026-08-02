import os
import re

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .services.llm import answer_from_pages
from .services.pdf import extract_pages

app = FastAPI(title="SmartLearn Lite API")
documents: dict[str, list[dict]] = {}

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    chat_id: str = Field(default="day2-demo", min_length=1)
    message: str = Field(min_length=2, max_length=2000)


@app.get("/")
def root():
    return {"message": "SmartLearn Lite API is running"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/upload")
async def upload(chat_id: str = Query(...), file: UploadFile = File(...)):
    if file.filename is None or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")
    content = await file.read()
    if not content:
        raise HTTPException(400, "File is empty")
    try:
        pages = extract_pages(content)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    total_chars = sum(len(p["text"]) for p in pages)
    if total_chars == 0:
        raise HTTPException(422, "No readable text found — OCR is not supported")
    documents[chat_id] = pages
    return {
        "status": "ok",
        "filename": file.filename,
        "pages": len(pages),
        "characters": total_chars,
    }


@app.post("/chat")
def chat(body: ChatRequest):
    pages = documents.get(body.chat_id)
    if pages is None:
        raise HTTPException(404, "Chat ID not found — upload a PDF first")

    try:
        answer = answer_from_pages(pages, body.message)
    except RuntimeError:
        raise HTTPException(502, "LLM service unavailable — check API key")
    except Exception:
        raise HTTPException(502, "Upstream AI call failed")

    citations = sorted({
        int(m.group(1))
        for m in re.finditer(r"\[Page\s+(\d+)\]", answer)
        if int(m.group(1)) <= len(pages)
    })

    return {"answer": answer, "citations": citations}
