import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from services.rag import answer_chat_turn, build_upload_response, prepare_rag_chat_record

app = FastAPI(title="SmartLearn Lite API")
documents: dict[str, dict] = {}

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

    # Build the Day 3 RAG-ready record (pages, chunks, embeddings, FAISS index, empty history)
    try:
        record = prepare_rag_chat_record(
            chat_id=chat_id,
            filename=file.filename,
            pdf_bytes=content,
            upload_root=Path("smartlearn-backend") / "uploads",
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    total_chars = sum(len(p["text"]) for p in record["pages"])
    if total_chars == 0:
        raise HTTPException(422, "No readable text found — OCR is not supported")

    documents[chat_id] = record
    return build_upload_response(record)


@app.get("/documents/{chat_id}/file")
def get_document_file(chat_id: str):
    """Serve the uploaded PDF for a chat session so the frontend can preview it."""
    record = documents.get(chat_id)
    if record is None:
        raise HTTPException(404, "Chat ID not found")
    file_path = record.get("file_path") or record.get("saved_pdf_path", "")
    if not file_path or not Path(file_path).exists():
        raise HTTPException(404, "PDF file not found")
    return FileResponse(file_path, media_type="application/pdf")


@app.post("/chat")
def chat(body: ChatRequest):
    document = documents.get(body.chat_id)
    if document is None:
        raise HTTPException(404, "Chat ID not found — upload a PDF first")

    try:
        result = answer_chat_turn(document, body.message)
    except Exception:
        raise HTTPException(502, "Upstream AI call failed")

    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "sources": result["sources"],
    }
