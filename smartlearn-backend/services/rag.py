"""rag.py — RAG pipeline helpers: text cleaning, page loading, chunking, embeddings, FAISS, retrieval."""

import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Union

from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()


# ── Text cleaning ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalize one extracted page of PDF text.

    Removes null bytes, soft hyphens, repeated whitespace, and noisy
    line breaks so downstream chunking sees clean paragraph text.
    """
    if not text:
        return ""

    # strip null bytes (common in PDF metadata)
    text = text.replace("\x00", "")

    # strip soft hyphens (invisible discretionary hyphens)
    text = text.replace("\xad", "")

    # collapse repeated whitespace (spaces, tabs, newlines) → single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ── Page loading ────────────────────────────────────────────────────────────

def extract_pages_for_rag(
    file_path: Union[str, Path],
    page_limit: Optional[int] = None,
) -> list[dict]:
    """Read a PDF page by page and return ``[{page, text}]`` records.

    - Keeps original PDF page numbers (1-indexed).
    - Skips pages whose extracted text is empty after cleaning.
    - No hard-coded page limit; an optional *page_limit* is available for
      very large files.

    Parameters
    ----------
    file_path : str or Path
        Path to the PDF file on disk.
    page_limit : int or None
        If set, raise ``ValueError`` when the PDF exceeds this many pages.

    Returns
    -------
    list[dict]
        ``{"page": int, "text": str}`` for each page with readable text.
    """
    file_path = Path(file_path)
    reader = PdfReader(str(file_path))

    if page_limit is not None and len(reader.pages) > page_limit:
        raise ValueError(
            f"PDF contains {len(reader.pages)} pages, exceeding the "
            f"limit of {page_limit}."
        )

    records: list[dict] = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw = (page.extract_text() or "").strip()
        cleaned = clean_text(raw)
        if cleaned:
            records.append({"page": page_number, "text": cleaned})

    return records


def extract_pages_from_bytes_for_rag(pdf_bytes: bytes) -> list[dict]:
    """Read PDF pages from uploaded bytes, returning ``[{page, text}]`` records.

    Keeps original PDF page numbers (1-indexed) and skips pages whose
    extracted text is empty after cleaning.  Designed for the backend
    upload route where the file arrives as an in-memory byte buffer.
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    records: list[dict] = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw = (page.extract_text() or "").strip()
        cleaned = clean_text(raw)
        if cleaned:
            records.append({"page": page_number, "text": cleaned})
    return records


# ── JSON artifact I/O ───────────────────────────────────────────────────────

def save_json(
    data: Any,
    output_path: Union[str, Path],
) -> Path:
    """Save *data* as a UTF-8 JSON file, creating parent folders as needed.

    Returns the resolved output path for convenience.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def load_json(input_path: Union[str, Path]) -> Any:
    """Read a JSON artifact back into Python."""
    input_path = Path(input_path)
    return json.loads(input_path.read_text(encoding="utf-8"))


# ── Chunking ─────────────────────────────────────────────────────────────────

def slice_long_text(text: str, chunk_size: int) -> list[str]:
    """Split a single oversized text block into pieces ≤ *chunk_size* chars.

    Tries to split on sentence boundaries (``. ``) first, then word
    boundaries (`` ``), then falls back to character-level cuts.
    """
    if len(text) <= chunk_size:
        return [text]

    pieces: list[str] = []
    remaining = text

    while len(remaining) > chunk_size:
        # Try period + space first
        cut = remaining.rfind(". ", 0, chunk_size)
        if cut == -1:
            # Try space
            cut = remaining.rfind(" ", 0, chunk_size)
        if cut == -1:
            # Fallback: hard cut
            cut = chunk_size

        pieces.append(remaining[: cut + 1].strip())
        remaining = remaining[cut + 1 :].strip()

    if remaining:
        pieces.append(remaining)

    return pieces


def chunk_by_paragraph(records: list[dict], chunk_size: int) -> list[dict]:
    """Split page *records* by paragraph boundaries, preserving page numbers.

    Splits each page's text on double-newline, then further splits any
    paragraph longer than *chunk_size* with :func:`slice_long_text`.
    """
    chunks: list[dict] = []
    chunk_index = 0

    for rec in records:
        paragraphs = rec["text"].split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            pieces = slice_long_text(para, chunk_size)
            for piece in pieces:
                chunks.append({
                    "chunk_id": f"chunk-{chunk_index:04d}",
                    "page": rec["page"],
                    "text": piece,
                    "chunk_mode": "paragraph",
                })
                chunk_index += 1

    return chunks


def chunk_by_characters(
    records: list[dict],
    chunk_size: int,
    overlap: int = 0,
) -> list[dict]:
    """Create fixed-size sliding-window chunks with optional *overlap*.

    When *overlap* is 0 the windows are plain non-overlapping slices.
    """
    chunks: list[dict] = []
    chunk_index = 0
    step = chunk_size - overlap
    step = max(step, 1)  # avoid zero or negative step

    for rec in records:
        text = rec["text"]
        start = 0
        while start < len(text):
            window = text[start : start + chunk_size].strip()
            if window:
                chunks.append({
                    "chunk_id": f"chunk-{chunk_index:04d}",
                    "page": rec["page"],
                    "text": window,
                    "chunk_mode": (
                        "character_overlap" if overlap > 0 else "character"
                    ),
                })
                chunk_index += 1
            start += step

    return chunks


def chunk_with_langchain_recursive(
    records: list[dict],
    chunk_size: int = 700,
    chunk_overlap: int = 120,
    separators: Optional[list[str]] = None,
) -> list[dict]:
    """Chunk pages with LangChain's ``RecursiveCharacterTextSplitter``.

    Tries larger separators first (``\\n\\n`` → ``\\n`` → `` `` → ``""``),
    so paragraph-like boundaries are preferred.  When PDF text is messy,
    this usually produces cleaner chunk starts than a plain paragraph split.

    Every returned chunk keeps the same schema as the rest of ``rag.py``:
    ``chunk_id``, ``page``, ``text``, ``chunk_mode``.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        raise ImportError(
            "langchain-text-splitters is required for langchain_recursive mode. "
            "Install it with: pip install langchain-text-splitters"
        )

    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks: list[dict] = []
    chunk_index = 0

    for rec in records:
        page_text = rec["text"]
        if not page_text.strip():
            continue
        page_chunks = splitter.split_text(page_text)
        for text in page_chunks:
            text = text.strip()
            if not text:
                continue
            chunks.append({
                "chunk_id": f"chunk-{chunk_index:04d}",
                "page": rec["page"],
                "text": text,
                "chunk_mode": "langchain_recursive",
            })
            chunk_index += 1

    return chunks


def build_chunks(
    records: list[dict],
    chunk_mode: str,
    chunk_size: int = 700,
    overlap: int = 0,
) -> list[dict]:
    """Dispatch to the correct chunking strategy and return uniform chunks."""
    if chunk_mode == "paragraph":
        return chunk_by_paragraph(records, chunk_size)
    elif chunk_mode == "character":
        return chunk_by_characters(records, chunk_size, overlap=0)
    elif chunk_mode == "character_overlap":
        return chunk_by_characters(records, chunk_size, overlap=overlap)
    elif chunk_mode == "langchain_recursive":
        return chunk_with_langchain_recursive(
            records, chunk_size=chunk_size, chunk_overlap=overlap,
        )
    else:
        raise ValueError(
            f"Unknown chunk_mode {chunk_mode!r}. "
            f"Expected: paragraph, character, character_overlap, "
            f"or langchain_recursive."
        )


# ── Embedding pipeline ──────────────────────────────────────────────────────

def model_tag(model_name: str) -> str:
    """Turn a model name into a safe filename suffix.

    ``"sentence-transformers/all-MiniLM-L6-v2"`` → ``"all_MiniLM_L6_v2"``
    """
    # Keep only the last part after the final "/" and sanitise
    short = model_name.rsplit("/", 1)[-1]
    return re.sub(r"[^a-zA-Z0-9]+", "_", short).strip("_")


def resolve_model_source(
    model_name: str,
    artifact_root: Union[str, Path, None] = None,
) -> Union[str, Path]:
    """Prefer a local cached model folder when it already exists.

    Checks ``<artifact_root>/hf_models/<tag>/`` first, then falls back
    to the original *model_name* string (which Hugging Face can download).
    """
    tag = model_tag(model_name)
    if artifact_root is not None:
        local = Path(artifact_root) / "hf_models" / tag
        if local.exists():
            return local
    return model_name


def get_device() -> str:
    """Return ``"cuda"`` if a GPU is available, otherwise ``"cpu"``."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


_load_model_cache: dict[str, Any] = {}


def load_model(
    model_name: str,
    model_source: Union[str, Path],
    device: str = "cpu",
):
    """Create (or reuse a cached) sentence-transformer model instance."""
    cache_key = f"{model_name}|{device}"
    if cache_key in _load_model_cache:
        return _load_model_cache[cache_key]

    import os
    from sentence_transformers import SentenceTransformer

    # If loading from a local folder, stay offline to avoid the HF Hub
    # "unauthenticated requests" warning every time the model is loaded.
    model_path = Path(str(model_source))
    if model_path.exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    model = SentenceTransformer(
        str(model_source),
        device=device,
        model_kwargs={"use_safetensors": False},
    )
    _load_model_cache[cache_key] = model
    return model


def embed_texts(
    texts: list[str],
    model_name: str,
    model_source: Union[str, Path, None] = None,
    model_cache_dir: Union[str, Path, None] = None,
    batch_size: int = 32,
    device: str = "cpu",
) -> "np.ndarray":
    """Encode a list of texts into normalised ``float32`` vectors.

    Returns a 2-D numpy array of shape ``(len(texts), dim)``.
    """
    import numpy as np

    source = model_source or model_name
    model = load_model(model_name, source, device=device)

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=(len(texts) > 50),
    )
    return np.asarray(vectors, dtype="float32")


def ensure_artifact_dirs(
    artifact_root: Union[str, Path, None] = None,
) -> dict[str, Path]:
    """Return a dict of artifact folder paths, creating them on demand."""
    if artifact_root is None:
        artifact_root = Path("artifacts")
    else:
        artifact_root = Path(artifact_root)

    dirs = {
        "root": artifact_root,
        "raw_pages": artifact_root / "raw_pages",
        "chunks": artifact_root / "chunks",
        "embeddings": artifact_root / "embeddings",
        "indexes": artifact_root / "indexes",
        "chroma": artifact_root / "chroma",
        "reports": artifact_root / "reports",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def artifact_paths_for(
    document_id: str,
    chunk_mode: str,
    model_name: str,
    chunk_size: int,
    overlap: int,
    artifact_root: Union[str, Path, None] = None,
) -> dict[str, Path]:
    """Decide where to save pages, chunks, embeddings, manifests, and indexes."""
    dirs = ensure_artifact_dirs(artifact_root)
    tag = model_tag(model_name)
    safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", document_id)

    prefix = f"{safe_id}_{chunk_mode}"
    if chunk_mode in ("character", "character_overlap"):
        prefix += f"_c{chunk_size}"
        if overlap:
            prefix += f"_o{overlap}"
    prefix += f"_{tag}"

    return {
        "raw_pages": dirs["raw_pages"] / f"{safe_id}_pages.json",
        "chunks": dirs["chunks"] / f"{prefix}.json",
        "embeddings": dirs["embeddings"] / f"{prefix}.npy",
        "manifest": dirs["embeddings"] / f"{prefix}.manifest.json",
        "index": dirs["indexes"] / f"{prefix}.faiss",
    }


def ensure_artifacts(
    document_id: str,
    pdf_name: str,
    pages: list[dict],
    chunk_mode: str = "character_overlap",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 700,
    overlap: int = 120,
    batch_size: int = 32,
    artifact_root: Union[str, Path, None] = None,
) -> dict:
    """Build (or reuse) the full pages → chunks → embeddings → manifest bundle.

    Returns a dict with keys ``chunks``, ``embeddings``, and ``manifest``.
    """
    import numpy as np

    paths = artifact_paths_for(
        document_id, chunk_mode, model_name, chunk_size, overlap, artifact_root,
    )

    # ── chunks ──
    if paths["chunks"].exists():
        chunks = load_json(paths["chunks"])
    else:
        chunks = build_chunks(pages, chunk_mode=chunk_mode, chunk_size=chunk_size, overlap=overlap)
        save_json(chunks, paths["chunks"])

    # ── embeddings ──
    if paths["embeddings"].exists():
        embeddings = np.load(paths["embeddings"], allow_pickle=False)
    else:
        device = get_device()
        model_source = resolve_model_source(model_name, artifact_root)
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(
            texts,
            model_name,
            model_source=model_source,
            batch_size=batch_size,
            device=device,
        )
        np.save(paths["embeddings"], embeddings)

    # ── manifest ──
    manifest = {
        "document_id": document_id,
        "pdf_name": pdf_name,
        "num_pages": len(pages),
        "chunk_mode": chunk_mode,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "model_name": model_name,
        "num_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "device": get_device(),
        "chunk_path": str(paths["chunks"]),
        "embedding_path": str(paths["embeddings"]),
        "raw_pages_path": str(paths["raw_pages"]),
    }
    save_json(manifest, paths["manifest"])

    return {"chunks": chunks, "embeddings": embeddings, "manifest": manifest}


# ── FAISS index ──────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: "np.ndarray"):
    """Build a FAISS inner-product index from normalised embedding vectors.

    When vectors are L2-normalised, inner product equals cosine similarity,
    so ``IndexFlatIP`` gives the same ranking as cosine search without the
    extra normalisation step at query time.
    """
    import faiss
    import numpy as np

    embeddings = np.asarray(embeddings, dtype="float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_faiss_index(index, index_path: Union[str, Path]) -> None:
    """Write a FAISS index to a binary ``.faiss`` file on disk."""
    import faiss

    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


def load_faiss_index(index_path: Union[str, Path]):
    """Read a saved FAISS index back into memory."""
    import faiss

    return faiss.read_index(str(index_path))


def ensure_index(
    document_id: str,
    pdf_name: str,
    pages: Optional[list[dict]] = None,
    pdf_path: Union[str, Path, None] = None,
    chunk_mode: str = "character_overlap",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 700,
    overlap: int = 120,
    batch_size: int = 32,
    artifact_root: Union[str, Path, None] = None,
) -> dict:
    """Build (or reuse) chunks, embeddings, and a FAISS index for one PDF.

    Returns a bundle with ``chunks``, ``embeddings``, ``manifest``, and
    ``faiss_index`` so callers can search immediately.
    """
    import numpy as np

    # Resolve pages: use cached, extract from pdf_path, or use caller-supplied
    if pages is None and pdf_path is not None:
        pages = extract_pages_for_rag(pdf_path)

    if pages is None:
        raise ValueError("Either pages or pdf_path must be provided.")

    paths = artifact_paths_for(
        document_id, chunk_mode, model_name, chunk_size, overlap, artifact_root,
    )

    # ── chunks + embeddings (reuse Lab A) ──
    if paths["chunks"].exists():
        chunks = load_json(paths["chunks"])
    else:
        chunks = build_chunks(pages, chunk_mode=chunk_mode, chunk_size=chunk_size, overlap=overlap)
        save_json(chunks, paths["chunks"])

    if paths["embeddings"].exists():
        embeddings = np.load(paths["embeddings"], allow_pickle=False)
    else:
        device = get_device()
        model_source = resolve_model_source(model_name, artifact_root)
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(
            texts, model_name, model_source=model_source,
            batch_size=batch_size, device=device,
        )
        np.save(paths["embeddings"], embeddings)

    # ── FAISS index ──
    if paths["index"].exists():
        index = load_faiss_index(paths["index"])
    else:
        index = build_faiss_index(embeddings)
        save_faiss_index(index, paths["index"])

    # ── model source (for later question embedding) ──
    model_source = resolve_model_source(model_name, artifact_root)

    # ── manifest ──
    manifest = {
        "document_id": document_id,
        "pdf_name": pdf_name,
        "num_pages": len(pages),
        "chunk_mode": chunk_mode,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "model_name": model_name,
        "model_source": str(model_source),
        "num_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "device": get_device(),
        "chunk_path": str(paths["chunks"]),
        "embedding_path": str(paths["embeddings"]),
        "raw_pages_path": str(paths["raw_pages"]),
        "index_path": str(paths["index"]),
    }
    save_json(manifest, paths["manifest"])

    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "faiss_index": index,
        "manifest": manifest,
        "model_source": model_source,
    }


# ── Project-facing document wrapper ──────────────────────────────────────────

def prepare_rag_document(
    document_id: str,
    filename: str,
    pages: list[dict],
    chunk_mode: str = "character_overlap",
    chunk_size: int = 700,
    overlap: int = 120,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    artifact_root: Union[str, Path, None] = None,
) -> dict:
    """Package one PDF into a server-side document record.

    The returned dict is ready to be stored in ``documents[document_id]``
    and reused across multiple questions without rebuilding the index.
    """
    bundle = ensure_index(
        document_id=document_id,
        pdf_name=filename,
        pages=pages,
        chunk_mode=chunk_mode,
        model_name=model_name,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size,
        artifact_root=artifact_root,
    )

    device = get_device()
    model_source = resolve_model_source(model_name, artifact_root)

    return {
        "document_id": document_id,
        "filename": filename,
        "pages": pages,
        "chunks": bundle["chunks"],
        "model_name": model_name,
        "model_source": str(model_source),
        "chunk_size": len(bundle["chunks"]),
        "embedding_dim": bundle["manifest"]["embedding_dim"],
        "artifacts": {
            "index": bundle["manifest"]["index_path"],
            "chunks": bundle["manifest"]["chunk_path"],
            "embeddings": bundle["manifest"]["embedding_path"],
            "raw_pages": bundle["manifest"]["raw_pages_path"],
        },
        "history": [],
    }


def prepare_rag_chat_record(
    chat_id: str,
    filename: str,
    pdf_bytes: Optional[bytes] = None,
    pages: Optional[list[dict]] = None,
    upload_root: Optional[Union[str, Path]] = None,
    chunk_mode: str = "character_overlap",
    chunk_size: int = 700,
    overlap: int = 120,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    artifact_root: Optional[Union[str, Path]] = None,
) -> dict:
    """Build a complete upload-time record for one chat session.

    Resolves pages from *pdf_bytes* when not provided, saves the uploaded
    PDF to *upload_root*, builds chunks/embeddings/FAISS index via
    :func:`prepare_rag_document`, and returns a dict ready to store in
    ``documents[chat_id]`` with an empty history list.
    """
    # Resolve pages
    if pages is None and pdf_bytes is not None:
        pages = extract_pages_from_bytes_for_rag(pdf_bytes)
    if pages is None:
        raise ValueError("Either pdf_bytes or pages must be provided.")

    # Save uploaded PDF to disk
    saved_pdf_path = ""
    if upload_root is not None:
        upload_root = Path(upload_root)
        upload_root.mkdir(parents=True, exist_ok=True)
        saved_pdf_path = str(upload_root / f"{chat_id}.pdf")
        if pdf_bytes is not None:
            Path(saved_pdf_path).write_bytes(pdf_bytes)

    # Delete old cached artifacts so a fresh upload always rebuilds
    artifact_paths = artifact_paths_for(
        chat_id, chunk_mode, model_name, chunk_size, overlap, artifact_root,
    )
    for key in ("chunks", "embeddings", "manifest", "index"):
        cached = artifact_paths.get(key)
        if cached and Path(cached).exists():
            Path(cached).unlink()

    # Build RAG document (chunks, embeddings, FAISS index)
    doc = prepare_rag_document(
        document_id=chat_id,
        filename=filename,
        pages=pages,
        chunk_mode=chunk_mode,
        chunk_size=chunk_size,
        overlap=overlap,
        model_name=model_name,
        batch_size=batch_size,
        artifact_root=artifact_root,
    )

    doc["chat_id"] = chat_id
    doc["file_path"] = saved_pdf_path
    doc["saved_pdf_path"] = saved_pdf_path

    return doc


def build_upload_response(document: dict) -> dict:
    """Build the Day 2-compatible upload success JSON from a Day 3 record.

    Keeps the visible frontend contract stable while the server stores a
    richer RAG-ready record internally.
    """
    total_chars = sum(len(p["text"]) for p in document["pages"])
    return {
        "status": "ok",
        "filename": document["filename"],
        "pages": len(document["pages"]),
        "characters": total_chars,
        "chat_id": document.get("chat_id", ""),
    }


# ── Retrieval helpers ────────────────────────────────────────────────────────

def keyword_set(text: str) -> set[str]:
    """Return lightweight lexical tokens from *text* for simple reranking.

    Lowercases, then extracts word-character runs so that e.g. "BM25"
    and "Llama-3-70B" stay as single tokens.
    """
    return set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()))


def split_sentences(text: str) -> list[str]:
    """Split *text* into candidate answer sentences.

    Splits on ``. `` ``! `` ``? `` and newline-then-sentence-start,
    keeping only fragments longer than 10 characters.
    """
    raw = re.split(r"(?<=[.!?])\s+|\n(?=[A-Z])", text)
    return [s.strip() for s in raw if len(s.strip()) > 10]


def search_bundle(
    question: str,
    bundle: dict,
    top_k: int = 3,
    candidate_pool: int = 60,
    batch_size: int = 1,
    history: Optional[list[dict]] = None,
) -> list[dict]:
    """Search an in-memory FAISS index bundle for the top-k chunks.

    Retrieves *candidate_pool* nearest neighbours, applies a light
    lexical boost, then returns the top *top_k* hits with page,
    chunk_id, text, and score fields.
    """
    import numpy as np

    chunks = bundle["chunks"]
    faiss_index = bundle["faiss_index"]
    manifest = bundle["manifest"]

    model_name = manifest.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    model_source = bundle.get("model_source") or manifest.get("model_source") or model_name
    device = get_device()

    # ── embed question (with history context when available) ──
    search_question = question
    if history:
        recent = history[-3:]
        # Only use previous questions (not full answers) to avoid noise
        history_prefix = " | ".join(
            f"Previous: {turn['question']} (page {turn.get('citations', ['?'])[0]})"
            for turn in recent
        )
        search_question = f"{history_prefix} | Now: {question}"

    q_vec = embed_texts(
        [search_question],
        model_name,
        model_source=model_source,
        batch_size=max(batch_size, 1),
        device=device,
    )

    # ── FAISS search ──
    k = min(candidate_pool, faiss_index.ntotal)
    scores, indices = faiss_index.search(q_vec, k)

    # ── map to hits with light lexical rerank ──
    q_keywords = keyword_set(question)
    raw_hits: list[dict] = []

    for i in range(k):
        idx = indices[0][i]
        if idx == -1:          # FAISS padding for short indexes
            continue
        score = float(scores[0][i])
        chunk = chunks[idx]

        # Light lexical boost: each overlapping keyword adds a small bonus
        chunk_keywords = keyword_set(chunk["text"])
        overlap = len(q_keywords & chunk_keywords)
        combined = score * (1.0 + 0.05 * overlap)

        # Boost chunks from pages cited in conversation history
        if history:
            cited_pages = set()
            for turn in history:
                cited_pages.update(turn.get("citations", []))
            if chunk["page"] in cited_pages:
                combined *= 1.1  # 10% boost for previously cited pages

        raw_hits.append({
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "score": round(combined, 4),
            "vector_score": round(score, 4),
        })

    # Re-rank by combined score, then pick top-1 chunk per page for diversity
    raw_hits.sort(key=lambda h: h["score"], reverse=True)
    diverse: list[dict] = []
    seen_pages: set[int] = set()
    for h in raw_hits:
        if h["page"] not in seen_pages:
            diverse.append(h)
            seen_pages.add(h["page"])
        if len(diverse) >= top_k:
            break
    return diverse


def search_document(
    question: str,
    document: dict,
    top_k: int = 3,
    candidate_pool: int = 60,
    history: Optional[list[dict]] = None,
) -> list[dict]:
    """Load the saved FAISS index for *document* and return top-k hits.

    This is the project-facing entry point: it reads the ``.faiss`` file
    from disk, rebuilds an in-memory bundle, and delegates to
    :func:`search_bundle`.
    """
    index_path = document["artifacts"]["index"]
    faiss_index = load_faiss_index(index_path)

    chunks = document.get("chunks")
    if chunks is None:
        chunks = load_json(document["artifacts"]["chunks"])

    bundle = {
        "chunks": chunks,
        "faiss_index": faiss_index,
        "manifest": {
            "model_name": document["model_name"],
            "model_source": document.get("model_source", document["model_name"]),
        },
        "model_source": document.get("model_source", document["model_name"]),
    }

    return search_bundle(
        question, bundle,
        top_k=top_k,
        candidate_pool=candidate_pool,
        history=history,
    )


def best_sentence_answer(question: str, hits: list[dict]) -> str:
    """Return one short answer sentence from *hits* with a page tag.

    Splits every hit into sentences, scores each by keyword overlap with
    the question, and keeps the best one.  Falls back to the first 200
    chars of the top hit when no sentence is long enough.
    """
    if not hits:
        return "No relevant passage found."

    # Collect candidate sentences
    candidates: list[tuple[str, int]] = []
    for hit in hits:
        for sent in split_sentences(hit["text"]):
            candidates.append((sent, hit["page"]))

    if not candidates:
        return f"{hits[0]['text'][:200]} [page {hits[0]['page']}]"

    # Best by keyword overlap
    q_keywords = keyword_set(question)
    best_sent, best_page, best_score = "", 1, -1

    for sent, page in candidates:
        sent_keywords = keyword_set(sent)
        overlap = len(q_keywords & sent_keywords)
        if overlap > best_score:
            best_score = overlap
            best_sent = sent
            best_page = page

    if not best_sent:
        best_sent, best_page = candidates[0]

    return f"{best_sent} [page {best_page}]"


# ── Project-facing answer wrapper ────────────────────────────────────────────

def extract_citations(
    answer: str,
    hits: Optional[list[dict]] = None,
) -> list[int]:
    """Return numeric PDF page citations from an answer string.

    Looks for ``[page N]`` tags in *answer* first, then falls back to
    unique page numbers in *hits*.
    """
    pages: set[int] = set()

    # Parse [page N] tags
    for match in re.finditer(r"\[page\s*(\d+)\]", answer):
        pages.add(int(match.group(1)))

    # Fallback: pages from retrieval hits
    if not pages and hits:
        pages.update(hit["page"] for hit in hits)

    return sorted(pages)


def build_sources(hits: list[dict]) -> list[dict]:
    """Convert retrieval hits into frontend-friendly source objects.

    Each source carries *page*, *chunk_id*, *score*, and a *preview*
    (first 200 chars of the chunk text).
    """
    return [
        {
            "page": h["page"],
            "chunk_id": h["chunk_id"],
            "score": h["score"],
            "preview": h["text"][:200],
        }
        for h in hits
    ]


def build_grounded_user_prompt(
    question: str,
    hits: list[dict],
    history: Optional[list[dict]] = None,
) -> str:
    """Build a grounded prompt with retrieved chunks and optional chat history.

    Includes the last 5 conversation turns (when available) so the LLM can
    use earlier context while still grounding its answer in fresh retrieval.
    """
    context = "\n\n".join(
        f"[{h['chunk_id']} page {h['page']}] {h['text']}" for h in hits
    )

    prompt_parts = [
        "You are a helpful teaching assistant. Answer the question "
        "based on the provided document context. Do not make up information "
        "that is not in the context. Keep the answer concise. "
        "Include [page N] at the end of your answer.\n",
    ]

    if history:
        recent = history[-5:]  # keep context manageable
        history_text = "\n".join(
            f"Q: {turn['question']}\nA: {turn['answer']}"
            for turn in recent
        )
        prompt_parts.append(f"Previous conversation:\n{history_text}\n")

    prompt_parts.append(f"Context:\n{context}\n")
    prompt_parts.append(f"Question: {question}\n")
    prompt_parts.append("Answer:")

    return "\n".join(prompt_parts)


def answer_document(
    document: dict,
    question: str,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "deepseek-chat",
) -> dict:
    """Answer one question against a prepared document record.

    Always returns ``answer``, ``citations``, and ``sources``:

    - When ``OPENROUTER_API_KEY`` is set, the LLM answers from
      retrieved chunks.
    - When the key is missing, a local sentence is extracted from
      the top retrieval hits.
    """
    # ── retrieval ──
    hits = search_document(
        question, document,
        top_k=top_k,
        candidate_pool=candidate_pool,
        history=document.get("history"),
    )

    # ── answer ──
    api_key = __import__("os").environ.get("OPENROUTER_API_KEY", "")
    if api_key:
        # LLM path: pass retrieved chunks and history as context
        answer = _llm_answer(question, hits, answer_model, api_key, history=document.get("history"))
    else:
        answer = best_sentence_answer(question, hits)

    # ── citations + sources ──
    citations = extract_citations(answer, hits)
    sources = build_sources(hits)

    return {
        "answer": answer,
        "citations": citations,
        "sources": sources,
    }


def _llm_answer(
    question: str,
    hits: list[dict],
    model: str,
    api_key: str,
    history: Optional[list[dict]] = None,
) -> str:
    """Ask an LLM to answer *question* using the retrieved *hits* as context.

    When *history* is provided, includes recent conversation turns so the
    LLM can resolve pronouns like "that page" correctly.
    """
    prompt = build_grounded_user_prompt(question, hits, history=history)
    try:
        import openai
        client = openai.OpenAI(
            base_url="https://api.deepseek.com",
            api_key=api_key,
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=256,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # Fall back to local extraction on any LLM error
        return best_sentence_answer(question, hits)


def append_history(
    document: dict,
    question: str,
    result: dict,
) -> list[dict]:
    """Append a Q&A turn to the document's in-memory history and return it."""
    document["history"].append({
        "question": question,
        "answer": result["answer"],
        "citations": result["citations"],
    })
    return document["history"]


def answer_document_turn(
    document: dict,
    question: str,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "deepseek-chat",
) -> dict:
    """Answer one question and update in-memory history.

    Calls :func:`answer_document` for retrieval + answering, then appends
    the turn via :func:`append_history`.  Returns the answer result with
    an extra ``history`` key so callers can inspect the updated list.
    """
    result = answer_document(
        document, question,
        top_k=top_k,
        candidate_pool=candidate_pool,
        answer_model=answer_model,
    )
    append_history(document, question, result)
    result["history"] = document["history"]
    return result


def answer_chat_turn(
    document: dict,
    message: str,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "deepseek-chat",
) -> dict:
    """Route-level entry point for one chat turn.

    Retrieves fresh evidence, answers the question, and stores the turn
    in the document's in-memory history.  Designed to be called directly
    from the ``POST /chat`` route.
    """
    return answer_document_turn(
        document, message,
        top_k=top_k,
        candidate_pool=candidate_pool,
        answer_model=answer_model,
    )


# ── Evaluation helpers ────────────────────────────────────────────────────────

def normalize_for_match(text: str) -> str:
    """Normalize *text* for simple string-based scoring.

    Lowercases, collapses whitespace, and strips punctuation so that
    ``"BM25"`` and ``"  bm25  "`` match each other.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip().rstrip(".")
    return text


def contains_any_answer(text: str, answers: list[str]) -> bool:
    """Return ``True`` when *text* contains at least one acceptable answer.

    Both *text* and every answer are normalised before comparison.
    """
    norm_text = normalize_for_match(text)
    return any(normalize_for_match(a) in norm_text for a in answers)


def evaluate_questions(
    eval_set: list[dict],
    documents_by_name: dict[str, dict],
    top_k: int = 3,
    candidate_pool: int = 60,
) -> "pd.DataFrame":
    """Run the *eval_set* against prepared documents and return a score table.

    Each row records the question, retrieved pages, local answer,
    ``retrieval_hit`` (gold answer appears in any retrieved chunk), and
    ``answer_hit`` (gold answer appears in the extracted answer sentence).
    """
    import pandas as pd

    rows: list[dict] = []

    for item in eval_set:
        pdf_name = item["pdf_name"]
        question = item["question"]
        gold_answers = item["answers"]

        doc = documents_by_name.get(pdf_name)
        if doc is None:
            rows.append({
                "pdf_name": pdf_name,
                "question": question,
                "pages": [],
                "local_answer": "DOCUMENT NOT FOUND",
                "retrieval_hit": False,
                "answer_hit": False,
            })
            continue

        # retrieval
        hits = search_document(
            question, doc,
            top_k=top_k,
            candidate_pool=candidate_pool,
        )
        local_answer = best_sentence_answer(question, hits)

        # scoring
        all_chunk_text = " ".join(h["text"] for h in hits)
        retrieval_hit = contains_any_answer(all_chunk_text, gold_answers)
        answer_hit = contains_any_answer(local_answer, gold_answers)

        rows.append({
            "pdf_name": pdf_name,
            "question": question,
            "pages": sorted({h["page"] for h in hits}),
            "local_answer": local_answer,
            "retrieval_hit": retrieval_hit,
            "answer_hit": answer_hit,
        })

    return pd.DataFrame(rows)


# ── Optional Chroma branch ───────────────────────────────────────────────────

def _require_chromadb():
    """Import ``chromadb`` or raise a clear ``ImportError``."""
    try:
        import chromadb
        return chromadb
    except ImportError:
        raise ImportError(
            "chromadb is required for the Chroma path. "
            "Install it with: pip install chromadb"
        )


def build_chroma_collection(
    document_id: str,
    chunks: list[dict],
    embeddings: "np.ndarray",
    persist_dir: Union[str, Path],
) -> dict:
    """Create (or reopen) a persistent Chroma collection from *chunks* and *embeddings*.

    Returns collection metadata: ``collection_name`` and ``item_count``.
    """
    import numpy as np

    chromadb = _require_chromadb()
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name=document_id)

    # Only add if the collection is empty (idempotent)
    if collection.count() == 0:
        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [{"page": c["page"], "chunk_id": c["chunk_id"]} for c in chunks]
        emb_list = np.asarray(embeddings, dtype="float32").tolist()

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=emb_list,
        )

    return {
        "collection_name": document_id,
        "item_count": collection.count(),
    }


def query_chroma_collection(
    document_id: str,
    query_embedding: "np.ndarray",
    persist_dir: Union[str, Path],
    top_k: int = 3,
) -> list[dict]:
    """Query a persistent Chroma collection for top-k matches.

    Returns hits with ``chunk_id``, ``page``, ``text``, and ``score``.
    """
    import numpy as np

    chromadb = _require_chromadb()
    persist_dir = Path(persist_dir)

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(name=document_id)

    q_vec = np.asarray(query_embedding, dtype="float32").tolist()
    result = collection.query(
        query_embeddings=q_vec,
        n_results=min(top_k, collection.count()),
    )

    hits: list[dict] = []
    if result["ids"] and result["ids"][0]:
        for i, chunk_id in enumerate(result["ids"][0]):
            metadata = result["metadatas"][0][i] if result["metadatas"] else {}
            doc_text = result["documents"][0][i] if result["documents"] else ""
            distance = result["distances"][0][i] if result["distances"] else 0
            hits.append({
                "chunk_id": chunk_id,
                "page": metadata.get("page", 1),
                "text": doc_text,
                "score": round(1.0 - float(distance), 4),  # distance → similarity
            })

    return hits


def search_document_with_chroma(
    question: str,
    document: dict,
    persist_dir: Union[str, Path],
    top_k: int = 3,
    batch_size: int = 1,
) -> list[dict]:
    """Search one document via its Chroma collection.

    Embeds *question*, queries the persisted collection, and returns
    hits in the same shape as :func:`search_document`.
    """
    device = get_device()
    model_name = document["model_name"]
    model_source = document.get("model_source", model_name)

    q_vec = embed_texts(
        [question], model_name,
        model_source=model_source, batch_size=max(batch_size, 1), device=device,
    )

    return query_chroma_collection(
        document["document_id"], q_vec, persist_dir, top_k=top_k,
    )


def answer_document_with_chroma(
    document: dict,
    question: str,
    persist_dir: Union[str, Path],
    top_k: int = 3,
    answer_model: str = "deepseek-chat",
) -> dict:
    """Answer a question using the Chroma retrieval path.

    Returns the same ``{answer, citations, sources}`` shape as
    :func:`answer_document`.
    """
    hits = search_document_with_chroma(
        question, document, persist_dir, top_k=top_k,
    )

    api_key = __import__("os").environ.get("OPENROUTER_API_KEY", "")
    if api_key:
        answer = _llm_answer(question, hits, answer_model, api_key)
    else:
        answer = best_sentence_answer(question, hits)

    return {
        "answer": answer,
        "citations": extract_citations(answer, hits),
        "sources": build_sources(hits),
    }


# ── Preview helpers ─────────────────────────────────────────────────────────

def preview_records(
    records: list[dict],
    columns: list[str],
    rows: int = 5,
):
    """Return a pandas DataFrame preview of *records* for the chosen columns."""
    import pandas as pd

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    usable = [col for col in columns if col in frame.columns]
    return frame[usable].head(rows)


def relative_path_str(path: Union[str, Path], base: Union[str, Path]) -> str:
    """Return *path* as a string relative to *base*, for display purposes."""
    try:
        return str(Path(path).relative_to(Path(base)))
    except ValueError:
        return str(path)
