"""rag.py — RAG pipeline helpers: text cleaning, page loading, chunking, embeddings, FAISS, retrieval."""

import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Union

from pypdf import PdfReader


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
    else:
        raise ValueError(
            f"Unknown chunk_mode {chunk_mode!r}. "
            f"Expected: paragraph, character, or character_overlap."
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

    from sentence_transformers import SentenceTransformer

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
