"""
doc_parse.py — PDF / Word 抽取为带页码或类型的文本块，供入库与溯源
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chunking import split_text


@dataclass(frozen=True)
class SourcedChunk:
    """单条待嵌入文本及其溯源字段（写入 Chroma metadata）。"""

    text: str
    page: str
    file_type: str


def _chunks_from_plain_text(text: str, *, chunk_chars: int, chunk_overlap: int, file_type: str) -> list[SourcedChunk]:
    parts = split_text(text, chunk_chars, chunk_overlap)
    return [SourcedChunk(text=p, page="", file_type=file_type) for p in parts]


def parse_txt_or_md(path: Path, *, chunk_chars: int, chunk_overlap: int) -> list[SourcedChunk]:
    ext = path.suffix.lower()
    ft = "md" if ext == ".md" else "txt"
    raw = path.read_text(encoding="utf-8")
    return _chunks_from_plain_text(raw, chunk_chars=chunk_chars, chunk_overlap=chunk_overlap, file_type=ft)


def parse_pdf(path: Path, *, chunk_chars: int, chunk_overlap: int) -> list[SourcedChunk]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    out: list[SourcedChunk] = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        page_label = str(i + 1)
        for piece in split_text(text, chunk_chars, chunk_overlap):
            out.append(SourcedChunk(text=piece, page=page_label, file_type="pdf"))
    return out


def parse_docx(path: Path, *, chunk_chars: int, chunk_overlap: int) -> list[SourcedChunk]:
    from docx import Document

    doc = Document(str(path))
    paras: list[str] = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            paras.append(t)
    raw = "\n\n".join(paras)
    if not raw.strip():
        return []
    parts = split_text(raw, chunk_chars, chunk_overlap)
    return [SourcedChunk(text=p, page="", file_type="docx") for p in parts]


def parse_document(path: Path, *, chunk_chars: int, chunk_overlap: int) -> list[SourcedChunk]:
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return parse_txt_or_md(path, chunk_chars=chunk_chars, chunk_overlap=chunk_overlap)
    if ext == ".pdf":
        return parse_pdf(path, chunk_chars=chunk_chars, chunk_overlap=chunk_overlap)
    if ext == ".docx":
        return parse_docx(path, chunk_chars=chunk_chars, chunk_overlap=chunk_overlap)
    return []
