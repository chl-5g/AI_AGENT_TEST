"""
chunking.py — 纯文本切片（无环境依赖，供 ingestion / doc_parse 共用）
"""

from __future__ import annotations


def split_long_paragraph(para: str, chunk_chars: int, chunk_overlap: int) -> list[str]:
    para = para.strip()
    if not para:
        return []
    if len(para) <= chunk_chars:
        return [para]

    chunks: list[str] = []
    i = 0
    n = len(para)
    while i < n:
        end = min(i + chunk_chars, n)
        piece = para[i:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        i = end - chunk_overlap
        if i < 0:
            i = 0
    return chunks


def split_text(text: str, chunk_chars: int, chunk_overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []

    out: list[str] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        out.extend(split_long_paragraph(block, chunk_chars, chunk_overlap))
    return out
