from typing import List
import logfire


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 300) -> List[str]:
    """
    Semantic-ish chunker that splits by paragraphs, packing greedily up to
    chunk_size. Carries the tail paragraph(s) of each chunk (up to `overlap`
    chars) forward into the start of the next chunk.

    FIX: the original version had zero overlap between chunks — a compound
    explanation that happened to straddle a chunk boundary (common in prose
    docs: "X is defined as... [boundary] ...and you configure it via Y()")
    got permanently split with no shared context, so no single retrieved
    chunk ever carried the whole concept. Diagnosed via keyword-overlap
    analysis against golden ground truths: retrieval was reliably finding
    *a* relevant chunk, but that chunk only ever covered part of the answer.
    Also bumped default chunk_size 1500 -> 2000 chars, since compound
    technical explanations (checkpointer setup, memory config, etc.) often
    need more than ~250 words to state fully.
    """
    with logfire.span("Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current_parts: List[str] = []
        current_len = 0

        def flush(carry_overlap: bool):
            """Finalize current_parts into a chunk. If carry_overlap, seed
            the next chunk's current_parts with the tail paragraph(s) of
            this chunk (up to `overlap` chars), so context isn't lost at
            the boundary."""
            nonlocal current_parts, current_len
            if not current_parts:
                return
            chunk_text_out = "\n\n".join(current_parts).strip()
            if chunk_text_out:
                chunks.append(chunk_text_out)

            if carry_overlap:
                tail: List[str] = []
                tail_len = 0
                for part in reversed(current_parts):
                    if tail_len + len(part) > overlap and tail:
                        break
                    tail.insert(0, part)
                    tail_len += len(part)
                current_parts = tail
                current_len = tail_len
            else:
                current_parts = []
                current_len = 0

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue

            # Hard-split any paragraph that is itself larger than chunk_size
            if len(p) >= chunk_size:
                flush(carry_overlap=False)
                step = max(chunk_size - overlap, 1)
                for start in range(0, len(p), step):
                    piece = p[start:start + chunk_size].strip()
                    if piece:
                        chunks.append(piece)
                continue

            if current_len + len(p) < chunk_size:
                current_parts.append(p)
                current_len += len(p)
            else:
                flush(carry_overlap=True)
                current_parts.append(p)
                current_len += len(p)

        flush(carry_overlap=False)

        valid_chunks = [c for c in chunks if c.strip()]
        logfire.info(f"Generated {len(valid_chunks)} chunks")
        return valid_chunks