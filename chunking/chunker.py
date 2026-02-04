import re
from typing import List, Dict

HEADING_RE = re.compile(
    r"(?im)^(?P<h>"
    r"(?:section|sec\.?)\s+\d+(?:\.\d+)*"
    r"|§\s*\d+(?:\.\d+)*"
    r"|\d+(?:\.\d+)+\s*[:\-]"
    r"|exhibit\s+[A-Z]\d*\b"
    r"|schedule\s+[A-Z0-9]+\b"
    r"|appendix\s+[A-Z0-9]+\b"
    r")\s*.*$"
)

def _para_split(text: str) -> List[str]:
    text = (text or "").replace("\r", "\n")
    parts = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in parts if p.strip()]

def split_into_section_blocks(text: str) -> List[Dict]:
    """
    Returns blocks: [{"heading": str|None, "text": str, "start": int, "end": int}]
    """
    text = (text or "").replace("\r", "\n").strip()
    if not text:
        return []

    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [{"heading": None, "text": text, "start": 0, "end": len(text)}]

    blocks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append(
            {
                "heading": m.group("h").strip(),
                "text": text[start:end].strip(),
                "start": start,
                "end": end,
            }
        )
    return blocks

def chunk_text(
    text: str,
    max_chars: int = 3000,
    overlap_chars: int = 300,   # <-- keep for compatibility (ignored)
    min_chars: int = 400,
    overlap_paragraphs: int = 1,
) -> List[Dict]:
    """
    Backward compatible:
    - accepts overlap_chars (ignored; we use overlap_paragraphs)
    - returns start_char/end_char (approx but consistent)
    - returns section_heading to improve quote traceability
    """
    raw = (text or "").replace("\r", "\n")
    blocks = split_into_section_blocks(raw)
    chunks: List[Dict] = []
    chunk_id = 0

    for block in blocks:
        heading = block["heading"]
        block_start = block["start"]
        paras = _para_split(block["text"])

        buf: List[str] = []
        buf_len = 0
        buf_start_char = block_start  # approximate start

        def flush(buf_end_char: int):
            nonlocal chunk_id, buf, buf_len, buf_start_char
            if not buf:
                return
            out = "\n\n".join(buf).strip()
            if len(out) >= min_chars:
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": out,
                        "section_heading": heading,
                        "start_char": max(buf_start_char, 0),
                        "end_char": max(buf_end_char, 0),
                    }
                )
                chunk_id += 1
            buf, buf_len = [], 0

        # Track end char approx within this block
        running = block_start

        for p in paras:
            p_len = len(p) + 2  # approx with separators

            if not buf:
                buf = [p]
                buf_len = p_len
                buf_start_char = running

            elif buf_len + p_len <= max_chars:
                buf.append(p)
                buf_len += p_len

            else:
                flush(running)
                # overlap by paragraphs (clean)
                overlap = buf[-overlap_paragraphs:] if overlap_paragraphs > 0 else []
                buf = overlap + [p]
                buf_len = sum(len(x) + 2 for x in buf)
                # approximate start shifts back a bit if we overlapped
                if overlap:
                    buf_start_char = max(running - sum(len(x) + 2 for x in overlap), block_start)
                else:
                    buf_start_char = running

            running += p_len

        flush(running)

    return chunks
