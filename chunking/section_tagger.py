import re
from typing import Optional

SECTION_RE = re.compile(r'\bSection\s+(\d+(?:\.\d+)*)\b', re.IGNORECASE)
PLAIN_NUM_RE = re.compile(r'^\s*(\d+(?:\.\d+)+)\s+', re.MULTILINE)
EXHIBIT_RE = re.compile(r'\bExhibit\s+([A-Z]\d*|[A-Z])\b', re.IGNORECASE)

def find_section_label(text: str) -> Optional[str]:
    """
    Returns a best-effort labels
    """
    m = SECTION_RE.search(text)
    if m:
        return f"Section {m.group(1)}"

    m = EXHIBIT_RE.search(text)
    if m:
        return f"Exhibit {m.group(1).upper()}"

    m = PLAIN_NUM_RE.search(text)
    if m:
        return f"Section {m.group(1)}"

    return None
