
from typing import List, Dict, Any
import re


def _map_state(status: str) -> str:
    """
    Supports:
      - "Compliant" | "Partial" | "Non-Compliant"
      - "Fully Compliant" | "Partially Compliant" | "Non-Compliant"
    """
    if not status:
        return ""

    s = status.strip()

    if s in ("Fully Compliant", "Partially Compliant", "Non-Compliant"):
        return s

    if s == "Compliant":
        return "Fully Compliant"
    if s == "Partial":
        return "Partially Compliant"
    return "Non-Compliant"


def _to_int_conf(v: Any) -> int:
    """Supports 95, '95', '95%', None."""
    if v is None:
        return 0
    if isinstance(v, str):
        v = v.strip().replace("%", "")
    try:
        return max(0, min(100, int(float(v))))
    except Exception:
        return 0


def _collect_grouped_quotes(
    result: Dict,
    max_sources: int = 8,
    max_quotes_per_source: int = 2,
) -> str:
    """
    Returns REAL contract excerpts grouped by Section/Exhibit label.

    Example:
      Section 6.7:
      - SAML 2.0 SSO is supported...
      - API access will use OAuth 2.0...

      Exhibit G13:
      - NET-01–NET-03 ...
    """
    groups: Dict[str, List[str]] = {}

    def norm_label(label: str, quote: str) -> str:
        # Prefer explicit "Section X.Y" / "Exhibit A" from label or quote
        blob = f"{label or ''} {quote or ''}"
        m = re.search(r"(Section\s+\d+(?:\.\d+)*|Exhibit\s+[A-Z]\d*)", blob, flags=re.IGNORECASE)
        if m:
            # normalize casing: "Section 6.7" / "Exhibit A"
            ref = m.group(1).strip()
            ref = re.sub(r"\s+", " ", ref)
            # Titlecase first word only
            if ref.lower().startswith("section"):
                return "Section " + ref.split(" ", 1)[1]
            if ref.lower().startswith("exhibit"):
                return "Exhibit " + ref.split(" ", 1)[1]
            return ref
        return (label or "Contract").strip() or "Contract"

    def add(label: str, quote: str):
        q = (quote or "").strip()
        if not q:
            return

        lab = norm_label(label, q)
        if lab not in groups:
            groups[lab] = []
        # Deduplicate identical quotes
        if q not in groups[lab]:
            groups[lab].append(q)

    # Prefer analyzer-style: controls[].evidence[]
    controls = result.get("controls") or []
    for c in controls:
        for e in (c.get("evidence") or []):
            add(e.get("label", ""), e.get("quote", ""))

    # Fallback: schema-style "Relevant Quotes"
    rq = result.get("Relevant Quotes")
    if isinstance(rq, list) and rq:
        for item in rq:
            if isinstance(item, dict):
                add(item.get("label", ""), item.get("quote", ""))
            else:
                add("Contract", str(item))

    if not groups:
        return "—"

    def sort_key(k: str):
        ks = k.lower()
        if ks.startswith("section"):
            nums = re.findall(r"\d+", k)
            return (0, [int(n) for n in nums])
        if ks.startswith("exhibit"):
            return (1, k)
        return (2, k)

    lines: List[str] = []
    for lab in sorted(groups.keys(), key=sort_key)[:max_sources]:
        lines.append(f"{lab}:")
        for q in groups[lab][:max_quotes_per_source]:
            lines.append(f"- {q}")
        lines.append("")  # blank line between groups

    return "\n".join(lines).strip() or "—"


def build_table_rows(results: List[Dict]) -> List[Dict]:
    rows = []
    for r in results or []:
        question = r.get("Compliance Question", r.get("requirement", ""))
        state_raw = r.get("Compliance State", r.get("status", ""))
        conf_raw = r.get("Confidence", r.get("confidence", 0))

        rows.append({
            "Compliance Question": question,
            "Compliance State": _map_state(state_raw),
            "Confidence": f"{_to_int_conf(conf_raw)}%",
            # ✅ real excerpts, grouped, deduped
            "Relevant Quotes": _collect_grouped_quotes(r, max_sources=8, max_quotes_per_source=2),
            "Rationale": r.get("Rationale", r.get("rationale", "")) or "—",
        })
    return rows
