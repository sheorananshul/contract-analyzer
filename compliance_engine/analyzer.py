


import os
import json
from typing import List, Dict
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def analyze_requirement(
    requirement_name: str,
    requirement_description: str,
    controls: List[str],
    retrieved_clauses: List[Dict],
    model: str = "gpt-4o-mini",
) -> Dict:

    # If no evidence, skip GPT and return deterministic Non-Compliant / Insufficient Evidence
    if not retrieved_clauses:
        return {
            "requirement": requirement_name,
            "status": "Non-Compliant",
            "confidence": 20,
            "controls": [{"name": c, "covered": False, "evidence": []} for c in controls],
            "rationale": "No sufficiently relevant contract language was retrieved for this requirement.",
            "gaps": ["No evidence found above similarity threshold."],
            "recommendations": ["Add explicit contract language covering these controls."],
        }

    # ✅ Build evidence text WITH chunk_id + label (minimal but important)
    parts = []
    for c in retrieved_clauses:
        chunk_id = c.get("chunk_id")
        label = c.get("label") or f"Chunk {chunk_id}"
        score = c.get("score", 0.0)
        text = c.get("text", "")

        parts.append(f"[chunk_id={chunk_id} | label={label} | score={score:.3f}]\n{text}")

    clauses_text = "\n\n".join(parts)

    prompt = f"""
You are a cybersecurity and contract compliance auditor.

Evaluate whether the contract satisfies the requirement below.

STRICT RULES:
1) You may ONLY use the provided contract evidence text.
2) For any control marked covered=true, you MUST provide at least one verbatim quote copied exactly from the evidence.
3) Each evidence item MUST include the chunk_id it came from.
4) If you cannot find an exact quote, mark covered=false.
5) Be strict: general statements like "reasonable security" do NOT count as coverage unless they explicitly match the control.
6) Use the label from the evidence header when writing evidence (e.g., "Section 6.7", "Exhibit G13").
7) IMPORTANT: When you cite evidence, copy chunk_id and label EXACTLY from the evidence header.
   Do NOT invent or change section numbers.

Return ONLY valid JSON with this schema:
{{
  "requirement": string,
  "status": "Fully Compliant" | "Partially Compliant" | "Non-Compliant",
  "confidence": integer,
  "controls": [
    {{
      "name": string,
      "covered": boolean,
      "evidence": [
        {{ "chunk_id": integer, "label": string, "quote": string }}
      ]
    }}
  ],
  "rationale": string,
  "gaps": [string],
  "recommendations": [string]
}}

Requirement: {requirement_name}
Description: {requirement_description}

Controls:
- """ + "\n- ".join(controls) + f"""

Contract Evidence:
{clauses_text}
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    raw = resp.choices[0].message.content.strip()

    # Parse JSON safely
    try:
        data = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        data = json.loads(raw[start:end + 1])
    # Remove LLM-provided confidence (we compute it deterministically)
    data.pop("confidence", None)

    # ✅ Safety: ensure controls exist
    if not isinstance(data.get("controls"), list):
        data["controls"] = [{"name": c, "covered": False, "evidence": []} for c in controls]

    # ✅ Post-validation: if covered but no evidence => flip to not covered
    for ctrl in data.get("controls", []):
        if ctrl.get("covered") and not ctrl.get("evidence"):
            ctrl["covered"] = False

    # ✅ Deterministic status recompute
    total = len(data.get("controls", [])) or len(controls)
    covered_count = sum(1 for c in data.get("controls", []) if c.get("covered") is True)

    if total > 0 and covered_count == total:
        data["status"] = "Fully Compliant"
    elif covered_count == 0:
        data["status"] = "Non-Compliant"
    else:
        data["status"] = "Partially Compliant"

    # ---------------------------
    # Confidence: evidence-based + never 100
    # ---------------------------
    MAX_CONF = 95          
    MIN_CONF = 30          # avoid misleading extremes
    NO_EVIDENCE_CONF = 20  # retrieval exists but no quotable evidence

    total_controls = len(controls) or 1
    covered_controls = sum(1 for c in data["controls"] if c.get("covered") is True)

    evidence_quotes = sum(len(c.get("evidence", [])) for c in data["controls"])

    scores = [
        c.get("score", 0.0)
        for c in retrieved_clauses
        if isinstance(c.get("score", None), (int, float))
    ]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    coverage_ratio = covered_controls / total_controls
    base = 20 + int(60 * coverage_ratio)  # 20 → 80

    if evidence_quotes > 0:
        base += 5
    else:
        base = min(base, NO_EVIDENCE_CONF)

    if avg_score >= 0.40:
        base += 5
    elif avg_score < 0.25:
        base -= 5

    if data["status"] == "Fully Compliant":
        base += 5
    elif data["status"] == "Non-Compliant":
        base -= 5

    # Final confidence assignment (special case: no quotable evidence)
    if evidence_quotes == 0:
        data["confidence"] = min(NO_EVIDENCE_CONF, MAX_CONF)
    else:
        data["confidence"] = max(MIN_CONF, min(base, MAX_CONF))




    return data






