
import streamlit as st
import hashlib
import json
import pandas as pd  # (optional now, but you can keep it)
from chunking.section_tagger import find_section_label

from ui.report_table import build_table_rows
from chatbot.chat import answer_question_with_rag

from ingestion.pdf_loader import load_contract_pdf_bytes
from chunking.chunker import chunk_text
from embeddings.embedder import embed_texts
from vector_store.faiss_store import FaissVectorStore
from rag.retriever import retrieve_clauses
from compliance_engine.analyzer import analyze_requirement


def load_standards(path: str = "standards/compliance_standards.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Badge for HTML table ----------
def badge_html(state: str) -> str:
    s = (state or "").strip().lower().replace("_", " ").replace("-", " ")

    if "fully" in s or s == "compliant":
        bg, fg = "#DCFCE7", "#166534"   # green
    elif "partial" in s:
        bg, fg = "#FEF9C3", "#854D0E"   # yellow
    elif "non" in s:
        bg, fg = "#FEE2E2", "#991B1B"   # red
    else:
        bg, fg = "#E5E7EB", "#111827"   # gray (unknown)

    label = state or "Unknown"
    return (
        f"<span style='background:{bg};color:{fg};"
        "padding:4px 10px;border-radius:999px;"
        "font-weight:600;font-size:0.85rem;white-space:nowrap;'>"
        f"{label}</span>"
    )


def _escape_html(s: str) -> str:
    """Prevent HTML injection inside our custom HTML table."""
    if s is None:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render_report_table(rows: list[dict]) -> None:
    """Render a full-width, wrapping HTML table with colored status badges."""
    css = """
    <style>
      .report-table-wrap { width: 100%; overflow-x: auto; }
      table.report-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.92rem;
        background: white;
      }
      table.report-table th, table.report-table td {
        border: 1px solid #E5E7EB;
        padding: 10px;
        vertical-align: top;
        white-space: normal;
        word-break: break-word;
      }
      table.report-table th {
        background: #F5F7FB;
        text-align: left;
        font-weight: 650;
      }
      /* nicer row hover */
      table.report-table tbody tr:hover {
        background: #FAFAFB;
      }
      /* set reasonable column widths */
      .col-q { min-width: 260px; }
      .col-s { min-width: 140px; }
      .col-c { min-width: 110px; }
      .col-quotes { min-width: 340px; }
      .col-rationale { min-width: 340px; }
      /* make newlines show in HTML */
      .prewrap { white-space: pre-wrap; }
    </style>
    """

    html = [css, "<div class='report-table-wrap'>", "<table class='report-table'>"]
    html.append(
        "<thead><tr>"
        "<th class='col-q'>Compliance Question</th>"
        "<th class='col-s'>Compliance State</th>"
        "<th class='col-c'>Confidence</th>"
        "<th class='col-quotes'>Relevant Quotes</th>"
        "<th class='col-rationale'>Rationale</th>"
        "</tr></thead><tbody>"
    )

    for r in rows:
        q = _escape_html(r.get("Compliance Question", ""))
        state = r.get("Compliance State", "")
        conf = _escape_html(r.get("Confidence", ""))
        quotes = _escape_html(r.get("Relevant Quotes", "") or "—")
        rationale = _escape_html(r.get("Rationale", "") or "—")

        # Preserve line breaks if you used "\n" in quotes/rationale
        quotes_html = f"<div class='prewrap'>{quotes}</div>"
        rationale_html = f"<div class='prewrap'>{rationale}</div>"

        html.append(
            "<tr>"
            f"<td>{q}</td>"
            f"<td>{badge_html(state)}</td>"
            f"<td>{conf}</td>"
            f"<td>{quotes_html}</td>"
            f"<td>{rationale_html}</td>"
            "</tr>"
        )

    html.append("</tbody></table></div>")
    st.markdown("".join(html), unsafe_allow_html=True)


# Cache embedding calls (helps when Streamlit reruns)
@st.cache_data(show_spinner=False)
def embed_cached(texts, model):
    return embed_texts(texts, model=model)


# ---------- Page config + header ----------
st.set_page_config(
    page_title="Contract Analyzer",
    page_icon="📄",
    layout="wide",
)
st.markdown("""
<style>
  .stButton>button { border-radius: 12px; padding: 0.6rem 1rem; }
  .stDownloadButton>button { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

st.title("📄 Contract Analyzer")
st.caption("Upload a contract PDF → run compliance checks → export structured JSON.")
st.divider()


# ---------------------------
# Session state init
# ---------------------------
if "store" not in st.session_state:
    st.session_state.store = None
if "contract_hash" not in st.session_state:
    st.session_state.contract_hash = None
if "contract_text" not in st.session_state:
    st.session_state.contract_text = None
if "results" not in st.session_state:
    st.session_state.results = None
if "debug_retrieval" not in st.session_state:
    st.session_state.debug_retrieval = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


uploaded = st.file_uploader("Upload contract PDF", type=["pdf"])
use_ocr = st.checkbox("Use OCR for scanned PDFs", value=True)
debug_mode = st.toggle("Debug mode (show retrieval chunks)", value=False)

if not uploaded:
    st.info("Upload a PDF to start.")
    st.stop()


# ---------------------------
# Step 1: Extract text
# ---------------------------
with st.status("Processing contract...", expanded=False) as status:
    status.update(label="1/3 Extracting text from PDF...", state="running")

    pdf_bytes = uploaded.read()
    contract_text = load_contract_pdf_bytes(pdf_bytes, use_ocr_fallback=use_ocr)

    if not contract_text or len(contract_text) < 200:
        status.update(label="Failed: No extractable text found.", state="error")
        st.error("Could not extract readable text from this PDF. Try enabling OCR or use a digital PDF.")
        st.stop()

    # Compute contract hash AFTER extraction
    contract_hash = hashlib.md5(contract_text.encode("utf-8")).hexdigest()

    # If new PDF uploaded -> reset index + results + chat (optional)
    if st.session_state.contract_hash != contract_hash:
        st.session_state.contract_hash = contract_hash
        st.session_state.contract_text = contract_text
        st.session_state.store = None
        st.session_state.results = None
        st.session_state.debug_retrieval = None
        st.session_state.chat_messages = []

    # ---------------------------
    # Step 2: Build index ONLY if needed (one-time per PDF)
    # ---------------------------
    if st.session_state.store is None:
        status.update(label="2/3 Building semantic index (chunk + embed + FAISS)...", state="running")

        chunks = chunk_text(contract_text, max_chars=3000, overlap_chars=300, min_chars=400)
        chunk_texts = [c["text"] for c in chunks]

        metas = []
        for c in chunks:
            label = find_section_label(c["text"]) or "Unlabeled"
            metas.append({
                "chunk_id": c["id"],
                "label": label,
                "start_char": c["start_char"],
                "end_char": c["end_char"],
            })

        embs = embed_cached(chunk_texts, "text-embedding-3-small")
        store = FaissVectorStore(dim=len(embs[0]))
        store.add(embs, chunk_texts, metas=metas)
        if debug_mode:
            st.write("DEBUG chunks:", len(chunks))
            st.write("DEBUG embeddings:", len(embs))
            st.write("DEBUG faiss ntotal:", store.index.ntotal)


        st.session_state.store = store
        status.update(label="Index ready ✅", state="running")
    else:
        status.update(label="2/3 Index already built (reusing cached index) ✅", state="running")

    status.update(label="3/3 Ready ✅", state="complete")


store = st.session_state.store
standards = load_standards()


# ---------------------------
# Compliance Analysis
# ---------------------------
st.subheader("✅ Compliance Analysis")

col1, col2 = st.columns([1, 2])
with col1:
    run_checks = st.button("🚀 Run compliance checks", type="primary", use_container_width=True)

with col2:
    st.caption("Tip: Re-uploading the same PDF won’t rebuild the index. New PDF will reset index + results.")

if run_checks:
    with st.spinner("Running GenAI compliance checks..."):
        results = []
        debug_retrieval = {}

        for req_name, req in standards.items():
            retrieved = retrieve_clauses(
                store=store,
                requirement_name=req_name,
                requirement_description=req["description"],
                controls=req["controls"],
                top_k=12,
                min_score = 0.25
            )
            

            if debug_mode:
                with st.expander(f"Debug: Retrieved evidence — {req_name}", expanded=False):
                    st.write("Number of retrieved chunks:", len(retrieved))

                    for i, chunk in enumerate(retrieved, start=1):
                        st.markdown(f"**Chunk {i}**")
                        st.write("Score:", chunk.get("score"))
                        st.write("label:", chunk.get("label"))
                        st.write(chunk.get("text", "")[:500] + "...")
                        st.divider()

            analysis = analyze_requirement(
                requirement_name=req_name,
                requirement_description=req["description"],
                controls=req["controls"],
                retrieved_clauses=retrieved,
                model="gpt-4o-mini",
            )

            results.append(analysis)

            if debug_mode:
                debug_retrieval[req_name] = retrieved

        st.session_state.results = results
        st.session_state.debug_retrieval = debug_retrieval

# ---------------------------
# Output: TABLE + JSON
# ---------------------------
if st.session_state.results:
    st.subheader("Structured Compliance Output")

    rows = build_table_rows(st.session_state.results)
    df = pd.DataFrame(rows)

    # dynamic height so you don't see lots of empty rows
    ROW_H = 38          # approx row height
    HEADER_H = 40       # header height
    MAX_H = 520         # cap so it doesn't get too tall

    table_h = min(MAX_H, HEADER_H + ROW_H * (len(df) + 1))

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=table_h
    )

    with st.expander("View raw JSON output"):
        st.json(st.session_state.results)

    st.download_button(
        "Download JSON output",
        data=json.dumps(st.session_state.results, indent=2).encode("utf-8"),
        file_name="compliance_output.json",
        mime="application/json",
    )
else:
    st.info("Click **Run compliance checks** to generate the report.")
# ---------------------------
# Bonus: Chatbot
# ---------------------------
with st.expander("💬 Chat with the Contract", expanded=True):
    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_q = st.chat_input(
        "Ask anything about the contract...",
        key="chat_input_contract_analyzer"
    )

    if user_q:
        st.session_state.chat_messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, retrieved = answer_question_with_rag(
                    store=store,
                    question=user_q,
                    top_k=15,
                    model="gpt-4o-mini",
                )
                st.markdown(answer)

                if debug_mode:
                    with st.expander("Show supporting contract excerpts"):
                        for c in retrieved:
                            st.markdown(f"**Chunk {c['chunk_id']} (score={c['score']:.3f})**")
                            st.write(c["text"][:1200] + ("..." if len(c["text"]) > 1200 else ""))

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
