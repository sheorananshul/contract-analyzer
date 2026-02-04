# Contract Analyzer – RAG-Based Compliance Engine

An AI-powered contract compliance analyzer built using **Retrieval-Augmented Generation (RAG)**.  
The system ingests contract PDFs, retrieves relevant clauses via semantic search, and evaluates compliance against predefined standards using **evidence-backed reasoning with deterministic confidence scoring**.

---

## What This Project Does

- Upload a contract PDF (digital or scanned)
- Split it into semantically meaningful chunks
- Store embeddings in a vector database (FAISS)
- Retrieve only the most relevant clauses per compliance requirement
- Use an LLM **strictly on retrieved evidence** to determine:
  - Compliance status
  - Verbatim supporting quotes
  - Gaps and recommendations
  - A confidence score that is **never 100%**

The result is a **trustworthy, explainable compliance report**, not a black-box answer.

---

## Key Features

- 📄 **PDF Ingestion** (OCR fallback for scanned contracts)
- ✂️ **Overlapping Semantic Chunking** with section labeling
- 🧠 **Vector Search using FAISS**
- 🔎 **RAG-based Compliance Analysis**
- 📊 **Structured JSON Output**
- 💬 **Contract Q&A Chatbot**
- 🛡️ **Deterministic, Evidence-Based Confidence Scoring**
- 🧩 **Fully Modular Architecture**

---

## Architecture (High Level)

