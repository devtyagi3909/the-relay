# The Relay: High-Throughput RAG Pipeline & Rate-Limit Resilience Engine

An end-to-end, rate-limit resilient Retrieval-Augmented Generation (RAG) system engineered for the Vaani ML Engineering assignment. It conducts accurate, grounded multi-paper synthesis across three version-locked arXiv papers (`1706.03762v7`, `2410.00037v1`, `2507.07955v2`) under strict multi-tier provider quotas (30 RPM, 6,000 TPM, 14,400 RPD, 500,000 TPD).

---

## 1. Submission Deliverables Summary

As mandated by the Vaani Assignment Brief (Pages 1–4) and Hiring Team email:

| Requirement | Path / Location | Description |
| :--- | :--- | :--- |
| **Technical Write-up (PDF)** | [`writeup/writeup.pdf`](file:///Users/devtyagi/Downloads/the-relay/writeup/writeup.pdf) | Exactly 2 pages, zero text clipping, strictly following all mandatory headings. |
| **Technical Write-up (Markdown)** | [`writeup/writeup.md`](file:///Users/devtyagi/Downloads/the-relay/writeup/writeup.md) | Source report containing all deterministic tables and failure analyses. |
| **Raw Telemetry File** | [`results.csv`](file:///Users/devtyagi/Downloads/the-relay/results.csv) | 450 requests across 7 load shapes and 3 retrieval ablations (12 columns, zero nulls). |
| **Conversation History** | [`conversation_history.md`](file:///Users/devtyagi/Downloads/the-relay/conversation_history.md) | Complete prompt log and unedited assistant interactions. |
| **Codebase** | [`src/`](file:///Users/devtyagi/Downloads/the-relay/src/) | Clean, modular Python implementation across ingestion, calibration, pipeline, and verification. |
| **Verification Script** | [`src/verify_submission.py`](file:///Users/devtyagi/Downloads/the-relay/src/verify_submission.py) | Zero-tolerance audit script enforcing table parity and schema constraints. |

---

## 2. Quickstart & Verification

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Zero-Tolerance Audit
Verifies `results.csv` schema (450 rows, 0 nulls, valid enums), checks zero credential leaks (`gsk_*`), confirms exact decimal table parity in `writeup.md`, and validates the 2-page PDF budget:
```bash
python3 -m src.verify_submission
```

### Recompute Tables from results.csv
Re-derives Tables 1, 2a, 2b, 3 and the Section 4 slow trace directly from `results.csv`:
```bash
python3 -m src.compute_tables
```

### Recompile 2-Page PDF
Compiles `writeup.md` into `writeup.pdf` using PyMuPDF:
```bash
python3 writeup/render_pdf.py
```

---

## 3. Architecture & Core Findings

### The 4.1 QPM Cliff
With `balanced_512_k5_rerank` averaging ~1,450 prompt tokens, Groq's 6,000 TPM limit trips at:
$$\frac{6,000\text{ tokens/min}}{1,450\text{ tokens/request}} \approx 4.137\text{ queries per minute}$$
Despite an advertised 30 RPM limit, unthrottled offered loads above 4.1 QPM trigger immediate HTTP 429 rate limit exceptions. The Relay overcomes this cliff via **Dynamic Retrieval Downshifting**: when remaining TPM drops below 4,200, the pipeline dynamically downshifts to `small_256_k3_norerank` (~262 prompt tokens), cutting token consumption by >75% and sustaining steady offered loads of 20 QPM at 100.0% completion.

### Asymmetric Pipeline Allocation
To prevent 4 LLM calls per user query from collapsing 30 provider RPM to 7.5 user QPM:
1. **Step 1 (Query Planner)**: 1ms deterministic lexical and entity planner (0 LLM tokens).
2. **Step 2 (Hybrid Library Retriever)**: In-memory BM25 + character & word n-gram TF-IDF cosine reranker with table preservation (<12ms CPU latency).
3. **Step 3 (Synthesizer)**: Streaming generation capturing exact Time-To-First-Token (TTFT) via Server-Sent Events.
4. **Step 4 (Verifier Loop)**: Deterministic inline citation and numeric grounding validator; loops back to Step 3 at most once if latency budget permits.

### 3:00 PM Daily Quota Exhaustion
When daily RPD or TPD hits 0, the pipeline returns a verified, zero-token **Extractive Citation Card** in <15ms containing top verbatim passages and exact citations, recording `status="rate_limited"` without leaving the user hanging.

---

## 4. Google Drive Submission Instructions

Per the hiring team instructions:
1. Create a Google Drive folder named `The-Relay-Dev-Tyagi`.
2. Place the following inside:
   - `writeup/writeup.pdf`
   - `writeup/writeup.md`
   - `results.csv`
   - `conversation_history.md`
   - The entire `the-relay` project directory (or zip)
3. Set folder sharing permissions to **"Anyone with the link can view"**.
4. Submit the link in the Vaani submission form.
