# Financial RAG Challenge — Self Discovery Lab

**Author:** Audrey Rah  
**Course:** AI Summer Class — Self Discovery Lab: Design RAG with Real World Data  
**Data source:** [Databricks OfficeQA](https://github.com/databricks/officeqa)  
**Corpus years:** 2022, 2023, 2024, 2025

A retrieval-augmented generation (RAG) system that searches U.S. Treasury Bulletin records to answer financial questions. The project compares a **Baseline** (simple) pipeline against an **Engineered** (improved) pipeline and reports retriever and generator metrics at **K=5**.

---

## Problem

Build a system that:

1. Indexes U.S. Treasury Bulletin text for recent years (minimum 4 years).
2. Retrieves relevant document chunks for OfficeQA questions.
3. Generates answers grounded in those chunks.
4. Compares Baseline vs Engineered designs with measurable improvements.

Questions and ground-truth answers come from `officeqa_pro.csv`. Document text is built from Treasury Bulletin PDFs (parsed to `.txt`).

---

## Results (K=5)

| Metric | Baseline | Engineered |
|--------|----------|------------|
| Hit Rate@5 | 33.33% | **66.67%** |
| MRR | 0.1111 | **0.3333** |
| Recall@5 | 0.12% | **7.72%** |
| Groundedness | 100.00% | 66.67% |
| Factual Accuracy | 0.00% | **33.33%** |
| Hallucination Rate | 0.00% | 33.33% |

Evaluated on 3 OfficeQA Pro questions filtered for 2022–2025: `UID0010`, `UID0086`, `UID0111`.

Full per-question breakdown: [`results/per_question_results.csv`](results/per_question_results.csv)  
Audit verification: [`results/AUDIT_REPORT.md`](results/AUDIT_REPORT.md)

---

## Architecture

### Vector index
- **Baseline:** In-memory TF-IDF (uni-grams) + cosine similarity (scikit-learn).
- **Engineered:** SentenceTransformer (`all-MiniLM-L6-v2`) + hybrid search (65% dense / 35% sparse TF-IDF), with two-stage retrieval (top 3 bulletins → top 5 chunks).

> Note: No persistent ChromaDB/FAISS store in this lab build; vectors are held in memory for the 4-year subset.

### Metadata (Year / Month)
Every chunk is tagged with **year** and **month** parsed from filenames like `treasury_bulletin_2022_12.txt`.

**Engineered retrieval uses metadata to:**
- Filter candidates by year inferred from the question.
- Apply soft rerank bonuses when year/month or ESF-related terms align with the query.

### Chunking
| System | Strategy |
|--------|------------|
| Baseline | 1,200 characters, 0 overlap |
| Engineered | 2,048 characters (~512 tokens), 512-character overlap |

### Generator
Extractive / rule-based answerer (not an LLM), with a helper for ESF QoQ table math. Scored with [`officeqa/reward.py`](officeqa/reward.py) at ±1% tolerance.

---

## Repository structure
