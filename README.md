# Self Discovery Lab: Financial RAG Challenge

Build and evaluate a **Baseline** vs **Engineered** RAG pipeline over U.S. Treasury Bulletin records using the Databricks [OfficeQA](https://github.com/databricks/officeqa) dataset.

**Student:** Audrey Rah  
**Recent Years Used:** 2022, 2023, 2024, 2025  
**GitHub:** https://github.com/YOUR_GITHUB_USERNAME/financial-rag-challenge

---

## Goal

Build a system that searches through U.S. Treasury records to answer financial questions. This project compares two pipelines:

- **Baseline (Simple):** Unoptimized RAG — basic TF-IDF retrieval and simple extractive answers.
- **Engineered (Improved):** Better embeddings, metadata filtering, hybrid search, and a strict grounded generator.

The learning objective is to prove improvements with real numbers — not just claim the engineered version is better.

---

## Quick Start

### 1. Install dependencies

```powershell
pip install -r requirements.txt
pip install python-docx
```

### 2. Run the full pipeline from scratch

```powershell
python run_all.py
```

### 3. Or run step by step

```powershell
python extract_csv.py
python build_corpus.py --years 2022 2023 2024 2025
python financial_rag_lab.py --student-name "Audrey Rah"
python create_submission_docx.py
```

### 4. Outputs

After running, check:

- `results/metrics.json`
- `results/per_question_results.csv`
- `results/DISCUSSION_BOARD_SUBMISSION.md`
- `results/Financial_RAG_Discussion_Board_Submission.docx`

---

## Architecture

| Component | Baseline (Simple) | Engineered (Improved) |
|-----------|-------------------|------------------------|
| **Vector index** | In-memory TF-IDF (uni-grams) | SentenceTransformer `all-MiniLM-L6-v2` + hybrid sparse/dense |
| **Chunking** | 1,200 characters, 0 overlap | 2,048 characters (~512 tokens), 512-character overlap |
| **Metadata** | Stored on chunks only | Year filter + file pre-retrieval + soft rerank bonuses |
| **Generator** | Simple extractive — verbatim context numbers only | Strict extractive + ESF table math (operand-backed derived values) |

### Metadata (Required)

Every chunk is tagged with **Year** and **Month** parsed from filenames like `treasury_bulletin_2022_12.txt`.

| Pipeline | Metadata usage |
|----------|----------------|
| Baseline | Tags stored on chunks but not used to filter search |
| Engineered | Year-level filtering before retrieval + soft bonuses during reranking |

### Chunking Strategy

| Pipeline | Strategy |
|----------|----------|
| Baseline | 1,200-char windows, no overlap |
| Engineered | 2,048-char windows (~512 tokens), 512-char overlap |

### Generator Refinement (Professor Feedback)

The first engineered version improved retrieval but hurt generation (Groundedness 100% → 67%, Hallucination 0% → 33%). The fix splits baseline and engineered generators. The engineered generator only outputs:

1. Numbers found **verbatim** in retrieved chunks, OR
2. Values **computed from table operands** present in context (e.g., ESF QoQ percent change)

This keeps all engineered metrics at or above baseline.

### Evaluation Setup

- **K = 5** (top 5 retrieved chunks)
- **Tolerance:** ±1% via `officeqa/reward.py`
- **Questions:** 3 OfficeQA Pro questions for 2022–2025 (UID0010, UID0086, UID0111)

---

## Data Sources

| Source | Location |
|--------|----------|
| Questions + answer key | `data/officeqa_pro.csv` |
| Treasury Bulletin text | `data/corpus/*.txt` (48 files) |
| Original PDFs | `data/raw_pdfs/` |
| Scoring logic | `officeqa/reward.py` |

---

## Final Results (Verified)

| Metric | Baseline (Simple) | Engineered (Improved) |
|--------|:-----------------:|:---------------------:|
| Hit Rate@5 | 33.33% | **66.67%** |
| MRR | 0.1111 | **0.3333** |
| Recall@5 | 0.12% | **7.72%** |
| Groundedness | 100.00% | **100.00%** |
| Factual Accuracy | 0.00% | **33.33%** |
| Hallucination Rate | 0.00% | **0.00%** |

All engineered metrics meet or exceed baseline.

---

## Submit to Canvas

| File | Purpose |
|------|---------|
| `results/Financial_RAG_Discussion_Board_Submission.docx` | Word upload |
| `results/DISCUSSION_BOARD_SUBMISSION.md` | Copy/paste for discussion board |

Replace `YOUR_GITHUB_USERNAME` with your real GitHub repo link before posting.

---

## Engineering Reflection (Summary)

**The Bottleneck:** Baseline failed mainly on **retrieval** (Hit Rate@5 = 33.33%, MRR = 0.1111) while groundedness stayed at 100%.

**The Metadata Fix:** Year/Month tags improved Hit Rate and MRR; groundedness stayed at 100% after generator refinement.

**Scaling Insight:** In-memory vector search would be the first bottleneck when scaling to the full 1939–2025 archive.

Full text: `results/DISCUSSION_BOARD_SUBMISSION.md`

---

## Project Structure

```
Complete version/
├── financial_rag_lab.py          # Main Baseline vs Engineered evaluation
├── build_corpus.py               # PDF → .txt corpus
├── extract_csv.py                # OfficeQA questions CSV
├── download_data.py              # Optional Hugging Face download
├── create_submission_docx.py     # Word submission generator
├── run_all.py                    # End-to-end runner
├── requirements.txt
├── README.md
├── data/
│   ├── officeqa_pro.csv
│   ├── corpus/                   # 48 Treasury .txt files (2022–2025)
│   └── raw_pdfs/
├── sources/
│   └── officeqa_pro_github.html
├── officeqa/
│   └── reward.py
└── results/
    ├── metrics.json
    ├── per_question_results.csv
    ├── DISCUSSION_BOARD_SUBMISSION.md
    ├── Financial_RAG_Discussion_Board_Submission.docx
    └── AUDIT_REPORT.md
```

---

## Known Limitations

1. Only 3 OfficeQA Pro questions match the 2022–2025 year filter.
2. UID0111 references bulletins outside the 4-year corpus.
3. UID0010 may require external FX data not in Treasury bulletins alone.
4. In-memory sklearn search (not ChromaDB/FAISS).
5. Rule-based extractive generator — reproducible without LLM API keys.
