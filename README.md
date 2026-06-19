# Self Discovery Lab: Financial RAG Challenge

Build and evaluate a **Baseline** vs **Engineered** RAG pipeline over U.S. Treasury Bulletin records (OfficeQA).

**Student:** Audrey Rah | **Years:** 2022, 2023, 2024, 2025

## Submit to Canvas

| File | Purpose |
|------|---------|
| `results/Financial_RAG_Discussion_Board_Submission.docx` | Discussion board Word submission |
| `results/CORRECTED_DISCUSSION_BOARD_SUBMISSION.md` | Same content in markdown |
| `results/metrics.json` | Baseline vs Engineered metrics |
| `results/per_question_results.csv` | Per-question details |
| `results/AUDIT_REPORT.md` | Metric verification audit |



## Start



## Architecture

| Component | Baseline | Engineered |
|-----------|----------|------------|
| Vector index | In-memory TF-IDF (uni-grams) | SentenceTransformer + hybrid sparse/dense |
| Chunking | 1,200 chars, 0 overlap | 2,048 chars (~512 tokens), 512-char overlap |
| Metadata | Stored on chunks | Year filter + file pre-retrieval + soft bonuses |
| Generator | Extractive/rule-based + ESF helper | Same |

**Metadata:** Each chunk tagged with `year` and `month` from `treasury_bulletin_YYYY_MM.txt`.

**Evaluation:** K=5, ±1% tolerance via `officeqa/reward.py`.

## Data sources

1. **Questions:** `data/officeqa_pro.csv` (from `extract_csv.py` or `download_data.py` with HF token)
2. **Corpus:** Quarterly Treasury Bulletin PDFs via `build_corpus.py` (2022–2024 from fiscal.treasury.gov; 2025 from ESF financial statements)

## Final metrics (verified)

| Metric | Baseline | Engineered |
|--------|----------|------------|
| Hit Rate@5 | 33.33% | 66.67% |
| MRR | 0.1111 | 0.3333 |
| Recall@5 | 0.12% | 7.72% |
| Groundedness | 100.00% | 66.67% |
| Factual Accuracy | 0.00% | 33.33% |
| Hallucination Rate | 0.00% | 33.33% |

Evaluated on 3 OfficeQA Pro questions: UID0010, UID0086, UID0111.
