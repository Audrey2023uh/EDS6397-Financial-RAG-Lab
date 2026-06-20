# Self Discovery Lab: Financial RAG Challenge

Build and evaluate a **Baseline vs Engineered RAG pipeline** over U.S. Treasury Bulletin records using the Databricks OfficeQA dataset.

**Student:** Audrey Rah
**Recent Years Used:** 2022, 2023, 2024, 2025
**GitHub:**(https://github.com/Audrey2023uh/EDS6397-Financial-RAG-Lab)

## Goal

Build a system that searches through U.S. Treasury records to answer financial questions. This project compares two pipelines:

* **Baseline (Simple):** Unoptimized RAG — basic TF-IDF retrieval and simple extractive answers.
* **Engineered (Improved):** Better embeddings, metadata filtering, hybrid search, and a strict grounded generator.

The learning objective is to prove improvements with real numbers — not just claim the engineered version is better.

## Submit to Canvas

| File                                                     | Purpose                                                    |
| -------------------------------------------------------- | ---------------------------------------------------------- |
| `results/Financial_RAG_Discussion_Board_Submission.docx` | Word file for Canvas upload                                |
| `results/DISCUSSION_BOARD_SUBMISSION.md`                 | Same content in markdown — copy/paste for discussion board |
| `results/metrics.json`                                   | Machine-readable Baseline vs Engineered metrics            |
| `results/per_question_results.csv`                       | Per-question retrieval and answer details                  |
| `results/AUDIT_REPORT.md`                                | Metric verification audit                                  |

Before posting, replace `YOUR_GITHUB_USERNAME` in the submission files with your real GitHub repo link.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python run_all.py
```

### 3. Or run step by step

```bash
python extract_csv.py
python build_corpus.py
python financial_rag_lab.py
python create_submission_docx.py
```

### 4. Outputs

After running, check:

```text
results/metrics.json
results/per_question_results.csv
results/DISCUSSION_BOARD_SUBMISSION.md
results/Financial_RAG_Discussion_Board_Submission.docx
```

## Architecture

### Component Comparison

| Component    | Baseline (Simple)                                 | Engineered (Improved)                                              |
| ------------ | ------------------------------------------------- | ------------------------------------------------------------------ |
| Vector index | In-memory TF-IDF (uni-grams)                      | SentenceTransformer all-MiniLM-L6-v2 + hybrid sparse/dense         |
| Chunking     | 1,200 characters, 0 overlap                       | 2,048 characters (~512 tokens), 512-character overlap              |
| Metadata     | Stored on chunks only                             | Year filter + file pre-retrieval + soft rerank bonuses             |
| Generator    | Simple extractive — verbatim context numbers only | Strict extractive + ESF table math (operand-backed derived values) |

## Vector Database

This project uses an in-memory vector search layer built with scikit-learn and SentenceTransformers rather than ChromaDB or FAISS. For a 4-year subset this is fast enough; at 80-year scale a persistent vector DB would be required.

## Metadata (Required)

Every chunk is tagged with Year and Month parsed from filenames like:

```text
treasury_bulletin_2022_12.txt  →  year=2022, month=12
```

How metadata is used:

| Pipeline   | Metadata usage                                                        |
| ---------- | --------------------------------------------------------------------- |
| Baseline   | Tags stored on chunks but not used to filter search                   |
| Engineered | Year-level filtering before retrieval + soft bonuses during reranking |

This metadata fix improved retrieval metrics significantly (Hit Rate and MRR doubled) without hurting generation quality.

## Chunking Strategy

| Pipeline   | Strategy                                           | Why                                             |
| ---------- | -------------------------------------------------- | ----------------------------------------------- |
| Baseline   | 1,200-char windows, no overlap                     | Simple starting point                           |
| Engineered | 2,048-char windows (~512 tokens), 512-char overlap | Reduces table splits; keeps numeric rows intact |

## Generator Design

**Baseline generator:** Picks numbers found verbatim in retrieved context. Conservative — high groundedness, low factual accuracy when retrieval misses the right bulletin.

**Engineered generator (refined after professor feedback):** Only outputs:

* Numbers found verbatim in retrieved chunks, OR
* Values computed from table operands that appear in context, such as ESF ratio math where source numbers are present

This prevents the engineered pipeline from inventing answers that are not supported by the retrieved text.

## Evaluation Setup

* **K = 5** (top 5 retrieved chunks)
* **Tolerance:** ±1% for numeric answers via `officeqa/reward.py`
* **Questions evaluated:** 3 OfficeQA Pro questions filtered for 2022–2025

  * UID0010
  * UID0086
  * UID0111

## Data Sources

| Source                 | Location                                   | How it was obtained                                 |
| ---------------------- | ------------------------------------------ | --------------------------------------------------- |
| Questions + answer key | `data/officeqa_pro.csv`                    | `extract_csv.py` from OfficeQA GitHub HTML snapshot |
| Treasury Bulletin text | `data/corpus/*.txt`                        | `build_corpus.py` — 48 files, 12 months × 4 years   |
| Original PDFs          | `data/raw_pdfs/`                           | fiscal.treasury.gov (2022–2024 quarterly bulletins) |
| 2025 corpus            | `data/corpus/treasury_bulletin_2025_*.txt` | ESF financial statement proxy                       |
| Scoring logic          | `officeqa/reward.py`                       | Databricks OfficeQA reward script                   |

## Optional: Hugging Face Download

If you have a Hugging Face token with access to the gated OfficeQA dataset:

```bash
python download_data.py
```

## Success Metrics

### Set A: Search Metrics (The Retriever — "The Librarian")

| Metric     | Formula                                                 | What it means                            |
| ---------- | ------------------------------------------------------- | ---------------------------------------- |
| Hit Rate@5 | Correct docs in Top 5 / Total queries                   | Did search find the right page?          |
| MRR        | Average of 1 / rank of first correct doc                | Did the best answer appear near the top? |
| Recall@5   | Relevant snippets found / Total relevant snippets in DB | Did you find all necessary info?         |

### Set B: Answer Metrics (The Generator — "The Student")

| Metric             | Formula                                   | What it means                     |
| ------------------ | ----------------------------------------- | --------------------------------- |
| Groundedness       | Claims supported by source / Total claims | Did the AI stick to the text?     |
| Factual Accuracy   | Correct answers / Total questions         | Does it match the CSV within ±1%? |
| Hallucination Rate | Fabricated claims / Total claims          | Did the AI make up numbers?       |

## Final Results

### Scorecard (K=5)

| Metric             | Baseline (Simple) | Engineered (Improved) |
| ------------------ | ----------------: | --------------------: |
| Hit Rate@5         |            33.33% |                66.67% |
| MRR                |            0.1111 |                0.3333 |
| Recall@5           |             0.12% |                 7.72% |
| Groundedness       |           100.00% |               100.00% |
| Factual Accuracy   |             0.00% |                33.33% |
| Hallucination Rate |             0.00% |                 0.00% |

All engineered metrics meet or exceed baseline.

## What Changed After Professor Feedback

The first engineered version improved retrieval but hurt generation:

| Metric             | Old Engineered (Problem) | New Engineered (Fixed) |
| ------------------ | -----------------------: | ---------------------: |
| Groundedness       |                   66.67% |                100.00% |
| Hallucination Rate |                   33.33% |                  0.00% |
| Factual Accuracy   |                   33.33% |                 33.33% |
| Hit Rate@5         |                   66.67% |                 66.67% |
| MRR                |                   0.3333 |                 0.3333 |

**Root cause:** The engineered generator computed correct ESF answers, such as UID0086 → 4.815, but those computed values were not verbatim in the retrieved chunks, so groundedness and hallucination metrics penalized them.

**Fix:** Split baseline and engineered generators. Engineered generator now only emits verbatim context numbers or operand-backed derived values traceable to source table rows in the retrieved text.

## Engineering Reflection

### The Bottleneck

Looking at the Baseline, the main failure was in **Finding the data (Retriever)**, not **Understanding the data (Generator)**.

Evidence:

* Hit Rate@5 = 33.33% and MRR = 0.1111 → search often returned the wrong bulletin in the top 5
* Groundedness = 100.00% → when the system did answer, it stuck to retrieved text
* Factual Accuracy = 0.00% → wrong answers, but driven by retrieval misses rather than misreading numbers

## The Metadata Fix

Adding Year/Month metadata changed scores mainly on the retrieval side.

| Metric           | Baseline | Engineered |
| ---------------- | -------: | ---------: |
| Hit Rate@5       |   33.33% |     66.67% |
| MRR              |   0.1111 |     0.3333 |
| Factual Accuracy |    0.00% |     33.33% |
| Groundedness     |  100.00% |    100.00% |

Metadata helped retrieval metrics more than generation metrics.

## Scaling Insight

If scaled from this 4-year subset to the full 80-year archive (1939–2025), the first component that would likely break is the in-memory vector search layer (embedding matrix + cosine similarity over all chunks). At production scale, this would need a persistent vector database such as ChromaDB, FAISS, or similar, with indexed approximate nearest-neighbor search.

## Project Structure

```text
Fiancial/
├── financial_rag_lab.py
├── build_corpus.py
├── extract_csv.py
├── download_data.py
├── create_submission_docx.py
├── run_all.py
├── requirements.txt
├── README.md
├── data/
│   ├── officeqa_pro.csv
│   ├── corpus/
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

## Known Limitations

* Only 3 OfficeQA Pro questions match the 2022–2025 year filter.
* UID0111 references bulletins outside the 4-year corpus.
* UID0010 may require external FX data not fully available in Treasury bulletins alone.
* Uses in-memory sklearn search, not ChromaDB/FAISS.
* Rule-based extractive generator — reproducible without LLM API keys.

## What Learned

* **Smart Organizing:** Chunking strategy matters as much as the model.
* **Librarian vs Student:** Low Hit Rate = search problem. High Groundedness but wrong answer = retrieval miss.
* **Data Integrity:** Metadata (Year/Month tags) keeps the AI on track.
* **The Power of a Baseline:** You cannot claim improvement without numbers to compare against.
* **Thinking at Scale:** A 4-year in-memory pipeline would need redesign for 80 years of data.
