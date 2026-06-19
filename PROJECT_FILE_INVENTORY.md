# Financial RAG Lab — Final Project File Inventory

**Project:** Self Discovery Lab: Design RAG with real world data  
**Location:** `C:\Users\audre\OneDrive\Ai Summer Class\AI Class\Practice Problem\Sec Week\Fiancial`  
**Student:** Audrey Rah | **Years:** 2022, 2023, 2024, 2025

---

## KEEP — Final submission & audit deliverables

| File | Role |
|------|------|
| `results/Financial_RAG_Discussion_Board_Submission.docx` | **Final Word submission** (Canvas upload) |
| `results/CORRECTED_DISCUSSION_BOARD_SUBMISSION.md` | Final markdown submission (source of truth) |
| `results/AUDIT_REPORT.md` | Strict audit evidence |
| `results/metrics.json` | Final Baseline vs Engineered metrics |
| `results/per_question_results.csv` | Per-question evaluation log |

---

## KEEP — Pipeline code (current, audited)

| File | Role |
|------|------|
| `financial_rag_lab.py` | Baseline + Engineered RAG evaluation |
| `build_corpus.py` | Download/parse Treasury Bulletin PDFs → `.txt` |
| `extract_csv.py` | Build `officeqa_pro.csv` from GitHub snapshot |
| `download_data.py` | Optional Hugging Face data download |
| `create_submission_docx.py` | Regenerate final Word submission |
| `run_all.py` | End-to-end pipeline runner |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

## KEEP — Data & dependencies

| File / folder | Role |
|---------------|------|
| `data/officeqa_pro.csv` | Answer key (268 questions) |
| `data/corpus/treasury_bulletin_2022_*.txt` | Corpus year 2022 (12 files) |
| `data/corpus/treasury_bulletin_2023_*.txt` | Corpus year 2023 (12 files) |
| `data/corpus/treasury_bulletin_2024_*.txt` | Corpus year 2024 (12 files) |
| `data/corpus/treasury_bulletin_2025_*.txt` | Corpus year 2025 (12 files) |
| `data/raw_pdfs/b2022-*.pdf` | Source PDFs for 2022–2024 rebuild |
| `data/raw_pdfs/b2023-*.pdf` | Source PDFs for 2022–2024 rebuild |
| `data/raw_pdfs/b2024-*.pdf` | Source PDFs for 2022–2024 rebuild |
| `sources/officeqa_pro_github.html` | Source snapshot for `extract_csv.py` |
| `officeqa/reward.py` | Factual scoring (±1% tolerance) |
| `officeqa/README.md` | OfficeQA benchmark reference |
| `officeqa/LICENSE-APACHE` | License |
| `officeqa/LICENSE-CC-BY-SA` | License |
| `officeqa/NOTICE` | Notice |

---

## DELETED — Superseded or unused (cleanup 2026-06-18)

| File / folder | Reason |
|---------------|--------|
| `Result/` (entire folder) | Duplicate of `results/`; merged into `results/` |
| `results/DISCUSSION_BOARD_SUBMISSION.md` | **Old draft** (wrong reflections; superseded by CORRECTED) |
| `data/corpus/treasury_bulletin_2020_*.txt` (12) | Not used — pipeline years are 2022–2025 |
| `data/corpus/treasury_bulletin_2021_*.txt` (9) | Not used — pipeline years are 2022–2025 |
| `data/raw_pdfs/b2020-*.pdf` (4) | Only fed removed 2020 corpus |
| `data/raw_pdfs/b2021-*.pdf` (3) | Only fed removed 2021 corpus |
| `data/raw_pdfs/test2025.pdf` | Debug download artifact |
| `data/raw_pdfs/b2025-1.pdf` | Invalid HTML file (not a real PDF) |
| `officeqa/.git/` | Git metadata not needed for submission |
| `officeqa/corpus_scripts/` | Not referenced by current pipeline |

---

## Final folder layout

```
Fiancial/
├── README.md
├── requirements.txt
├── financial_rag_lab.py
├── build_corpus.py
├── extract_csv.py
├── download_data.py
├── create_submission_docx.py
├── run_all.py
├── PROJECT_FILE_INVENTORY.md          ← this file
├── data/
│   ├── officeqa_pro.csv
│   ├── corpus/                          ← 48 txt files (2022–2025)
│   └── raw_pdfs/                        ← b2022–b2024 quarterly PDFs
├── results/                             ← ALL submission outputs here
│   ├── Financial_RAG_Discussion_Board_Submission.docx
│   ├── CORRECTED_DISCUSSION_BOARD_SUBMISSION.md
│   ├── AUDIT_REPORT.md
│   ├── metrics.json
│   └── per_question_results.csv
├── sources/
│   └── officeqa_pro_github.html
└── officeqa/
    ├── reward.py
    ├── README.md
    └── LICENSE*
```

---

## Regenerate final outputs

```powershell
cd "C:\Users\audre\OneDrive\Ai Summer Class\AI Class\Practice Problem\Sec Week\Fiancial"
python financial_rag_lab.py --student-name "Audrey Rah"
python create_submission_docx.py
```
