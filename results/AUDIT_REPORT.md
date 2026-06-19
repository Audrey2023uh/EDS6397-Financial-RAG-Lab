# Strict Audit: Financial RAG Discussion Board Submission

**Audited against:** `results/metrics.json`, `results/per_question_results.csv`, re-run of `financial_rag_lab.py` (2026-06-18), and `financial_rag_lab.py` source.

---

## 1. Metric verification (PASS with caveats)

| Metric | Baseline (reported) | Baseline (verified) | Engineered (reported) | Engineered (verified) | Verdict |
|--------|--------------------:|--------------------:|----------------------:|----------------------:|---------|
| Hit Rate@5 | 33.33% | 33.33% (1/3) | 66.67% | 66.67% (2/3) | **PASS** |
| MRR | 0.1111 | 0.1111 | 0.3333 | 0.3333 | **PASS** |
| Recall | 0.12% | 0.12% | 7.72% | 7.72% | **PASS** (in `metrics.json` / `.md` only) |
| Groundedness | 100.00% | 100.00% | 66.67% | 66.67% | **PASS** |
| Factual Accuracy | 0.00% | 0.00% | 33.33% | 33.33% (1/3) | **PASS** |
| Hallucination Rate | 0.00% | 0.00% | 33.33% | 33.33% | **PASS** |

**Per-question evidence (`per_question_results.csv`):**

| UID | Baseline Hit | Baseline MRR | Baseline Factual | Eng Hit | Eng MRR | Eng Factual |
|-----|:------------:|:------------:|:----------------:|:-------:|:-------:|:-----------:|
| UID0010 | No | 0 | No | Yes | 0.5 | No |
| UID0086 | Yes | 0.333 | No | Yes | 0.5 | **Yes** |
| UID0111 | No | 0 | No | No | 0 | No |

**Caveats:**
- Only **3 evaluation questions** (not 133). Metrics are statistically fragile.
- **Groundedness** marks UID0086 engineered as 0% grounded because `4.815` is **computed**, not verbatim in chunks—so “Hallucination Rate 33%” is a **metric-definition artifact**, not true fabrication.
- **Recall** is computed as `(relevant chunks in top-5) / (all chunks from gold source files in DB)` and is very low because gold files split into hundreds of chunks.

---

## 2. Assignment requirement checklist

| Requirement | Verdict | Evidence / issue |
|-------------|---------|------------------|
| ≥4 recent years in corpus (2022–2025) | **PASS** | 48 monthly files loaded for 2022–2025 |
| Baseline + Engineered scorecard | **PASS** | `metrics.json` |
| Hit Rate, MRR, Groundedness, Factual Accuracy, Hallucination @ K=5 | **PASS** | Values match logs |
| Two reflection parts (3 questions) | **PASS** in Word | Present in `.docx` |
| Filter `officeqa_pro` to chosen years | **PARTIAL FAIL** | Filter uses *any* source year in window, not *all* years; **UID0111** needs 2015/2020 bulletins **not in corpus** |
| Document architecture (Section 3) | **FAIL** in Word | In `DISCUSSION_BOARD_SUBMISSION.md` only; **missing from `.docx`** |
| Vector DB documented | **PARTIAL** | Uses in-memory scikit-learn matrices, **not** ChromaDB/FAISS |
| Metadata Year/Month described | **PASS** | Tagged from filenames; year filter + soft bonuses in engineered path |
| Chunking strategy described | **PASS** in `.md` only | 1200/0 vs 2048/512 chars |
| Recall reported (Set A, Section 4) | **PARTIAL** | In `.md` + `metrics.json`; **omitted from Word scorecard** |
| Discussion template exact match | **PARTIAL** | `.md` adds Recall row not in template; Word omits Recall |

---

## 3. Inaccuracies / unsupported claims in current submission

### Part 1 scorecard
- **No numeric errors** relative to `metrics.json`.

### Part 2 reflections — issues found

| Claim | Verdict | Why |
|-------|---------|-----|
| “Factual Accuracy 0% confirms retrieval—not reading—was the limiting step” | **UNSUPPORTED / WRONG** | UID0086 **baseline retrieved** `treasury_bulletin_2022_12.txt` (rank 3) but answered `30,` not `4.815`. Retrieval worked; **generator failed**. |
| “When context was retrieved the answers stayed tied to source numbers” | **FALSE** | Baseline predictions: `1`, `30,`, `2024` — all **incorrect** vs ground truth. |
| “Failed to surface December 2022 bulletin” (as baseline example) | **MISLEADING** | Baseline **did** surface `2022_12` for UID0086; it failed on UID0010 and UID0111. |
| “Metadata helped Groundedness” | **FALSE** | Groundedness **fell** 100% → 66.67%. |
| “Year/**Month** metadata filtering” | **OVERSTATED** | Code applies **year filter** + **soft** month/entity bonuses, not hard month filtering. |
| “Vector store / embedding index” scaling | **SUPPORTED** | No persistent vector DB; brute-force cosine over in-memory matrices. |

### Architecture (`.md` only)
| Claim | Verdict |
|-------|---------|
| “SentenceTransformer embeddings (engineered)” | **PASS** |
| “TF-IDF baseline” | **PASS** (uni-grams only: `ngram_range=(1,1)`) |
| “Soft month/entity bonuses at rerank” | **PASS** |
| “Two-stage file→chunk retrieval” | **MISSING from submission text** — implemented in code |
| “Hybrid 65% dense + 35% sparse” | **MISSING from submission text** |
| “Extractive generator (not LLM)” | **MISSING** — important for interpreting Generator metrics |
| “ChromaDB / FAISS” | **NOT USED** |

---

## 4. Should Recall be reported?

**Yes, for full assignment compliance.** Section 4 Set A lists Recall as a required retriever metric (“Report these numbers for both Baseline and Engineered”). The **discussion board template** omits it, but instructors grading against the full rubric may expect it. **Recommendation:** add a Recall row or a one-line Set A footnote under the scorecard.

---

## 5. Name: Audre vs Audrey Rah

No name appears in generated metrics or code logs. For an academic submission, use your **official Canvas name**. If that is **Audrey Rah**, change it—**“Audre” alone is likely a grading risk**.

---

## 6. Grading risks (summary)

1. Only **3** eval questions from OfficeQA Pro for 2022–2025.
2. **UID0111** evaluated without required 2015/2020 source bulletins in corpus.
3. **2025 corpus** partly built from ESF financial PDF proxy, not full Treasury Bulletin parsed text.
4. **Reflection logic errors** (factual accuracy ≠ proof of retrieval bottleneck).
5. **Architecture not in Word** submission.
6. **No ChromaDB/FAISS** (acceptable if documented honestly).
7. **Generator is rule-based/extractive**, not an LLM—should be stated when discussing “Generator” metrics.

---

## 7. Overall audit verdict

| Artifact | Verdict |
|----------|---------|
| Numeric scorecard | **PASS** (matches code output) |
| Word `.docx` vs assignment template | **PASS** (5 metrics + 3 reflections) |
| Full assignment compliance | **FAIL** (architecture, Recall, question filter, reflection accuracy) |
| Safe to post as-is | **NO** — use corrected submission below |
