Name: Audrey Rah | Recent Years Used: 2022, 2023, 2024, 2025

Part 1: The Scorecard

| Metric | Baseline (Simple) | Engineered (Improved) |
|---|---:|---:|
| Hit Rate (K=5) | 33.33% | 66.67% |
| MRR | 0.1111 | 0.3333 |
| Groundedness | 100.00% | 66.67% |
| Factual Accuracy | 0.00% | 33.33% |
| Hallucination Rate | 0.00% | 33.33% |

Set A (Retriever) — Recall@5: Baseline 0.12% | Engineered 7.72%
(Evaluated on 3 OfficeQA Pro questions touching 2022–2025: UID0010, UID0086, UID0111.)

Part 2: Engineering Reflection

**The Bottleneck:** At the aggregate level, the baseline failed more on retrieval than generation. Hit Rate@5 was only 33.33% and MRR was 0.1111, while Groundedness was 100.00%. That gap suggests the librarian often did not return the right bulletin in the top 5. However, the story is not one-sided: for UID0086 the baseline did retrieve `treasury_bulletin_2022_12.txt` (rank 3) yet still answered incorrectly (`30,` vs `4.815`), so the generator/table parser also failed when the right document was present. Baseline Factual Accuracy of 0.00% therefore reflects both retrieval misses and answer-extraction failures—not retrieval alone.

**The Metadata Fix:** Tagging chunks with Year and Month from filenames (e.g., `treasury_bulletin_2022_12.txt`) and using year-level filtering plus soft metadata bonuses in the engineered pipeline raised Hit Rate@5 from 33.33% to 66.67% and MRR from 0.1111 to 0.3333. Recall@5 rose from 0.12% to 7.72%. Factual Accuracy improved from 0.00% to 33.33% (UID0086 answered correctly after engineered retrieval + ESF table math). Groundedness fell from 100.00% to 66.67% because the correct engineered answer `4.815` was computed from table values, not quoted verbatim in the chunks—so retrieval metrics improved more than generation-trust metrics.

**Scaling Insight:** Scaling from this 4-year subset to the full 1939–2025 archive, the first component to break would be the in-memory vector search layer (embedding matrix + cosine similarity over all chunks). Chunk count grows with years, so brute-force search and re-embedding would become too slow and memory-heavy before any LLM-style generator became the bottleneck. Production would need sharded storage, approximate nearest-neighbor indexes (e.g., FAISS), and metadata-first year/month filtering.

Technical Stack (documented for Section 3)

- **Vector index:** In-memory scikit-learn (TF-IDF cosine baseline; SentenceTransformer `all-MiniLM-L6-v2` + hybrid sparse/dense engineered). No ChromaDB/FAISS persisted store.
- **Metadata:** Year and Month on every chunk from `treasury_bulletin_YYYY_MM.txt`. Engineered path: year filter, file-level pre-retrieval (top 3 bulletins), then chunk search with soft year/month/ESF bonuses.
- **Chunking:** Baseline — 1,200 characters, 0 overlap. Engineered — 2,048 characters (~512 tokens), 512-character overlap.
- **Generator:** Extractive/rule-based answerer (not an LLM), with ESF QoQ helper for table math. Scored with `officeqa/reward.py` at ±1% tolerance, K=5.
