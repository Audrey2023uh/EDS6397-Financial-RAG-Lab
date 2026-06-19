"""Create corrected Word submission document."""
from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "results" / "metrics.json"
OUTPUT_PATH = ROOT / "results" / "Financial_RAG_Discussion_Board_Submission.docx"

STUDENT_NAME = "Audrey Rah"


def fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def fmt_mrr(value: float) -> str:
    return f"{value:.4f}"


def add_heading(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(12)


def main() -> None:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    baseline = metrics["baseline"]
    engineered = metrics["engineered"]
    years = ", ".join(str(y) for y in metrics["years"])

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    title = doc.add_paragraph()
    title_run = title.add_run("Self Discovery Lab: The Financial RAG Challenge")
    title_run.bold = True
    title_run.font.size = Pt(14)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph()
    doc.add_paragraph(f"Name: {STUDENT_NAME} | Recent Years Used: {years}")

    add_heading(doc, "Part 1: The Scorecard")
    doc.add_paragraph()

    table = doc.add_table(rows=6, cols=3)
    table.style = "Table Grid"
    headers = ["Metric", "Baseline (Simple)", "Engineered (Improved)"]
    for col, header in enumerate(headers):
        cell = table.rows[0].cells[col]
        cell.text = header
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True

    rows = [
        ("Hit Rate (K=5)", fmt_pct(baseline["hit_rate_at_k"]), fmt_pct(engineered["hit_rate_at_k"])),
        ("MRR", fmt_mrr(baseline["mrr"]), fmt_mrr(engineered["mrr"])),
        ("Groundedness", fmt_pct(baseline["groundedness"]), fmt_pct(engineered["groundedness"])),
        ("Factual Accuracy", fmt_pct(baseline["factual_accuracy"]), fmt_pct(engineered["factual_accuracy"])),
        ("Hallucination Rate", fmt_pct(baseline["hallucination_rate"]), fmt_pct(engineered["hallucination_rate"])),
    ]
    for row_idx, (metric, base_val, eng_val) in enumerate(rows, start=1):
        table.rows[row_idx].cells[0].text = metric
        table.rows[row_idx].cells[1].text = base_val
        table.rows[row_idx].cells[2].text = eng_val

    for row in table.rows:
        row.height = Inches(0.28)

    doc.add_paragraph(
        f"Set A (Retriever) — Recall@5: Baseline {fmt_pct(baseline['recall'])} | "
        f"Engineered {fmt_pct(engineered['recall'])} "
        f"(3 OfficeQA Pro questions: UID0010, UID0086, UID0111)."
    )

    doc.add_paragraph()
    add_heading(doc, "Part 2: Engineering Reflection")
    doc.add_paragraph()

    doc.add_paragraph(
        "The Bottleneck: At the aggregate level, the baseline failed more on retrieval than generation. "
        f"Hit Rate@5 was {fmt_pct(baseline['hit_rate_at_k'])} and MRR was {fmt_mrr(baseline['mrr'])}, while "
        f"Groundedness was {fmt_pct(baseline['groundedness'])}. That gap suggests the librarian often did not "
        "return the right bulletin in the top 5. However, for UID0086 the baseline did retrieve "
        "treasury_bulletin_2022_12.txt (rank 3) yet still answered incorrectly (30, vs 4.815), so the "
        "generator/table parser also failed when the right document was present. Baseline Factual Accuracy of "
        f"{fmt_pct(baseline['factual_accuracy'])} reflects both retrieval misses and answer-extraction failures."
    )

    doc.add_paragraph(
        "The Metadata Fix: Tagging chunks with Year and Month from filenames and using year-level filtering plus "
        "soft metadata bonuses in the engineered pipeline raised Hit Rate@5 from "
        f"{fmt_pct(baseline['hit_rate_at_k'])} to {fmt_pct(engineered['hit_rate_at_k'])} and MRR from "
        f"{fmt_mrr(baseline['mrr'])} to {fmt_mrr(engineered['mrr'])}. Recall@5 rose from "
        f"{fmt_pct(baseline['recall'])} to {fmt_pct(engineered['recall'])}. Factual Accuracy improved from "
        f"{fmt_pct(baseline['factual_accuracy'])} to {fmt_pct(engineered['factual_accuracy'])}. Groundedness fell "
        f"from {fmt_pct(baseline['groundedness'])} to {fmt_pct(engineered['groundedness'])} because the correct "
        "engineered answer 4.815 was computed from table values, not quoted verbatim in chunks—so retrieval metrics "
        "improved more than generation-trust metrics."
    )

    doc.add_paragraph(
        "Scaling Insight: Scaling from this 4-year subset to the full 1939–2025 archive, the first component to "
        "break would be the in-memory vector search layer (embedding matrix + cosine similarity over all chunks). "
        "Chunk count grows with years, so brute-force search and re-embedding would become too slow and memory-heavy "
        "before any LLM-style generator became the bottleneck. Production would need sharded storage, approximate "
        "nearest-neighbor indexes (e.g., FAISS), and metadata-first year/month filtering."
    )

    doc.add_paragraph()
    add_heading(doc, "Technical Stack (Section 3)")
    doc.add_paragraph(
        "Vector index: In-memory scikit-learn (TF-IDF cosine baseline; SentenceTransformer all-MiniLM-L6-v2 "
        "hybrid sparse/dense engineered). No ChromaDB/FAISS persisted store."
    )
    doc.add_paragraph(
        "Metadata: Year and Month on every chunk from treasury_bulletin_YYYY_MM.txt. Engineered path uses year "
        "filter, file-level pre-retrieval (top 3 bulletins), then chunk search with soft year/month/ESF bonuses."
    )
    doc.add_paragraph(
        "Chunking: Baseline — 1,200 characters, 0 overlap. Engineered — 2,048 characters (~512 tokens), "
        "512-character overlap."
    )
    doc.add_paragraph(
        "Generator: Extractive/rule-based answerer (not an LLM), with ESF QoQ helper. Scored with officeqa/reward.py "
        "at ±1% tolerance, K=5."
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
