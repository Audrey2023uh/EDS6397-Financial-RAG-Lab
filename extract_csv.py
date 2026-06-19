"""One-time helper: extract officeqa_pro.csv from GitHub blob HTML snapshot."""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path

SOURCE_HTML = Path(__file__).resolve().parent / "sources" / "officeqa_pro_github.html"
OUTPUT_CSV = Path(__file__).resolve().parent / "data" / "officeqa_pro.csv"


def extract_csv_from_html(html: str) -> str:
    match = re.search(r"```csv\n(.*?)```", html, re.DOTALL)
    if not match:
        raise ValueError("Could not find ```csv block in HTML snapshot")
    return match.group(1)


def main() -> None:
    if not SOURCE_HTML.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE_HTML}. Save the GitHub officeqa_pro.csv page HTML there."
        )
    html = SOURCE_HTML.read_text(encoding="utf-8")
    csv_text = extract_csv_from_html(html)
    rows = list(csv.reader(io.StringIO(csv_text)))
    if len(rows) < 2:
        raise ValueError("CSV appears empty")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.write_text(csv_text, encoding="utf-8")
    print(f"Wrote {len(rows) - 1} questions to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
