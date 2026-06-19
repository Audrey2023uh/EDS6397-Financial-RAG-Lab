"""Build Treasury Bulletin text corpus from fiscal.treasury.gov quarterly PDFs."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import requests
from pypdf import PdfReader

# Quarterly Treasury Bulletin PDFs on fiscal.treasury.gov (bYEAR-Q.pdf).
QUARTER_TO_MONTHS = {
    1: (1, 2, 3),
    2: (4, 5, 6),
    3: (7, 8, 9),
    4: (10, 11, 12),
}
LEGACY_BASE_URL = (
    "https://fiscal.treasury.gov/system/files/files/reports-statements/treasury-bulletin/b{year}-{quarter}.pdf"
)
MODERN_BASE_URL = (
    "https://www.fiscal.treasury.gov/files/reports-statements/treasury-bulletin/{year}/b{year}-{quarter}.pdf"
)


def pdf_urls(year: int, quarter: int) -> list[str]:
    if year >= 2025:
        return [MODERN_BASE_URL.format(year=year, quarter=quarter)]
    return [
        LEGACY_BASE_URL.format(year=year, quarter=quarter),
        MODERN_BASE_URL.format(year=year, quarter=quarter),
    ]


def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(__import__("io").BytesIO(pdf_bytes))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)


def download_pdf(urls: list[str], dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 10_000:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        response = requests.get(
            url,
            timeout=120,
            headers={"User-Agent": "Mozilla/5.0 (compatible; OfficeQA-RAG-Lab/1.0)"},
        )
        if response.status_code == 200 and len(response.content) >= 10_000:
            if not response.content.startswith(b"%PDF"):
                continue
            dest.write_bytes(response.content)
            return True
    return False


def write_monthly_txt(year: int, month: int, text: str, corpus_dir: Path) -> Path:
    name = f"treasury_bulletin_{year}_{month:02d}.txt"
    header = (
        f"# U.S. Treasury Bulletin\n"
        f"# Source: fiscal.treasury.gov quarterly PDF\n"
        f"# Year: {year} | Month: {month:02d}\n\n"
    )
    path = corpus_dir / name
    path.write_text(header + text, encoding="utf-8")
    return path


def build_corpus(years: tuple[int, ...], data_dir: Path) -> int:
    raw_dir = data_dir / "raw_pdfs"
    corpus_dir = data_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for year in years:
        for quarter, months in QUARTER_TO_MONTHS.items():
            urls = pdf_urls(year, quarter)
            pdf_path = raw_dir / f"b{year}-{quarter}.pdf"
            if not download_pdf(urls, pdf_path):
                print(f"SKIP missing PDF: {urls[0]}")
                continue
            text = extract_pdf_text(pdf_path.read_bytes())
            if len(text.strip()) < 500:
                print(f"SKIP low text volume: {pdf_path.name}")
                continue
            for month in months:
                write_monthly_txt(year, month, text, corpus_dir)
                written += 1
            print(f"OK {pdf_path.name} -> months {months}")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and parse Treasury Bulletin PDFs")
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    args = parser.parse_args()
    count = build_corpus(tuple(args.years), args.data_dir)
    print(f"Wrote {count} monthly text files under {args.data_dir / 'corpus'}")


if __name__ == "__main__":
    main()
