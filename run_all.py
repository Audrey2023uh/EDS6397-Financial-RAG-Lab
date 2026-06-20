"""Run the full Financial RAG lab pipeline end-to-end."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print(f"\n>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    run([sys.executable, "extract_csv.py"])
    run([sys.executable, "build_corpus.py", "--years", "2022", "2023", "2024", "2025"])
    run([sys.executable, "financial_rag_lab.py", "--student-name", "Audrey Rah"])
    run([sys.executable, "create_submission_docx.py"])


if __name__ == "__main__":
    main()
