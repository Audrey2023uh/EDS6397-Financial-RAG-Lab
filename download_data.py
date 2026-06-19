"""Download OfficeQA benchmark CSV and Treasury Bulletin text corpus from Hugging Face."""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

YEARS_DEFAULT = (2022, 2023, 2024, 2025)
MONTHS = tuple(f"{m:02d}" for m in range(1, 13))


def year_file_pattern(years: tuple[int, ...]) -> list[str]:
    patterns: list[str] = []
    for year in years:
        for month in MONTHS:
            patterns.append(
                f"treasury_bulletins_parsed/transformed/treasury_bulletin_{year}_{month}.txt"
            )
    return patterns


def download_from_hf(years: tuple[int, ...], data_dir: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub: pip install huggingface_hub") from exc

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit(
            "Hugging Face token required. Request access at "
            "https://huggingface.co/datasets/databricks/officeqa then set HF_TOKEN."
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = hf_hub_download(
        repo_id="databricks/officeqa",
        filename="officeqa_pro.csv",
        repo_type="dataset",
        token=token,
        local_dir=str(data_dir / "hf_cache"),
        local_dir_use_symlinks=False,
    )
    # copy to stable path
    target_csv = data_dir / "officeqa_pro.csv"
    target_csv.write_bytes(Path(csv_path).read_bytes())
    print(f"Downloaded benchmark CSV -> {target_csv}")

    corpus_dir = data_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    patterns = year_file_pattern(years)
    local_dir = snapshot_download(
        repo_id="databricks/officeqa",
        repo_type="dataset",
        allow_patterns=patterns,
        token=token,
        local_dir=str(corpus_dir / "hf_snapshot"),
        local_dir_use_symlinks=False,
    )
    transformed = Path(local_dir) / "treasury_bulletins_parsed" / "transformed"
    copied = 0
    for txt in transformed.glob("treasury_bulletin_*.txt"):
        year = int(re.search(r"_(\d{4})_", txt.name).group(1))
        if year in years:
            dest = corpus_dir / txt.name
            dest.write_bytes(txt.read_bytes())
            copied += 1
    print(f"Copied {copied} bulletin files into {corpus_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS_DEFAULT))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    args = parser.parse_args()
    download_from_hf(tuple(args.years), args.data_dir)


if __name__ == "__main__":
    main()
