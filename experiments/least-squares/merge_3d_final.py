"""Merge the per-width three-dimensional campaign outputs.

``run_final_3d.sh`` writes one directory per width so that the large SVD jobs
can be resumed independently.  This small post-processing step restores the
usual ``results/<model>/<study>`` layout consumed by ``plot_convergence.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from study_runner import aggregate, save_records


GROUP_FIELDS = (
    "study",
    "model",
    "configuration",
    "algorithm",
    "system_backend",
    "direct_solver",
    "direct_rcond",
    "manufactured_solution",
    "activation_power",
    "ritz_degree",
    "N",
    "Q",
    "K",
    "nu",
)


def load_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(payload.get("records", []))
    if not records:
        raise RuntimeError("no records found in the requested campaign directory")
    return records


def merge(source_study: str, dest_study: str, source: Path, destination: Path) -> None:
    # The parallel campaign stores one result file per repetition under
    # ``parts/<study>/N-<width>``.  Keep accepting the original
    # one-file-per-width layout so the already completed N=500 run and
    # older campaigns remain mergeable.  If parts exist for a width they
    # take precedence over any stale direct file for that width.
    paths: list[Path] = []
    direct_root = source / source_study
    parts_root = source / "parts" / source_study
    widths = {
        path.name
        for path in direct_root.glob("N-*")
        if path.is_dir()
    } | {
        path.name
        for path in parts_root.glob("N-*")
        if path.is_dir()
    }
    for width in sorted(widths):
        part_paths = sorted((parts_root / width).glob("run-*/results.json"))
        if part_paths:
            paths.extend(part_paths)
        else:
            direct = direct_root / width / "results.json"
            if direct.exists():
                paths.append(direct)
    if not paths or any(not path.exists() for path in paths):
        raise FileNotFoundError(f"incomplete {source_study} campaign under {source}")
    records = load_records(paths)
    for record in records:
        record["study"] = dest_study
    summary = aggregate(records, GROUP_FIELDS)
    save_records(destination / "elasticity-3d" / dest_study, records, summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results-remote/2026-08-28-3d-final"))
    parser.add_argument("--destination", type=Path, default=Path("results"))
    parser.add_argument("--source-study", default="main")
    parser.add_argument("--destination-study", default="order")
    args = parser.parse_args()
    merge(args.source_study, args.destination_study, args.source, args.destination)
    print(f"merged three-dimensional records from {args.source} into {args.destination}")


if __name__ == "__main__":
    main()
