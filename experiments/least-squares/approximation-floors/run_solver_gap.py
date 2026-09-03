"""Compare the campaign's measured errors with the trial space's floors.

The least-squares functional is equivalent to the graph norm, so a correctly
implemented solver should land within a small constant of the best
approximation the trial space admits.  This script quantifies that constant.  A
ratio near one means the remaining error is approximation, not solver: no
choice of ``rcond``, ridge parameter, coefficient budget, quadrature rule, or
backend can improve it.

Usage::

    python run_solver_gap.py                       # every model found
    python run_solver_gap.py --model elasticity-2d --study order
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent / "results"
RESULTS = HERE / "results"

# Campaign metric names paired with the floor columns that bound them.
METRIC_FLOORS = {
    "elasticity-2d": {"u_h1_error": "scalar_attainable_graph",
                      "sigma_hdiv_error": "tensor_raw_graph"},
    "elasticity-3d": {"u_h1_error": "scalar_attainable_graph",
                      "sigma_hdiv_error": "tensor_raw_graph"},
    "plane-stress": {"u_h1_error": "scalar_attainable_graph",
                     "sigma_hdiv_error": "tensor_raw_graph"},
    "plate": {"w_h2_error": "scalar_attainable_graph",
              "M_hdivdiv_error": "tensor_raw_graph"},
}
PAPER_ALGORITHM = "ball"


def load_floors(model: str) -> list[dict]:
    path = RESULTS / f"floors-{model}.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [
            {key: (float(value) if _is_number(value) else value)
             for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _is_number(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def load_campaign(model: str, study: str) -> list[dict]:
    path = CAMPAIGN / model / study / "results.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["summary"]


def match_floor(floors: list[dict], row: dict) -> dict | None:
    """Find the floor row measured at this configuration's N, k and degree."""

    power = float(row.get("activation_power", 3))
    degree = float(row.get("ritz_degree", 3))
    candidates = [
        floor for floor in floors
        if floor["N"] == float(row["N"]) and floor["activation_power"] == power
    ]
    exact = [floor for floor in candidates if floor["ritz_degree"] == degree]
    return (exact or candidates or [None])[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(METRIC_FLOORS))
    parser.add_argument("--study", default="order")
    parser.add_argument("--algorithm", default=PAPER_ALGORITHM)
    args = parser.parse_args()

    models = [args.model] if args.model else list(METRIC_FLOORS)
    output: list[dict] = []
    for model in models:
        floors = load_floors(model)
        summary = load_campaign(model, args.study)
        if not floors or not summary:
            print(f"[{model}] skipped: missing floors or campaign results")
            continue
        print(f"\n=== {model} / {args.study} / {args.algorithm} ===")
        header = f"{'N':>6} {'k':>3} {'metric':>18} {'measured':>11} {'floor':>11} {'ratio':>7}"
        print(header)
        for row in sorted(summary, key=lambda item: item["N"]):
            if row.get("algorithm", PAPER_ALGORITHM) != args.algorithm:
                continue
            floor = match_floor(floors, row)
            if floor is None:
                continue
            for metric, floor_key in METRIC_FLOORS[model].items():
                measured = row.get(f"{metric}_mean")
                bound = floor[floor_key]
                if measured is None or not bound:
                    continue
                ratio = measured / bound
                print(f"{int(row['N']):>6} {int(floor['activation_power']):>3} "
                      f"{metric:>18} {measured:>11.3e} {bound:>11.3e} {ratio:>7.2f}")
                output.append({
                    "model": model, "study": args.study, "algorithm": args.algorithm,
                    "N": row["N"], "activation_power": floor["activation_power"],
                    "metric": metric, "measured": measured, "floor": bound,
                    "ratio": ratio,
                })

    if output:
        RESULTS.mkdir(exist_ok=True)
        # One file per study, so the k=7 and k=3 ladders (``order`` and
        # ``order-k3``) do not overwrite each other.
        path = RESULTS / f"solver-gap-{args.study}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
        worst = max(item["ratio"] for item in output)
        print(f"\nWrote {path}   (largest measured/floor ratio: {worst:.2f})")


if __name__ == "__main__":
    main()
