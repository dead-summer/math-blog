"""Coefficient growth as a diagnostic of dictionary saturation.

The total-error estimate carries the coefficient budget ``B`` through the
training term ``B^2 K Q^{-1/2}``, so ``B`` is not free.  Sweeping the truncation
level on a fixed assembled system separates two regimes:

* saturated dictionary -- the small singular directions carry no approximation
  power, so tightening ``rcond`` inflates ``||c||`` by orders of magnitude while
  the graph error stalls or worsens;
* unsaturated dictionary -- those directions carry real signal, so the graph
  error falls by orders of magnitude while ``||c||`` barely moves.

Usage::

    python run_coefficient_saturation.py --model elasticity-2d --powers 3,7
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from solvers import run_solver  # noqa: E402
from study_runner import MODEL_SPECS, load_model  # noqa: E402

RESULTS = HERE / "results"
DEFAULT_RCONDS = (1e-6, 1e-8, 1e-10, 1e-12, 1e-14)


def parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def evaluate(module, spec, data, coefficients):
    """Model-specific graph errors for one coefficient vector."""

    if spec.key == "plate":
        result = module.evaluate_feature_result(
            "tsvd", 0.0,
            coefficients[: data.dim_m], coefficients[data.dim_m :],
            data.eval_data,
        )
        return {"scalar_graph": result.w_h2_error, "tensor_graph": result.M_hdivdiv_error}
    import elasticity_common as ec

    stress = ec.lift_stress_coefficients(
        data.stress_adapter, coefficients[: data.solved_dim_s]
    )
    result = ec.evaluate_feature_result(
        module.PROBLEM, "tsvd", 0.0, stress,
        coefficients[data.solved_dim_s :], data.eval_data,
    )
    return {"scalar_graph": result.u_h1_error, "tensor_graph": result.sigma_hdiv_error}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="elasticity-2d", choices=tuple(MODEL_SPECS))
    parser.add_argument("--powers", type=parse_ints, default=(3, 7))
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--rconds", type=parse_floats, default=DEFAULT_RCONDS)
    parser.add_argument("--threads", type=int, default=28)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    spec = MODEL_SPECS[args.model]
    module = load_model(spec)
    rows: list[dict] = []

    for power in args.powers:
        cfg = module.default_config()
        cfg.activation_power = power
        if args.width is not None:
            setattr(cfg, spec.width_fields[0], args.width)
            setattr(cfg, spec.width_fields[1], args.width)
            cfg.Q_train = 4 * (args.width + 1)
        cfg.algorithms_to_run = ["tsvd"]
        width = int(getattr(cfg, spec.width_fields[0]))

        benchmark_kwargs = dict(
            E=cfg.E, nu=cfg.nu, Q_train=cfg.Q_train, Q_test=cfg.Q_test,
            sampling_method=cfg.sampling_method,
            manufactured_solution=cfg.manufactured_solution,
        )
        if spec.key == "plate":
            benchmark_kwargs["h"] = cfg.h
        else:
            benchmark_kwargs["body_force_batch_size"] = cfg.body_force_batch_size
        benchmark = module.build_shared_benchmark(**benchmark_kwargs)

        feature_kwargs = {
            spec.feature_width_names[0]: width,
            spec.feature_width_names[1]: width,
            "activation_power": power,
            "ritz_degree": cfg.ritz_degree,
            "ritz_ratio": cfg.ritz_ratio,
            "projection_samples": cfg.projection_samples,
        }
        feature_space = module.build_shared_feature_space(**feature_kwargs)
        data = module.prepare_experiment(cfg, benchmark, feature_space)

        print(f"\n=== {args.model}  k={power}  N={width}  m={data.column_count} ===")
        print(f"{'rcond':>8} {'rank':>6} {'||c||_2':>11} {'B':>11} "
              f"{'scalar graph':>13} {'tensor graph':>13}")
        for rcond in args.rconds:
            output = run_solver(
                "tsvd", data.residual_design, data.rhs, cfg, hyperparameter=rcond
            )
            coefficients = output.coefficients
            norm = float(torch.linalg.vector_norm(coefficients))
            errors = evaluate(module, spec, data, coefficients)
            budget = norm * math.sqrt(data.column_count)
            print(f"{rcond:>8.0e} {output.rank:>6d} {norm:>11.3e} {budget:>11.3e} "
                  f"{errors['scalar_graph']:>13.3e} {errors['tensor_graph']:>13.3e}")
            rows.append({
                "model": args.model, "activation_power": power, "N": width,
                "columns": data.column_count, "rcond": rcond, "rank": output.rank,
                "coefficient_norm": norm, "budget": budget, **errors,
            })
        del data, feature_space, benchmark

    RESULTS.mkdir(exist_ok=True)
    stem = RESULTS / f"coefficient-saturation-{args.model}"
    stem.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {stem}.csv and {stem}.json")

    for power in args.powers:
        family = [row for row in rows if row["activation_power"] == power]
        if len(family) < 2:
            continue
        norm_growth = family[-1]["coefficient_norm"] / family[0]["coefficient_norm"]
        error_gain = family[0]["tensor_graph"] / family[-1]["tensor_graph"]
        print(f"  k={power}: over the rcond ladder ||c|| grows {norm_growth:.3g}x "
              f"while the tensor graph error improves {error_gain:.3g}x")


if __name__ == "__main__":
    main()
