"""Run the two-dimensional near-incompressibility benchmark.

The manufactured displacement is Example 1 of Li--Yang (CMAME, 2020), also
used by Grieshaber--McBride--Reddy.  For every dictionary width and repetition,
all Lamé parameters share the dictionary, training points, Ritz quadrature,
validation rule, and test rule.  The coefficient budget is fixed in advance;
it is not tuned separately for different material parameters.

The paper-scale defaults are intentionally expensive::

    OPENBLAS_NUM_THREADS=4 LS_TORCH_THREADS=4 \
      conda run -n dl python run_near_incompressible_2d.py
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

import study_runner as sr


ROOT = Path(__file__).resolve().parent
SPEC = sr.MODEL_SPECS["elasticity-2d"]
DEFAULT_WIDTHS = (200, 400, 600, 800, 1000)
DEFAULT_LAMBDAS = (10.0, 1000.0, 100000.0)
DEFAULT_OUTPUT_DIR = ROOT / "results" / "elasticity-2d" / "near-incompressible"
MANUFACTURED_SOLUTION = "grieshaber_li_yang"
MU = 1.0


def material_parameters(lam: float) -> tuple[float, float]:
    """Return ``(E, nu)`` that gives the prescribed ``mu=1`` and ``lambda``."""

    if not math.isfinite(lam) or lam <= 0.0:
        raise ValueError("lambda values must be finite and positive")
    nu = lam / (2.0 * (lam + MU))
    young = 2.0 * MU * (1.0 + nu)
    return young, nu


def parse_int_list(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return parsed


def parse_float_list(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(not math.isfinite(item) or item <= 0.0 for item in parsed):
        raise argparse.ArgumentTypeError(
            "expected finite positive comma-separated numbers"
        )
    return parsed


def load_existing_records(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "results.json"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Malformed results file: {path}")
    return records


def configure_threads() -> None:
    token = os.environ.get("LS_TORCH_THREADS")
    if token is None:
        return
    try:
        count = int(token)
    except ValueError as exc:
        raise ValueError("LS_TORCH_THREADS must be a positive integer") from exc
    if count < 1:
        raise ValueError("LS_TORCH_THREADS must be a positive integer")
    torch.set_num_threads(count)
    torch.set_num_interop_threads(1)


def save_progress(output_dir: Path, records: list[dict[str, Any]]) -> None:
    group_fields = (
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
        "lambda",
        "E",
        "nu",
        "mu",
        "fixed_budget",
    )
    summary = sr.aggregate(records, group_fields)
    sr.save_records(output_dir, records, summary)


def run(args: argparse.Namespace) -> None:
    module = sr.load_model(SPEC)
    output_dir = Path(args.output_dir)
    records = load_existing_records(output_dir) if args.resume else []
    completed = {
        (int(record["N"]), int(record["run"]), float(record["lambda"]))
        for record in records
    }

    for width in args.widths:
        base_cfg = replace(
            module.default_config(),
            N_s=width,
            N_u=width,
            Q_train=args.q_ratio * (width + 1),
            Q_test=args.test_points,
            activation_power=7,
            ritz_degree=9,
            ritz_ratio=2.0,
            projection_samples=12808,
            sampling_method="mc",
            manufactured_solution=MANUFACTURED_SOLUTION,
            algorithms_to_run=["ball"],
            direct_rcond=1.0e-14,
            system_backend="direct",
            direct_solver="streaming_tsqr",
            coefficient_budget=args.fixed_budget,
        )

        for local_run in range(args.repeats):
            run_index = args.run_offset + local_run
            missing_lambdas = tuple(
                lam
                for lam in args.lambdas
                if (width, run_index, float(lam)) not in completed
            )
            if not missing_lambdas:
                print(
                    f"SKIP N={width}, run={run_index + 1}: all lambda values exist",
                    flush=True,
                )
                continue

            seed = 100_000 * (run_index + 1)
            projection_seed = seed + 23
            training_seed = seed + 31
            validation_seed = seed + 37
            test_seed = seed + 41
            feature_space = sr.build_feature_space(
                module,
                SPEC,
                base_cfg,
                projection_seed=projection_seed,
            )
            parameter_diagnostics = sr.parameter_set_diagnostics(feature_space.theta_s)
            projection = feature_space.projected_u

            for lam in missing_lambdas:
                young, nu = material_parameters(lam)
                cfg = replace(base_cfg, E=young, nu=nu)
                print(
                    f"\n=== near-incompressible, N={width}, "
                    f"lambda={lam:g}, run={run_index + 1}/{args.run_offset + args.repeats} ===",
                    flush=True,
                )

                validation_benchmark = module.build_shared_benchmark(
                    **sr.benchmark_kwargs(
                        SPEC,
                        cfg,
                        q_test=args.validation_points,
                        interior_seed=training_seed,
                        test_seed=validation_seed,
                    )
                )
                validation_data = module.prepare_experiment(
                    cfg,
                    validation_benchmark,
                    feature_space,
                )
                validation_result = module.run_experiment(
                    cfg,
                    print_table=False,
                    plot_results=False,
                    benchmark=validation_benchmark,
                    feature_space=feature_space,
                    experiment_data=validation_data,
                )[0]
                validation_score = math.hypot(
                    validation_result.sigma_hdiv_error,
                    validation_result.u_h1_error,
                )

                test_benchmark = module.build_shared_benchmark(
                    **sr.benchmark_kwargs(
                        SPEC,
                        cfg,
                        q_test=args.test_points,
                        interior_seed=training_seed,
                        test_seed=test_seed,
                    )
                )
                test_data = module.retarget_experiment_data(
                    cfg,
                    validation_data,
                    test_benchmark,
                    feature_space,
                )
                result = module.run_experiment(
                    cfg,
                    print_table=False,
                    plot_results=False,
                    benchmark=test_benchmark,
                    feature_space=feature_space,
                    experiment_data=test_data,
                )[0]
                material = module.PROBLEM.make_material(cfg)
                record = {
                    "study": "near-incompressible",
                    "model": SPEC.key,
                    "configuration": f"N-{width}-lambda-{lam:g}",
                    "run": run_index,
                    "N": width,
                    "Q": cfg.Q_train,
                    "K": int(projection.space.dimension),
                    "lambda": float(lam),
                    "material_lambda": float(material.lam),
                    "E": float(young),
                    "nu": float(nu),
                    "mu": MU,
                    "activation_power": cfg.activation_power,
                    "ritz_degree": int(projection.degree),
                    "ritz_ratio": cfg.ritz_ratio,
                    "projection_samples": int(projection.quadrature_samples),
                    "projection_gram_residual": float(projection.gram_residual),
                    "boundary_residual": float(projection.boundary_residual()),
                    "fixed_budget": float(args.fixed_budget),
                    "validation_score": validation_score,
                    "validation_points": args.validation_points,
                    "test_points": args.test_points,
                    "test_residual": math.hypot(
                        result.constitutive_residual,
                        result.equilibrium_residual,
                    ),
                    "parameter_covering_radius": float(
                        parameter_diagnostics.covering_radius
                    ),
                    "parameter_separation": float(parameter_diagnostics.separation),
                    "projection_seed": projection_seed,
                    "training_seed": training_seed,
                    "validation_seed": validation_seed,
                    "test_seed": test_seed,
                    "system_backend": cfg.system_backend,
                    "direct_solver": cfg.direct_solver,
                    "direct_rcond": cfg.direct_rcond,
                    "manufactured_solution": cfg.manufactured_solution,
                    **asdict(result),
                }
                records.append(
                    {key: sr.finite_or_text(value) for key, value in record.items()}
                )
                completed.add((width, run_index, float(lam)))
                save_progress(output_dir, records)
                print(
                    f"Saved {len(records)} records to {output_dir}; "
                    f"validation graph error={validation_score:.6e}",
                    flush=True,
                )
                del (
                    validation_data,
                    validation_benchmark,
                    test_data,
                    test_benchmark,
                )
                gc.collect()

            del feature_space
            gc.collect()

    save_progress(output_dir, records)
    print(f"Completed {len(records)} records in {output_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=parse_int_list, default=DEFAULT_WIDTHS)
    parser.add_argument("--lambdas", type=parse_float_list, default=DEFAULT_LAMBDAS)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--run-offset", type=int, default=0)
    parser.add_argument("--q-ratio", type=int, default=8)
    parser.add_argument("--validation-points", type=int, default=64**2)
    parser.add_argument("--test-points", type=int, default=128**2)
    parser.add_argument("--fixed-budget", type=float, default=3.0e6)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append missing (N, run, lambda) tuples to an existing results.json",
    )
    return parser


def main() -> None:
    configure_threads()
    args = build_parser().parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.run_offset < 0:
        raise ValueError("--run-offset must be nonnegative")
    if args.q_ratio < 1:
        raise ValueError("--q-ratio must be positive")
    if args.validation_points < 1 or args.test_points < 1:
        raise ValueError("quadrature point counts must be positive")
    if not math.isfinite(args.fixed_budget) or args.fixed_budget <= 0.0:
        raise ValueError("--fixed-budget must be finite and positive")
    run(args)


if __name__ == "__main__":
    main()
