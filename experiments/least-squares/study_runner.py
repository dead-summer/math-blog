"""Reproducible repeated studies for the projected least-squares models.

Examples
--------
Run the ten-repeat width study for two-dimensional elasticity::

    python study_runner.py order --model elasticity-2d

Run the prescribed training-sample and Ritz-dimension sweeps::

    python study_runner.py q --model elasticity-2d
    python study_runner.py k --model plate

The hidden-layer parameters are a deterministic quasi-uniform point set, so
repeats only randomize the training samples and the projection quadrature.
Each configured algorithm (ball / ridge / tsvd) selects its hyperparameter
on an independent validation rule before the final test evaluation; all
candidates share one spectral factorization of the assembled system.  The
paper reports ``ball`` only -- see ``README.md`` and ``solvers.py`` for why the
three are one spectral-filter family.  The runner writes raw records and
mean/sample-standard-deviation summaries in both JSON and CSV.
Full default studies are intentionally expensive.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ls_common import VALID_DIRECT_SOLVERS  # noqa: E402
from rfm_core import parameter_set_diagnostics  # noqa: E402
from solvers import SOLVERS, get_solver_spec  # noqa: E402
from system_backends import VALID_SYSTEM_BACKENDS  # noqa: E402

DEFAULT_WIDTHS = (200, 400, 600, 800, 1000)
DEFAULT_Q_RATIOS = (1, 2, 4, 8, 16, 32)
DEFAULT_K_RATIOS = (1.0, 2.0, 4.0)
DEFAULT_ACTIVATION_POWERS = (3, 5, 7, 9)
# Width at which the power study is run, per model.  The 2-D
# elasticity width is the outcome of a capacity probe: N=500 leaves enough
# rows for Q=16(N+1) while keeping the k=9 solve above the algebraic floor.
# Other models retain the top of their order ladder.
POWER_STUDY_WIDTHS = {
    "elasticity-2d": 500,
    "plane-stress": 1000,
    "plate": 1000,
    "elasticity-3d": 1000,
}
# Model-specific power-study capacity settings.  These are applied before
# CLI overrides, so an explicit --q-ratio, --ritz-degree, or
# --projection-samples still wins.  The two 2-D studies that resolve k=9 use a
# matched p=10 Ritz space and a fixed projection rule; the plate needs only
# Q=8(N+1), as established by the paired Q=8 versus Q=16 capacity probe.
POWER_STUDY_Q_RATIOS = {
    "elasticity-2d": 16,
    "plate": 8,
}
POWER_STUDY_RITZ_DEGREES = {
    "elasticity-2d": 10,
    "plate": 10,
}
POWER_STUDY_PROJECTION_SAMPLES = {
    "elasticity-2d": 12_808,
    "plate": 12_808,
}
# The plate's default deflection is a polynomial of total degree eight, which a
# rho_k dictionary reproduces exactly once its P_k supplement covers it.  That
# would make the power study measure polynomial reproduction instead of
# ridge approximation, so this study uses an analytic non-polynomial solution.
POWER_STUDY_SOLUTIONS = {"plate": "trig"}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    relative_path: str
    width_fields: tuple[str, str]
    feature_width_names: tuple[str, str]
    primary_metrics: tuple[str, str]
    supports_k_sweep: bool


MODEL_SPECS = {
    "elasticity-2d": ModelSpec(
        key="elasticity-2d",
        relative_path="linear-elasticity-2d/linear_elasticity_2d.py",
        width_fields=("N_s", "N_u"),
        feature_width_names=("N_s", "N_u"),
        primary_metrics=("sigma_hdiv_error", "u_h1_error"),
        supports_k_sweep=True,
    ),
    "elasticity-3d": ModelSpec(
        key="elasticity-3d",
        relative_path="linear-elasticity-3d/linear_elasticity_3d.py",
        width_fields=("N_s", "N_u"),
        feature_width_names=("N_s", "N_u"),
        primary_metrics=("sigma_hdiv_error", "u_h1_error"),
        supports_k_sweep=False,
    ),
    "plane-stress": ModelSpec(
        key="plane-stress",
        relative_path="plane-stress/plane_stress.py",
        width_fields=("N_s", "N_u"),
        feature_width_names=("N_s", "N_u"),
        primary_metrics=("sigma_hdiv_error", "u_h1_error"),
        supports_k_sweep=False,
    ),
    "plate": ModelSpec(
        key="plate",
        relative_path="plate-bending/plate_bending.py",
        width_fields=("N_m", "N_u"),
        feature_width_names=("N_m", "N_u"),
        primary_metrics=("M_hdivdiv_error", "w_h2_error"),
        supports_k_sweep=True,
    ),
}


def load_model(spec: ModelSpec) -> ModuleType:
    """Load one driver without relying on hyphenated directory package names."""

    path = ROOT / spec.relative_path
    module_spec = importlib.util.spec_from_file_location(
        f"least_squares_{spec.key.replace('-', '_')}",
        path,
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Cannot load model driver {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def benchmark_kwargs(
    spec: ModelSpec,
    cfg: Any,
    *,
    q_test: int,
    interior_seed: int,
    test_seed: int,
) -> dict[str, Any]:
    """Translate a model configuration into its benchmark-builder arguments."""

    kwargs: dict[str, Any] = {
        "E": cfg.E,
        "nu": cfg.nu,
        "Q_train": cfg.Q_train,
        "Q_test": q_test,
        "sampling_method": cfg.sampling_method,
        "interior_seed": interior_seed,
        # Every model records ``manufactured_solution``, so it must reach the
        # builder for every model: a configuration that names a solution the
        # run then ignores is indistinguishable in the results file from one
        # that honoured it.
        "manufactured_solution": cfg.manufactured_solution,
        "test_seed": test_seed,
    }
    if spec.key != "plate":
        kwargs["body_force_batch_size"] = cfg.body_force_batch_size
    else:
        kwargs["h"] = cfg.h
    return kwargs


def build_feature_space(
    module: ModuleType,
    spec: ModelSpec,
    cfg: Any,
    *,
    projection_seed: int,
) -> Any:
    kwargs = {
        spec.feature_width_names[0]: getattr(cfg, spec.width_fields[0]),
        spec.feature_width_names[1]: getattr(cfg, spec.width_fields[1]),
        "activation_power": cfg.activation_power,
        "ritz_degree": cfg.ritz_degree,
        "ritz_ratio": cfg.ritz_ratio,
        "projection_samples": cfg.projection_samples,
        "projection_seed": projection_seed,
    }
    if hasattr(cfg, "projection_batch_size"):
        kwargs["projection_batch_size"] = cfg.projection_batch_size
    return module.build_shared_feature_space(**kwargs)


def validation_score(result: Any, spec: ModelSpec) -> float:
    """Use independent manufactured-solution graph errors for budget selection."""

    values = [float(getattr(result, name)) for name in spec.primary_metrics]
    return math.sqrt(sum(value * value for value in values))


def finite_or_text(value: Any) -> Any:
    """Make non-finite floats portable in JSON and CSV."""

    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "nan"
        if math.isinf(float(value)):
            return "inf" if float(value) > 0 else "-inf"
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def run_configuration(
    module: ModuleType,
    spec: ModelSpec,
    cfg: Any,
    *,
    run_index: int,
    ladders: dict[str, tuple[float, ...] | None],
    validation_points: int,
) -> list[dict[str, Any]]:
    """Select each algorithm's hyperparameter on independent points, then test.

    Returns one record per configured algorithm.  All candidates and all
    algorithms share the same assembled system and its single spectral
    factorization, so the ladder sweeps only repeat cheap filtering.  The
    validation and test rules differ only in their quadrature points, and the
    residual system depends on the training points alone, so the same system
    and factorization carry over from selection to reporting.
    """

    seed = 100_000 * (run_index + 1)
    projection_seed = seed + 23
    feature_space = build_feature_space(
        module,
        spec,
        cfg,
        projection_seed=projection_seed,
    )
    parameter_diagnostics = parameter_set_diagnostics(
        feature_space.theta_m if spec.key == "plate" else feature_space.theta_s
    )
    train_seed = seed + 31
    validation_benchmark = module.build_shared_benchmark(
        **benchmark_kwargs(
            spec,
            cfg,
            q_test=validation_points,
            interior_seed=train_seed,
            test_seed=seed + 37,
        )
    )

    algorithm_ids = [get_solver_spec(name).id for name in cfg.algorithms_to_run]
    validation_data = module.prepare_experiment(cfg, validation_benchmark, feature_space)
    selections: dict[str, tuple[float, float, dict[str, Any]]] = {}
    for algorithm_id in algorithm_ids:
        solver = SOLVERS[algorithm_id]
        ladder = ladders.get(algorithm_id) or solver.default_ladder
        scores: dict[str, Any] = {}
        selected_hyper = ladder[0]
        selected_score = math.inf
        for hyper in ladder:
            candidate_cfg = replace(
                cfg,
                algorithms_to_run=[algorithm_id],
                **{solver.config_field: hyper},
            )
            candidate = module.run_experiment(
                candidate_cfg,
                print_table=False,
                plot_results=False,
                benchmark=validation_benchmark,
                feature_space=feature_space,
                experiment_data=validation_data,
            )[0]
            score = validation_score(candidate, spec)
            scores[str(finite_or_text(hyper))] = finite_or_text(score)
            if score < selected_score:
                selected_score = score
                selected_hyper = hyper
        selections[algorithm_id] = (selected_hyper, selected_score, scores)

    test_benchmark = module.build_shared_benchmark(
        **benchmark_kwargs(
            spec,
            cfg,
            q_test=cfg.Q_test,
            interior_seed=train_seed,
            test_seed=seed + 41,
        )
    )
    test_data = module.retarget_experiment_data(
        cfg,
        validation_data,
        test_benchmark,
        feature_space,
    )
    del validation_data, validation_benchmark
    projection = (
        feature_space.projected_w
        if spec.key == "plate"
        else feature_space.projected_u
    )
    records: list[dict[str, Any]] = []
    for algorithm_id in algorithm_ids:
        solver = SOLVERS[algorithm_id]
        selected_hyper, selected_score, scores = selections[algorithm_id]
        final_cfg = replace(
            cfg,
            algorithms_to_run=[algorithm_id],
            **{solver.config_field: selected_hyper},
        )
        result = module.run_experiment(
            final_cfg,
            print_table=False,
            plot_results=False,
            benchmark=test_benchmark,
            feature_space=feature_space,
            experiment_data=test_data,
        )[0]
        record = {
            "model": spec.key,
            "run": run_index,
            "N": int(getattr(cfg, spec.width_fields[0])),
            "Q": int(cfg.Q_train),
            "K": int(projection.space.dimension),
            "activation_power": int(cfg.activation_power),
            "ritz_degree": int(projection.degree),
            "ritz_ratio": float(cfg.ritz_ratio),
            "projection_samples": int(projection.quadrature_samples),
            "projection_gram_residual": float(projection.gram_residual),
            "boundary_residual": float(projection.boundary_residual()),
            "validation_score": float(selected_score),
            "hyperparameter_scores": scores,
            "parameter_covering_radius": float(parameter_diagnostics.covering_radius),
            "parameter_separation": float(parameter_diagnostics.separation),
            "projection_seed": projection_seed,
            "training_seed": train_seed,
            "system_backend": str(getattr(cfg, "system_backend", "direct")),
            "direct_solver": str(cfg.direct_solver),
            "direct_rcond": float(cfg.direct_rcond),
            # The power study replaces the plate's polynomial deflection,
            # so the target field is part of a record's provenance.
            "manufactured_solution": str(getattr(cfg, "manufactured_solution", "default")),
            "nu": float(cfg.nu),
            **asdict(result),
        }
        records.append({key: finite_or_text(value) for key, value in record.items()})
    del test_data
    return records


def numeric_columns(records: list[dict[str, Any]], group_fields: set[str]) -> list[str]:
    columns: list[str] = []
    for key in records[0]:
        if (
            key in group_fields
            or key.endswith("seed")
            or key in {"run", "hyperparameter"}
        ):
            continue
        values = [record.get(key) for record in records]
        if all(isinstance(value, (int, float, bool)) for value in values):
            columns.append(key)
    return columns


def aggregate(
    records: list[dict[str, Any]],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Aggregate numeric diagnostics as mean and sample standard deviation."""

    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = tuple(record[field] for field in group_fields)
        buckets.setdefault(key, []).append(record)
    metrics = numeric_columns(records, set(group_fields))
    summary: list[dict[str, Any]] = []
    for key, rows in buckets.items():
        item = dict(zip(group_fields, key))
        item["repeats"] = len(rows)
        item["hyperparameters"] = [row.get("hyperparameter") for row in rows]
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in rows], dtype=float)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(item)
    return summary


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    flattened = dict(record)
    for key, value in tuple(flattened.items()):
        if isinstance(value, dict):
            flattened[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return flattened


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    flattened = [flatten_record(row) for row in rows]
    fieldnames = list(dict.fromkeys(key for row in flattened for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def save_records(
    output_dir: Path,
    records: list[dict[str, Any]],
    summary: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"records": records, "summary": summary}
    (output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_csv(output_dir / "records.csv", records)
    write_csv(output_dir / "summary.csv", summary)


def parse_float_list(value: str) -> tuple[float, ...]:
    parsed: list[float] = []
    for item in value.split(","):
        token = item.strip().lower()
        parsed.append(math.inf if token in {"inf", "infinity"} else float(token))
    if not parsed:
        raise argparse.ArgumentTypeError("expected a nonempty comma-separated list")
    return tuple(parsed)


def parse_int_list(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return parsed


def parse_str_list(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected a nonempty comma-separated list")
    return parsed


def study_configurations(
    study: str,
    base_cfg: Any,
    spec: ModelSpec,
    *,
    widths: tuple[int, ...],
    q_ratio: int | None = None,
) -> Iterable[tuple[str, Any]]:
    """Yield one configuration per requested study bucket.

    ``q_ratio`` is an optional explicit training-sample ratio.  The historical
    defaults remain unchanged when it is omitted; exposing it lets large
    three-dimensional runs use a deliberately chosen MC density instead of the
    old hard-coded ``4(N+1)`` in both the order and power studies.
    """

    if q_ratio is not None and q_ratio <= 0:
        raise ValueError("q_ratio must be positive when provided")
    if study == "order":
        ratio = q_ratio or 4
        for width in widths:
            yield (
                f"N-{width}",
                replace(
                    base_cfg,
                    **{
                        spec.width_fields[0]: width,
                        spec.width_fields[1]: width,
                        "Q_train": ratio * (width + 1),
                    },
                ),
            )
        return
    if study == "q":
        width = 400
        for ratio in DEFAULT_Q_RATIOS:
            yield (
                f"Qratio-{ratio}",
                replace(
                    base_cfg,
                    **{
                        spec.width_fields[0]: width,
                        spec.width_fields[1]: width,
                        "Q_train": ratio * (width + 1),
                    },
                ),
            )
        return
    if study == "k":
        if not spec.supports_k_sweep:
            raise ValueError(f"K sweep is not prescribed for {spec.key}")
        width = 400
        for ratio in DEFAULT_K_RATIOS:
            yield (
                f"Kratio-{ratio:g}",
                replace(
                    base_cfg,
                    **{
                        spec.width_fields[0]: width,
                        spec.width_fields[1]: width,
                        "Q_train": 4 * (width + 1),
                        "ritz_ratio": ratio,
                    },
                ),
            )
        return
    if study == "power":
        # The prescribed default width is model-specific.  A single explicit
        # ``--widths`` value overrides it, which is useful for the large 3-D
        # power campaign where N is selected by a prior capacity probe.
        width = (
            POWER_STUDY_WIDTHS[spec.key]
            if widths == DEFAULT_WIDTHS
            else widths[0]
        )
        overrides = {
            spec.width_fields[0]: width,
            spec.width_fields[1]: width,
            "Q_train": (
                q_ratio
                if q_ratio is not None
                else POWER_STUDY_Q_RATIOS.get(spec.key, 4)
            )
            * (width + 1),
            # Hold one auxiliary space fixed across all powers.  Models with a
            # prescribed matched degree receive it when base_cfg is built.
            "ritz_degree": base_cfg.ritz_degree,
        }
        if spec.key in POWER_STUDY_SOLUTIONS:
            overrides["manufactured_solution"] = POWER_STUDY_SOLUTIONS[spec.key]
        for power in DEFAULT_ACTIVATION_POWERS:
            yield (
                f"k-{power}",
                replace(base_cfg, **overrides, activation_power=power),
            )
        return
    raise ValueError(f"Unknown study {study}")


def run_study(args: argparse.Namespace) -> None:
    spec = MODEL_SPECS[args.model]
    module = load_model(spec)
    base_cfg = (
        module.default_config()
        if hasattr(module, "default_config")
        else module.LeastSquaresConfig()
    )
    if args.study == "power":
        power_defaults: dict[str, Any] = {}
        if spec.key in POWER_STUDY_RITZ_DEGREES:
            power_defaults["ritz_degree"] = POWER_STUDY_RITZ_DEGREES[spec.key]
        if spec.key in POWER_STUDY_PROJECTION_SAMPLES:
            power_defaults["projection_samples"] = (
                POWER_STUDY_PROJECTION_SAMPLES[spec.key]
            )
        if power_defaults:
            base_cfg = replace(base_cfg, **power_defaults)
    if args.test_points is not None:
        base_cfg = replace(base_cfg, Q_test=args.test_points)
    if args.projection_samples is not None:
        base_cfg = replace(base_cfg, projection_samples=args.projection_samples)
    if args.projection_batch_size is not None:
        if not hasattr(base_cfg, "projection_batch_size"):
            raise ValueError("--projection-batch-size is unsupported by this model")
        base_cfg = replace(base_cfg, projection_batch_size=args.projection_batch_size)
    if args.system_backend is not None:
        base_cfg = replace(base_cfg, system_backend=args.system_backend)
    if args.direct_solver is not None:
        base_cfg = replace(base_cfg, direct_solver=args.direct_solver)
    if args.direct_batch_size is not None:
        base_cfg = replace(base_cfg, direct_batch_size=args.direct_batch_size)
    if args.direct_qr_block_size is not None:
        base_cfg = replace(base_cfg, direct_qr_block_size=args.direct_qr_block_size)
    if args.evaluation_batch_size is not None:
        base_cfg = replace(base_cfg, evaluation_batch_size=args.evaluation_batch_size)
    if args.body_force_batch_size is not None:
        base_cfg = replace(base_cfg, body_force_batch_size=args.body_force_batch_size)
    if args.direct_rcond is not None:
        base_cfg = replace(base_cfg, direct_rcond=args.direct_rcond)
    if args.ritz_degree is not None:
        base_cfg = replace(base_cfg, ritz_degree=args.ritz_degree)
    if args.ritz_ratio is not None:
        if args.study == "k":
            raise ValueError("--ritz-ratio conflicts with the k study, which sweeps it")
        base_cfg = replace(base_cfg, ritz_ratio=args.ritz_ratio)
    if args.activation_power is not None:
        base_cfg = replace(base_cfg, activation_power=args.activation_power)
    if args.manufactured_solution is not None:
        base_cfg = replace(
            base_cfg,
            manufactured_solution=args.manufactured_solution,
        )
    if args.algorithms is not None:
        base_cfg = replace(
            base_cfg,
            algorithms_to_run=[get_solver_spec(name).id for name in args.algorithms],
        )

    base_ladders: dict[str, tuple[float, ...] | None] = {
        "ball": args.budgets,
        "ridge": args.ridge_lambdas,
        "tsvd": args.tsvd_rconds,
    }
    records: list[dict[str, Any]] = []
    configurations = list(
        study_configurations(
            args.study,
            base_cfg,
            spec,
            widths=args.widths,
            q_ratio=args.q_ratio,
        )
    )
    if args.activation_powers is not None:
        if args.study != "power":
            raise ValueError("--activation-powers is only valid for the power study")
        requested_powers = set(args.activation_powers)
        configurations = [
            (label, cfg)
            for label, cfg in configurations
            if cfg.activation_power in requested_powers
        ]
        found_powers = {cfg.activation_power for _, cfg in configurations}
        missing_powers = requested_powers - found_powers
        if missing_powers:
            raise ValueError(
                "activation powers are not prescribed by this study: "
                f"{sorted(missing_powers)}"
            )
    # ``run_offset`` lets independent worker processes execute disjoint
    # repetitions concurrently.  The default remains zero, preserving the
    # original serial campaign and its deterministic seeds.
    width_overrides: dict[str, int] = {}
    if args.stress_width is not None:
        width_overrides[spec.width_fields[0]] = args.stress_width
    if args.displacement_width is not None:
        width_overrides[spec.width_fields[1]] = args.displacement_width
    if width_overrides:
        configurations = [
            (label, replace(cfg, **width_overrides)) for label, cfg in configurations
        ]

    for label, cfg in configurations:
        for local_run_index in range(args.repeats):
            run_index = args.run_offset + local_run_index
            print(
                f"\n=== study={args.study}, model={spec.key}, "
                f"config={label}, run={run_index + 1} "
                f"(worker offset={args.run_offset}, local {local_run_index + 1}/{args.repeats}) ==="
            )
            ladders = dict(base_ladders)
            new_records = run_configuration(
                module,
                spec,
                cfg,
                run_index=run_index,
                ladders=ladders,
                validation_points=(
                    args.validation_points
                    if args.validation_points is not None
                    else (16**3 if spec.key == "elasticity-3d" else 32**2)
                ),
            )
            for record in new_records:
                record["study"] = args.study
                record["configuration"] = label
            records.extend(new_records)

    # Fields that identify a bucket rather than measure it.  activation_power
    # and ritz_degree belong here even though they are integers: aggregating
    # them into a mean would hide them from every consumer that keys a summary
    # row back to its dictionary (plot_convergence's reference rates, the
    # floor comparison in approximation-floors/run_solver_gap.py).
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
        "nu",
    )
    summary = aggregate(records, group_fields)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else ROOT / "results" / spec.key / (args.output_name or args.study)
    )
    save_records(output_dir, records, summary)
    print(f"Saved raw records and mean ± std summaries to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("order", "q", "k", "power"))
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument(
        "--output-name",
        default=None,
        help="results subdirectory name; defaults to the study name. Use it to "
             "keep one study's runs at different activation powers apart.",
    )
    parser.add_argument(
        "--activation-power",
        type=int,
        default=None,
        help="override the driver's default activation power for every configuration",
    )
    parser.add_argument(
        "--manufactured-solution",
        default=None,
        help="override the driver's manufactured solution for every configuration",
    )
    parser.add_argument(
        "--activation-powers",
        type=parse_int_list,
        default=None,
        help="comma-separated subset of the prescribed power-study powers; "
             "only valid with study=power",
    )
    parser.add_argument(
        "--ritz-degree",
        type=int,
        default=None,
        help="override the auxiliary B-spline degree for every configuration",
    )
    parser.add_argument(
        "--stress-width",
        type=int,
        default=None,
        help="override the first width field (N_s / N_m) for every configuration, "
             "leaving the second at the value the study prescribes; the paper's "
             "runs keep the two equal",
    )
    parser.add_argument(
        "--displacement-width",
        type=int,
        default=None,
        help="override the second width field (N_u) for every configuration",
    )
    parser.add_argument(
        "--ritz-ratio",
        type=float,
        default=None,
        help="override the auxiliary space ratio K/(N+1) for every configuration; "
             "the k study sweeps it, but every other study keeps the model default",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--run-offset",
        type=int,
        default=0,
        help=(
            "zero-based repetition index offset; useful when several workers "
            "run disjoint repeats concurrently (default: 0)"
        ),
    )
    parser.add_argument("--widths", type=parse_int_list, default=DEFAULT_WIDTHS)
    parser.add_argument(
        "--q-ratio",
        type=int,
        default=None,
        help=(
            "power-study MC training ratio Q/(N+1); model default is "
            "16 for elasticity-2d, 8 for plate, and 4 otherwise"
        ),
    )
    parser.add_argument(
        "--algorithms",
        type=parse_str_list,
        help=f"comma-separated algorithm ids (default: model config; valid: {list(SOLVERS)})",
    )
    parser.add_argument(
        "--budgets",
        type=parse_float_list,
        help=f"ball budget ladder (default: {SOLVERS['ball'].default_ladder})",
    )
    parser.add_argument(
        "--ridge-lambdas",
        type=parse_float_list,
        help=f"ridge relative-lambda ladder (default: {SOLVERS['ridge'].default_ladder})",
    )
    parser.add_argument(
        "--tsvd-rconds",
        type=parse_float_list,
        help=f"tsvd rcond ladder (default: {SOLVERS['tsvd'].default_ladder})",
    )
    parser.add_argument("--validation-points", type=int)
    parser.add_argument("--test-points", type=int)
    parser.add_argument("--projection-samples", type=int)
    parser.add_argument(
        "--projection-batch-size",
        type=int,
        default=None,
        help="batch size for the Monte Carlo Ritz projection (elasticity models)",
    )
    parser.add_argument(
        "--system-backend",
        choices=VALID_SYSTEM_BACKENDS,
        help="least-squares system backend (default: model config)",
    )
    parser.add_argument(
        "--direct-solver",
        choices=VALID_DIRECT_SOLVERS,
        help="direct backend assembly/compression method (default: model config)",
    )
    parser.add_argument(
        "--direct-batch-size",
        type=int,
        default=None,
        help="number of training points per streaming-TSQR block (default: model config)",
    )
    parser.add_argument(
        "--direct-qr-block-size",
        type=int,
        default=None,
        help="TSQR panel block size passed to DTPQRT (default: model config)",
    )
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=None,
        help="test quadrature block size used during error evaluation (default: model config)",
    )
    parser.add_argument(
        "--body-force-batch-size",
        type=int,
        default=None,
        help="batch size for manufactured body-force autodiff (default: model config)",
    )
    parser.add_argument(
        "--direct-rcond",
        type=float,
        default=None,
        help="relative truncation level of the pseudo-inverse used by the ball "
             "solve (default: model config)",
    )
    parser.add_argument("--output-dir")
    return parser


def main() -> None:
    # The rented host has 32 allocated CPUs.  When several independent widths
    # run concurrently, PyTorch otherwise keeps its process-wide default
    # thread pool (typically all visible cores), causing severe oversubscription
    # even though OPENBLAS_NUM_THREADS is capped.  An explicit opt-in keeps
    # single-process runs fast while making the campaign's parallel groups
    # predictable.
    thread_token = os.environ.get("LS_TORCH_THREADS")
    if thread_token:
        try:
            thread_count = int(thread_token)
        except ValueError as exc:
            raise ValueError("LS_TORCH_THREADS must be a positive integer") from exc
        if thread_count < 1:
            raise ValueError("LS_TORCH_THREADS must be a positive integer")
        torch.set_num_threads(thread_count)
        torch.set_num_interop_threads(1)
    args = build_parser().parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.run_offset < 0:
        raise ValueError("--run-offset must be nonnegative")
    run_study(args)


if __name__ == "__main__":
    main()
