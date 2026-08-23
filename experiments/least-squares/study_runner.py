"""Reproducible repeated studies for the projected least-squares models.

Examples
--------
Run the ten-repeat width study for two-dimensional elasticity::

    python study_runner.py main --model elasticity-2d

Run the prescribed training-sample and Ritz-dimension ablations::

    python study_runner.py q --model elasticity-2d
    python study_runner.py k --model plate

The hidden-layer parameters are a deterministic quasi-uniform point set, so
repeats only randomize the training samples and the projection quadrature.
Each configured algorithm (ball / ridge / tsvd) selects its hyperparameter
on an independent validation rule before the final test evaluation; all
candidates share one spectral factorization of the assembled system.  The
runner writes raw records and mean/sample-standard-deviation summaries in both
JSON and CSV.
Full default studies are intentionally expensive.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ls_common import VALID_DIRECT_SOLVERS  # noqa: E402
from rfm_core import parameter_set_diagnostics  # noqa: E402
from solvers import SOLVERS, get_solver_spec  # noqa: E402
from system_backends import VALID_SYSTEM_BACKENDS  # noqa: E402

DEFAULT_WIDTHS = (200, 400, 600, 800, 1000)
DEFAULT_Q_RATIOS = (1, 2, 4, 8, 16, 32)
DEFAULT_K_RATIOS = (1.0, 2.0, 4.0)
DEFAULT_NU_VALUES = (0.49, 0.499, 0.4999, 0.49999, 0.499999)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    relative_path: str
    width_fields: tuple[str, str]
    feature_width_names: tuple[str, str]
    primary_metrics: tuple[str, str]
    supports_k_ablation: bool


MODEL_SPECS = {
    "elasticity-2d": ModelSpec(
        key="elasticity-2d",
        relative_path="linear-elasticity-2d/linear_elasticity_2d.py",
        width_fields=("N_s", "N_u"),
        feature_width_names=("N_s", "N_u"),
        primary_metrics=("sigma_hdiv_error", "u_h1_error"),
        supports_k_ablation=True,
    ),
    "elasticity-3d": ModelSpec(
        key="elasticity-3d",
        relative_path="linear-elasticity-3d/linear_elasticity_3d.py",
        width_fields=("N_s", "N_u"),
        feature_width_names=("N_s", "N_u"),
        primary_metrics=("sigma_hdiv_error", "u_h1_error"),
        supports_k_ablation=False,
    ),
    "plane-stress": ModelSpec(
        key="plane-stress",
        relative_path="plane-stress/plane_stress.py",
        width_fields=("N_s", "N_u"),
        feature_width_names=("N_s", "N_u"),
        primary_metrics=("sigma_hdiv_error", "u_h1_error"),
        supports_k_ablation=False,
    ),
    "plate": ModelSpec(
        key="plate",
        relative_path="plate-bending/plate_bending.py",
        width_fields=("N_m", "N_u"),
        feature_width_names=("N_m", "N_u"),
        primary_metrics=("M_hdivdiv_error", "w_h2_error"),
        supports_k_ablation=True,
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
        "test_seed": test_seed,
    }
    if spec.key in {"elasticity-2d", "elasticity-3d"}:
        kwargs.update(
            body_force_batch_size=cfg.body_force_batch_size,
            manufactured_solution=cfg.manufactured_solution,
        )
    elif spec.key == "plane-stress":
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
        "ritz_ratio": cfg.ritz_ratio,
        "projection_samples": cfg.projection_samples,
        "projection_seed": projection_seed,
    }
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
    factorization, so the ladder sweeps only repeat cheap filtering.
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
    del validation_data

    test_benchmark = module.build_shared_benchmark(
        **benchmark_kwargs(
            spec,
            cfg,
            q_test=cfg.Q_test,
            interior_seed=train_seed,
            test_seed=seed + 41,
        )
    )
    test_data = module.prepare_experiment(cfg, test_benchmark, feature_space)
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
) -> Iterable[tuple[str, Any]]:
    if study == "main":
        for width in widths:
            yield (
                f"N-{width}",
                replace(
                    base_cfg,
                    **{
                        spec.width_fields[0]: width,
                        spec.width_fields[1]: width,
                        "Q_train": 4 * (width + 1),
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
        if not spec.supports_k_ablation:
            raise ValueError(f"K ablation is not prescribed for {spec.key}")
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
    if study == "nu":
        if spec.key != "elasticity-3d":
            raise ValueError("The near-incompressible nu scan is defined for elasticity-3d")
        for nu in DEFAULT_NU_VALUES:
            yield (f"nu-{nu:.6g}", replace(base_cfg, nu=nu))
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
    if args.test_points is not None:
        base_cfg = replace(base_cfg, Q_test=args.test_points)
    if args.projection_samples is not None:
        base_cfg = replace(base_cfg, projection_samples=args.projection_samples)
    if args.system_backend is not None:
        base_cfg = replace(base_cfg, system_backend=args.system_backend)
    if args.direct_solver is not None:
        base_cfg = replace(base_cfg, direct_solver=args.direct_solver)
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
    shared_nu_ladders: dict[str, tuple[float, ...]] | None = None
    configurations = list(
        study_configurations(args.study, base_cfg, spec, widths=args.widths)
    )
    for label, cfg in configurations:
        for run_index in range(args.repeats):
            print(
                f"\n=== study={args.study}, model={spec.key}, "
                f"config={label}, run={run_index + 1}/{args.repeats} ==="
            )
            ladders = dict(base_ladders)
            if args.study == "nu" and shared_nu_ladders is not None:
                ladders.update(shared_nu_ladders)
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
            if args.study == "nu" and shared_nu_ladders is None:
                shared_nu_ladders = {}
                for record in new_records:
                    value = record["hyperparameter"]
                    shared_nu_ladders[record["algorithm"]] = (
                        math.inf if value == "inf" else float(value),
                    )

    group_fields = (
        "study",
        "model",
        "configuration",
        "algorithm",
        "system_backend",
        "direct_solver",
        "N",
        "Q",
        "K",
        "nu",
    )
    summary = aggregate(records, group_fields)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else ROOT / "results" / spec.key / args.study
    )
    save_records(output_dir, records, summary)
    print(f"Saved raw records and mean ± std summaries to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("main", "q", "k", "nu"))
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--widths", type=parse_int_list, default=DEFAULT_WIDTHS)
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
        "--system-backend",
        choices=VALID_SYSTEM_BACKENDS,
        help="least-squares system backend (default: model config)",
    )
    parser.add_argument(
        "--direct-solver",
        choices=VALID_DIRECT_SOLVERS,
        help="direct backend assembly/compression method (default: model config)",
    )
    parser.add_argument("--output-dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    run_study(args)


if __name__ == "__main__":
    main()
