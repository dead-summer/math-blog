"""Measure best-approximation floors for one model across activation powers.

Usage::

    python run_floors.py --model elasticity-2d --powers 3,5,7,9 --widths 250,500,1000
    python run_floors.py --model plate --powers 3,7 --degrees 3,7

Writes ``results/floors-<model>.csv`` and ``.json``.  Each row records the
floors of one ``(power, width, degree)`` configuration together with the
theoretical rate ``beta = (s_cap(d) - m)/d`` that the fitted order is compared
against.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from floors import (
    approximation_rate,
    field_norms,
    fitted_order,
    load_benchmark,
    moment_hdivdiv_floor,
    projected_dictionary,
    quasi_uniform_features,
    raw_dictionary,
    ritz_projection,
    saturation_index,
    scalar_floor,
    spline_space,
    stress_hdiv_floor,
    tensor_l2_floor,
    TensorSplineSpace,
)

RESULTS = Path(__file__).resolve().parent / "results"


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def measure(model: str, power: int, width: int, degree: int, ritz_ratio: float,
            q_test: int | None, solution: str | None = None) -> dict:
    """All floors for one configuration of one model."""

    started = time.perf_counter()
    benchmark = load_benchmark(model, width, q_test, solution)
    scalar_scale, tensor_scale = field_norms(benchmark)
    dimension = benchmark.dimension
    order = benchmark.sobolev_order
    parameters = quasi_uniform_features(width, dimension, power=power)
    feature_count = width + 1
    evaluate_raw = raw_dictionary(parameters, power, dimension)

    common = dict(
        points=benchmark.points,
        weights=benchmark.weights,
        values=benchmark.scalar_values,
        gradients=benchmark.scalar_gradients,
        hessians=benchmark.scalar_hessians,
    )
    # Pure L2 fit: the superconvergent baseline the graph norms are compared to.
    raw_l2 = scalar_floor(evaluate_raw, feature_count, order=0, **common)
    # Fit in the graph norm's own order: what the method actually minimizes.
    raw_graph = scalar_floor(evaluate_raw, feature_count, order=order, **common)

    space = TensorSplineSpace.with_minimum_dimension(
        dimension,
        order,
        max(width + 1, int(ritz_ratio * (width + 1))),
        degree=degree,
    )
    spline_only = scalar_floor(spline_space(space), space.dimension, order=order, **common)

    projected = ritz_projection(parameters, order, power, degree, ritz_ratio, width)
    projected_floor = scalar_floor(
        projected_dictionary(projected), feature_count, order=order, **common
    )

    if model == "plate":
        tensor = moment_hdivdiv_floor(
            evaluate_raw,
            feature_count,
            benchmark.points,
            benchmark.weights,
            benchmark.tensor_values,
            benchmark.load,
        )
    else:
        tensor = stress_hdiv_floor(
            evaluate_raw,
            feature_count,
            benchmark.pairs,
            benchmark.voigt_weight,
            benchmark.points,
            benchmark.weights,
            benchmark.tensor_values,
            benchmark.load,
        )
    # Measured separately because ``tensor.l2`` is a component of the
    # graph-optimal fit rather than the L2 optimum.
    tensor_l2 = tensor_l2_floor(
        evaluate_raw,
        feature_count,
        benchmark.voigt_weight,
        benchmark.points,
        benchmark.weights,
        benchmark.tensor_values,
    )

    scalar_graph = raw_graph.h2 if order == 2 else raw_graph.h1
    projected_graph = projected_floor.h2 if order == 2 else projected_floor.h1
    spline_graph = spline_only.h2 if order == 2 else spline_only.h1
    return {
        "model": model,
        "solution": benchmark.solution,
        "dimension": dimension,
        "sobolev_order": order,
        "activation_power": power,
        "N": width,
        "K": space.dimension,
        "ritz_degree": degree,
        "ritz_ratio": ritz_ratio,
        "saturation_index": saturation_index(dimension, power),
        "beta_scalar_l2": approximation_rate(dimension, power, 0),
        "beta_scalar_graph": approximation_rate(dimension, power, order),
        "scalar_raw_l2": raw_l2.l2,
        "scalar_raw_graph": scalar_graph,
        "scalar_spline_graph": spline_graph,
        "scalar_projected_graph": projected_graph,
        # The trial space the driver actually searches is the projected one, so
        # its floor is the largest of the dictionary and spline caps.
        "scalar_attainable_graph": max(projected_graph, spline_graph),
        "tensor_raw_l2": tensor.l2,
        "tensor_l2_floor": tensor_l2,
        "tensor_raw_divergence": tensor.divergence,
        "tensor_raw_graph": tensor.graph,
        "scalar_exact_norm": scalar_scale,
        "tensor_exact_norm": tensor_scale,
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        choices=("elasticity-2d", "elasticity-3d", "plane-stress", "plate"))
    parser.add_argument("--powers", type=parse_ints, default=(3, 5, 7, 9))
    parser.add_argument("--widths", type=parse_ints, default=(250, 500, 1000))
    parser.add_argument("--degrees", type=parse_ints, default=(7,))
    parser.add_argument("--ritz-ratio", type=float, default=2.0)
    parser.add_argument("--q-test", type=int, default=None)
    parser.add_argument(
        "--solution",
        default=None,
        help="manufactured solution name; defaults to the driver's own default",
    )
    parser.add_argument("--suffix", default="", help="tag appended to the output stem")
    parser.add_argument("--threads", type=int, default=28)
    parser.add_argument(
        "--refit",
        action="store_true",
        help="recompute the fitted-order columns from an existing results file",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    rows: list[dict] = []
    if args.refit:
        rows = json.loads(
            (RESULTS / f"floors-{args.model}{args.suffix}.json").read_text(encoding="utf-8")
        )
        args.degrees = tuple(sorted({int(row["ritz_degree"]) for row in rows}))
        args.powers = tuple(sorted({int(row["activation_power"]) for row in rows}))
        if "scalar_exact_norm" not in rows[0]:
            # Older results carry no exact-field reference norms; recover them
            # from the benchmark to keep the output schema current.
            cache: dict[int, tuple[float, float]] = {}
            for row in rows:
                width = int(row["N"])
                if width not in cache:
                    cache[width] = field_norms(
                        load_benchmark(args.model, width, None, args.solution)
                    )
                row["scalar_exact_norm"], row["tensor_exact_norm"] = cache[width]
    for degree in args.degrees if not args.refit else ():
        for power in args.powers:
            for width in args.widths:
                row = measure(args.model, power, width, degree,
                              args.ritz_ratio, args.q_test, args.solution)
                rows.append(row)
                print(
                    f"[{args.model}] k={power} N={width} degree={degree}: "
                    f"scalar L2={row['scalar_raw_l2']:.3e} "
                    f"graph={row['scalar_raw_graph']:.3e} "
                    f"projected={row['scalar_projected_graph']:.3e} "
                    f"spline={row['scalar_spline_graph']:.3e} | "
                    f"tensor L2={row['tensor_l2_floor']:.3e} "
                    f"graph={row['tensor_raw_graph']:.3e} "
                    f"({row['seconds']:.0f}s)",
                    flush=True,
                )

    # Fitted orders per (degree, power) family, appended as extra columns.
    for degree in args.degrees:
        for power in args.powers:
            family = [r for r in rows
                      if r["ritz_degree"] == degree and r["activation_power"] == power]
            widths = [r["N"] for r in family]
            for source, target in (
                ("scalar_raw_l2", "order_scalar_l2"),
                ("scalar_raw_graph", "order_scalar_graph"),
                ("tensor_l2_floor", "order_tensor_l2"),
                ("tensor_raw_graph", "order_tensor_graph"),
            ):
                value = fitted_order(
                    widths,
                    [r[source] for r in family],
                )
                for row in family:
                    row[target] = value

    RESULTS.mkdir(exist_ok=True)
    stem = RESULTS / f"floors-{args.model}{args.suffix}"
    stem.with_suffix(".json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {stem}.csv and {stem}.json")

    print("\nfitted order vs theory (raw dictionary):")
    print(f"  {'k':>2} {'s_cap':>6} {'beta_L2':>8} {'fit_L2':>8} "
          f"{'beta_graph':>11} {'fit_graph':>10} {'fit_tenL2':>10} {'fit_tensor':>11}")
    seen = set()
    for row in rows:
        key = (row["ritz_degree"], row["activation_power"])
        if key in seen:
            continue
        seen.add(key)
        fmt = lambda value: "     nan" if value != value else f"{value:>8.2f}"
        print(f"  {int(row['activation_power']):>2} {row['saturation_index']:>6.1f} "
              f"{row['beta_scalar_l2']:>8.2f} {fmt(row['order_scalar_l2'])} "
              f"{row['beta_scalar_graph']:>11.2f} {fmt(row['order_scalar_graph'])[-8:]:>10} "
              f"{fmt(row['order_tensor_l2'])[-8:]:>10} "
              f"{fmt(row['order_tensor_graph'])[-8:]:>11}")


if __name__ == "__main__":
    main()
