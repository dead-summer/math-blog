"""Plane-stress driver.

Plane stress is two-dimensional elasticity with the plane-stress compliance
law and no trace gauge; only the manufactured displacement and defaults are
specific to this experiment.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import elasticity_common as ec  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "public" / "images" / "least-squares" / "plane-stress"


def default_displacement(material: ec.Material) -> ec.DisplacementFn:
    """The manufactured in-plane displacement field."""

    def displacement(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[:, 0], x[:, 1]
        boundary_factor = x1 * (1.0 - x1) * x2 * (1.0 - x2)
        u1 = torch.exp(x1 - x2) * boundary_factor
        u2 = torch.sin(math.pi * x1) * torch.sin(math.pi * x2)
        return torch.stack([u1, u2], dim=1)

    return displacement


PROBLEM = ec.ElasticityProblem(
    name="plane-stress",
    spec=ec.make_voigt_spec(2),
    material_law=ec.plane_stress_material,
    solutions={"default": default_displacement},
    default_solution="default",
    use_trace_constraint=False,
    output_dir=OUTPUT_DIR,
    nu_upper_inclusive=True,
)


@dataclass
class LeastSquaresConfig(ec.LeastSquaresConfig):
    """Plane-stress defaults."""

    E: float = 1.5
    nu: float = 0.5
    manufactured_solution: str = "default"


def default_config() -> LeastSquaresConfig:
    """Load the folder's ``defaults.json`` on top of the dataclass defaults."""

    return ec.load_config_defaults(LeastSquaresConfig, Path(__file__).resolve().parent)


def build_shared_benchmark(**kwargs) -> ec.SharedBenchmarkData:
    return ec.build_shared_benchmark(PROBLEM, **kwargs)


def build_shared_feature_space(**kwargs) -> ec.SharedFeatureSpace:
    return ec.build_shared_feature_space(PROBLEM, **kwargs)


def prepare_experiment(
    cfg: LeastSquaresConfig,
    benchmark: ec.SharedBenchmarkData,
    feature_space: ec.SharedFeatureSpace,
) -> ec.LeastSquaresExperimentData:
    return ec.prepare_experiment(PROBLEM, cfg, benchmark, feature_space)


def run_experiment(
    cfg: LeastSquaresConfig | None = None,
    print_table: bool = True,
    plot_results: bool = True,
    benchmark: ec.SharedBenchmarkData | None = None,
    feature_space: ec.SharedFeatureSpace | None = None,
    experiment_data: ec.LeastSquaresExperimentData | None = None,
) -> list[ec.AlgorithmResult]:
    return ec.run_experiment(
        PROBLEM,
        default_config() if cfg is None else cfg,
        print_table=print_table,
        plot_results=plot_results,
        benchmark=benchmark,
        feature_space=feature_space,
        experiment_data=experiment_data,
    )


def main(cfg: LeastSquaresConfig | None = None) -> None:
    """Script entrypoint."""

    run_experiment(cfg, print_table=True, plot_results=True)


if __name__ == "__main__":
    main()
