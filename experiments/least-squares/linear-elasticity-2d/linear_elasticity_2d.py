"""Two-dimensional linear elasticity driver.

Everything problem-specific lives here: the manufactured displacement fields
and the default configuration.  The numerics are shared with the other
stress--displacement experiments through ``elasticity_common``.
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
OUTPUT_DIR = PROJECT_ROOT / "public" / "images" / "least-squares" / "linear-elasticity-2d"


def hu_zhang_displacement(material: ec.Material) -> ec.DisplacementFn:
    """The Hu-Zhang 2D manufactured displacement field."""

    def displacement(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[:, 0], x[:, 1]
        u1 = torch.exp(x1 - x2) * x1 * (1.0 - x1) * x2 * (1.0 - x2)
        u2 = torch.sin(math.pi * x1) * torch.sin(math.pi * x2)
        return torch.stack([u1, u2], dim=1)

    return displacement


def div_free_displacement(material: ec.Material) -> ec.DisplacementFn:
    """A divergence-free manufactured displacement with zero boundary trace."""

    def displacement(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[:, 0], x[:, 1]
        pi = math.pi
        h1 = torch.sin(2.0 * pi * x1)
        h2 = torch.sin(2.0 * pi * x2)
        g1 = (1.0 - torch.cos(2.0 * pi * x1)) / (2.0 * pi)
        g2 = (1.0 - torch.cos(2.0 * pi * x2)) / (2.0 * pi)
        return torch.stack([-g1 * h2, h1 * g2], dim=1)

    return displacement


def near_incompressible_displacement(material: ec.Material) -> ec.DisplacementFn:
    """A lambda-scaled nearly incompressible zero-boundary displacement."""

    lam = material.lam
    if not math.isfinite(lam) or lam == 0.0:
        raise ValueError("lam must be finite and nonzero for near_incompressible.")
    div_free = div_free_displacement(material)

    def displacement(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[:, 0], x[:, 1]
        pi = math.pi
        perturbation = torch.stack(
            [
                torch.sin(2.0 * pi * x1) * x2 * (1.0 - x2),
                torch.sin(2.0 * pi * x2) * x1 * (1.0 - x1),
            ],
            dim=1,
        )
        return div_free(x) + perturbation / lam

    return displacement


PROBLEM = ec.ElasticityProblem(
    name="linear-elasticity-2d",
    spec=ec.make_voigt_spec(2),
    material_law=ec.isotropic_material,
    solutions={
        "hu_zhang": hu_zhang_displacement,
        "div_free": div_free_displacement,
        "near_incompressible": near_incompressible_displacement,
    },
    default_solution="hu_zhang",
    use_trace_constraint=True,
    output_dir=OUTPUT_DIR,
)


@dataclass
class LeastSquaresConfig(ec.LeastSquaresConfig):
    """2D linear elasticity defaults (inherits the shared field set)."""


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
