# Pure-random direct residual least squares

This branch keeps the paper's feature family unchanged: each scalar space uses
one constant basis function and `N` fixed random `tanh` features. No polynomial,
Fourier, learned, or problem-specific features are added.

The only solver-side changes are:

1. assemble the weighted residual design matrix `A` and right-hand side `b`;
2. scale every column of `A` to unit Euclidean norm;
3. solve `A z ~= b` with the rank-revealing LAPACK `GELSD` driver and
   `rcond = 1e-14`, rather than solving the normal equations.

Small consistency checks give relative errors of about `2e-16` to `4e-16` for
`A.T @ A` versus the legacy Gram matrix, and `2e-16` to `6e-16` for
`A.T @ b` versus the legacy load vector.

## Experiment setting

The explicit direct matrix is memory intensive, so these diagnostics use
Gauss-Legendre quadrature with `Q_train = 4096`. The 2D and plate tests use
`Q_test = 16384`; 3D uses `Q_test = 8000`. These settings differ from the
paper's original Sobol/Gram tables and are therefore reported as a separate
solver diagnostic rather than silently replacing the original data.

The tuned pure-random shape parameters are:

- 2D elasticity and plane stress: `gamma = 3`;
- plate bending: `gamma = 2`;
- 3D elasticity: `gamma = 2`.

All random seeds remain the same as in the paper. Runs at different `N` use a
fixed seed, but the generated bias arrays are not strict prefixes of one common
`N = 1000` dictionary; the curves should not be described as nested-space
convergence.

## Reproduction

```bash
python experiments/least-squares/linear-elasticity-2d/direct_qr.py \
  --N 1000 --q-train 4096 --q-test 16384

python experiments/least-squares/plane-stress/direct_qr.py \
  --N 1000 --q-train 4096 --q-test 16384

python experiments/least-squares/plate-bending/direct_qr.py \
  --N 1000 --q-train 4096 --q-test 16384 --driver gelsd --rcond 1e-14

python experiments/least-squares/linear-elasticity-3d/direct_qr.py \
  --N 400 --q-train 4096 --q-test 8000 --manufactured-solution hu_zhang
```

The complete measured values are stored in `direct-qr-results.csv`.

## Result summary

- 2D elasticity reaches `u = 2.12e-13`, `sigma = 3.42e-11` at `N = 1000`.
- Plane stress reaches `u = 1.63e-13`, `sigma = 2.98e-11` at `N = 1000`.
- Plate bending reaches `u = 2.37e-11`, but the moment error remains
  `4.92e-9` at `N = 1000`.
- 3D Hu-Zhang reaches `u = 2.22e-5`, `sigma = 1.51e-3` at `N = 400`.

Thus direct factorization removes much of the normal-equation accuracy loss,
but pure random `tanh` features do not uniformly reach `1e-10` within the
tested budget. The remaining limitation is the approximation space, especially
for plate moments and 3D stresses.

At `N = 1000`, the explicit 3D matrix would require roughly 2.5 GiB before
assembly temporaries and LAPACK workspace. It is not safe on the current
7.6-GiB machine; a matrix-free iterative solver or streaming factorization is
needed for a full 3D direct scan.
