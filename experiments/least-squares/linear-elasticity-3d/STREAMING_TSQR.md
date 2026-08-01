# 3D direct least squares with streaming TSQR

## Why the dense solve runs out of memory

For the active 3D residual design,

\[
m=9Q,\qquad n=6N_s+3N_u+8.
\]

At `N_s=N_u=1000` and `Q=32768`, the dense matrix has shape
`(294912, 9008)` and occupies 19.79 GiB before feature temporaries and
DGELSD workspace are counted.

The `streaming_tsqr` backend never materializes this matrix. It performs:

1. one batched pass to compute the global stress-basis mean;
2. batched assembly of an augmented block `C_i = [A_i, b_i]`;
3. an in-place LAPACK `DTPQRT` update of the augmented triangular factor;
4. the existing column scaling and DGELSD solve on the reduced factor.

This is Householder QR, not a normal-equation method. In exact arithmetic,

\[
\|Ax-b\|_2^2=\|Rx-c\|_2^2+\beta^2,
\qquad \|A_{:,j}\|_2=\|R_{:,j}\|_2,
\]

so the reduced solve preserves the dense backend's column scaling, `rcond`,
numerical rank, and scaled-coordinate minimum-norm semantics. No singular
directions are truncated during TSQR.

## Recommended configuration

```python
LeastSquaresConfig(
    Q_train=32**3,
    direct_solver="streaming_tsqr",
    direct_batch_size=1024,
    direct_qr_block_size=64,
)
```

Use `direct_solver="dense"` only for small regression problems. The streaming
peak storage is approximately `O(9 * batch_size * n + n**2)`, rather than
`O(9 * Q * n)`. Reducing `direct_batch_size` lowers peak memory at the cost of
more QR updates.

## Measured Q comparison

The following runs use `N_s=N_u=1000`, 8,000 Gauss test points, float64,
`rcond=1e-14`, batch size 1,024, and QR block size 64 on the development
workstation.

| Q | Dense A | TSQR compression | Reduced solve | Solver total | Peak RSS | displacement L2 | stress L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,728 | 1.04 GiB | 16.29 s | 139.38 s | 155.67 s | 1.82 GiB | 7.70e-6 | 7.34e-5 |
| 4,096 | 2.47 GiB | 39.62 s | 139.59 s | 179.21 s | 1.94 GiB | 2.30e-7 | 2.31e-5 |
| 32,768 | 19.79 GiB | 317.20 s | 139.27 s | 456.47 s | 1.97 GiB | 2.30e-7 | 2.30e-5 |

`Q=1728` is under-resolved for `N=1000`. Increasing to `Q=4096` removes the
observed error increase. For this feature space and manufactured solution,
`Q=4096` and `Q=32768` are already at essentially the same test-error plateau.
Thus `Q=4096` is the practical setting for repeated ablations, while
`Q=32768` is useful as the final sampling-independent reference.

A useful diagnostic is the residual-row oversampling ratio

\[
\rho=\frac{9Q}{6N_s+3N_u+8}.
\]

For `N_s=N_u=1000`, it is only 1.73 at `Q=1728`, but 4.09 at `Q=4096`.
For repeated studies, choosing the next valid tensor-product cube with
`rho >= 4` is a reasonable empirical rule; it should still be checked when
the feature bandwidth, PDE coefficients, or sampling rule changes.

The timings also show that feature assembly is no longer the bottleneck. At
`Q=32768`, batch assembly takes 9.48 s, QR updates take 307.50 s, and the
reduced SVD takes about 139 s.

## Relation to arXiv:2409.15818

The sketching/matrix-free ideas in [arXiv:2409.15818](https://arxiv.org/abs/2409.15818)
are promising for a faster second backend, especially as a CountSketch-based
preconditioner for LSQR/LSMR. They do not directly replace this high-accuracy
reference solve: explicitly materializing the sketched construction can still
retain a dense matrix, while the fully implicit iterative variant introduces
stopping-tolerance and conditioning errors. Streaming TSQR is therefore the
safer first optimization; a matrix-free sketched solver can be compared
against it later.

