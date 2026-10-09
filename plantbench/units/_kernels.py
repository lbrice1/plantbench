"""CUDA kernels for the Peng–Robinson evaluations of a batch on a GPU.

On a GPU every array operation is a kernel launch that costs the host about 16 us
whatever the size of the batch, and the closed-loop right-hand side of an NGL case is made
of tens of thousands of them, most elementwise on a few hundred rows.  The kernels here
do in one launch, one thread per row, what the NumPy code of `peng_robinson` does in many:

- `ln_phi` and `state`: the mixture parameters, the root of the cubic with its two Newton
  steps, and the log fugacity coefficients of a phase;
- `rachford_rice`: the vapour fraction, each row stopping as soon as it has converged,
  where the NumPy loop runs every row until the slowest has.

They follow the NumPy code formula by formula, and agree with it to round-off (tests in
`tests/units/test_kernels.py`, run on a GPU); they are used only on CuPy arrays, so the
path on the host does not change.  `SOURCE` is plain C apart from the kernel entry points,
so that the per-row functions can also be compiled and checked on the host.
"""

from __future__ import annotations

import functools

import numpy as np

from plantbench import backend

NC_MAX = 16  # the most components a kernel row holds
_THREADS = 128

# The functions of one row, valid C and CUDA C++.  `DEVICE` is `__device__` on the GPU.
DEVICE_SOURCE = r"""
#define NC_MAX 16
#define PB_EPS 2.220446049250313e-16

/* The real roots of Z^3 + c2 Z^2 + c1 Z + c0 = 0, as `_cubic_real_roots`. */
DEVICE void pb_cubic_real_roots(double c2, double c1, double c0, double *r)
{
    double shift = c2 / 3.0;
    double p = c1 - c2 * shift;
    double q = 2.0 * shift * shift * shift - shift * c1 + c0;
    double disc = (q / 2.0) * (q / 2.0) + (p / 3.0) * (p / 3.0) * (p / 3.0);
    if (disc > 0.0 || p >= 0.0) {
        double sq = sqrt(disc > 0.0 ? disc : 0.0);
        double one = cbrt(-q / 2.0 + sq) + cbrt(-q / 2.0 - sq) - shift;
        r[0] = one;
        r[1] = one;
        r[2] = one;
    } else {
        double pn = p < 0.0 ? p : -1.0;
        double rr = 2.0 * sqrt(-pn / 3.0);
        double arg = 3.0 * q / (pn * rr);
        arg = arg < -1.0 ? -1.0 : (arg > 1.0 ? 1.0 : arg);  /* NaN stays NaN */
        double theta = acos(arg) / 3.0;
        for (int k = 0; k < 3; k++)
            r[k] = rr * cos(theta - 2.0 * 3.141592653589793 / 3.0 * k) - shift;
    }
}

/* The double roots round-off can hide, as `_near_double_roots`: each stationary point
   of the cubic where the cubic is zero to round-off, NaN otherwise. */
DEVICE void pb_near_double_roots(double c2, double c1, double c0, double *r)
{
    double d = c2 * c2 - 3.0 * c1;
    double s = sqrt(d > 0.0 ? d : 0.0);
    r[0] = (-c2 - s) / 3.0;
    r[1] = (-c2 + s) / 3.0;
    for (int k = 0; k < 2; k++) {
        double Z = r[k];
        double f = ((Z + c2) * Z + c1) * Z + c0;
        double size = fabs(Z) * Z * Z + fabs(c2) * Z * Z + fabs(c1 * Z) + fabs(c0);
        if (!(d >= 0.0 && fabs(f) <= 16.0 * PB_EPS * size))
            r[k] = PB_NAN;
    }
}

/* The root of a phase from the cubic's coefficients, as `_select_root`. */
DEVICE double pb_select_root(double c2, double c1, double c0, double B, int liquid)
{
    double r[5];
    pb_cubic_real_roots(c2, c1, c0, r);
    pb_near_double_roots(c2, c1, c0, r + 3);
    double Z = 0.0;
    int found = 0;
    for (int k = 0; k < 5; k++) {
        if (r[k] > B) {  /* false for NaN */
            if (!found || (liquid ? r[k] < Z : r[k] > Z))
                Z = r[k];
            found = 1;
        }
    }
    if (!found)
        return PB_NAN;
    double f = ((Z + c2) * Z + c1) * Z + c0;
    for (int it = 0; it < 2; it++) {
        double next = Z - f / ((3.0 * Z + 2.0 * c2) * Z + c1);
        double f_next = ((next + c2) * next + c1) * next + c0;
        if (fabs(f_next) <= fabs(f)) {  /* false for NaN */
            Z = next;
            f = f_next;
        }
    }
    return Z;
}

/* The compressibility root of a phase, as `PengRobinson._Z` with roots "closed". */
DEVICE double pb_cubic_Z(double A, double B, int liquid)
{
    double c2 = -(1.0 - B);
    double c1 = A - 3.0 * B * B - 2.0 * B;
    double c0 = -(A * B - B * B - B * B * B);
    return pb_select_root(c2, c1, c0, B, liquid);
}

/* The state of one row, as `PengRobinson._state`: s = (a, da, b, A, B, Z, L), and
   sum_j = sum_j x_j a_ij. */
DEVICE void pb_state(int nc, double T, double P, const double *x, int liquid,
                     const double *Tc, const double *m, const double *ac,
                     const double *omk, const double *bi, double R, double sqrt2,
                     double *sum_j, double *s)
{
    double sq[NC_MAX], ri[NC_MAX];
    for (int i = 0; i < nc; i++) {
        double root = 1.0 + m[i] * (1.0 - sqrt(T / Tc[i]));
        double a_i = ac[i] * root * root;
        double da_i = -ac[i] * m[i] * root / sqrt(T * Tc[i]);
        sq[i] = sqrt(a_i);
        ri[i] = da_i / a_i;
    }
    double a = 0.0, da = 0.0, b = 0.0;
    for (int i = 0; i < nc; i++) {
        double si = 0.0, di = 0.0;
        for (int j = 0; j < nc; j++) {
            double a_ij = sq[i] * sq[j] * omk[i * nc + j];
            si += a_ij * x[j];
            di += x[j] * a_ij * (ri[i] + ri[j]);
        }
        sum_j[i] = si;
        a += x[i] * si;
        da += x[i] * di;
        b += x[i] * bi[i];
    }
    da *= 0.5;
    double A = a * P / ((R * T) * (R * T));
    double B = b * P / (R * T);
    double Z = pb_cubic_Z(A, B, liquid);
    s[0] = a;
    s[1] = da;
    s[2] = b;
    s[3] = A;
    s[4] = B;
    s[5] = Z;
    s[6] = log((Z + (1.0 + sqrt2) * B) / (Z + (1.0 - sqrt2) * B));
}

/* Log fugacity coefficients of one row, as `PengRobinson.ln_phi`. */
DEVICE void pb_ln_phi(int nc, double T, double P, const double *x, int liquid,
                      const double *Tc, const double *m, const double *ac,
                      const double *omk, const double *bi, double R, double sqrt2,
                      double *out)
{
    double sum_j[NC_MAX], s[7];
    pb_state(nc, T, P, x, liquid, Tc, m, ac, omk, bi, R, sqrt2, sum_j, s);
    double a = s[0], b = s[2], A = s[3], B = s[4], Z = s[5], L = s[6];
    for (int i = 0; i < nc; i++) {
        double bi_b = bi[i] / b;
        out[i] = bi_b * (Z - 1.0) - log(Z - B)
                 - (A / (2.0 * sqrt2 * B)) * (2.0 * sum_j[i] / a - bi_b) * L;
    }
}

/* The vapour fraction of one row, as `_rachford_rice`, the row stopping once converged. */
DEVICE double pb_rachford_rice(int nc, const double *z, const double *K, double beta0,
                               double tol, int max_iter)
{
    double Kmax = K[0], Kmin = K[0];
    for (int i = 1; i < nc; i++) {
        Kmax = K[i] > Kmax ? K[i] : Kmax;
        Kmin = K[i] < Kmin ? K[i] : Kmin;
    }
    if (Kmax <= 1.0)
        return -1.0;
    if (Kmin >= 1.0)
        return 2.0;
    double lo = 1.0 / (1.0 - Kmax);
    double hi = 1.0 / (1.0 - Kmin);
    double b_lo = lo + 1e-9 * (hi - lo), b_hi = hi - 1e-9 * (hi - lo);
    double b = beta0 < b_lo ? b_lo : (beta0 > b_hi ? b_hi : beta0);
    for (int it = 0; it < max_iter; it++) {
        double f = 0.0, df = 0.0;
        for (int i = 0; i < nc; i++) {
            double Km1 = K[i] - 1.0;
            double d = 1.0 + b * Km1;
            f += z[i] * Km1 / d;
            df -= z[i] * Km1 * Km1 / (d * d);
        }
        if (f > 0.0)
            lo = b;
        if (f < 0.0)
            hi = b;
        double next = b - f / df;
        if (next < lo || next > hi)
            next = 0.5 * (lo + hi);
        int done = fabs(next - b) <= tol * (1.0 + fabs(b));
        b = next;
        if (done)
            break;
    }
    return b;
}
"""

_KERNELS = r"""
extern "C" __global__ void pb_ln_phi_rows(
    int n, int nc, const double *T, const double *P, const double *x, const bool *liquid,
    const double *Tc, const double *m, const double *ac, const double *omk,
    const double *bi, double R, double sqrt2, double *out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n)
        pb_ln_phi(nc, T[r], P[r], x + r * nc, liquid[r], Tc, m, ac, omk, bi, R, sqrt2,
                  out + r * nc);
}

extern "C" __global__ void pb_state_rows(
    int n, int nc, const double *T, const double *P, const double *x, const bool *liquid,
    const double *Tc, const double *m, const double *ac, const double *omk,
    const double *bi, double R, double sqrt2, double *sum_j, double *s)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n) {
        double row[7];
        pb_state(nc, T[r], P[r], x + r * nc, liquid[r], Tc, m, ac, omk, bi, R, sqrt2,
                 sum_j + r * nc, row);
        for (int k = 0; k < 7; k++)
            s[k * n + r] = row[k];
    }
}

extern "C" __global__ void pb_rachford_rice_rows(
    int n, int nc, const double *z, const double *K, const double *beta0, double tol,
    int max_iter, double *beta)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n)
        beta[r] = pb_rachford_rice(nc, z + r * nc, K + r * nc, beta0[r], tol, max_iter);
}
"""

SOURCE = ("#define DEVICE __device__\n"
          "#define PB_NAN __longlong_as_double(0x7ff8000000000000LL)\n"
          + DEVICE_SOURCE + _KERNELS)


def applies(xp) -> bool:
    """Whether arrays of the module `xp` are evaluated by these kernels."""
    return getattr(xp, "__name__", "") == "cupy"


@functools.cache
def _module():
    return backend.module("cuda").RawModule(code=SOURCE)


def _launch(name: str, n: int, args: tuple) -> None:
    _module().get_function(name)(((n + _THREADS - 1) // _THREADS,), (_THREADS,), args)


def _rows(eos, T, P, x, phase):
    # Imported here: `peng_robinson` imports this module.
    from plantbench.units.peng_robinson import SQRT2, R

    cp = backend.module("cuda")
    n, nc = x.shape
    if nc > NC_MAX:
        raise ValueError(f"the GPU kernels hold at most {NC_MAX} components, not {nc}")
    if isinstance(phase, str):
        liquid = cp.full(n, phase == "liquid", dtype=bool)
    else:
        liquid = cp.ascontiguousarray(phase, dtype=bool)
    f = lambda v: cp.ascontiguousarray(v, dtype=cp.float64)  # noqa: E731
    return (np.int32(n), np.int32(nc), f(T), f(P), f(x), liquid,
            f(eos.c.Tc), f(eos.m_i), f(eos.ac_i), f(eos.one_minus_k), f(eos.b_i),
            np.float64(R), np.float64(SQRT2))


def ln_phi(eos, T, P, x, phase):
    """`PengRobinson.ln_phi` of every row in one launch; `phase` is "liquid", "vapour",
    or a boolean array true on the liquid rows."""
    cp = backend.module("cuda")
    args = _rows(eos, T, P, x, phase)
    out = cp.empty(x.shape)
    _launch("pb_ln_phi_rows", x.shape[0], args + (out,))
    return out


def state(eos, T, P, x, phase):
    """`PengRobinson._state` of every row in one launch: a, da, b, sum_j, A, B, Z, L."""
    cp = backend.module("cuda")
    n = x.shape[0]
    args = _rows(eos, T, P, x, phase)
    sum_j = cp.empty(x.shape)
    s = cp.empty((7, n))
    _launch("pb_state_rows", n, args + (sum_j, s))
    a, da, b, A, B, Z, L = s
    return a, da, b, sum_j, A, B, Z, L


def rachford_rice(z, K, beta0, tol: float, max_iter: int):
    """`_rachford_rice` of every row in one launch, each row stopping once converged."""
    cp = backend.module("cuda")
    n, nc = z.shape
    f = lambda v: cp.ascontiguousarray(v, dtype=cp.float64)  # noqa: E731
    beta = cp.empty(n)
    _launch("pb_rachford_rice_rows", n,
            (np.int32(n), np.int32(nc), f(z), f(K), f(cp.broadcast_to(beta0, (n,))),
             np.float64(tol), np.int32(max_iter), beta))
    return beta
