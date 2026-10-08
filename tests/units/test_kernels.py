"""The GPU kernels of `plantbench.units._kernels` against the NumPy code they replace.

The per-row C is checked twice: compiled for the host with the system C compiler, which
needs no GPU and catches the two drifting apart, and as the CUDA kernels themselves on a
GPU (`pixi run -e cuda test-gpu`).  Both compare with `PengRobinson.on("cpu")`, the NumPy
code with the closed-form cubic, as the GPU path runs it."""

from __future__ import annotations

import ctypes
import shutil
import subprocess

import numpy as np
import pytest

pytest.importorskip("thermo")

from plantbench import backend  # noqa: E402
from plantbench.units import _kernels  # noqa: E402
from plantbench.units import peng_robinson as prm  # noqa: E402

NAMES = ("nitrogen", "methane", "ethane", "propane", "n-butane", "n-pentane", "n-hexane")
FEED = np.array([0.010, 0.930, 0.030, 0.015, 0.009, 0.003, 0.003])

_HOST = r"""
#include <math.h>
#define DEVICE static
#define PB_NAN NAN
""" + _kernels.DEVICE_SOURCE + r"""
void host_ln_phi(int n, int nc, const double *T, const double *P, const double *x,
                 const unsigned char *liquid, const double *Tc, const double *m,
                 const double *ac, const double *omk, const double *bi, double R,
                 double sqrt2, double *out)
{
    for (int r = 0; r < n; r++)
        pb_ln_phi(nc, T[r], P[r], x + r * nc, liquid[r], Tc, m, ac, omk, bi, R, sqrt2,
                  out + r * nc);
}
void host_rachford_rice(int n, int nc, const double *z, const double *K,
                        const double *beta0, double tol, int max_iter, double *beta)
{
    for (int r = 0; r < n; r++)
        beta[r] = pb_rachford_rice(nc, z + r * nc, K + r * nc, beta0[r], tol, max_iter);
}
"""

_D = ctypes.POINTER(ctypes.c_double)


@pytest.fixture(scope="module")
def eos():
    return prm.PengRobinson(prm.Components.from_names(NAMES)).on("cpu")


@pytest.fixture(scope="module")
def rows(eos):
    """Rows over the range of a natural-gas plant, 150 to 400 K and 1 to 60 bar, with
    compositions scattered about a natural gas, and a phase for each."""
    rng = np.random.default_rng(0)
    n = 2000
    return (rng.uniform(150.0, 400.0, n), rng.uniform(1e5, 6e6, n),
            rng.dirichlet(200.0 * FEED + 0.2, n), rng.random(n) < 0.5)


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("no C compiler")
    d = tmp_path_factory.mktemp("kernels")
    (d / "k.c").write_text(_HOST)
    subprocess.run([cc, "-O2", "-shared", "-fPIC", "-Wall", "-Werror", "-Wno-unused-function",
                    str(d / "k.c"), "-o", str(d / "k.so"), "-lm"], check=True)
    return ctypes.CDLL(str(d / "k.so"))


def _p(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def _assert_same(got, want, rtol):
    finite = np.isfinite(want)
    np.testing.assert_array_equal(np.isfinite(got), finite)
    np.testing.assert_allclose(got[finite], want[finite], rtol=rtol, atol=rtol)


def test_the_fugacity_row_compiled_for_the_host_is_the_numpy_one(eos, rows, host):
    T, P, x, liquid = rows
    n, nc = x.shape
    keep = [np.ascontiguousarray(v, dtype=float) for v in (T, P, x)]
    consts = [np.ascontiguousarray(v, dtype=float)
              for v in (eos.c.Tc, eos.m_i, eos.ac_i, eos.one_minus_k, eos.b_i)]
    flags = liquid.astype(np.uint8)
    out = np.empty((n, nc))
    host.host_ln_phi(ctypes.c_int(n), ctypes.c_int(nc), *map(_p, keep), _p(flags),
                     *map(_p, consts), ctypes.c_double(prm.R),
                     ctypes.c_double(float(prm.SQRT2)), _p(out))
    with np.errstate(all="ignore"):
        _assert_same(out, eos.ln_phi(T, P, x, liquid), rtol=1e-10)


def test_the_rachford_rice_row_compiled_for_the_host_is_the_numpy_one(eos, rows, host):
    T, P, x, _ = rows
    n, nc = x.shape
    K = np.ascontiguousarray(eos.wilson_K(T, P))
    beta0 = np.full(n, 0.5)
    beta = np.empty(n)
    host.host_rachford_rice(ctypes.c_int(n), ctypes.c_int(nc), _p(np.ascontiguousarray(x)),
                            _p(K), _p(beta0), ctypes.c_double(1e-14), ctypes.c_int(100),
                            _p(beta))
    np.testing.assert_allclose(beta, prm._rachford_rice(x, K, beta0), rtol=0, atol=1e-12)


@pytest.mark.parametrize("phase", ["liquid", "vapour", "rows"])
@pytest.mark.gpu
@pytest.mark.skipif(not backend.available("cuda"), reason="no CuPy or no GPU")
def test_the_state_and_fugacities_on_the_device_are_the_numpy_ones(eos, rows, phase):
    T, P, x, liquid = rows
    which = liquid if phase == "rows" else phase
    dev = eos.on("cuda")
    assert dev._use_kernels
    d = lambda v: backend.to_device(v, "cuda")  # noqa: E731
    which_d = d(liquid).astype(bool) if phase == "rows" else phase
    with np.errstate(all="ignore"):
        for got, want in zip(dev._state(d(T), d(P), d(x), which_d), eos._state(T, P, x, which)):
            _assert_same(backend.to_host(got), want, rtol=1e-12)
        _assert_same(backend.to_host(dev.ln_phi(d(T), d(P), d(x), which_d)),
                     eos.ln_phi(T, P, x, which), rtol=1e-10)


@pytest.mark.gpu
@pytest.mark.skipif(not backend.available("cuda"), reason="no CuPy or no GPU")
def test_rachford_rice_on_the_device_is_the_numpy_one(eos, rows):
    T, P, x, _ = rows
    K = eos.wilson_K(T, P)
    beta0 = np.full(len(T), 0.5)
    d = lambda v: backend.to_device(v, "cuda")  # noqa: E731
    got = backend.to_host(prm._rachford_rice(d(x), d(K), d(beta0)))
    np.testing.assert_allclose(got, prm._rachford_rice(x, K, beta0), rtol=0, atol=1e-12)


@pytest.mark.gpu
@pytest.mark.skipif(not backend.available("cuda"), reason="no CuPy or no GPU")
def test_a_flash_on_the_device_is_the_numpy_one(eos, rows):
    T, P, z, _ = rows
    d = lambda v: backend.to_device(v, "cuda")  # noqa: E731
    want = eos.flash_PT(T, P, z)
    got = eos.on("cuda").flash_PT(d(T), d(P), d(z))
    np.testing.assert_allclose(backend.to_host(got.V), want.V, rtol=0, atol=1e-9)
    np.testing.assert_allclose(backend.to_host(got.h), want.h, rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(backend.to_host(got.s), want.s, rtol=1e-9, atol=1e-9)
