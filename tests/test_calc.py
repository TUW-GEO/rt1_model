import pytest
import numpy as np
import cloudpickle
from rt1_model import RT1, surface, volume, set_lambda_backend


def test_init():
    V = volume.Rayleigh()
    S = surface.Isotropic()

    R = RT1(V=V, SRF=S)
    R.set_geometry(t_0=np.deg2rad(60.0), p_0=0.0, geometry="mono")

    assert R.geometry == "mono"
    assert R.t_0 == np.deg2rad(60.0)
    assert R.t_ex == R.t_0
    assert R.p_0 == 0.0
    assert R.p_ex == np.pi


@pytest.mark.parametrize("backend", ["sympy", "symengine"])
def test_calc(backend):
    set_lambda_backend(backend)

    V = volume.Rayleigh()
    S = surface.Isotropic()

    # just try to get it running simply without further testing
    R = RT1(V=V, SRF=S, dB=False)
    R.set_geometry(t_0=np.deg2rad(60.0), p_0=0.0)
    R.update_params(tau=0.7, omega=0.3, NormBRDF=0.3)

    Itot, Isurf, Ivol, Iint = R.calc()
    assert np.allclose(Itot, Isurf + Ivol + Iint)

    # check values for sig0 = False
    R = RT1(V=V, SRF=S, dB=False, sig0=False)
    R.set_geometry(t_0=np.deg2rad(60.0), p_0=0.0, geometry="mono")
    R.update_params(tau=0.7, omega=0.3, NormBRDF=0.3)

    Itot, Isurf, Ivol, Iint = R.calc()
    assert np.allclose(Itot, Isurf + Ivol + Iint)

    # check values in dB
    R = RT1(V=V, SRF=S, dB=True)
    R.set_geometry(t_0=np.deg2rad(60.0), p_0=0.0, geometry="mono")
    R.update_params(tau=0.7, omega=0.3, NormBRDF=0.3)

    Itot, Isurf, Ivol, Iint = R.calc()
    assert np.allclose(
        Itot,
        10 * np.log10(10 ** (Isurf / 10) + 10 ** (Ivol / 10) + 10 ** (Iint / 10)),
    )

    # check values in dB for sig0 = False
    R = RT1(V=V, SRF=S, dB=True, sig0=False)
    R.set_geometry(t_0=np.deg2rad(60.0), p_0=0.0, geometry="mono")
    R.update_params(tau=0.7, omega=0.3, NormBRDF=0.3)

    Itot, Isurf, Ivol, Iint = R.calc()
    assert np.allclose(
        Itot,
        10 * np.log10(10 ** (Isurf / 10) + 10 ** (Ivol / 10) + 10 ** (Iint / 10)),
    )

    # test results for tau=0 / omega=0
    V = volume.Rayleigh()
    R = RT1(V=V, SRF=S, dB=False)
    R.set_geometry(t_0=np.deg2rad(60.0), p_0=0.0, geometry="mono")
    R.update_params(tau=0.0, omega=0.0, NormBRDF=0.3)

    Itot, Isurf, Ivol, Iint = R.calc()
    assert np.allclose(Ivol, 0.0)
    assert np.allclose(Iint, 0.0)
    assert np.allclose(Itot, Isurf)
    assert Isurf > 0.0


@pytest.mark.parametrize("backend", ["sympy", "symengine"])
def test_zero_tau(backend):
    set_lambda_backend(backend)

    t_0 = np.deg2rad(60.0)
    t_ex = np.deg2rad(60.0)
    p_0 = 0.0
    p_ex = np.pi

    V = volume.Rayleigh()
    S = surface.Isotropic()

    R = RT1(V=V, SRF=S, sig0=False, dB=False)
    R.tau = 0
    R.omega = 0
    R.bsf = 0
    R.NormBRDF = 4

    R.set_geometry(t_0=t_0, p_0=p_0, t_ex=t_ex, p_ex=p_ex, geometry="ffff")

    Itot, Isurf, Ivol, Iint = R.calc()
    assert np.allclose(Isurf, 2.0 / np.pi, 15)
    assert np.allclose(Ivol, 0.0)


@pytest.mark.parametrize("backend", ["sympy", "symengine"])
def test_pickle(backend):
    set_lambda_backend(backend)

    SRF = surface.HenyeyGreenstein(t="t_s", ncoefs=8)
    V = volume.HGRayleigh(t="t_v", ncoefs=8)

    R = RT1(V=V, SRF=SRF)
    R.set_geometry(t_0=0.1, p_0=0.2, geometry="mono")
    R.calc(omega=0.3, tau=0.1, NormBRDF=0.3, t_s=0.3, t_v=0.4)

    dump = cloudpickle.dumps(R)

    load = cloudpickle.loads(dump)
    assert np.allclose(load.calc(), R.calc())
