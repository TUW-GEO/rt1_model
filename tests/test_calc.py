import pytest
import numpy as np
import cloudpickle
from rt1_model import RT1, surface, volume, set_lambda_backend, get_lambda_backend


@pytest.mark.xfail(raises=AssertionError)
def test_init_wrong():
    V = volume.Rayleigh()
    S = surface.Isotropic()

    R = RT1(V=V, SRF=S)
    R.set_monostatic(p_0=0)
    R.set_geometry(t_0=np.deg2rad(60.0), p_0=29.0)


def test_init():
    V = volume.Rayleigh()
    S = surface.Isotropic()

    R = RT1(V=V, SRF=S)
    R.set_monostatic(p_0=0)
    R.set_geometry(t_0=np.deg2rad(60.0))

    assert R._monostatic is True
    assert R.t_0 == np.deg2rad(60.0)
    assert R.t_ex == R.t_0
    assert R.p_0 == 0.0
    assert R.p_ex == np.pi


@pytest.mark.parametrize("tau", [0.3, 0])
@pytest.mark.parametrize("p_0_fixed", [True, False])
@pytest.mark.parametrize("sig0", [True, False])
@pytest.mark.parametrize("dB", [True, False])
@pytest.mark.parametrize("backend", ["sympy", "symengine"])
def test_calc_monostatic(backend, dB, sig0, p_0_fixed, tau):
    p_0 = 0.3
    t_0 = 0.2

    set_lambda_backend(backend)

    dynamic_angles = {"t_0": t_0}
    fixed_angles = {}

    if p_0_fixed is False:
        dynamic_angles["p_0"] = p_0
    else:
        fixed_angles["p_0"] = p_0

    V = volume.Rayleigh()
    S = surface.Isotropic()

    # just try to get it running simply without further testing
    R = RT1(V=V, SRF=S, dB=dB, sig0=sig0)
    R.set_monostatic(**fixed_angles)
    R.set_geometry(**dynamic_angles)
    R.update_params(tau=tau, omega=0.3, NormBRDF=0.3)

    Itot, Isurf, Ivol, Iint = R.calc()

    if dB is False:
        assert np.allclose(Itot, Isurf + Ivol + Iint)
        if tau == 0:
            assert Isurf >= 0
            assert Ivol == 0
            assert Iint == 0
    else:
        assert np.allclose(
            Itot,
            10 * np.log10(10 ** (Isurf / 10) + 10 ** (Ivol / 10) + 10 ** (Iint / 10)),
        )
        if tau == 0:
            assert Isurf != 0
            assert Ivol == -np.inf
            assert Iint == -np.inf


@pytest.mark.parametrize("t_ex_fixed", [True, False])
@pytest.mark.parametrize("t_0_fixed", [True, False])
@pytest.mark.parametrize("p_ex_fixed", [True, False])
@pytest.mark.parametrize("p_0_fixed", [True, False])
@pytest.mark.parametrize("sig0", [True, False])
@pytest.mark.parametrize("dB", [True, False])
@pytest.mark.parametrize("backend", ["sympy", "symengine"])
def test_calc_bistatic(backend, dB, sig0, p_0_fixed, p_ex_fixed, t_0_fixed, t_ex_fixed):
    tau = 0.2
    angles = dict(p_0=0.3, t_0=0.2, p_ex=0.45, t_ex=0.23)

    fixed = dict(p_0=p_0_fixed, t_0=t_0_fixed, p_ex=p_ex_fixed, t_ex=t_ex_fixed)
    dynamic_angles = {key: val for key, val in angles.items() if not fixed[key]}
    fixed_angles = {key: val for key, val in angles.items() if fixed[key]}

    set_lambda_backend(backend)

    V = volume.Rayleigh()
    S = surface.Isotropic()

    # just try to get it running simply without further testing
    R = RT1(V=V, SRF=S, dB=dB, sig0=sig0)
    R.set_bistatic(**fixed_angles)
    R.set_geometry(**dynamic_angles)
    R.update_params(tau=tau, omega=0.3, NormBRDF=0.3)

    Itot, Isurf, Ivol, Iint = R.calc()

    if dB is False:
        assert np.allclose(Itot, Isurf + Ivol + Iint)
        if tau == 0:
            assert Isurf >= 0
            assert Ivol == 0
            assert Iint == 0
    else:
        assert np.allclose(
            Itot,
            10 * np.log10(10 ** (Isurf / 10) + 10 ** (Ivol / 10) + 10 ** (Iint / 10)),
        )
        if tau == 0:
            assert Isurf != 0
            assert Ivol == -np.inf
            assert Iint == -np.inf


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
    R.set_bistatic(p_0=p_0, t_ex=t_ex, p_ex=p_ex)
    R.set_geometry(t_0=t_0)

    Itot, Isurf, Ivol, Iint = R.calc()
    assert np.allclose(Isurf, 2.0 / np.pi, 15)
    assert np.allclose(Ivol, 0.0)


@pytest.mark.parametrize("backend", ["sympy", "symengine"])
def test_pickle(backend):
    set_lambda_backend(backend)

    SRF = surface.HenyeyGreenstein(t="t_s", ncoefs=8)
    V = volume.HGRayleigh(t="t_v", ncoefs=8)

    R = RT1(V=V, SRF=SRF)
    R.set_monostatic(0.2)
    R.set_geometry(t_0=0.1)
    R.calc(omega=0.3, tau=0.1, NormBRDF=0.3, t_s=0.3, t_v=0.4)

    dump = cloudpickle.dumps(R)

    load = cloudpickle.loads(dump)
    assert np.allclose(load.calc(), R.calc())


@pytest.mark.parametrize("dB", [True, False], ids=["dB", "linear"])
@pytest.mark.parametrize("sig0", [True, False], ids=["sig0", "intensity"])
def test_sympy_symengine_equality(sig0, dB):
    # create some random parameter arrays
    n, n_incs = 100, 20

    t_0 = np.tile(np.deg2rad(np.linspace(30, 75, n_incs)), (n, 1))
    x = np.linspace(np.linspace(0.1, 0.2, n_incs), np.linspace(0.3, 0.4, n_incs), n)

    SRF = surface.HenyeyGreenstein(t="x", ncoefs=3)
    V = volume.HGRayleigh(t="x", ncoefs=3)

    # evaluate result with sympy
    set_lambda_backend("sympy")
    assert get_lambda_backend() == "sympy", "Backend was not correctly set"

    R_sympy = RT1(V=V, SRF=SRF, sig0=sig0, dB=dB)
    R_sympy.omega = "x"
    R_sympy.NormBRDF = "x"
    R_sympy.tau = "x"

    R_sympy.set_monostatic(p_0=0.2)
    R_sympy.set_geometry(t_0=t_0)
    R_sympy.omega, R_sympy.NormBRDF, R_sympy.tau = "x", "x", "x"
    sympy_res = R_sympy.calc(x=x)

    # evaluate result with symengine
    set_lambda_backend("symengine")
    assert get_lambda_backend() == "symengine", "Backend was not correctly set"

    R_seng = RT1(V=V, SRF=SRF, sig0=sig0, dB=dB)
    R_seng.omega, R_seng.NormBRDF, R_seng.tau = "x", "x", "x"
    R_seng.set_monostatic(p_0=0.2)

    R_seng.set_geometry(t_0=t_0)
    seng_res = R_seng.calc(x=x)

    assert np.allclose(sympy_res, seng_res)
