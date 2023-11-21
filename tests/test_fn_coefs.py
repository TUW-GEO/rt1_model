import pytest
import numpy as np
from scipy.special import gamma
from rt1_model import surface, volume, RT1, set_lambda_backend


@pytest.mark.parametrize("backend", ["sympy", "symengine"])
def test_fn_coefficients_HGIso(backend):
    set_lambda_backend(backend)
    # test if calculation of fn coefficients is correct
    # this is done by comparing the obtained coefficients
    # against the analytical solution using a Isotropic volume
    # and isotropic surface scattering phase function
    S = surface.Isotropic()
    V = volume.Isotropic()

    RT = RT1(V=V, SRF=S)

    r = RT._fnevals(theta_0=0, phi_0=0, theta_ex=0, phi_ex=np.pi, bsf=0)

    assert np.allclose(r[0], 1.0 / (2.0 * np.pi)), "fn coefs not equal!"


@pytest.mark.parametrize("backend", ["sympy", "symengine"])
def test_fn_coefficients_RayCosine(backend):
    set_lambda_backend(backend)
    # test if calculation of fn coefficients is correct
    # for a cosine lobe with reduced number of coefficients
    # this is done by comparing the obtained coefficients
    # against the analytical solution using a Rayleigh volume
    # and isotropic surface scattering phase function
    S = surface.CosineLobe(ncoefs=1, i=5)
    V = volume.Rayleigh()
    # --> cosTHETA = 0.

    # tests are using full Volume phase function, but only
    # ncoef times the coefficients from the surface

    t_0 = np.pi / 2.0
    t_ex = 0.234234
    p_0 = np.pi / 2.0
    p_ex = 0.0

    RT = RT1(V=V, SRF=S)
    res = RT._fnevals(theta_0=t_0, phi_0=p_0, theta_ex=t_ex, phi_ex=p_ex, bsf=0)

    # ncoefs = 1
    # analtytical solution for ncoefs = 1 --> n=0

    a0 = (3.0 / (16.0 * np.pi)) * (4.0 / 3.0)
    a2 = (3.0 / (16.0 * np.pi)) * (2.0 / 3.0)
    b0 = (15.0 * np.sqrt(np.pi)) / (16.0 * gamma(3.5) * gamma(4.0))

    ref0 = 1.0 / 4.0 * b0 * (8.0 * a0 - a2 - 3.0 * a2 * np.cos(2.0 * t_0))
    ref2 = 3.0 / 4.0 * a2 * b0 * (1.0 + 3.0 * np.cos(2.0 * t_0))

    assert np.allclose([ref0, ref2], [res[0], res[2]]), "fn coefs not equal!"

    # ncoefs = 2
    # first and third coef should be the same as for ncoefs=1
    S = surface.CosineLobe(ncoefs=2, i=5)
    RT = RT1(V=V, SRF=S)
    res2 = RT._fnevals(theta_0=t_0, phi_0=p_0, theta_ex=t_ex, phi_ex=p_ex, bsf=0)

    assert np.allclose([ref0, ref2], [res2[0], res2[2]]), "fn coefs not equal!"


@pytest.mark.parametrize("backend", ["sympy", "symengine"])
def test_fn_coefficients_RayIso(backend):
    set_lambda_backend(backend)
    # test if calculation of fn coefficients is correct
    # this is done by comparing the obtained coefficients
    # against the analytical solution using a Rayleigh volume
    # and isotropic surface scattering phase function
    S = surface.Isotropic()
    V = volume.Rayleigh()
    t_0 = np.deg2rad(60.0)
    t_ex = np.deg2rad(60.0)
    p_0 = 0.0
    p_ex = np.pi

    R = RT1(V=V, SRF=S)

    # the reference solutions should be (see rayleighisocoefficients.pdf)
    f0 = 3.0 / (16.0 * np.pi) * (3.0 - np.cos(t_0) ** 2.0)
    f1 = 0.0
    f2 = 3.0 / (16.0 * np.pi) * (3.0 * np.cos(t_0) ** 2.0 - 1.0)
    # and all others are 0.

    assert np.allclose(
        [f0, f1, f2],
        R._fnevals(theta_0=t_0, phi_0=p_0, theta_ex=t_ex, phi_ex=p_ex, bsf=0),
    ), "fn-coefs not equal!"
