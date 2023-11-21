import pytest
from rt1_model import surface, volume, set_lambda_backend

# find all names of distribution functions (to make sure all are tested)
SRFnames = [
    key
    for key, val in surface.__dict__.items()
    if (
        isinstance(val, type)
        and issubclass(val, surface.SurfaceScatter)
        and not key.startswith("_")
        and not key in ["LinComb"]
    )
]

Vnames = [
    key
    for key, val in volume.__dict__.items()
    if (
        isinstance(val, type)
        and issubclass(val, volume.VolumeScatter)
        and not key.startswith("_")
        and not key in ["LinComb"]
    )
]


@pytest.mark.parametrize("backend", ["sympy", "symengine"])
def test_surface_init(backend):
    set_lambda_backend(backend)

    a = [0.1, 0.2, 0.3]

    choices = dict(
        Isotropic=dict(),
        CosineLobe=dict(i=3, ncoefs=5, a=a),
        HenyeyGreenstein=dict(t=0.4, ncoefs=6, a=a),
        HG_nadirnorm=dict(t=0.4, ncoefs=4, a=a),
    )

    assert all(
        i in choices for i in SRFnames if i != "SurfaceScatter"
    ), f"Surface functions {set(SRFnames).difference(choices)} are not tested!"

    # Check SRF initialization
    for name, params in choices.items():
        SRF = getattr(surface, name)(**params)

        for key, val in params.items():
            assert (
                getattr(SRF, key) == val
            ), f"Parameter {key} incorrecty assigned for surface.{name}!"

        SRF.calc(0.1, 0.2, 0.3, 0.4)
        SRF.legexpansion(0.1, 0.2, 0.3, 0.4)
        SRF._func


@pytest.mark.parametrize("backend", ["sympy", "symengine"])
def test_volume_init(backend):
    set_lambda_backend(backend)

    a = [-0.5, 0.6, 0.4]

    choices = dict(
        Isotropic=dict(),
        Rayleigh=dict(a=a),
        HenyeyGreenstein=dict(t=0.4, ncoefs=5, a=a),
        HGRayleigh=dict(t=0.4, ncoefs=6, a=a),
    )

    assert all(
        i in choices for i in Vnames if i != "VolumeScatter"
    ), f"Volume functions {set(Vnames).difference(choices)} are not tested!"

    # Check SRF initialization
    for name, params in choices.items():
        V = getattr(volume, name)(**params)

        for key, val in params.items():
            assert (
                getattr(V, key) == val
            ), f"Parameter {key} incorrecty assigned for volume.{name}!"

        # evaluate function numerical
        V.calc(0.1, 0.2, 0.3, 0.4)
        V.legexpansion(0.1, 0.2, 0.3, 0.4)
        V._func


@pytest.mark.parametrize("backend", ["sympy", "symengine"])
def test_linear_combinations_SRF(backend):
    set_lambda_backend(backend)

    a = [0.1, 0.2, 0.3]

    choices = dict(
        Isotropic=dict(),
        CosineLobe=dict(i=3, ncoefs=5, a=a),
        HenyeyGreenstein=dict(t=0.4, ncoefs=6, a=a),
        HG_nadirnorm=dict(t=0.4, ncoefs=3, a=a),
    )

    choices = [
        (1 / len(choices), getattr(surface, name)(**kwargs))
        for name, kwargs in choices.items()
    ]

    SRF = surface.LinComb(choices)

    SRF.calc(0.1, 0.2, 0.3, 0.4)
    SRF.legexpansion(0.1, 0.2, 0.3, 0.4)
    SRF._func


@pytest.mark.parametrize("backend", ["sympy", "symengine"])
def test_linear_combinations_V(backend):
    set_lambda_backend(backend)

    a = [0.1, 0.2, 0.3]

    choices = dict(
        Isotropic=dict(),
        Rayleigh=dict(a=a),
        HenyeyGreenstein=dict(t=0.4, ncoefs=5, a=a),
        HGRayleigh=dict(t=0.3, ncoefs=4, a=a),
    )

    choices = [
        (1 / len(choices), getattr(volume, name)(**kwargs))
        for name, kwargs in choices.items()
    ]

    V = volume.LinComb(choices)

    V.calc(0.1, 0.2, 0.3, 0.4)
    V.legexpansion(0.1, 0.2, 0.3, 0.4)
    V._func
