"""Definition of volume phase scattering functions."""

from functools import wraps

import sympy as sp

from ._scatter import _Scatter, _LinComb, _parse_sympy_param
from .helpers import append_numpy_docstring


class VolumeScatter(_Scatter):
    """
    Class for use as volume scattering distribution.

    Parameters
    ----------
    ncoefs : int
             Number of coefficients used for the Legendre-approximation.

    a : [ float , float , float ] , optional (default = [-1.,1.,1.])
        Generalized scattering angle parameters used for defining the
        scat_angle() of the distribution function. For more details, see:
        https://rt1-model.rtfd.io/en/latest/theory.html#equation-general_scat_angle

    """

    def __init__(self, ncoefs=None, a=None):
        # register plot-functions
        self._register_plotfuncs()

        # set scattering angle generalization-matrix to [-1,1,1] if it is not
        # explicitly provided by the chosen class this results in a peak in
        # forward-direction which is suitable for describing volume-scattering
        # phase-functions
        if a is None:
            a = getattr(self, "a", [-1.0, 1.0, 1.0])

        self.a = [_parse_sympy_param(i) for i in a]
        self._ncoefs = ncoefs

        assert len(self.a) == 3, "Generalization-parameter 'a' must contain 3 values"

    def legcoefs(self):
        """Legendre coefficients of the BRDF."""
        raise NotImplementedError

    def _func(self):
        """Phase function as sympy object."""
        raise NotImplementedError


class LinComb(_LinComb, VolumeScatter):
    """Class to create linear combinations of volume scattering distributions."""

    @wraps(_LinComb.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


LinComb.__doc__ = _LinComb.__init__.__doc__


@append_numpy_docstring(VolumeScatter)
class Isotropic(VolumeScatter):
    """
    Define an isotropic scattering distribution.

    Notes
    -----
    - Only 1 expansion coefficient is required, so `ncoefs` is always set to 1!
    - Since the distribution is independent of the scattering angle, the `a` parameter
      has no effect!

    """

    def __init__(self, **kwargs):
        super(Isotropic, self).__init__(**kwargs)

    @property
    def ncoefs(self):
        """The number of coefficients used to approximate the phase function."""
        # Only 1 coefficient is needed to correctly represent the scattering function
        return 1

    @property
    def legcoefs(self):
        """Legendre coefficients of the phase function."""
        n = sp.Symbol("n")
        return (1.0 / (4.0 * sp.pi)) * sp.KroneckerDelta(0, n)

    @property
    def _func(self):
        """Phase function as sympy object."""
        return 1.0 / (4.0 * sp.pi)


@append_numpy_docstring(VolumeScatter)
class Rayleigh(VolumeScatter):
    """
    Rayleigh scattering function.

    Notes
    -----
    - Only 3 expansion coefficient are required, so `ncoefs` is always set to 3!

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def ncoefs(self):
        """The number of coefficients used to approximate the BRDF."""
        # Only 3 coefficients are needed to correctly represent the scattering function
        return 3

    @property
    def _func(self):
        """Phase function as sympy object."""
        theta_0 = sp.Symbol("theta_0")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_ex = sp.Symbol("phi_ex")
        x = self.scat_angle(theta_0, theta_ex, phi_0, phi_ex, self.a)
        return 3.0 / (16.0 * sp.pi) * (1.0 + x**2.0)

    @property
    def legcoefs(self):
        """Legendre coefficients of the phase function."""
        # only 3 coefficients are needed to correctly represent
        # the Rayleigh scattering function
        n = sp.Symbol("n")
        return (
            (3.0 / (16.0 * sp.pi))
            * (
                (4.0 / 3.0) * sp.KroneckerDelta(0, n)
                + (2.0 / 3.0) * sp.KroneckerDelta(2, n)
            )
        ).expand()


@append_numpy_docstring(VolumeScatter)
class HenyeyGreenstein(VolumeScatter):
    """
    HenyeyGreenstein scattering function.

        Henyey, L. G. and Greenstein, J. L., Diffuse radiation in the Galaxy.,
        The Astrophysical Journal, vol. 93, pp. 70–83, 1941. doi:10.1086/144246.

    Parameters
    ----------
    t : float
        Asymmetry parameter (must be in the range -1 < t < 1).

    """

    def __init__(self, t=None, **kwargs):
        super().__init__(**kwargs)

        assert t is not None, "The asymmetry parameter t needs to be provided!"

        self.t = _parse_sympy_param(t)

    @property
    def _func(self):
        """Phase function as sympy object."""
        theta_0 = sp.Symbol("theta_0")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_ex = sp.Symbol("phi_ex")
        x = self.scat_angle(theta_0, theta_ex, phi_0, phi_ex, self.a)
        func = (1.0 - self.t**2.0) / (
            (4.0 * sp.pi) * (1.0 + self.t**2.0 - 2.0 * self.t * x) ** 1.5
        )

        return func

    @property
    def legcoefs(self):
        """Legendre coefficients of the phase function."""
        n = sp.Symbol("n")
        legcoefs = (1.0 / (4.0 * sp.pi)) * (2.0 * n + 1) * self.t**n
        return legcoefs


@append_numpy_docstring(VolumeScatter)
class HGRayleigh(VolumeScatter):
    """
    HenyeyGreenstein-Rayleigh scattering function.

        Quanhua Liu and Fuzhong Weng: Combined henyey-greenstein and
        rayleigh phase function,
        Appl. Opt., 45(28):7475-7479, Oct 2006. doi: 10.1364/AO.45.

    Parameters
    ----------
    t : float
        Asymmetry parameter (must be in the range -1 < t < 1).

    """

    def __init__(self, t=None, **kwargs):
        super().__init__(**kwargs)

        assert t is not None, "The asymmetry parameter t needs to be provided!"

        self.t = _parse_sympy_param(t)

    @property
    def _func(self):
        """Phase function as sympy object."""
        theta_0 = sp.Symbol("theta_0")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_ex = sp.Symbol("phi_ex")
        x = self.scat_angle(theta_0, theta_ex, phi_0, phi_ex, self.a)
        return (
            3.0
            / (8.0 * sp.pi)
            * (
                1.0
                / (2.0 + self.t**2)
                * (1 + x**2)
                * (1.0 - self.t**2.0)
                / ((1.0 + self.t**2.0 - 2.0 * self.t * x) ** 1.5)
            )
        )

    @property
    def legcoefs(self):
        """Legendre coefficients of the phase function."""
        n = sp.Symbol("n")
        return sp.Piecewise(
            (
                3.0
                / (8.0 * sp.pi)
                * 1.0
                / (2.0 + self.t**2)
                * (
                    (n + 2.0) * (n + 1.0) / (2.0 * n + 3) * self.t ** (n + 2.0)
                    + (n + 1.0) ** 2.0 / (2.0 * n + 3.0) * self.t**n
                    + (5.0 * n**2.0 - 1.0) / (2.0 * n - 1.0) * self.t**n
                ),
                n < 2,
            ),
            (
                3.0
                / (8.0 * sp.pi)
                * 1.0
                / (2.0 + self.t**2)
                * (
                    n * (n - 1.0) / (2.0 * n - 1.0) * self.t ** (n - 2.0)
                    + (n + 2.0) * (n + 1.0) / (2.0 * n + 3) * self.t ** (n + 2.0)
                    + (n + 1.0) ** 2.0 / (2.0 * n + 3.0) * self.t**n
                    + (5.0 * n**2.0 - 1.0) / (2.0 * n - 1.0) * self.t**n
                ),
                True,
            ),
        )
