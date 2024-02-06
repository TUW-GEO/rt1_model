"""Definition of surface scattering functions (BRDF)."""

from abc import abstractmethod, ABCMeta
from functools import wraps

import sympy as sp

from ._scatter import _Scatter, _LinComb, _parse_sympy_param
from .helpers import append_numpy_docstring


class SurfaceScatter(_Scatter, metaclass=ABCMeta):
    """
    Abstract base class for use as surface scattering distribution.

    Parameters
    ----------
    ncoefs : int
             Number of coefficients used for the Legendre-approximation.

    a : [ float , float , float ] , optional (default = [1.,1.,1.])
        Generalized scattering angle parameters used for defining the
        scat_angle() of the distribution function. For more details, see:
        https://rt1-model.rtfd.io/en/latest/theory.html#equation-general_scat_angle

    """

    def __init__(self, ncoefs=None, a=None):
        # register plot-functions
        self._register_plotfuncs()

        # set scattering angle generalization-matrix to [1,1,1] if it is not
        # explicitly provided by the chosen class.
        # this results in a peak in specular-direction which is suitable
        # for describing surface BRDF's, see:
        # https://rt1-model.rtfd.io/en/latest/model_specification.html#evaluation-angles
        if a is None:
            a = getattr(self, "a", [1.0, 1.0, 1.0])

        self.a = [_parse_sympy_param(i) for i in a]
        self._ncoefs = ncoefs

        assert len(self.a) == 3, "Generalization-parameter 'a' must contain 3 values"

    @abstractmethod
    def legendre_coefficients(self):
        """Legendre coefficients of the BRDF."""
        ...

    @abstractmethod
    def phase_function(self):
        """Phase function as sympy object."""
        ...


class LinComb(_LinComb, SurfaceScatter):
    """Class to create linear combinations of surface scattering distributions."""

    @wraps(_LinComb.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


LinComb.__doc__ = _LinComb.__init__.__doc__


@append_numpy_docstring(SurfaceScatter)
class Isotropic(SurfaceScatter):
    """
    Isotropic (Lambertian) surface brdf.

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
        """The number of coefficients used in the legendre expansion."""
        # Only 1 coefficient is needed to correctly represent the scattering function
        return 1

    @property
    def legendre_coefficients(self):
        """Legendre coefficients of the BRDF."""
        n = sp.Symbol("n")
        return (1.0 / sp.pi) * sp.KroneckerDelta(0, n)

    @property
    def phase_function(self):
        """Phase function as sympy object."""
        return 1.0 / sp.pi


@append_numpy_docstring(SurfaceScatter)
class CosineLobe(SurfaceScatter):
    """
    Cosine-lobe of power i.

    Parameters
    ----------
    i : int
        Power of the cosine lobe, i.e. cos(x)^i
    """

    def __init__(self, i=None, **kwargs):
        super().__init__(**kwargs)

        assert i is not None, "Cosine lobe power must be specified!"
        assert isinstance(i, int), "Cosine lobe power must be an integer!"
        assert i >= 0, "Power of Cosine-Lobe must be greater than 0!"

        self.i = i

    @property
    def legendre_coefficients(self):
        """Legendre coefficients of the BRDF."""
        n = sp.Symbol("n")
        # A13   The Rational(is needed as otherwise a Gamma function Pole error is issued)
        return (
            1.0
            / sp.pi
            * (
                (
                    2 ** (-2 - self.i)
                    * (1 + 2 * n)
                    * sp.sqrt(sp.pi)
                    * sp.gamma(1 + self.i)
                )
                / (
                    sp.gamma((2 - n + self.i) * sp.Rational(1, 2))
                    * sp.gamma((3 + n + self.i) * sp.Rational(1, 2))
                )
            )
        )

    @property
    def phase_function(self):
        """Phase function as sympy object."""
        return (
            1.0
            / sp.pi
            * (
                self.scat_angle_symbolic
                * (1.0 + sp.sign(self.scat_angle_symbolic))
                / 2.0
            )
            ** self.i
        )


@append_numpy_docstring(SurfaceScatter)
class HenyeyGreenstein(SurfaceScatter):
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
    def phase_function(self):
        """Phase function as sympy object."""

        return (
            1.0
            * (1.0 - self.t**2.0)
            / (
                (sp.pi)
                * (1.0 + self.t**2.0 - 2.0 * self.t * self.scat_angle_symbolic) ** 1.5
            )
        )

    @property
    def legendre_coefficients(self):
        """Legendre coefficients of the BRDF."""
        n = sp.Symbol("n")
        return 1.0 * (1.0 / (sp.pi)) * (2.0 * n + 1) * self.t**n


@append_numpy_docstring(SurfaceScatter)
class HG_nadirnorm(SurfaceScatter):
    """
    Nadir-normalized HenyeyGreenstein scattering function.

        R.Quast, C.Albergel, J.C.Calvet, W.Wagner, A Generic First-Order Radiative
        Transfer Modelling Approach for the Inversion of Soil and Vegetation Parameters
        from Scatterometer Observations, Remote Sensing (2019), doi:10.3390/rs11030285

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
    def phase_function(self):
        """Define Phase function as sympy object."""

        nadir_hemreflect = 4 * (
            (1.0 - self.t**2.0)
            * (
                1.0
                - self.t * (-self.t + self.a[0])
                - sp.sqrt(
                    (1 + self.t**2 - 2 * self.a[0] * self.t) * (1 + self.t**2)
                )
            )
            / (
                2.0
                * self.a[0] ** 2.0
                * self.t**2.0
                * sp.sqrt(1.0 + self.t**2.0 - 2.0 * self.a[0] * self.t)
            )
        )

        func = (1.0 / nadir_hemreflect) * (
            (1.0 - self.t**2.0)
            / (
                (sp.pi)
                * (1.0 + self.t**2.0 - 2.0 * self.t * self.scat_angle_symbolic) ** 1.5
            )
        )

        return func

    @property
    def legendre_coefficients(self):
        """Legendre coefficients of the BRDF."""
        nadir_hemreflect = 4 * (
            (1.0 - self.t**2.0)
            * (
                1.0
                - self.t * (-self.t + self.a[0])
                - sp.sqrt(
                    (1 + self.t**2 - 2 * self.a[0] * self.t) * (1 + self.t**2)
                )
            )
            / (
                2.0
                * self.a[0] ** 2.0
                * self.t**2.0
                * sp.sqrt(1.0 + self.t**2.0 - 2.0 * self.a[0] * self.t)
            )
        )

        n = sp.Symbol("n")
        legendre_coefficients = (1.0 / nadir_hemreflect) * (
            (1.0 / (sp.pi)) * (2.0 * n + 1) * self.t**n
        )

        return legendre_coefficients
