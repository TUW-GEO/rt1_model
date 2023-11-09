"""Definition of surface scattering functions (BRDF)."""

from functools import partial, update_wrapper, lru_cache

import numpy as np
import sympy as sp

from ._scatter import _Scatter, _LinComb


class _Surface(_Scatter):
    """Base class for surface scattering functions."""

    name = "RT1_Surface_base_class"
    _param_names = ["a"]

    def __init__(self, **kwargs):
        # set scattering angle generalization-matrix to [1,1,1] if it is not
        # explicitly provided by the chosen class.
        # this results in a peak in specular-direction which is suitable
        # for describing surface BRDF's
        self.a = getattr(self, "a", [1.0, 1.0, 1.0])

        try:
            from .plot import polarplot, hemreflect

            # quick way for visualizing the functions as polarplot
            self.polarplot = partial(polarplot, X=self)
            update_wrapper(self.polarplot, polarplot)
            # quick way for visualizing the associated hemispherical reflectance
            self.hemreflect = partial(hemreflect, SRF=self)
            update_wrapper(self.hemreflect, hemreflect)
        except ImportError:
            pass

    def __repr__(self):
        try:
            return (
                self.name
                + "("
                + (",\n" + " " * (len(self.name) + 1)).join(
                    [f"{param}={getattr(self, param)}" for param in self._param_names]
                )
                + ")"
            )
        except Exception:
            return object.__repr__(self)

    @property
    def init_dict(self):
        """Get a dict that can be used to initialize the BRDF."""
        if self.name.startswith("LinComb"):
            d = dict()
            for key in self._param_names:
                val = self.__dict__[key]
                if isinstance(val, sp.Basic):
                    d[key] = str(val)
                else:
                    d[key] = val
            d["SRF_name"] = "LinComb"
            srfchoices = []
            for frac, srf in d["choices"]:
                if isinstance(frac, sp.Basic):
                    srfchoices.append([str(frac), srf.init_dict])
                else:
                    srfchoices.append([frac, srf.init_dict])

            d["choices"] = srfchoices
        else:
            d = dict()
            for key in self._param_names:
                val = self.__dict__[key]
                if isinstance(val, sp.Basic):
                    d[key] = str(val)
                else:
                    d[key] = val
            d["SRF_name"] = self.name
        return d


class LinComb(_LinComb, _Surface):
    # docstring hinherited
    name = "LinComb"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Isotropic(_Surface):
    """Define an isotropic surface brdf."""

    name = "Isotropic"
    _param_names = []

    def __init__(self, **kwargs):
        super(Isotropic, self).__init__(**kwargs)

    @property
    def ncoefs(self):
        """The number of coefficients used to approximate the BRDF."""
        # make ncoefs a property since it is fixed and should not be changed
        # only 1 coefficient is needed to correctly represent
        # the Isotropic scattering function
        return 1

    @property
    @lru_cache()
    def legcoefs(self):
        """Legendre coefficients of the BRDF."""
        n = sp.Symbol("n")
        return (1.0 / sp.pi) * sp.KroneckerDelta(0, n)

    @property
    @lru_cache()
    def _func(self):
        """Phase function as sympy object for later evaluation."""
        return 1.0 / sp.pi


class CosineLobe(_Surface):
    """
    Define a (possibly generalized) cosine-lobe of power i.

    Parameters
    ----------
    i : scalar(int)
        Power of the cosine lobe, i.e. cos(x)^i

    ncoefs : scalar(int)
             Number of coefficients used within the Legendre-approximation

    a : [ float , float , float ] , optional (default = [1.,1.,1.])
        generalized scattering angle parameters used for defining the
        scat_angle() of the BRDF
        (http://rt1.readthedocs.io/en/latest/theory.html#equation-general_scat_angle)
    """

    name = "CosineLobe"
    _param_names = ["i", "ncoefs", "a"]

    def __init__(self, ncoefs=None, i=None, a=[1.0, 1.0, 1.0], **kwargs):
        assert ncoefs is not None, (
            "Error: number of coefficients " + "needs to be provided!"
        )
        assert i is not None, "Error: Cosine lobe power needs to be specified!"
        super(CosineLobe, self).__init__(**kwargs)
        assert ncoefs > 0
        self.i = i
        assert isinstance(self.i, int), (
            "Error: Cosine lobe power needs " + "to be an integer!"
        )
        assert i >= 0, "ERROR: Power of Cosine-Lobe needs to be greater than 0"
        assert isinstance(a, list), (
            "Error: Generalization-parameter " + "needs to be a list"
        )
        assert len(a) == 3, (
            "Error: Generalization-parameter list must " + "contain 3 values"
        )

        self.a = [self._parse_sympy_param(i) for i in a]
        self.ncoefs = int(ncoefs)

    @property
    @lru_cache()
    def legcoefs(self):
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
    @lru_cache()
    def _func(self):
        """Phase function as sympy object for later evaluation."""
        theta_0 = sp.Symbol("theta_0")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_ex = sp.Symbol("phi_ex")

        # self._func = sp.Max(self.scat_angle(theta_i,
        #                                    theta_s,
        #                                    phi_i,
        #                                    phi_s,
        #                                    a=self.a), 0.)**self.i  # eq. A13

        # alternative formulation avoiding the use of sp.Max()
        #     (this is done because   sp.lambdify('x',sp.Max(x), "numpy")
        #      generates a function that can not interpret array inputs.)
        x = self.scat_angle(theta_0, theta_ex, phi_0, phi_ex, a=self.a)
        return 1.0 / sp.pi * (x * (1.0 + sp.sign(x)) / 2.0) ** self.i


class HenyeyGreenstein(_Surface):
    """
    HenyeyGreenstein scattering function for use as BRDF.

    Parameters
    ----------
    t : scalar(float)
        Asymmetry parameter of the Henyey-Greenstein function

    ncoefs : scalar(int)
             Number of coefficients used within the Legendre-approximation

    a : [ float , float , float ] , optional (default = [1.,1.,1.])
        generalized scattering angle parameters used for defining the
        scat_angle() of the BRDF
        (http://rt1.readthedocs.io/en/latest/theory.html#equation-general_scat_angle)
    """

    name = "HenyeyGreenstein"
    _param_names = ["t", "ncoefs", "a"]

    def __init__(self, t=None, ncoefs=None, a=[1.0, 1.0, 1.0], **kwargs):
        assert t is not None, "t parameter needs to be provided!"
        assert ncoefs is not None, "Number of coeff. needs to be specified"
        super(HenyeyGreenstein, self).__init__(**kwargs)

        assert isinstance(a, list), (
            "Error: Generalization-parameter " + "needs to be a list"
        )
        assert len(a) == 3, (
            "Error: Generalization-parameter list must " + "contain 3 values"
        )

        self.t = self._parse_sympy_param(t)
        self.a = [self._parse_sympy_param(i) for i in a]

        self.ncoefs = ncoefs
        assert self.ncoefs > 0

    @property
    @lru_cache()
    def _func(self):
        """Phase function as sympy object for later evaluation."""
        theta_0 = sp.Symbol("theta_0")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_ex = sp.Symbol("phi_ex")

        x = self.scat_angle(theta_0, theta_ex, phi_0, phi_ex, a=self.a)

        return (
            1.0
            * (1.0 - self.t**2.0)
            / ((sp.pi) * (1.0 + self.t**2.0 - 2.0 * self.t * x) ** 1.5)
        )

    @property
    @lru_cache()
    def legcoefs(self):
        """Legendre coefficients of the BRDF."""
        n = sp.Symbol("n")
        return 1.0 * (1.0 / (sp.pi)) * (2.0 * n + 1) * self.t**n


class HG_nadirnorm(_Surface):
    """
    Nadir-normalized HenyeyGreenstein scattering function.

    Parameters
    ----------
    t : scalar(float)
        Asymmetry parameter of the Henyey-Greenstein function

    ncoefs : scalar(int)
             Number of coefficients used within the Legendre-approximation

    a : [ float , float , float ] , optional (default = [1.,1.,1.])
        generalized scattering angle parameters used for defining the
        scat_angle() of the BRDF
        (http://rt1.readthedocs.io/en/latest/theory.html#equation-general_scat_angle)
    """

    name = "HG_nadirnorm"
    _param_names = ["t", "ncoefs", "a"]

    def __init__(self, t=None, ncoefs=None, a=[1.0, 1.0, 1.0], **kwargs):
        assert t is not None, "t parameter needs to be provided!"
        assert ncoefs is not None, "Number of coeffs needs to be specified"
        super(HG_nadirnorm, self).__init__(**kwargs)

        assert isinstance(a, list), (
            "Error: Generalization-parameter " + "needs to be a list"
        )
        assert len(a) == 3, (
            "Error: Generalization-parameter list must " + "contain 3 values"
        )

        self.t = self._parse_sympy_param(t)
        self.a = [self._parse_sympy_param(i) for i in a]

        self.ncoefs = ncoefs
        assert self.ncoefs > 0

        self._param_names = ["t", "ncoefs", "a"]

    @property
    @lru_cache()
    def _func(self):
        """Define phase function as sympy object for later evaluation."""
        theta_0 = sp.Symbol("theta_0")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_ex = sp.Symbol("phi_ex")

        x = self.scat_angle(theta_0, theta_ex, phi_0, phi_ex, a=self.a)

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
            / ((sp.pi) * (1.0 + self.t**2.0 - 2.0 * self.t * x) ** 1.5)
        )

        return func

    @property
    @lru_cache()
    def legcoefs(self):
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
        legcoefs = (1.0 / nadir_hemreflect) * (
            (1.0 / (sp.pi)) * (2.0 * n + 1) * self.t**n
        )

        return legcoefs
