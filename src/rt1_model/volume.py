"""Definition of volume phase scattering functions."""

from functools import partial, update_wrapper, lru_cache

import numpy as np
import sympy as sp

from ._scatter import _Scatter, _LinComb


class _Volume(_Scatter):
    """Base class for volume scattering functions."""

    name = "RT1_Volume_base_class"
    _param_names = ["a"]

    def __init__(self, **kwargs):
        # set scattering angle generalization-matrix to [-1,1,1] if it is not
        # explicitly provided by the chosen class this results in a peak in
        # forward-direction which is suitable for describing volume-scattering
        # phase-functions
        self.a = getattr(self, "a", [-1.0, 1.0, 1.0])

        try:
            from .plot import polarplot

            # add a quick way for visualizing the functions as polarplot
            self.polarplot = partial(polarplot, X=self)
            update_wrapper(self.polarplot, polarplot)
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
            d["V_name"] = "LinComb"
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
            d["V_name"] = self.name
        return d


class LinComb(_LinComb, _Volume):
    name = "LinComb"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Isotropic(_Volume):
    """Define an isotropic scattering function."""

    name = "Isotropic"
    _param_names = []

    def __init__(self, **kwargs):
        super(Isotropic, self).__init__(**kwargs)

    @property
    def ncoefs(self):
        """The number of coefficients used to approximate the phase function."""
        # make ncoefs a property since it is fixed and should not be changed
        # only 1 coefficient is needed to correctly represent
        # the Isotropic scattering function
        return 1

    @property
    @lru_cache()
    def legcoefs(self):
        """Legendre coefficients of the phase function."""
        n = sp.Symbol("n")
        return (1.0 / (4.0 * sp.pi)) * sp.KroneckerDelta(0, n)

    @property
    @lru_cache()
    def _func(self):
        """Phase function as sympy object for later evaluation."""
        return 1.0 / (4.0 * sp.pi)


class Rayleigh(_Volume):
    """
    Rayleigh scattering function.

    Parameters
    ----------
    ncoefs : scalar(int)
             Number of coefficients used within the Legendre-approximation

    a : [ float , float , float ] , optional (default = [-1.,1.,1.])
        generalized scattering angle parameters used for defining the
        scat_angle() of the BRDF
        (http://rt1.readthedocs.io/en/latest/theory.html#equation-general_scat_angle)
    """

    name = "Rayleigh"
    _param_names = []

    def __init__(self, a=[-1.0, 1.0, 1.0], **kwargs):
        super().__init__(**kwargs)

        assert isinstance(a, list), (
            "Error: Generalization-parameter " + "needs to be a list"
        )
        assert len(a) == 3, (
            "Error: Generalization-parameter list must " + "contain 3 values"
        )
        self.a = [self._parse_sympy_param(i) for i in a]

    @property
    def ncoefs(self):
        """The number of coefficients used to approximate the BRDF."""
        # make ncoefs a property since it is fixed and should not be changed
        # only 3 coefficients are needed to correctly represent
        # the Rayleigh scattering function
        return 3

    @property
    @lru_cache()
    def _func(self):
        """Phase function as sympy object for later evaluation."""
        theta_0 = sp.Symbol("theta_0")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_ex = sp.Symbol("phi_ex")
        x = self.scat_angle(theta_0, theta_ex, phi_0, phi_ex, self.a)
        return 3.0 / (16.0 * sp.pi) * (1.0 + x**2.0)

    @property
    @lru_cache()
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


class HenyeyGreenstein(_Volume):
    """
    HenyeyGreenstein scattering function.

    Parameters
    ----------
    t : scalar(float)
        Asymmetry parameter of the Henyey-Greenstein phase function

    ncoefs : scalar(int)
             Number of coefficients used within the Legendre-approximation

    a : [ float , float , float ] , optional (default = [-1.,1.,1.])
        generalized scattering angle parameters used for defining the
        scat_angle() of the BRDF
        (http://rt1.readthedocs.io/en/latest/theory.html#equation-general_scat_angle)
    """

    name = "HenyeyGreenstein"
    _param_names = ["ncoefs", "t", "a"]

    def __init__(self, t=None, ncoefs=None, a=[-1.0, 1.0, 1.0], **kwargs):
        assert t is not None, "t parameter needs to be provided!"
        assert ncoefs is not None, "Number of coeffs needs to be specified"
        super().__init__(**kwargs)
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
        x = self.scat_angle(theta_0, theta_ex, phi_0, phi_ex, self.a)
        func = (1.0 - self.t**2.0) / (
            (4.0 * sp.pi) * (1.0 + self.t**2.0 - 2.0 * self.t * x) ** 1.5
        )

        return func

    @property
    @lru_cache()
    def legcoefs(self):
        """Legendre coefficients of the phase function."""
        n = sp.Symbol("n")
        legcoefs = (1.0 / (4.0 * sp.pi)) * (2.0 * n + 1) * self.t**n
        return legcoefs


class HGRayleigh(_Volume):
    """
    HenyeyGreenstein-Rayleigh scattering function.

        'Quanhua Liu and Fuzhong Weng: Combined henyey-greenstein and
        rayleigh phase function,
        Appl. Opt., 45(28):7475-7479, Oct 2006. doi: 10.1364/AO.45.'

    Parameters
    ----------
    t : scalar(float)
        Asymmetry parameter of the Henyey-Greenstein-Rayleigh phase function

    ncoefs : scalar(int)
             Number of coefficients used within the Legendre-approximation

    a : [ float , float , float ] , optional (default = [-1.,1.,1.])
        generalized scattering angle parameters used for defining the
        scat_angle() of the BRDF
        (http://rt1.readthedocs.io/en/latest/theory.html#equation-general_scat_angle)
    """

    name = "HGRayleigh"
    _param_names = ["ncoefs", "t", "a"]

    def __init__(self, t=None, ncoefs=None, a=[-1.0, 1.0, 1.0], **kwargs):
        assert t is not None, "t parameter needs to be provided!"
        assert ncoefs is not None, "Number of coeffs needs to be specified"
        super().__init__(**kwargs)

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
    @lru_cache()
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
