"""Definition of surface scattering functions (BRDF)."""

from functools import partial, update_wrapper, lru_cache

import numpy as np
import sympy as sp

from ._scatter import _Scatter


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
        if self.name.startswith("LinCombSRF"):
            d = dict()
            for key in self._param_names:
                val = self.__dict__[key]
                if isinstance(val, sp.Basic):
                    d[key] = str(val)
                else:
                    d[key] = val
            d["SRF_name"] = "LinCombSRF"
            srfchoices = []
            for frac, srf in d["SRFchoices"]:
                if isinstance(frac, sp.Basic):
                    srfchoices.append([str(frac), srf.init_dict])
                else:
                    srfchoices.append([frac, srf.init_dict])

            d["SRFchoices"] = srfchoices
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

class LinCombSRF(_Surface):
    """
    Class to generate linear-combinations of volume-class elements.

    For details please look at the documentation
    (http://rt1.readthedocs.io/en/latest/model_specification.html#linear-combination-of-scattering-distributions)

    Parameters
    ----------
    SRFchoices : [ [float, Surface]  ,  [float, Surface]  ,  ...]
                 A list that contains the the individual BRDF's
                 (Surface-objects) and the associated weighting-factors
                 (floats) for the linear-combination.
    """

    name = "LinCombSRF"
    _param_names = ["SRFchoices"]

    def __init__(self, SRFchoices=None, **kwargs):
        super(LinCombSRF, self).__init__(**kwargs)

        # cast fractions passed as strings to sympy symbols
        self.SRFchoices = [(self._parse_sympy_param(i), j) for i, j in SRFchoices]

        self._set_legexpansion()

        name = "LinCombSRF"
        for c in self.SRFchoices:
            name += f"_({c[0]}, {c[1].name})"
        self.name = name

    @property
    @lru_cache()
    def _func(self):
        """Phase function as sympy object for later evaluation."""
        return self._SRFcombiner()._func

    def _set_legexpansion(self):
        """Set legexpansion to the combined legexpansion."""
        self.ncoefs = self._SRFcombiner().ncoefs
        self.legexpansion = self._SRFcombiner().legexpansion

    def _SRFcombiner(self):
        """
        Get a combined Surface object based on an input-array of Surface objects.

        The array must be shaped in the form:
            SRFchoices = [  [ weighting-factor   ,   Surface-class element ],
                            [ weighting-factor   ,   Surface-class element ],
                        ...]

        ATTENTION: the .legexpansion()-function of the combined surface-class
        element is no longer related to its legcoefs (which are set to 0.)
                   since the individual legexpansions of the combined surface-
                   class elements are possibly evaluated with a different
                   a-parameter of the generalized scattering angle! This does
                   not affect any calculations, since the evaluation is
                   only based on the use of the .legexpansion()-function.
        """

        class BRDFfunction(_Surface):
            """Dummy-class used to generate linear-combinations of BRDFs."""

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

                self._func = 0.0
                self.legcoefs = 0.0

        # initialize a combined phase-function class element
        SRFcomb = BRDFfunction()
        # set ncoefs of the combined volume-class element to the maximum
        SRFcomb.ncoefs = max([SRF[1].ncoefs for SRF in self.SRFchoices])
        #   number of coefficients within the chosen functions.
        #   (this is necessary for correct evaluation of fn-coefficients)

        # find BRDF functions with equal a parameters
        equals = [
            np.where(
                (np.array([VV[1].a for VV in self.SRFchoices]) == tuple(V[1].a)).all(
                    axis=1
                )
            )[0]
            for V in self.SRFchoices
        ]

        # evaluate index of BRDF-functions that have equal a parameter

        # find phase functions where a-parameter is equal
        equal_a = list({tuple(row) for row in equals})

        # evaluation of combined expansion in legendre-polynomials
        dummylegexpansion = []
        for i in range(0, len(equal_a)):
            SRFdummy = BRDFfunction()
            # select SRF choices where a parameter is equal
            SRFequal = np.take(self.SRFchoices, equal_a[i], axis=0)
            # set ncoefs to the maximum number within the choices
            # with equal a-parameter
            SRFdummy.ncoefs = max([SRF[1].ncoefs for SRF in SRFequal])
            # loop over phase-functions with equal a-parameter
            for SRF in SRFequal:
                # set parameters based on chosen phase-functions and evaluate
                # combined legendre-expansion
                SRFdummy.a = SRF[1].a
                SRFdummy._func = SRFdummy._func + SRF[1]._func * SRF[0]
                SRFdummy.legcoefs += SRF[1].legcoefs * SRF[0]

            dummylegexpansion = dummylegexpansion + [SRFdummy.legexpansion]

        # combine legendre-expansions for each a-parameter based on given
        # combined legendre-coefficients
        SRFcomb.legexpansion = lambda t_0, t_ex, p_0, p_ex: np.sum(
            [lexp(t_0, t_ex, p_0, p_ex) for lexp in dummylegexpansion]
        )

        for SRF in self.SRFchoices:
            # set parameters based on chosen classes to define analytic
            # function representation
            SRFcomb._func = SRFcomb._func + SRF[1]._func * SRF[0]
        return SRFcomb


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
