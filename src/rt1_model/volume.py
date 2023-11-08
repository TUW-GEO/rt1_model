"""Definition of volume phase scattering functions."""

from functools import partial, update_wrapper, lru_cache

import numpy as np
import sympy as sp

from ._scatter import _Scatter


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
        if self.name.startswith("LinCombV"):
            d = dict()
            for key in self._param_names:
                val = self.__dict__[key]
                if isinstance(val, sp.Basic):
                    d[key] = str(val)
                else:
                    d[key] = val
            d["V_name"] = "LinCombV"
            srfchoices = []
            for frac, srf in d["Vchoices"]:
                if isinstance(frac, sp.Basic):
                    srfchoices.append([str(frac), srf.init_dict])
                else:
                    srfchoices.append([frac, srf.init_dict])

            d["Vchoices"] = srfchoices
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

    def p(self, t_0, t_ex, p_0, p_ex, param_dict={}):
        """
        Evaluate the phase-function for chosen incidence- and exit angles.

        Parameters
        ----------
        t_0 : array_like(float)
              array of incident zenith-angles in radians

        p_0 : array_like(float)
              array of incident azimuth-angles in radians

        t_ex : array_like(float)
               array of exit zenith-angles in radians

        p_ex : array_like(float)
               array of exit azimuth-angles in radians

        Returns
        -------
        array_like(float)
            Numerical value of the volume-scattering phase-function

        """
        # if an explicit numeric function is provided, use it, otherwise
        # lambdify the available sympy-function
        if hasattr(self, "_func_numeric"):
            pfunc = self._func_numeric
        else:
            pfunc = self._lambda_func(*param_dict.keys())
        # in case _func is a constant, lambdify will produce a function with
        # scalar output which is not suitable for further processing
        # (this happens e.g. for the Isotropic brdf).
        # The following query is implemented to ensure correct array-output:
        # TODO this is not a proper test !
        if not isinstance(
            pfunc(
                np.array([0.1, 0.2, 0.3]),
                0.1,
                0.1,
                0.1,
                **{key: 0.12 for key in param_dict.keys()},
            ),
            np.ndarray,
        ):
            pfunc = np.vectorize(pfunc)

        return pfunc(t_0, t_ex, p_0, p_ex, **param_dict)


class LinCombV(_Volume):
    """
    Class to generate linear-combinations of volume-class elements.

    For details please look at the documentation
    (http://rt1.readthedocs.io/en/latest/model_specification.html#linear-combination-of-scattering-distributions)

    .. note::
        Since the normalization of a volume-scattering phase-function is fixed,
        the weighting-factors must equate to 1!

    Parameters
    ----------
    Vchoices : [ [float, Volume]  ,  [float, Volume]  ,  ...]
               a list that contains the the individual phase-functions
               (Volume-objects) and the associated weighting-factors
               (floats) of the linear-combination.
    """

    name = "LinCombV"
    _param_names = ["Vchoices"]

    def __init__(self, Vchoices=None, **kwargs):
        super().__init__(**kwargs)

        # cast fractions passed as strings to sympy symbols
        self.Vchoices = [(self._parse_sympy_param(i), j) for i, j in Vchoices]

        self._set_legexpansion()

        name = "LinCombV"
        for c in Vchoices:
            name += f"_({c[0]}, {c[1].name})"
        self.name = name

    @property
    @lru_cache()
    def _func(self):
        """Phase function as sympy object for later evaluation."""
        return self._Vcombiner()._func

    def _set_legexpansion(self):
        """Set legexpansion to the combined legexpansion."""
        self.ncoefs = self._Vcombiner().ncoefs
        self.legexpansion = self._Vcombiner().legexpansion

    def _Vcombiner(self):
        """
        Get a combined Volume object based on an input-array of Volume objects.

        The array must be shaped in the form:
            Vchoices = [  [ weighting-factor   ,   Volume-class element ]  ,
                          [ weighting-factor   ,   Volume-class element ]  ,
                          ...]

        In order to keep the normalization of the phase-functions correct,
        the sum of the weighting factors must equate to 1!

        Attention
        ---------
            the .legexpansion()-function of the combined volume-class
            element is no longer related to its legcoefs (which are set to 0.)
            since the individual legexpansions of the combined volume-class
            elements are possibly evaluated with a different a-parameter
            of the generalized scattering angle! This does not affect any
            calculations, since the evaluation is exclusively based on the
            use of the .legexpansion()-function.
        """

        class Phasefunction(_Volume):
            """Dummy-class used to generate linear-combinations of phase functions."""

            def __init__(self, **kwargs):
                super(Phasefunction, self).__init__(**kwargs)
                self._func = 0.0
                self.legcoefs = 0.0

        # find phase functions with equal a parameters
        equals = [
            np.where(
                (np.array([VV[1].a for VV in self.Vchoices]) == tuple(V[1].a)).all(
                    axis=1
                )
            )[0]
            for V in self.Vchoices
        ]

        # evaluate index of phase-functions that have equal a parameter
        equal_a = list({tuple(row) for row in equals})

        # initialize a combined phase-function class element

        # initialize the combined phase-function
        Vcomb = Phasefunction()
        # set ncoefs of the combined volume-class element to the maximum
        Vcomb.ncoefs = max([V[1].ncoefs for V in self.Vchoices])
        #   number of coefficients within the chosen functions.
        #   (this is necessary for correct evaluation of fn-coefficients)

        # evaluation of combined expansion in legendre-polynomials
        dummylegexpansion = []
        for i in range(0, len(equal_a)):
            Vdummy = Phasefunction()
            # select V choices where a parameter is equal
            Vequal = np.take(self.Vchoices, equal_a[i], axis=0)
            # set ncoefs to the maximum number within the choices with
            # equal a-parameter
            Vdummy.ncoefs = max([V[1].ncoefs for V in Vequal])
            # loop over phase-functions with equal a-parameter
            for V in Vequal:
                # set parameters based on chosen phase-functions and evaluate
                # combined legendre-expansion
                Vdummy.a = V[1].a
                Vdummy._func = Vdummy._func + V[1]._func * V[0]
                Vdummy.legcoefs = Vdummy.legcoefs + V[1].legcoefs * V[0]

            dummylegexpansion = dummylegexpansion + [Vdummy.legexpansion]

        # combine legendre-expansions for each a-parameter based on given
        # combined legendre-coefficients
        Vcomb.legexpansion = lambda t_0, t_ex, p_0, p_ex: np.sum(
            [lexp(t_0, t_ex, p_0, p_ex) for lexp in dummylegexpansion]
        )

        for V in self.Vchoices:
            # set parameters based on chosen classes to define analytic
            # function representation
            Vcomb._func = Vcomb._func + V[1]._func * V[0]

        return Vcomb


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

    # def _func_numeric(self, theta_0, theta_ex, phi_0, phi_ex, **kwargs):
    #     """Direct numeric version of _func."""
    #     if isinstance(self.t, sp.Symbol):
    #         t = kwargs[str(self.t)]
    #     else:
    #         t = self.t
    #     x = self._scat_angle_numeric(theta_0, theta_ex, phi_0, phi_ex, self.a)
    #     func = (1.0 - t**2.0) / (
    #         (4.0 * np.pi) * (1.0 + t**2.0 - 2.0 * t * x) ** 1.5
    #     )

    #     return func

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
