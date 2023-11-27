"""General object for scattering distribution functions."""

from functools import lru_cache, wraps, partial, update_wrapper

import sympy as sp
from . import _log
from .helpers import _lambdify, _parse_sympy_param


class _Scatter:
    """The base object for any Surface and Volume objects."""

    @property
    def ncoefs(self):
        """The number of coefficients used in the legendre expansion."""
        if not hasattr(self, "_ncoefs") or self._ncoefs is None:
            raise AttributeError(
                "You must specify the number of approximation coefficients for "
                f"a {self.__class__.__name__} scattering function to calculate "
                "first-order corrections! (or use `RT1(..., int_Q=False)` "
                "to omit calculating the interaction-term."
            )
        return self._ncoefs

    def scat_angle(self, t_0, t_ex, p_0, p_ex, a):
        """
        Generalized scattering angle with respect to the given zenith-angles.

        (http://rt1.readthedocs.io/en/latest/theory.html#equation-general_scat_angle)

        Standard-choices assigned in the volume- and surface class:

        - Surface: a=[ 1,1,1] ... pi-shifted scattering angle
          cos[t_0]*cos[t_ex] + sin[t_0]*sin[t_ex]*cos[p_0 - p_ex]
        - Volume:  a=[-1,1,1] ... ordinary scattering angle
          -cos[t_0]*cos[t_ex] + sin[t_0]*sin[t_ex]*cos[p_0 - p_ex]

        .. note::
            The scattering angle is defined with respect to the incident
            zenith-angle t_0, and not with respect to the incidence-angle in
            a spherical coordinate system (t_i)! The two definitions are
            related via t_i = pi - t_0

        Parameters
        ----------
        t_0 : array_like(float)
              incident zenith-angle
        p_0 : array_like(float)
              incident azimuth-angle
        t_ex : array_like(float)
               exit zenith-angle
        p_ex : array_like(float)
               exit azimuth-angle
        a : [ float , float , float ]
            generalized scattering angle parameters

        Returns
        -------
        float
              the generalized scattering angle

        """
        return (
            a[0] * sp.cos(t_0) * sp.cos(t_ex)
            + a[1] * sp.sin(t_0) * sp.sin(t_ex) * sp.cos(p_0) * sp.cos(p_ex)
            + a[2] * sp.sin(t_0) * sp.sin(t_ex) * sp.sin(p_0) * sp.sin(p_ex)
        )

    @lru_cache()
    def _lambda_func(self, *args):
        # define sympy objects
        theta_0 = sp.Symbol("theta_0")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_ex = sp.Symbol("phi_ex")

        # replace arguments and evaluate expression
        args = (theta_0, theta_ex, phi_0, phi_ex) + tuple(args)
        pfunc = _lambdify(args, self._func)

        # TODO check requirement for this!
        # if _func is a constant, lambdify will create a function that returns a scalar
        # which is not suitable for further processing. in that case, vectorize the
        # obtained function

        # TODO maybe find a better check for this
        # if self._func.is_constant():   # this is too slow!
        # if len(self._func.free_symbols) == 0:
        #     pfunc = np.vectorize(pfunc)

        return pfunc

    def legexpansion(self, t_0, t_ex, p_0, p_ex):
        """
        Legendre-expansion of the scattering distribution function.

        .. note::
            The incidence-angle argument of the legexpansion() is different
            to the documentation due to the direct definition of the argument
            as the zenith-angle (t_0) instead of the incidence-angle
            defined in a spherical coordinate system (t_i).
            They are related via: t_i = pi - t_0


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
        sympy - expression
            The Legendre expansion of the scattering distribution function.

        """
        assert self.ncoefs > 0

        n = sp.Symbol("n")

        return sp.Sum(
            self.legcoefs
            * sp.legendre(n, self.scat_angle(t_0, t_ex, p_0, p_ex, self.a)),
            (n, 0, self.ncoefs - 1),
        )

    def calc(self, t_0, t_ex, p_0, p_ex, param_dict={}):
        """
        Calculate numerical value of the scattering function.

        Parameters
        ----------
        t_0 : array_like(float)
            Incident zenith-angles in radians
        p_0 : array_like(float)
            Incident azimuth-angles in radians
        t_ex : array_like(float)
            Exit zenith-angles in radians
        p_ex : array_like(float)
            Exit azimuth-angles in radians
        param_dict: dict
            A dict of parameter-values (or arrays of values) that are required
            to fully specify the scattering function.

        Returns
        -------
        array_like(float)
            Numerical value of the BRDF

        """
        func = self._lambda_func(*param_dict.keys())

        return func(t_0, t_ex, p_0, p_ex, **param_dict)

    def _register_plotfuncs(self):
        # register quick-plot functions for distribution functions

        try:
            from .plot import polarplot, hemreflect
            from .surface import SurfaceScatter

            # quick way for visualizing the functions as polarplot
            self.polarplot = partial(polarplot, V_SRF=self)
            update_wrapper(self.polarplot, polarplot)

            if isinstance(self, SurfaceScatter):
                # quick way for visualizing the associated hemispherical reflectance
                self.hemreflect = partial(hemreflect, SRF=self)
                update_wrapper(self.hemreflect, hemreflect)
        except ImportError:
            _log.debug("Unable to register plotting functions.", exc_info=True)
            pass


class _LinComb(_Scatter):
    def __init__(self, choices=None, **kwargs):
        """
        Class to generate linear-combinations of scattering distribution functions.

        For details please look at the documentation
        (http://rt1.readthedocs.io/en/latest/model_specification.html#linear-combination-of-scattering-distributions)

        Parameters
        ----------
        choices : [ (float, ScatterObject) , (float, ScatterObject) ,  ...]
            A list that contains the the individual scattering functions
            and the associated weighting-factors for the linear-combination.

            The weights can be either numerical values or strings
            (which will be parsed as sympy expressions)

        Examples
        --------
        Defining linear-combinations of volume- or surface scattering distributions
        works completely similar:

        >>> from rt1_model import volume
        >>> V = volume.LinComb([(0.5, volume.Isotropic()), (0.5, volume.Rayleigh())])

        >>> from rt1_model import surface
        >>> V = surface.LinComb([(0.5, surface.Isotropic()),
                                 (0.5, surface.HenyeyGreenstein(t="t", ncoefs=10))])

        You can also use expressions for the weights!

        >>> from rt1_model import surface
        >>> V = surface.LinComb([("x", surface.Isotropic()),
                                 ("1 - x", surface.HenyeyGreenstein(t="t", ncoefs=10))])

        """
        self._weights, self._objs = [], []
        for w, o in choices:
            # cast weights passed as strings to sympy symbols
            self._weights.append(_parse_sympy_param(w))
            self._objs.append(o)

        # group weights and functions with respect to the "a" parameter
        # {a1 : [(w1, f1), (w2, f2), ...], a2 : [(w3, f3), (w4, f4), ...]}
        self._a_groups = dict()
        for frac, func in zip(self._weights, self._objs):
            self._a_groups.setdefault(tuple(func.a), []).append((frac, func))

        # this must be done at the end so that _objs and _weights are properly defined!
        super().__init__(**kwargs)

    @property
    def _func(self):
        """Phase function as sympy object for later evaluation."""
        _func = 0
        for c, o in zip(self._weights, self._objs):
            _func += c * o._func

        return _func

    @property
    def ncoefs(self):
        # set ncoefs of the combined scattering function to the maximum
        # number of coefficients within the chosen functions.
        # (this is necessary for correct evaluation of fn-coefficients)
        return max([o.ncoefs for o in self._objs])

    @property
    def legcoefs(self):
        raise NotImplementedError(
            "Legendre coefficients of linear combinations are not defined. "
            "Use `.legexpansion(...)` to get the combined Legendre expansion!"
        )

    @wraps(_Scatter.legexpansion)
    def legexpansion(self, t_0, t_ex, p_0, p_ex):
        # evaluate the combined legendre expansion
        n = sp.Symbol("n")

        exp = 0
        for a, choices in self._a_groups.items():
            # get max. ncoefs for each a-parameter group
            ncoefs = max(i[1].ncoefs for i in choices)
            # sum up legendre coefficients
            legcoefs = sum(frac * func.legcoefs for frac, func in choices)

            exp += sp.Sum(
                legcoefs * sp.legendre(n, self.scat_angle(t_0, t_ex, p_0, p_ex, a)),
                (n, 0, ncoefs - 1),
            )

        return exp
