"""General object for scattering distribution functions."""

from functools import lru_cache

import sympy as sp
import numpy as np


class _Scatter(object):
    """The base object for any Surface and Volume objects."""

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

    def _parse_sympy_param(self, val):
        # convenience function to set parameters as sympy.Symbols if a string
        # was used as value
        if isinstance(val, str):
            return sp.parse_expr(val)
        else:
            return val

    @lru_cache()
    def _lambda_func(self, *args):
        # define sympy objects
        theta_0 = sp.Symbol("theta_0")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_ex = sp.Symbol("phi_ex")

        # replace arguments and evaluate expression
        # sp.lambdify is used to allow array-inputs
        args = (theta_0, theta_ex, phi_0, phi_ex) + tuple(args)
        pfunc = sp.lambdify(args, self._func, modules=["numpy", "sympy"])

        # if _func is a constant, lambdify will create a function that returns a scalar
        # which is not suitable for further processing. in that case, vectorize the
        # obtained function

        # TODO maybe find a better check for this
        #if self._func.is_constant():   # this is too slow!
        if len(self._func.free_symbols) == 0:
            pfunc = np.vectorize(pfunc)

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
        brdffunc = self._lambda_func(*param_dict.keys())

        return brdffunc(t_0, t_ex, p_0, p_ex, **param_dict)

class _LinComb(_Scatter):
    """
    Class to generate linear-combinations of scattering distribution functions.

    For details please look at the documentation
    (http://rt1.readthedocs.io/en/latest/model_specification.html#linear-combination-of-scattering-distributions)

    Parameters
    ----------
    choices : [ [float, ScatterObject] , [float, ScatterObject] ,  ...]
        A list that contains the the individual scattering functions
        and the associated weighting-factors (floats) for the linear-combination.

    """
    name = "LinComb"
    _param_names = ["choices"]

    def __init__(self, choices=None, **kwargs):
        super().__init__(**kwargs)
        # cast fractions passed as strings to sympy symbols
        self.choices = [(self._parse_sympy_param(i), j) for i, j in choices]

        self._comb = self._combine()
        self._set_legexpansion()

        name = "LinComb"
        for c in self.choices:
            name += f"_({c[0]}, {c[1].name})"
        self.name = name


    @property
    @lru_cache()
    def _func(self):
        """Phase function as sympy object for later evaluation."""
        return self._comb._func

    def _set_legexpansion(self):
        """Set legexpansion to the combined legexpansion."""
        self.ncoefs = self._comb.ncoefs
        self.legexpansion = self._comb.legexpansion

    def _combine(self):
        """
        Get a combined Surface object based on an input-array of Surface objects.

        The array must be shaped in the form:
            choices = [  [ weighting-factor   ,   Surface-class element ],
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

        class _Dummy(_Scatter):
            """Dummy-class used to generate linear-combinations of BRDFs."""

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

                self._func = 0.0
                self.legcoefs = 0.0

        # initialize a combined phase-function class element
        comb = _Dummy()
        # set ncoefs of the combined volume-class element to the maximum
        comb.ncoefs = max([i[1].ncoefs for i in self.choices])
        #   number of coefficients within the chosen functions.
        #   (this is necessary for correct evaluation of fn-coefficients)

        # find BRDF functions with equal a parameters
        equals = [
            np.where(
                (np.array([cc[1].a for cc in self.choices]) == tuple(c[1].a)).all(
                    axis=1
                )
            )[0]
            for c in self.choices
        ]

        # evaluate index of functions that have equal a parameter

        # find phase functions where a-parameter is equal
        equal_a = list({tuple(row) for row in equals})

        # evaluation of combined expansion in legendre-polynomials
        dummylegexpansion = []
        for i in range(0, len(equal_a)):
            dummy = _Dummy()
            # select SRF choices where a parameter is equal
            equals = np.take(self.choices, equal_a[i], axis=0)
            # set ncoefs to the maximum number within the choices
            # with equal a-parameter
            dummy.ncoefs = max([SRF[1].ncoefs for SRF in equals])
            # loop over phase-functions with equal a-parameter
            for eq in equals:
                # set parameters based on chosen phase-functions and evaluate
                # combined legendre-expansion
                dummy.a = eq[1].a
                dummy._func = dummy._func + eq[1]._func * eq[0]
                dummy.legcoefs += eq[1].legcoefs * eq[0]

            dummylegexpansion = dummylegexpansion + [dummy.legexpansion]

        # combine legendre-expansions for each a-parameter based on given
        # combined legendre-coefficients
        comb.legexpansion = lambda t_0, t_ex, p_0, p_ex: np.sum(
            [lexp(t_0, t_ex, p_0, p_ex) for lexp in dummylegexpansion]
        )

        for c in self.choices:
            # set parameters based on chosen classes to define analytic
            # function representation
            comb._func = comb._func + c[1]._func * c[0]
        return comb