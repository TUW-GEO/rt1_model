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

    def _scat_angle_numeric(self, t_0, t_ex, p_0, p_ex, a):
        # a numeric version of scat_angle
        return (
            a[0] * np.cos(t_0) * np.cos(t_ex)
            + a[1] * np.sin(t_0) * np.sin(t_ex) * np.cos(p_0) * np.cos(p_ex)
            + a[2] * np.sin(t_0) * np.sin(t_ex) * np.sin(p_0) * np.sin(p_ex)
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
