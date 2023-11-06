"""
Core module for implementation of RT1 1st order scattering model.

References
----------
Quast & Wagner (2016): doi:10.1364/AO.55.005379
"""

from functools import lru_cache, wraps
from itertools import chain
import timeit

import numpy as np
import sympy as sp
from scipy.special import expi, expn

from . import _log

try:
    # if symengine is available, use it to perform series-expansions
    from symengine import expand, Lambdify

    _init_lambda_backend = "symengine"
except ImportError:
    from sympy import expand

    _init_lambda_backend = "sympy"


def _parse_sympy_param(val):
    # convenience function to set parameters as sympy.Symbols if a string
    # was used as value
    if isinstance(val, str):
        return sp.parse_expr(val, local_dict=dict(N=sp.Symbol("N")))
    else:
        return np.atleast_1d(val)


class RT1(object):
    """
    Main class to perform RT-simulations.

    Parameters
    ----------
    I0 : scalar(float)
         Incident intensity. (Only relevant if sig0 = False)

    t_0 : array_like(float)
          Array of incident zenith-angles in radians.

    p_0 : array_like(float)
          Array of incident azimuth-angles in radians.

    t_ex : array_like(float)
           Array of exit zenith-angles in radians.

           Only relevant for bi-static geometry! For monostatic calculations
           (e.g. `geometry="mono"`), theta_ex is automatically set to t_0

    p_ex : array_like(float)
           Array of exit azimuth-angles in radians.

           Only relevant for bi-static geometry! For monostatic calculations
           (e.g. `geometry="mono"`), phi_ex is automatically set to p_0 + np.pi

    V : rt1.Volume object
        The volume-scattering phase function to use.

    SRF : rt1.Surface object
        The surface-scattering phase function (BRDF) to use.

    geometry : str (default = 'mono')
        A 4 character string specifying which components of the angles should
        be fixed or variable. This is done to significantly speed up the
        evaluation-process of the fn-coefficient generation!

        - Passing  geometry = 'mono'  indicates a monstatic measurement geometry.
          (i.e.:  t_ex = t_0, p_ex = p_0 + pi)
          Note: Inputs for `t_ex` and `p_ex` are ignored for monostatic calculations!

        - For bi-static geometry, The 4 characters represent in order the properties of:
          t_0, t_ex, p_0, p_ex

          - 'f' indicates that the angle is treated 'fixed'
            (i.e. as a numerical constant)
          - 'v' indicates that the angle is treated 'variable'
            (i.e. as a sympy-variable)

        For detailed information on the specification of the
        geometry-parameter, please have a look at the "Evaluation Geometries"
        section of the documentation:
        (http://rt1.readthedocs.io/en/latest/model_specification.html#evaluation-geometries)

    bsf : float (default = 0.)
        Fraction of bare-soil (e.g. signal with no attenuation due to vegetation).

    int_Q : bool (default = True)
        Indicator whether the interaction-term should be calculated or not.

    param_dict : dict (default = {})
        A dictionary that is used to assign numerical values to the free parameters
        of the model. (e.g. "tau", "omega", "NormBRDF" and all parameters required to
        fully specify `V` and `SRF`)

    fn_input : array_like(sympy expression), optional (default = None)
        Optional input of pre-calculated array of sympy-expressions
        to speedup calculations where the same fn-coefficients can be used.
        If None, the coefficients will be calculated automatically at the
        initialization of the RT1-object.

    fnevals_input : callable, optional (default = None)
        Optional input of pre-compiled function to numerically evaluate
        the fn_coefficients. if None, the function will be compiled
        using the fn-coefficients provided.
        Note that once the _fnevals function is provided, the
        fn-coefficients are no longer needed and have no effect on the
        calculated results!

    lambda_backend : str (default = 'symengine' if possible, else 'sympy')
        The backend that will be used to evaluate and compile functions for
        numerical evaluation of the fn-coefficients.

        Possible values are:
            - 'sympy' : sympy.lambdify is used to compile
              the _fnevals function
            - 'symengine' : symengine.LambdifyCSE is used to
              compile the _fnevals function. This results in
              considerable speedup for long fn-coefficients
    """

    def __init__(
        self,
        I0=1,
        t_0=0,
        t_ex=None,
        p_0=0,
        p_ex=None,
        V=None,
        SRF=None,
        geometry="mono",
        int_Q=True,
        param_dict=None,
        fn_input=None,
        fnevals_input=None,
        lambda_backend=_init_lambda_backend,
    ):
        assert isinstance(geometry, str), (
            "ERROR: geometry must be " + "a 4-character string"
        )
        assert len(geometry) == 4, "ERROR: geometry must be " + "a 4-character string"
        self.geometry = geometry

        self.I0 = I0

        if param_dict is None:
            param_dict = dict(bsf=0)
        self.param_dict = param_dict

        self.lambda_backend = lambda_backend
        self.int_Q = int_Q

        assert V is not None, "You must provide a volume scattering phase function!"
        self.V = V

        assert SRF is not None, "You must provide a BRDF!"
        self.SRF = SRF

        self.fn_input = fn_input
        self.fnevals_input = fnevals_input

        self.t_0 = t_0
        self.p_0 = p_0
        if self.geometry != "mono":
            self.t_ex = t_ex
            self.p_ex = p_ex

        self._bsf = "bsf"
        self._NormBRDF = "NormBRDF"
        self._tau = "tau"
        self._omega = "omega"

        try:
            from .plot import Analyze, Analyze3D

            @wraps(
                Analyze.__init__,
                assigned=("__module__", "__qualname__", "__doc__", "__annotations__"),
            )
            def analyze(*args, **kwargs):
                return Analyze(R=self, **kwargs)

            @wraps(
                Analyze3D.__init__,
                assigned=("__module__", "__qualname__", "__doc__", "__annotations__"),
            )
            def analyze3d(*args, **kwargs):
                return Analyze3D(R=self, **kwargs)

            self.analyze = analyze
            self.analyze3d = analyze3d

        except ImportError as ex:
            # to provide a useful error message in case matpltolib was not found

            def getfunc(ex):
                def f(*args, **kwargs):
                    raise ex

                return f

            self.analyze = getfunc(ex)
            self.analyze3d = getfunc(ex)

    def __getstate__(self):
        # this is required since functions created by
        # symengine are currently not pickleable!
        _log.warning("Dropping fn-coefficients to allow pickling...")
        for delkey in ["_RT1__fn"]:
            if delkey in self.__dict__:
                _log.warning(f"Removing {delkey}")
                del self.__dict__[delkey]
        for Nonekey in ["_fn_input"]:
            if Nonekey in self.__dict__:
                _log.warning(f"Setting {Nonekey} to None")
                self.__dict__[Nonekey] = None

        if self.lambda_backend == "symengine":
            _log.warning(
                "The dump of the _fnevals functions "
                + "generated by symengine will be platform-dependent!"
            )

        return self.__dict__

    @property
    def _cached_props(self):
        """A list of the names of the properties that are cached."""
        names = [
            "_mu_0",
            "_mu_0_x",
            "_mu_ex",
            "_mu_ex_x",
            "_get_param_funcs",
            "_get_param_symbs",
        ]
        return names

    def _clear_cache(self):
        type(self)._mu_0.fget.cache_clear()
        type(self)._mu_0_x.fget.cache_clear()

        type(self)._mu_ex.fget.cache_clear()
        type(self)._mu_ex_x.fget.cache_clear()

        self._get_param_symbs.cache_clear()
        self._get_param_funcs.cache_clear()

    def _cache_info(self):
        text = []
        for name in self._cached_props:
            try:
                try:
                    cinfo = getattr(self, name).cache_info()
                except Exception:
                    cinfo = getattr(type(self), name).fget.cache_info()
                text += [f"{name:<18}:   " + f"{cinfo}"]
            except Exception:
                text += [f"{name:<18}:   " + "???"]

        _log.info("\n" + "\n".join(text))

    @property
    def NormBRDF(self):
        """Normalization factor for the BRDF."""
        return self._eval_param("NormBRDF")

    @property
    def tau(self):
        """Optical depth of the volume scattering layer."""
        return self._eval_param("tau")

    @property
    def omega(self):
        """Single scattering albedo of the volume scattering layer."""
        return self._eval_param("omega")

    @property
    def bsf(self):
        """Bare soil fraction."""
        return self._eval_param("bsf")

    @property
    def t_0(self):
        """Incident zenith angle."""
        return self._t_0

    @t_0.setter
    def t_0(self, t_0):
        # if t_0 is given as scalar input, convert it to 1d numpy array
        if np.isscalar(t_0):
            t_0 = np.array([t_0])
        self._t_0 = t_0

    @property
    def p_0(self):
        """Incident azimuth angle."""
        return self._p_0

    @p_0.setter
    def p_0(self, p_0):
        # if p_o is given as scalar input, convert it to 1d numpy array
        if np.isscalar(p_0):
            p_0 = np.array([p_0])
        self._p_0 = p_0

    @property
    def t_ex(self):
        """Exit zenith angle."""
        if self.geometry == "mono":
            return self.t_0
        else:
            return self._t_ex

    @t_ex.setter
    def t_ex(self, t_ex):
        # if geometry is mono, set t_ex to t_0
        if self.geometry == "mono":
            _log.warning('t_ex is always equal to t_0 if geometry is "mono"!')
            pass
        else:
            # if t_ex is given as scalar input, convert it to 1d numpy array
            if np.isscalar(t_ex):
                t_ex = np.array([t_ex])
            self._t_ex = t_ex

    @property
    def p_ex(self):
        """Exit azimuth angle."""
        if self.geometry == "mono":
            return self.p_0 + np.pi
        else:
            return self._p_ex

    @p_ex.setter
    def p_ex(self, p_ex):
        # if geometry is mono, set p_ex to p_0
        if self.geometry == "mono":
            _log.warning('p_ex is equal to (p_0 + PI) if geometry is "mono"!')
            pass
        else:
            # if p_ex is given as scalar input, convert it to 1d numpy array
            if np.isscalar(p_ex):
                p_ex = np.array([p_ex])
            self._p_ex = p_ex

    def set_geometry(self, t_0=None, p_0=None, t_ex=None, p_ex=None, geometry="mono"):
        """
        Set the observation geometry (e.g. incidence-angles and mono/bistatic geometry).

        Parameters
        ----------
        t_0 : array_like(float)
              Array of incident zenith-angles in radians

        p_0 : array_like(float)
              Array of incident azimuth-angles in radians

        t_ex : array_like(float)
               Array of exit zenith-angles in radians
               (if geometry is 'mono', theta_ex is automatically set to t_0)

        p_ex : array_like(float)
               Array of exit azimuth-angles in radians
               (if geometry is 'mono', phi_ex is automatically set to p_0 + np.pi)

        geometry : str (default = 'vvvv')
            4 character string specifying which components of the angles should
            be fixed or variable. This is done to significantly speed up the
            evaluation-process of the fn-coefficient generation

            The 4 characters represent in order the properties of:
                t_0, t_ex, p_0, p_ex

            - 'f' indicates that the angle is treated 'fixed'
              (i.e. as a numerical constant)
            - 'v' indicates that the angle is treated 'variable'
              (i.e. as a sympy-variable)
            - Passing  geometry = 'mono'  indicates a monstatic geometry
              (i.e.:  t_ex = t_0, p_ex = p_0 + pi)
              If monostatic geometry is used, the input-values of t_ex and p_ex
              have no effect on the calculations!

            For detailed information on the specification of the
            geometry-parameter, please have a look at the "Evaluation Geometries"
            section of the documentation:
            (http://rt1.readthedocs.io/en/latest/model_specification.html#evaluation-geometries)

        """
        self._clear_cache()

        if geometry != self.geometry:
            self._fn_ = None
            self._fnevals_ = None

        self.geometry = geometry

        if t_0 is not None:
            self.t_0 = t_0
        if p_0 is not None:
            self.p_0 = p_0
        if t_ex is not None:
            self.t_ex = t_ex
        if p_ex is not None:
            self.p_ex = p_ex

    def update_params(
        self,
        # omega=None, tau=None, NormBRDF=None, bsf=None,
        **kwargs,
    ):
        """
        Update the model parameters.

        Parameters
        ----------
        omega : array-like
                The single-scattering albedo of the volume-scattering layer
        tau : array-like
              The optical depth of the volume-scattering layer
        bsf : float (default = 0.)
              fraction of bare-soil contribution (no attenuation due to vegetation)
        kwargs :
            Any additional parameters required to fully specify the model
            (e.g. variable phase-function parameters).

        """
        # if omega is not None:
        #     self._omega = _parse_sympy_param(omega)
        # if tau is not None:
        #     self._tau = _parse_sympy_param(tau)
        # if NormBRDF is not None:
        #     self._NormBRDF = _parse_sympy_param(NormBRDF)
        # if bsf is not None:
        #     self._bsf = _parse_sympy_param(bsf)

        if "bsf" not in self.param_dict:
            kwargs.setdefault("bsf", 0)

        self.param_dict.update({key: np.atleast_1d(val) for key, val in kwargs.items()})

    def calc(self, sig0=True, dB=True, **params):
        """
        Calculate model and return result.

        Perform actual calculation of bistatic scattering at top of the
        random volume (z=0) for the specified geometry. For details please
        have a look at the documentation:
        (http://rt1.readthedocs.io/en/latest/theory.html#first-order-solution-to-the-rte)

        Parameters
        ----------
        sig0 : bool
            Indicator if sigma0 (True) or intensity (False) values should be calculated.

        dB : boold
            Indicator if results are returned in dB (True) or linear units (False).
            The default is True.

        params :
            Additional parameters required to evaluate the model definition.
            (see :py:meth:`update_params`)

        Returns
        -------
        Itot : array_like(float)
               Total scattered intensity (Itot = Isurf + Ivol + Iint)

        Isurf : array_like(float)
                Surface contribution

        Ivol : array_like(float)
               Volume contribution

        Iint : array_like(float)
               Interaction contribution
        """

        self.update_params(**params)

        if isinstance(self.tau, (int, float)):
            Isurf = self.surface()
            # differentiation for non-existing canopy, as otherwise NAN values
            if self.tau > 0.0:
                Ivol = self.volume()
                if self.int_Q is True:
                    Iint = self.interaction()
                else:
                    Iint = np.array([0.0])
            else:
                Ivol = np.full_like(Isurf, 0.0)
                Iint = np.full_like(Isurf, 0.0)
        else:
            Isurf = self.surface()
            Ivol = self.volume()
            # TODO this should be fixed more properly
            # (i.e. for tau=0, no interaction-term should be calculated)
            if self.int_Q is True:
                Iint = self.interaction()
                # check if there are nan-values present that result from
                # (self.tau = 0) and replace them with 0
                wherenan = np.isnan(Iint)
                if np.any(wherenan) and np.allclose(
                    *np.broadcast_arrays(wherenan, self.tau == 0.0)
                ):
                    _log.debug(
                        "Warning replacing nan-values caused by tau=0 "
                        "in the interaction-term with 0!",
                    )
                    Iint[np.where(wherenan)] = 0.0
                else:
                    pass

        if self.int_Q is True:
            ret = self._convert_sig0_db(
                np.stack((Isurf + Ivol + Iint, Isurf, Ivol, Iint)),
                sig0=sig0,
                dB=dB,
                )
        else:
            ret = self._convert_sig0_db(
                np.stack((Isurf + Ivol + Iint, Isurf, Ivol, Iint)),
                sig0=sig0,
                dB=dB,
                )

        return ret

    def _convert_sig0_db(self, val, sig0=True, dB=True):
        if sig0 is True:
            signorm = 4. * np.pi * self._mu_0 * val
        else:
            signorm = 1.

        ret = val * signorm

        if dB is True:
            ret = 10. * np.log10(ret)

        return ret

    def surface(self):
        """
        Numerical evaluation of the surface-contribution.

        (http://rt1.readthedocs.io/en/latest/theory.html#surface_contribution)

        Returns
        -------
        array_like(float)
            Numerical value of the surface-contribution for the
            given set of parameters
        """
        # bare soil contribution
        I_bs = (
            self.I0
            * self._mu_0
            * self.SRF.brdf(
                self.t_0,
                self.t_ex,
                self.p_0,
                self.p_ex,
                param_dict=self.param_dict,
            )
        )

        Isurf = (np.exp(-(self.tau / self._mu_0) - (self.tau / self._mu_ex))) * I_bs

        return self.NormBRDF * ((1.0 - self.bsf) * Isurf + self.bsf * I_bs)

    def volume(self):
        """
        Numerical evaluation of the volume-contribution.

        (http://rt1.readthedocs.io/en/latest/theory.html#volume_contribution)

        Returns
        -------
        array_like(float)
            Numerical value of the volume-contribution for the
            given set of parameters

        """
        vol = (
            (self.I0 * self.omega * self._mu_0 / (self._mu_0 + self._mu_ex))
            * (1.0 - np.exp(-(self.tau / self._mu_0) - (self.tau / self._mu_ex)))
            * self.V.p(
                self.t_0,
                self.t_ex,
                self.p_0,
                self.p_ex,
                param_dict=self.param_dict,
            )
        )

        return (1.0 - self.bsf) * vol

    def interaction(self):
        """
        Numerical evaluation of the interaction-contribution.

        (http://rt1.readthedocs.io/en/latest/theory.html#interaction_contribution)

        Returns
        -------
        array_like(float)
            Numerical value of the interaction-contribution for
            the given set of parameters

        """
        Fint1 = self._calc_Fint_1()
        Fint2 = self._calc_Fint_2()

        Iint = (
            self.I0
            * self._mu_0
            * self.omega
            * (
                np.exp(-self.tau / self._mu_ex) * Fint1
                + np.exp(-self.tau / self._mu_0) * Fint2
            )
        )

        return self.NormBRDF * (1.0 - self.bsf) * Iint

    def _surface_volume(self, sig0=True, dB=True):
        # convenience function to get surface + volume (e.g. without interaction)

        ret = self._convert_sig0_db(
            self.surface() + self.volume(),
            sig0=sig0,
            dB=dB,
            )

        return ret

    @property
    def _fn(self):
        fn = getattr(self, "_fn_", None)
        if fn is not None:
            return fn
        elif self.fn_input is not None:
            _log.debug("Using provided fn-coefficients.")
            return self.fn_input
        elif self.int_Q is True:
            # set the fn-coefficients and generate lambdified versions
            # of the fn-coefficients for evaluation
            # only evaluate fn-coefficients if _fnevals funcions are not
            # already available!
            _log.info("Evaluating coefficients for interaction-term...")

            tic = timeit.default_timer()
            # precalculate the expansiion coefficients for the interaction term
            expr_int = self._calc_interaction_expansion()
            toc = timeit.default_timer()
            _log.debug("Expansion calculated, it took " + str(toc - tic) + " sec.")

            # extract the expansion coefficients
            tic = timeit.default_timer()
            self._fn_ = self._extract_coefficients(expr_int)
            toc = timeit.default_timer()
            _log.info(f"Coefficients extracted, it took {toc - tic:.5f} sec.")
            return self._fn_
        else:
            return None

    @property
    def _fnevals(self):
        fnevals = getattr(self, "_fnevals_", None)
        if fnevals is not None:
            return fnevals
        elif self.fnevals_input is not None:
            _log.debug("Using provided fnevals-functions.")
            return self.fnevals_input
        elif self.int_Q is True:
            _log.debug("Generation of fnevals functions...")

            tic = timeit.default_timer()

            # define new lambda-functions for each fn-coefficient
            variables = sp.var(
                ("theta_0", "phi_0", "theta_ex", "phi_ex")
                + tuple(map(str, self.param_dict.keys()))
            )

            # use symengine's Lambdify if symengine has been used within
            # the fn-coefficient generation
            if self.lambda_backend == "symengine":
                _log.debug("Symengine set as backend.")
                # using symengines own "common subexpression elimination"
                # routine to perform lambdification

                # llvm backend is used to allow pickling of the functions
                # see https://github.com/symengine/symengine.py/issues/294
                self._fnevals_ = Lambdify(
                    list(variables),
                    self._fn,
                    order="F",
                    cse=True,
                    backend="llvm",
                )
                return self._fnevals_
            elif self.lambda_backend == "sympy":
                # using sympy's lambdify without "common subexpression
                # elimination" to perform lambdification

                _log.debug("Sympy set as backend.")

                sympy_fn = list(map(sp.sympify, self._fn))

                self._fnevals_ = sp.lambdify(
                    (variables),
                    sp.sympify(sympy_fn),
                    modules=["numpy", "sympy"],
                    dummify=False,
                )

                self._fnevals_.__doc__ = """
                                    A function to numerically evaluate the
                                    fn-coefficients a for given set of
                                    incidence angles and parameter-values
                                    as defined in the param_dict dict.

                                    The call-signature is:
                                        RT1-object._fnevals(theta_0, phi_0, \
                                        theta_ex, phi_ex, *param_dict.values())
                                    """

                return self._fnevals_
            else:
                raise TypeError(
                    'Lambda_backend "' + self.lambda_backend + '" is not available',
                )

            toc = timeit.default_timer()
            _log.debug(
                "Lambdification finished, it took " + str(toc - tic) + " sec.",
            )

        else:
            return None

    def _extract_coefficients(self, expr):
        """
        Extract Fn coefficients from given forumula.

        This is done by collecting the terms of expr with respect to powers
        of cos(theta_s) and simplifying the gained coefficients by applying
        a simple trigonometric identity.

        Parameters
        ----------
        expr : sympy expression
               prepared sympy-expression to be used for extracting
               the fn-coefficients (output of _calc_interaction_expansion())

        Returns
        -------
        fn : list(sympy expressions)
             A list of sympy expressions that represent the fn-coefficients
             associated with the given input-equation (expr).

        """
        theta_s = sp.Symbol("theta_s")

        N_fn = self.SRF.ncoefs + self.V.ncoefs - 1

        fn = []

        # find f_0 coefficient
        repl0 = dict([[sp.cos(theta_s), 0]])
        fn = fn + [expr.xreplace(repl0)]

        # find f_1 coefficient
        repl1 = dict(
            [[sp.cos(theta_s) ** i, 0] for i in list(range(N_fn, 0, -1)) if i != 1]
            + [[sp.cos(theta_s), 1]]
        )
        fn = fn + [expr.xreplace(repl1) - fn[0]]

        for n in np.arange(2, N_fn, dtype=int):
            repln = dict([[sp.cos(theta_s) ** int(n), 1]])
            fn = fn + [(expr.xreplace(repln)).xreplace(repl0) - fn[0]]

        return fn

    def _calc_interaction_expansion(self):
        """
        Evaluate the polar-integral from the definition of the fn-coefficients.

        (http://rt1.readthedocs.io/en/latest/theory.html#equation-fn_coef_definition)

        The approach is as follows:

            1. Expand the Legrende coefficents of the surface and volume
               phase functions
            2. Apply the function _integrate_0_2pi_phis() to evaluate
               the integral
            3. Replace remaining sin(theta_s) terms in the Legrende polynomials
               by cos(theta_s) to prepare for fn-coefficient extraction
            4. Expand again to ensure that a fully expanded expression
               is returned (to be used as input in _extract_coefficients() )

        Returns
        -------
        res : sympy expression
              A fully expanded expression that can be used as
              input for _extract_coefficients()

        """
        # preevaluate expansions for volume and surface phase functions
        # this returns symbolic code to be then further used

        volexp = self.V.legexpansion(
            self.t_0, self.t_ex, self.p_0, self.p_ex, self.geometry
        ).doit()

        brdfexp = self.SRF.legexpansion(
            self.t_0, self.t_ex, self.p_0, self.p_ex, self.geometry
        ).doit()

        # preparation of the product of p*BRDF for coefficient retrieval
        # this is the eq.23. and would need to be integrated from 0 to 2pi
        fPoly = expand(2 * sp.pi * volexp * brdfexp)

        # do integration of eq. 23
        expr = self._integrate_0_2pi_phis(fPoly)

        # now we do still simplify the expression to be able to express
        # things as power series of cos(theta_s)
        theta_s = sp.Symbol("theta_s")
        replacements = [
            (
                sp.sin(theta_s) ** i,
                expand((1.0 - sp.cos(theta_s) ** 2) ** sp.Rational(i, 2)),
            )
            for i in range(1, self.SRF.ncoefs + self.V.ncoefs - 1)
            if i % 2 == 0
        ]

        res = expand(expr.xreplace(dict(replacements)))

        return res

    def _cosintegral(self, i):
        """
        Analytical solution to the integral of cos(x)**i in the inteVal 0 ... 2*pi.

        Parameters
        ----------
        i : scalar(int)
            Power of the cosine function to be integrated, i.e.  cos(x)^i

        Returns
        -------
        - : float
              Numerical value of the integral of cos(x)^i
              in the inteVal 0 ... 2*pi

        """
        if i % 2 == 0:
            return (2 ** (-i)) * sp.binomial(i, i * sp.Rational(1, 2))
        else:
            # for odd exponents result is always zero
            return 0.0

    def _integrate_0_2pi_phis(self, expr):
        """
        Symbolic power-series integration.

        Perforn symbolic integration of a pre-expanded power-series
        in sin(phi_s) and cos(phi_s) over the variable phi_s
        in the inteVal 0 ... 2*pi

        The approach is as follows:

            1. Replace all appearing sin(phi_s)^odd with 0 since the
               integral vanishes
            2. Replace all remaining sin(phi_s)^even with their representation
               in terms of cos(phi_s)
            3. Replace all cos(phi_s)^i terms with _cosintegral(i)
            4. Expand the gained solution for further processing

        Parameters
        ----------
        expr : sympy expression
               pre-expanded product of the legendre-expansions of
               V.legexpansion() and SRF.legexpansion()

        Returns
        -------
        res : sympy expression
              resulting symbolic expression that results from integrating
              expr over the variable phi_s in the inteVal 0 ... 2*pi

        """
        phi_s = sp.Symbol("phi_s")

        # replace first all odd powers of sin(phi_s) as these are
        # all zero for the integral
        replacements1 = [
            (sp.sin(phi_s) ** i, 0.0)
            for i in range(1, self.SRF.ncoefs + self.V.ncoefs + 1)
            if i % 2 == 1
        ]

        # then substitute the sine**2 by 1-cos**2
        replacements1 = replacements1 + [
            (
                sp.sin(phi_s) ** i,
                expand((1.0 - sp.cos(phi_s) ** 2) ** sp.Rational(i, 2)),
            )
            for i in range(2, self.SRF.ncoefs + self.V.ncoefs + 1)
            if i % 2 == 0
        ]

        res = expand(expr.xreplace(dict(replacements1)))

        # replacements need to be done simultaneously, otherwise all
        # remaining sin(phi_s)**even will be replaced by 0

        # integrate the cosine terms
        replacements3 = [
            (sp.cos(phi_s) ** i, self._cosintegral(i))
            for i in range(1, self.SRF.ncoefs + self.V.ncoefs + 1)
        ]

        res = expand(res.xreplace(dict(replacements3)))
        return res

    @property
    @lru_cache()
    def _mu_0(self):
        return np.cos(self.t_0)

    @property
    @lru_cache()
    def _mu_ex(self):
        return np.cos(self.t_ex)

    @property
    def _all_param_symbs(self):
        return list(
            chain(
                *(
                    self._get_param_symbs(key)
                    for key in ("tau", "omega", "NormBRDF", "bsf")
                )
            )
        )

    @lru_cache()
    def _get_param_symbs(self, param):
        """Symbols used to define tau, omega, NormBRDF and bsf."""
        try:
            expr = sp.parse_expr(
                getattr(self, f"_{param}"), local_dict=dict(N=sp.Symbol("N"))
            )

            symbs = list(map(str, expr.free_symbols))
        except Exception:
            symbs = [param]
        return symbs

    @lru_cache()
    def _get_param_funcs(self, param):
        """Lambdified functions used to define tau, omega, NormBRDF and bsf."""
        if not isinstance(getattr(self, f"_{param}"), (str, sp.Basic)):
            return None

        try:
            func = sp.lambdify(
                self._get_param_symbs(param),
                getattr(self, f"_{param}"),
                modules=["numpy"],
            )
        except Exception:
            func = None
        return func

    def _eval_param(self, param):
        """
        Numerical evaluation of tau, omega, NormBRDF and bsf.

        All required parameters must be provided in `param_dict`!
        """
        func = self._get_param_funcs(param)

        if func is not None:
            return self._get_param_funcs(param)(
                **{
                    key: val
                    for key, val in self.param_dict.items()
                    if key in self._get_param_symbs(param)
                }
            )
        else:
            return getattr(self, f"_{param}")

    def _S2_mu(self, mu, tau):
        nmax = self.V.ncoefs + self.SRF.ncoefs - 1
        hlp1 = (
            np.exp(-tau / mu) * np.log(mu / (1.0 - mu))
            - expi(-tau)
            + np.exp(-tau / mu) * expi(tau / mu - tau)
        )

        # cache the results of the expn-evaluations to speed up the S2 loop
        @lru_cache()
        def innerfunc(k):
            return mu ** (-k) * (expn(k + 1.0, tau) - np.exp(-tau / mu) / k)

        S2 = np.array(
            [sum(innerfunc(k) for k in range(1, (n + 1) + 1)) for n in range(nmax)]
        )

        # clear the cache since tau might have changed!
        innerfunc.cache_clear()

        return S2 + hlp1

    def _calc_Fint_1(self):
        """
        Numerical evaluation of the F_int() function.

        (used in the definition of the interaction-contribution)
        (http://rt1.readthedocs.io/en/latest/theory.html#F_int)

        Returns
        -------
        S : array_like(float)
            Numerical value of F_int for the given set of parameters

        """
        mu1, mu2, phi1, phi2 = self._mu_0, self._mu_ex, self.p_0, self.p_ex

        # evaluate fn-coefficients
        if self.lambda_backend == "symengine":
            args = np.broadcast_arrays(
                np.arccos(mu1),
                phi1,
                np.arccos(mu2),
                phi2,
                *self.param_dict.values(),
            )

            # to correct for 0 dimensional arrays if a fn-coefficient
            # is identical to 0 (in a symbolic manner)
            fn = np.broadcast_arrays(*self._fnevals(args))
        else:
            args = np.broadcast_arrays(
                np.arccos(mu1),
                phi1,
                np.arccos(mu2),
                phi2,
                *self.param_dict.values(),
            )
            # to correct for 0 dimensional arrays if a fn-coefficient
            # is identical to 0 (in a symbolic manner)
            fn = np.broadcast_arrays(*self._fnevals(*args))

        multip = self._mu_0_x * self._S2_mu(mu1, self.tau)
        S = np.sum(fn * multip, axis=0)
        return S

    def _calc_Fint_2(self):
        """
        Numerical evaluation of the F_int() function.

        (used in the definition of the interaction-contribution)
        (http://rt1.readthedocs.io/en/latest/theory.html#F_int)

        Returns
        -------
        S : array_like(float)
            Numerical value of F_int for the given set of parameters

        """
        mu1, mu2, phi1, phi2 = self._mu_ex, self._mu_0, self.p_ex, self.p_0

        # evaluate fn-coefficients
        if self.lambda_backend == "symengine":
            args = np.broadcast_arrays(
                np.arccos(mu1),
                phi1,
                np.arccos(mu2),
                phi2,
                *self.param_dict.values(),
            )

            # to correct for 0 dimensional arrays if a fn-coefficient
            # is identical to 0 (in a symbolic manner)
            fn = np.broadcast_arrays(*self._fnevals(args))
        else:
            args = np.broadcast_arrays(
                np.arccos(mu1),
                phi1,
                np.arccos(mu2),
                phi2,
                *self.param_dict.values(),
            )
            # to correct for 0 dimensional arrays if a fn-coefficient
            # is identical to 0 (in a symbolic manner)
            fn = np.broadcast_arrays(*self._fnevals(*args))

        multip = self._mu_ex_x * self._S2_mu(mu1, self.tau)
        S = np.sum(fn * multip, axis=0)
        return S

    @property
    @lru_cache()
    def _mu_0_x(self):
        # pre-evaluate and cache required cosine powers
        nmax = self.V.ncoefs + self.SRF.ncoefs - 1
        mux = np.array([self._mu_0 ** (n + 1) for n in range(nmax)])
        return mux

    @property
    @lru_cache()
    def _mu_ex_x(self):
        # pre-evaluate and cache required cosine powers
        nmax = self.V.ncoefs + self.SRF.ncoefs - 1
        mux = np.array([self._mu_ex ** (n + 1) for n in range(nmax)])
        return mux

    # %% derivatives

    def _dvolume_dtau(self):
        """
        Get the derivative of the volume-contribution with respect to tau.

        Returns
        -------
        dvdt : array_like(float)
               Numerical value of dIvol/dtau for the given set of parameters

        """
        dvdt = (
            self.I0
            * self.omega
            * (self._mu_0 / (self._mu_0 + self._mu_ex))
            * (
                (1.0 / self._mu_0 + 1.0 / self._mu_ex ** (-1))
                * np.exp(-self.tau / self._mu_0 - self.tau / self._mu_ex)
            )
            * self.V.p(self.t_0, self.t_ex, self.p_0, self.p_ex, self.param_dict)
        )

        return (1.0 - self.bsf) * dvdt

    def _dvolume_domega(self):
        """
        Get the derivative of the volume-contribution with respect to omega.

        Returns
        -------
        dvdo : array_like(float)
               Numerical value of dIvol/domega for the given set of parameters

        """
        dvdo = (
            (self.I0 * self._mu_0 / (self._mu_0 + self._mu_ex))
            * (1.0 - np.exp(-(self.tau / self._mu_0) - (self.tau / self._mu_ex)))
            * self.V.p(self.t_0, self.t_ex, self.p_0, self.p_ex, self.param_dict)
        )

        return (1.0 - self.bsf) * dvdo

    def _dvolume_dbsf(self):
        """
        Get the derivative of the volume-contribution with respect to bsf.

        Returns
        -------
        dvdo : array_like(float)
               Numerical value of dIvol/dbsf for the given set of parameters

        """
        vol = (
            (self.I0 * self.omega * self._mu_0 / (self._mu_0 + self._mu_ex))
            * (1.0 - np.exp(-(self.tau / self._mu_0) - (self.tau / self._mu_ex)))
            * self.V.p(
                self.t_0,
                self.t_ex,
                self.p_0,
                self.p_ex,
                param_dict=self.param_dict,
            )
        )

        return -vol

    def _dvolume_dR(self):
        """
        Get the derivative of the volume-contribution with respect to NormBRDF.

        Returns
        -------
        dvdr : array_like(float)
               Numerical value of dIvol/dNormBRDF for the given set of parameters

        """
        dvdr = 0.0

        return dvdr

    def _dsurface_dtau(self):
        """
        Get the derivative of the surface-contribution with respect to tau.

        Returns
        -------
        dsdt : array_like(float)
               Numerical value of dIsurf/dtau for the given set of parameters

        """
        dsdt = (
            self.I0
            * (-1.0 / self._mu_0 - 1.0 / self._mu_ex)
            * np.exp(-self.tau / self._mu_0 - self.tau / self._mu_ex)
            * self._mu_0
            * self.SRF.brdf(self.t_0, self.t_ex, self.p_0, self.p_ex, self.param_dict)
        )

        # Incorporate BRDF-normalization factor
        dsdt = self.NormBRDF * (1.0 - self.bsf) * dsdt

        return dsdt

    def _dsurface_domega(self):
        """
        Get the derivative of the surface-contribution with respect to omega.

        Returns
        -------
        dsdo : array_like(float)
               Numerical value of dIsurf/domega for the given set of parameters

        """
        dsdo = 0.0

        return dsdo

    def _dsurface_dR(self):
        """
        Get the derivative of the surface-contribution with respect to NormBRDF.

        Returns
        -------
        dsdr : array_like(float)
               Numerical value of dIsurf/dNormBRDF for the given set of parameters

        """
        I_bs = (
            self.I0
            * self._mu_0
            * self.SRF.brdf(
                self.t_0,
                self.t_ex,
                self.p_0,
                self.p_ex,
                param_dict=self.param_dict,
            )
        )

        Isurf = (np.exp(-(self.tau / self._mu_0) - (self.tau / self._mu_ex))) * I_bs

        return (1.0 - self.bsf) * Isurf + self.bsf * I_bs

    def _dsurface_dbsf(self):
        """
        Numerical evaluation of the surface-contribution.

        (http://rt1.readthedocs.io/en/latest/theory.html#surface_contribution)

        Returns
        -------
        array_like(float)
            Numerical value of the surface-contribution for the
            given set of parameters

        """
        # bare soil contribution
        I_bs = (
            self.I0
            * self._mu_0
            * self.SRF.brdf(
                self.t_0,
                self.t_ex,
                self.p_0,
                self.p_ex,
                param_dict=self.param_dict,
            )
        )

        Isurf = (
            (np.exp(-(self.tau / self._mu_0) - (self.tau / self._mu_ex)))
            * I_bs
            * np.ones_like(self.t_0)
        )

        return self.NormBRDF * (I_bs - Isurf)

    @lru_cache(20)
    def _d_surface_dummy_lambda(self, key):
        """
        Get a function to compute direct surface-contribution parameter derivatives.

        A cached lambda-function for computing the derivative of the surface-function
        with respect to a given parameter.

        Parameters
        ----------
        key : str
            the parameter to use.

        Returns
        -------
        callable
            A function that calculates the derivative with respect to key.

        """
        theta_0 = sp.Symbol("theta_0")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_ex = sp.Symbol("phi_ex")

        args = (theta_0, theta_ex, phi_0, phi_ex) + tuple(self.param_dict.keys())

        return sp.lambdify(
            args,
            sp.diff(self.SRF._func, sp.Symbol(key)),
            modules=["numpy", "sympy"],
        )

    @lru_cache(20)
    def _d_volume_dummy_lambda(self, key):
        """
        Get a function to compute direct volume-contribution parameter derivatives.

        A cached lambda-function for computing the derivative of the volume-function
        with respect to a given parameter.

        Parameters
        ----------
        key : str
            the parameter to use.

        Returns
        -------
        callable
            A function that calculates the derivative with respect to key.

        """
        theta_0 = sp.Symbol("theta_0")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_ex = sp.Symbol("phi_ex")

        args = (theta_0, theta_ex, phi_0, phi_ex) + tuple(self.param_dict.keys())

        return sp.lambdify(
            args,
            sp.diff(self.V._func, sp.Symbol(key)),
            modules=["numpy", "sympy"],
        )

    def _d_surface_ddummy(self, key):
        """
        Surface contribution derivative with respect to a given parameter (incl. bsf).

        Parameters
        ----------
        key : The parameter to use

        Returns
        -------
        array_like(float)
            Numerical value of dIsurf/dkey for the given set of parameters

        """
        dI_bs = (
            self.I0
            * self._mu_0
            * self._d_surface_dummy_lambda(key)(
                self.t_0, self.t_ex, self.p_0, self.p_ex, **self.param_dict
            )
        )

        dI_s = (np.exp(-(self.tau / self._mu_0) - (self.tau / self._mu_ex))) * dI_bs

        return self.NormBRDF * ((1.0 - self.bsf) * dI_s + self.bsf * dI_bs)

    def _d_volume_ddummy(self, key):
        """
        Volume contribution derivative with respect to a given parameter (incl. bsf).

        Parameters
        ----------
        key : The parameter to use

        Returns
        -------
        array_like(float)
            Numerical value of dIvol/dkey for the given set of parameters

        """
        dIvol = (
            self.I0
            * self.omega
            * self._mu_0
            / (self._mu_0 + self._mu_ex)
            * (1.0 - np.exp(-(self.tau / self._mu_0) - (self.tau / self._mu_ex)))
            * self._d_volume_dummy_lambda(key)(
                self.t_0, self.t_ex, self.p_0, self.p_ex, **self.param_dict
            )
        )
        return (1.0 - self.bsf) * dIvol

    def jacobian(self, dB=True, sig0=True, param_list=["omega", "tau", "NormBRDF"]):
        """
        Return the jacobian of the total backscatter (without interaction contribution).

        (With respect to the parameters provided in param_list.)
        (default: param_list = ['omega', 'tau', 'NormBRDF'])

        The jacobian can be evaluated for measurements in linear or dB
        units, and for either intensity- or sigma_0 values.

        Note:
            The contribution of the interaction-term is currently
            NOT considered in the calculation of the jacobian!

        Parameters
        ----------
        dB : boolean (default = False)
             Indicator whether linear or dB units are used.
             The applied relation is given by:

             dI_dB(x)/dx =
             10 / [log(10) * I_linear(x)] * dI_linear(x)/dx

        sig0 : boolean (default = False)
               Indicator wheather intensity- or sigma_0-values are used
               The applied relation is given by:

               sig_0 = 4 * pi * cos(inc) * I

               where inc denotes the incident zenith-angle and I is the
               corresponding intensity
        param_list : list
                     a list of strings that correspond to the parameters
                     for which the jacobian should be evaluated.

                     possible values are: 'omega', 'tau' 'NormBRDF' and
                     any string corresponding to a sympy.Symbol used in the
                     definition of V or SRF

        Returns
        -------
        jac : array-like(float)
              The jacobian of the total backscatter with respect to
              omega, tau and NormBRDF

        """
        if sig0 is True and dB is False:
            norm = 4.0 * np.pi * np.cos(self.t_0)
        elif dB is True:
            norm = 10.0 / (np.log(10.0) * (self.surface() + self.volume()))
        else:
            norm = 1.0

        jac = []
        for key in param_list:
            if key == "omega":
                jac += [(self._dsurface_domega() + self._dvolume_domega()) * norm]
            elif key == "tau":
                jac += [(self._dsurface_dtau() + self._dvolume_dtau()) * norm]
            elif key == "NormBRDF":
                jac += [(self._dsurface_dR() + self._dvolume_dR()) * norm]
            elif key == "bsf":
                jac += [(self._dsurface_dbsf() + self._dvolume_dbsf()) * norm]
            elif key in self.param_dict:
                jac += [
                    (self._d_surface_ddummy(key) + self._d_volume_ddummy(key)) * norm
                ]
            else:
                assert False, (
                    "error in jacobian calculation... "
                    + str(key)
                    + " is not in param_dict"
                )

        return jac
