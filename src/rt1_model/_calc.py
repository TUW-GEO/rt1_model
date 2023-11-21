"""
Core module for implementation of RT1 1st order scattering model.

References
----------
Quast & Wagner (2016): doi:10.1364/AO.55.005379
Quast, Albergel, Calvet, Wagner (2019) : doi:10.3390/rs11030285

"""

from functools import lru_cache, wraps
from itertools import chain
import timeit

import numpy as np
import sympy as sp
from scipy.special import expi, expn
from scipy.sparse import vstack, block_diag, csr_matrix

from . import _log

try:
    # if symengine is available, use it to perform series-expansions
    from symengine import expand, Lambdify

    _init_lambda_backend = "symengine"

except ImportError:
    from sympy import expand

    _init_lambda_backend = "sympy"


def set_lambda_backend(lambda_backend):
    """
    Set the backend that will be used to evaluate symbolic expressions.

    Parameters
    ----------
    lambda_backend : str (default = 'symengine' if possible, else 'sympy')
        The backend that will be used to evaluate and compile functions for
        numerical evaluation of symbolic expressions.

        Possible values are:
            - 'sympy' : sympy.lambdify is used to compile functions with numpy and scipy
            - 'symengine' : symengine.LambdifyCSE is used to compile functions.
              This results in considerable speedup for more complex model calculations!

    """
    global _init_lambda_backend
    assert lambda_backend in [
        "sympy",
        "symengine",
    ], f"Lambda backend {lambda_backend} is not defined!"
    _init_lambda_backend = lambda_backend

    _log.debug("Backend set to {lambda_backend}")


_local_variable_symbols = dict(N=sp.Symbol("N"))


def _parse_sympy_param(val):
    # convenience function to set parameters as sympy.Symbols if a string
    # was used as value
    if isinstance(val, str):
        return sp.parse_expr(val, local_dict=_local_variable_symbols)
    else:
        return val


def _lambdify(variables, functions):
    """
    Lambdify a list of functions with the selected lambda_backend.

    Parameters
    ----------
    variables : list
        A list of strings or sympy.Symbols defining the function variables.
    functions : list
        A list of sympy expressions or strings that can be parsed as expressions.
    """
    # lambdify provided functions with sympy or symengine and unify call signature

    var = []
    for v in np.atleast_1d(variables):
        if isinstance(v, str):
            var.append(sp.Symbol(v))
        else:
            var.append(v)

    funcs = []
    for f in np.atleast_1d(functions):
        if isinstance(f, str):
            funcs.append(sp.parse_expr(f, local_dict=_local_variable_symbols))
        else:
            funcs.append(f)

    # make sure that we don't add additional axes to the returned dataset
    # if a single function is evaluated
    if len(funcs) == 1 and not isinstance(functions, list):
        funcs = funcs[0]

    # use symengine's Lambdify if symengine has been used within
    # the fn-coefficient generation
    if _init_lambda_backend == "symengine":
        # using symengines own "common subexpression elimination"
        # routine to perform lambdification

        # llvm backend is used to allow pickling of the functions
        # see https://github.com/symengine/symengine.py/issues/294
        seng_lambda_func = Lambdify(
            var,
            funcs,
            order="F",
            cse=True,
            backend="llvm",
        )

        var_names = list(map(str, variables))

        def lambda_func(*args, **kwargs):
            # allow similar call signature as sympy functions
            # (e.g. both args and kwargs)
            return seng_lambda_func(
                *np.broadcast_arrays(
                    *args, *(kwargs[key] for key in var_names[len(args) :])
                )
            )

    elif _init_lambda_backend == "sympy":
        # using sympy's lambdify without "common subexpression
        # elimination" to perform lambdification

        lambda_func = sp.lambdify(
            var,
            funcs,
            modules=["numpy", "sympy"],
            dummify=False,
        )
    else:
        raise TypeError(f"lambda_backend {_init_lambda_backend} is not available")

    return lambda_func


class RT1(object):
    """
    Main class to perform RT-simulations.

    Parameters
    ----------
    V : :py:class:`rt1_model.volume.VolumeScatter` object
        The volume-scattering phase function to use.
    SRF : :py:class:`rt1_model.volume.SurfaceScatter` object
        The surface-scattering phase function (BRDF) to use.
    sig0 : bool
        Indicator if sigma0 (True) or intensity (False) values should be calculated.
    dB : bool
        Indicator if results are returned in dB (True) or linear units (False).
        The default is True.
    int_Q : bool (default = True)
        Indicator whether the interaction-term should be calculated or not.
    fn_input : array_like(sympy expressions), optional (default = None)
        Optional input of pre-calculated array of sympy-expressions
        to speedup calculations where the same fn-coefficients can be used.
        If None, the coefficients will be calculated automatically at the
        initialization of the RT1-object.
    fnevals_input : callable, optional (default = None)
        Optional input of pre-compiled function to numerically evaluate
        the fn_coefficients. if None, the function will be compiled
        using the fn-coefficients provided.
        Note that once the fnevals functions are provided, the
        fn-coefficients are no longer needed and have no effect on the
        calculated results!
    I0 : array-like (float)
         Incident intensity. (Only relevant if sig0 = False)
         The default is 1.

    Attributes
    ----------
    param_dict : dict
        A dictionary that is used to assign numerical values to the free parameters
        of the model. (e.g. "tau", "omega", "NormBRDF", "bsf" and all parameters
        required to fully specify `V` and `SRF`). The default is {"bsf": 0}.
        Use :py:meth:`update_params` to update the used parameters!

    """

    # list of names of cached properties
    _cached_properties = [
        "_mu_0",
        "_mu_0_x",
        "_mu_ex",
        "_mu_ex_x",
    ]

    # list of names of cached functions
    _cached_functions = [
        "_get_param_funcs",
        "_get_param_symbs",
        "_d_volume_dummy_lambda",
        "_d_surface_dummy_lambda",
    ]

    def __init__(
        self,
        V=None,
        SRF=None,
        int_Q=True,
        sig0=True,
        dB=True,
        fn_input=None,
        fnevals_input=None,
        I0=1.0,
    ):
        assert V is not None, "You must provide a volume scattering phase function!"
        self.V = V

        assert SRF is not None, "You must provide a BRDF!"
        self.SRF = SRF

        self.int_Q = int_Q

        self.fn_input = fn_input
        self.fnevals_input = fnevals_input

        self.sig0 = sig0
        self.dB = dB

        self.I0 = I0

        # default parameter names
        self._bsf = "bsf"
        self._NormBRDF = "NormBRDF"
        self._tau = "tau"
        self._omega = "omega"

        self._param_dict = dict(bsf=0)

        # set default geometry
        self._geometry = "mono"
        self.set_geometry(t_0=np.pi / 4, p_0=0)

        self._register_plotfuncs()

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

        if _init_lambda_backend == "symengine":
            _log.warning(
                "The dump of the _fnevals functions "
                + "generated by symengine will be platform-dependent!"
            )

        # drop plot-functions (they are not pickleable and re-attached on unpickle)
        for key in ["analyze", "analyze3d"]:
            self.__dict__[key] = None

        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._register_plotfuncs()

    def _register_plotfuncs(self):
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
            def _getf(ex):
                def f(*args, **kwargs):
                    raise ex

            self.analyze = _getf(ex)
            self.analyze3d = _getf(ex)

    def _clear_cache(self, *keys):
        if keys:
            props = (i for i in keys if i in self._cached_properties)
            funcs = (i for i in keys if i in self._cached_functions)
        else:
            props = self._cached_properties
            funcs = self._cached_functions

        for key in props:
            getattr(type(self), key).fget.cache_clear()

        for key in funcs:
            getattr(self, key).cache_clear()

    def _clear_param_cache(self):
        # clear chaced functions & properties that might change if a parameter is set
        self._clear_cache("_get_param_funcs", "_get_param_symbs")

    def _clear_geom_cache(self):
        # clear chaced functions & properties that might change if geometry is set
        self._clear_cache("_mu_0", "_mu_0_x", "_mu_ex", "_mu_ex_x")

    def _cache_info(self):
        text = []

        for name in self._cached_functions:
            try:
                cinfo = getattr(self, name).cache_info()
                text += [f"{name:<18}:   " + f"{cinfo}"]
            except Exception:
                text += [f"{name:<18}:   " + "???"]

        for name in self._cached_properties:
            try:
                cinfo = getattr(type(self), name).fget.cache_info()
                text += [f"{name:<18}:   " + f"{cinfo}"]
            except Exception:
                text += [f"{name:<18}:   " + "???"]

        _log.info("\n" + "\n".join(text))

    @property
    def param_dict(self):
        """Dictionary holding the numerical values assigned to the model parameters."""
        return self._param_dict

    @property
    def NormBRDF(self):
        """Normalization factor for the surface BRDF."""
        return self._eval_param("NormBRDF")

    @NormBRDF.setter
    def NormBRDF(self, value):
        self._clear_param_cache()
        self._NormBRDF = value

    @property
    def tau(self):
        """Optical depth of the volume scattering layer."""
        return self._eval_param("tau")

    @tau.setter
    def tau(self, value):
        self._clear_param_cache()
        self._tau = value

    @property
    def omega(self):
        """Single scattering albedo of the volume scattering layer."""
        return self._eval_param("omega")

    @omega.setter
    def omega(self, value):
        self._clear_param_cache()
        self._omega = value

    @property
    def bsf(self):
        """Fraction of bare-soil (e.g. signal with no attenuation due to vegetation)."""
        return self._eval_param("bsf")

    @bsf.setter
    def bsf(self, value):
        self._clear_param_cache()
        self._bsf = value

    @property
    def t_0(self):
        """Incident zenith angle."""
        return self._t_0

    @t_0.setter
    def t_0(self, t_0):
        self._clear_geom_cache()
        # if t_0 is given as scalar input, convert it to 1d numpy array
        if np.isscalar(t_0):
            t_0 = np.array([t_0])
        self._t_0 = t_0

    @property
    @lru_cache()
    def _mu_0(self):
        return np.cos(self.t_0)

    @property
    def p_0(self):
        """Incident azimuth angle."""
        return self._p_0

    @p_0.setter
    def p_0(self, p_0):
        self._clear_geom_cache()
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
            self._clear_geom_cache()

            # if t_ex is given as scalar input, convert it to 1d numpy array
            if np.isscalar(t_ex):
                t_ex = np.array([t_ex])
            self._t_ex = t_ex

    @property
    @lru_cache()
    def _mu_ex(self):
        return np.cos(self.t_ex)

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
            self._clear_geom_cache()

            # if p_ex is given as scalar input, convert it to 1d numpy array
            if np.isscalar(p_ex):
                p_ex = np.array([p_ex])
            self._p_ex = p_ex

    @property
    def geometry(self):
        """
        The geometry for which the model should be evaluated.

        A 4 character string specifying which components of the angles should
        be fixed or variable. This is done to significantly speed up the
        evaluation-process of the fn-coefficient generation.

        The 4 characters represent in order the properties of:
            t_0, t_ex, p_0, p_ex

        - 'f' indicates that the angle is treated 'fixed'
          (i.e. treated as a numerical constant)
        - 'v' indicates that the angle is treated 'variable'
          (i.e. treated as a sympy-variable)
        - Passing  geometry = 'mono'  indicates a monstatic geometry
          (i.e.:  t_ex = t_0, p_ex = p_0 + pi)
          If monostatic geometry is used, the input-values of t_ex and p_ex
          have no effect on the calculations!

        For detailed information on the specification of the
        geometry-parameter, please have a look at the "Evaluation Geometries"
        section of the documentation:
        (http://rt1.readthedocs.io/en/latest/model_specification.html#evaluation-geometries)

        """
        return self._geometry

    @geometry.setter
    def geometry(self, geometry):
        assert (
            isinstance(geometry, str) and len(geometry) == 4
        ), "ERROR: geometry must be a 4-character string!"

        if geometry != self._geometry:
            self._fn_ = None
            self._fnevals_ = None
            self._clear_geom_cache()

        self._geometry = geometry

    def set_geometry(self, t_0=None, p_0=None, t_ex=None, p_ex=None, geometry=None):
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

        geometry : str (default = 'mono')
            The geometry for which the model should be evaluated.

            A 4 character string specifying which components of the angles should
            be fixed or variable. This is done to significantly speed up the
            evaluation-process of the fn-coefficient generation.

            The 4 characters represent in order the properties of:
                t_0, t_ex, p_0, p_ex

            - 'f' indicates that the angle is treated 'fixed'
              (i.e. treated as a numerical constant)
            - 'v' indicates that the angle is treated 'variable'
              (i.e. treated as a sympy-variable)
            - Passing  geometry = 'mono'  indicates a monstatic geometry
              (i.e.:  t_ex = t_0, p_ex = p_0 + pi)
              If monostatic geometry is used, the input-values of t_ex and p_ex
              have no effect on the calculations!

            For detailed information on the specification of the
            geometry-parameter, please have a look at the "Evaluation Geometries"
            section of the documentation:
            (http://rt1.readthedocs.io/en/latest/model_specification.html#evaluation-geometries)

        """
        if geometry is not None:
            self.geometry = geometry

        if t_0 is not None:
            self.t_0 = t_0
        if p_0 is not None:
            self.p_0 = p_0
        if t_ex is not None:
            self.t_ex = t_ex
        if p_ex is not None:
            self.p_ex = p_ex

    def update_params(self, **kwargs):
        """
        Update the numerical values used for the model parameters.

        These values will be used as default when calling `R.calc()`
        to compute the model!

        The default parameter names are:

        - "omega" : The single-scattering albedo of the volume-scattering layer
        - "tau" : The optical depth of the volume-scattering layer
        - "NormBRDF" : Normalization factor for the surface BRDF
        - "bsf" : Fraction of bare-soil contribution (no attenuation due to vegetation)

        The currently set values can be accessed via :py:attr:`RT1.param_dict`.

        Parameters
        ----------
        kwargs :
            Any parameters required to fully specify the model (e.g. omega, tau,
            NormBRDF, bsf and all required scattering distribution parameters).

        Examples
        --------
        >>> R = RT1(V=volume.Rayleigh(),
        >>>         SRF=surface.HenyeyGreenstein(t="t", ncoefs=10))
        >>> R.update_params(omega=0.2, tau=0.3, t=0.3, NormBRDF=0.2)

        """
        # if bsf is not explicitly provided (and still the default), set it to 0
        if (
            isinstance(self._bsf, str)
            and self._bsf == "bsf"
            and "bsf" not in self.param_dict
        ):
            kwargs.setdefault("bsf", 0)

        self.param_dict.update({key: np.atleast_1d(val) for key, val in kwargs.items()})

    def calc(self, **params):
        """
        Calculate all model components and return result.

        Perform actual calculation of bistatic scattering at top of the
        random volume (z=0) for the specified geometry. For details please
        have a look at the documentation:
        (http://rt1.readthedocs.io/en/latest/theory.html#first-order-solution-to-the-rte)

        Parameters
        ----------
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
        try:
            self.update_params(**params)

            if isinstance(self.tau, (int, float)):
                Isurf = self._surface()
                # differentiation for non-existing canopy, as otherwise NAN values
                if self.tau > 0.0:
                    Ivol = self._volume()
                    if self.int_Q is True:
                        Iint = self._interaction()
                    else:
                        Iint = np.array([0.0])
                else:
                    Ivol = np.full_like(Isurf, 0.0)
                    Iint = np.full_like(Isurf, 0.0)
            else:
                Isurf = self._surface()
                Ivol = self._volume()
                # TODO this should be fixed more properly
                # (i.e. for tau=0, no interaction-term should be calculated)
                if self.int_Q is True:
                    Iint = self._interaction()
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
                )
            else:
                ret = self._convert_sig0_db(
                    np.stack((Isurf + Ivol, Isurf, Ivol)),
                )

            return ret

        except TypeError as ex:
            symbs = (
                {
                    *self._all_param_symbs,
                    *map(
                        str, [*self.V._func.free_symbols, *self.SRF._func.free_symbols]
                    ),
                }
                - {"phi_0", "phi_ex", "theta_0", "theta_ex"}
                - set(self.param_dict)
            )

            if len(symbs) > 0:
                raise TypeError(
                    f"Model evaluation requires parameter specification for {symbs}."
                )
            else:
                raise ex

    def surface(self):
        """
        Numerical evaluation of the surface-contribution.

        (http://rt1.readthedocs.io/en/latest/theory.html#surface_contribution)

        Returns
        -------
        array_like(float)
            Numerical value of the surface-contribution with respect to the currently
            set parameters in :py:attr:`param_dict`.

        """

        return self._convert_sig0_db(self._surface())

    def _surface(self):
        # bare soil contribution (intensity)
        I_bs = (
            self.I0
            * self._mu_0
            * self.SRF.calc(
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
            Numerical value of the volume-contribution with respect to the currently
            set parameters in :py:attr:`param_dict`.

        """
        return self._convert_sig0_db(self._volume())

    def _volume(self):
        # volume contribution (intensity)
        vol = (
            (self.I0 * self.omega * self._mu_0 / (self._mu_0 + self._mu_ex))
            * (1.0 - np.exp(-(self.tau / self._mu_0) - (self.tau / self._mu_ex)))
            * self.V.calc(
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
            Numerical value of the interaction-contribution with respect to the
            currently set parameters in :py:attr:`param_dict`.

        """
        return self._convert_sig0_db(self._interaction())

    def _interaction(self):
        # interaction contribution (intensity)
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

    def _surface_volume(self):
        # convenience function to get surface + volume (e.g. without interaction)

        ret = self._convert_sig0_db(
            self._surface() + self._volume(),
        )

        return ret

    def _convert_sig0_db(self, val, sig0=None, dB=None):
        if sig0 is None:
            sig0 = self.sig0
        if dB is None:
            dB = self.dB

        if sig0 is True:
            signorm = 4.0 * np.pi * self._mu_0 / self.I0
        else:
            signorm = 1.0

        ret = val * signorm

        if dB is True:
            ret = 10.0 * np.log10(ret)

        return ret

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
        p = getattr(self, f"_{param}")
        if isinstance(p, str):
            expr = _parse_sympy_param(getattr(self, f"_{param}"))

            symbs = list(map(str, expr.free_symbols))
        else:
            symbs = []
        return symbs

    @lru_cache()
    def _get_param_funcs(self, param):
        """Lambdified functions used to define tau, omega, NormBRDF and bsf."""
        # only attempt to lambdify strings and sympy.Basic objects
        if not isinstance(getattr(self, f"_{param}"), (str, sp.Basic)):
            return None

        func = _lambdify(
            self._get_param_symbs(param),
            getattr(self, f"_{param}"),
        )

        return func

    def _eval_param(self, param):
        """
        Numerical evaluation of tau, omega, NormBRDF and bsf.

        All required parameters must be provided in `param_dict`!
        """
        func = self._get_param_funcs(param)

        if func is not None:
            try:
                params = self._get_param_symbs(param)
                return func(**{key: self.param_dict[key] for key in params})
                # return func(*[self.param_dict[key] for key in params])

            except Exception as ex:
                # check for missing parameters in case evaluation fails
                func_def = getattr(self, f"_{param}")
                missing = set(params).difference(self.param_dict)

                if missing:
                    raise TypeError(
                        "Missing specification for the parameters "
                        f"{missing} to compute the value of '{param} = {func_def}'."
                    )
                else:
                    raise ex
        else:
            return getattr(self, f"_{param}")

    # %% interaction contribution
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

            self._fnevals_ = _lambdify(variables, self._fn)
            return self._fnevals_

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

        theta_s = sp.Symbol("theta_s")
        phi_s = sp.Symbol("phi_s")

        if self.geometry == "mono":
            # handle monostatic geometry
            theta_0 = sp.Symbol("theta_0")

            p0 = np.unique(self.p_0)
            assert len(p0) == 1, (
                "p_0 must contain only a "
                + "single unique value for monostatic geometry"
            )

            angs = [theta_0, theta_0, p0[0], p0[0]]
        else:
            # handle all possible bistatic geometry definitions
            theta_0 = sp.Symbol("theta_0")
            phi_0 = sp.Symbol("phi_0")
            theta_ex = sp.Symbol("theta_ex")
            phi_ex = sp.Symbol("phi_ex")

            angs = []
            for symb, val, g in zip(
                (theta_0, theta_ex, phi_0, phi_ex),
                (self.t_0, self.t_ex, self.p_0, self.p_ex),
                self.geometry,
            ):
                if g == "f":
                    fixval = np.unique(val)
                    assert (
                        len(fixval) == 1
                    ), "fixed geometries must contain only a single unique value"
                    angs.append(fixval[0])
                elif g == "v":
                    angs.append(symb)
                else:
                    raise TypeError(f"{g} is not a valid geometry specifier!")

        brdfexp = self.SRF.legexpansion(theta_s, angs[1], phi_s, angs[3] + sp.pi).doit()
        volexp = self.V.legexpansion(sp.pi - angs[0], theta_s, angs[2], phi_s).doit()

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
        i : int
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
        args = np.broadcast_arrays(
            self.t_0,
            self.p_0,
            self.t_ex,
            self.p_ex,
            *self.param_dict.values(),
        )

        fn = np.broadcast_arrays(*self._fnevals(*args))

        multip = self._mu_0_x * self._S2_mu(self._mu_0, self.tau)
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
        args = np.broadcast_arrays(
            self.t_ex,
            self.p_ex,
            self.t_0,
            self.p_0,
            *self.param_dict.values(),
        )

        fn = np.broadcast_arrays(*self._fnevals(*args))

        multip = self._mu_ex_x * self._S2_mu(self._mu_ex, self.tau)
        S = np.sum(fn * multip, axis=0)
        return S

    # %% derivatives (tau, omega, NormBRDF, bsf)
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
            * self.V.calc(self.t_0, self.t_ex, self.p_0, self.p_ex, self.param_dict)
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
            * self.V.calc(self.t_0, self.t_ex, self.p_0, self.p_ex, self.param_dict)
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
            * self.V.calc(
                self.t_0,
                self.t_ex,
                self.p_0,
                self.p_ex,
                param_dict=self.param_dict,
            )
        )

        return -vol

    def _dvolume_dNormBRDF(self):
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
            * self.SRF.calc(self.t_0, self.t_ex, self.p_0, self.p_ex, self.param_dict)
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

    def _dsurface_dNormBRDF(self):
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
            * self.SRF.calc(
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
            * self.SRF.calc(
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

    # %% derivatives (other parameters)
    def _dparam_ddummy(self, param, key):
        # symbolic derivative of the parameters (NormBRDF, tau, omega, bsf) with
        # respect to a given symbol

        # make sure to explicitly parse the equation to support parameter-names
        # such as "N", "S", or "Q"

        if isinstance(getattr(self, f"_{param}"), str):
            diff = sp.diff(
                _parse_sympy_param(getattr(self, f"_{param}")), sp.Symbol(key)
            )
            return diff
        else:
            return 1

    def _dparam_ddummy_lambda(self, param, key):
        diff = self._dparam_ddummy(param, key)

        theta_0 = sp.Symbol("theta_0")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_ex = sp.Symbol("phi_ex")

        args = (theta_0, theta_ex, phi_0, phi_ex) + tuple(self.param_dict.keys())

        return _lambdify(args, diff)

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
        inner = None
        # Check symbols used to specify parameters.
        # If yes, use chain rule to evaluate derivatives
        if str(key) in self._get_param_symbs("tau"):
            inner = self._dparam_ddummy_lambda("tau", key)(
                self.t_0, self.t_ex, self.p_0, self.p_ex, **self.param_dict
            )
            return self._dvolume_dtau() * inner
        elif str(key) in self._get_param_symbs("omega"):
            inner = self._dparam_ddummy_lambda("omega", key)(
                self.t_0, self.t_ex, self.p_0, self.p_ex, **self.param_dict
            )
            return self._dvolume_domega() * inner

        elif str(key) in self._get_param_symbs("bsf"):
            inner = self._dparam_ddummy_lambda("bsf", key)(
                self.t_0, self.t_ex, self.p_0, self.p_ex, **self.param_dict
            )
            return self._dvolume_dbsf() * inner
        else:
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

    @lru_cache()
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

        return _lambdify(
            args,
            sp.diff(self.V._func, sp.Symbol(key)),
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
        inner = None
        # Check symbols used to specify parameters.
        # If yes, use chain rule to evaluate derivatives
        if str(key) in self._get_param_symbs("NormBRDF"):
            inner = self._dparam_ddummy_lambda("NormBRDF", key)(
                self.t_0, self.t_ex, self.p_0, self.p_ex, **self.param_dict
            )
            return self._dsurface_dNormBRDF() * inner
        elif str(key) in self._get_param_symbs("bsf"):
            inner = self._dparam_ddummy_lambda("bsf", key)(
                self.t_0, self.t_ex, self.p_0, self.p_ex, **self.param_dict
            )
            return self._dsurface_dbsf() * inner
        else:
            dI_bs = (
                self.I0
                * self._mu_0
                * self._d_surface_dummy_lambda(key)(
                    self.t_0, self.t_ex, self.p_0, self.p_ex, **self.param_dict
                )
            )

            dI_s = (np.exp(-(self.tau / self._mu_0) - (self.tau / self._mu_ex))) * dI_bs

            return self.NormBRDF * ((1.0 - self.bsf) * dI_s + self.bsf * dI_bs)

    @lru_cache()
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

        return _lambdify(
            args,
            sp.diff(self.SRF._func, sp.Symbol(key)),
        )

    def jacobian(self, param_list=["omega", "tau", "NormBRDF"], format="list"):
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
        param_list : list
            A list of strings that correspond to the names of the parameters
            for which the jacobian should be evaluated.

            Possible values are: 'omega', 'tau' 'NormBRDF' and
            any string corresponding to a sympy.Symbol used in the
            definition of V or SRF.
        format : str
            - "list": Return a list where each entry represents the 2D jacobian matrix
              of the corresponding parameter.
            - "scipy_least_squares": Return a `scipy.sparse.csr_matrix` suitable for
              use with `scipy.optimize.least_squares`.

              NOTE:
                At the moment, the refactoring for scipy least_squares is only suitable
                for constant parameters and/or dynamic parameters that represent
                timeseries of unique values for each observation.

                Have a look at the
                `examples <https://rt1-model.readthedocs.io/en/dev/examples.html>`_
                in the docs for more details!

        Returns
        -------
        jac : array-like(float)
              The jacobian of the total backscatter with respect to
              omega, tau and NormBRDF

        """

        if self.sig0 is True and self.dB is False:
            norm = 4.0 * np.pi * self._mu_0
        elif self.dB is True:
            norm = 10.0 / (np.log(10.0) * (self._surface() + self._volume()))
        else:
            norm = 1.0

        jac = []
        for key in param_list:
            if key == "omega":
                jac += [(self._dsurface_domega() + self._dvolume_domega()) * norm]
            elif key == "tau":
                jac += [(self._dsurface_dtau() + self._dvolume_dtau()) * norm]
            elif key == "NormBRDF":
                jac += [(self._dsurface_dNormBRDF() + self._dvolume_dNormBRDF()) * norm]
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
        if format == "list":
            return jac
        elif format == "scipy_least_squares":
            # Reshape jacobian to fit scipy.optimize requirements
            # (e.g. a 2D matrix of the shape (# measurements, # parameters) where each
            # column represents the derivatives with respect to the optimized parameter)
            jac_columns = []

            for key, j in zip(param_list, jac):
                if np.size(self.param_dict[key]) == 1:
                    # Static parameters are affected by all measurements, so the
                    # corresponding row of the scipy-jacobian is given by the ravelled
                    # rt1-jacobian values.
                    jac_columns += csr_matrix(j.ravel())
                else:
                    # Dynamic parameters represent timeseries of independent variables
                    # (a unique value for each timestamp). Therefore we need to convert
                    # the rt1-jacobian into a block-diagonal matrix so that each
                    # parameter value is only affected by measurements of the
                    # corresponding timestamp.
                    jac_columns += block_diag(j.tolist(), "csr")

            # stack and transpose to comply to scipy.optimize requirements
            return vstack(jac_columns).T
        else:
            raise TypeError(
                f"{format} is not a valid output format for the jacobian!"
                "Use one of: ('list', 'scipy_least_squares')."
            )
