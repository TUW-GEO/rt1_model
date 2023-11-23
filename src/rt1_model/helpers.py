from . import _log, get_lambda_backend
from ._numpydoc_docscrape import NumpyDocString

import sympy as sp
import numpy as np

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
        funcs.append(_parse_sympy_param(f))

    # make sure that we don't add additional axes to the returned dataset
    # if a single function is evaluated
    if len(funcs) == 1 and not isinstance(functions, list):
        funcs = funcs[0]

    # use symengine's Lambdify if symengine has been used within
    # the fn-coefficient generation
    backend = get_lambda_backend()
    if backend == "symengine":
        from symengine import Lambdify

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

    elif backend == "sympy":
        # using sympy's lambdify without "common subexpression
        # elimination" to perform lambdification

        lambda_func = sp.lambdify(
            var,
            funcs,
            modules=["numpy", "sympy"],
            dummify=False,
            cse=True,
        )
    else:
        raise TypeError(f"lambda_backend {backend} is not available")

    return lambda_func


def append_numpy_docstring(*funcs):
    """
    A decorator to combine numpy docstrings.

    Parameter descriptions that already exist in the inherited docstrings will be
    replaced by the description of the parameter in the function docstring!

    Parameters
    ----------
    funcs :
        Objects with __doc__ attribute defined.

    Returns
    -------
    f:
        The object with combined __doc__ sections.

    """

    def inner(decorated_f):
        try:
            parent_docstring = decorated_f.__doc__
            if parent_docstring is None:
                parent_docstring = ""

            comb = NumpyDocString(parent_docstring)

            for f in funcs:
                if not f.__doc__:
                    continue

                f_doc = NumpyDocString(f.__doc__)

                # combine lists of summaries with a newline
                for key in ["Summary", "Extended Summary"]:
                    if f_doc[key]:
                        comb[key] += ["\n", *f_doc[key]]

                # combine references and example strings with a newline
                for key in ["References", "Examples"]:
                    if f_doc[key]:
                        comb[key] += "\n" + f_doc[key]

                # combine parameter descriptions (override existing descriptions)
                for p in f_doc["Parameters"]:
                    # check if parameter is already defined
                    old_p = any(i for i in comb["Parameters"] if i.name == p.name)
                    # if no description exists, append the parameter
                    if not old_p:
                        comb["Parameters"] += [p]

                for key in [
                    "Returns",
                    "Yields",
                    "Receives",
                    "Raises",
                    "Warns",
                    "Other Parameters",
                    "Attributes",
                    "Methods",
                    "See Also",
                    "Notes",
                    "Warnings",
                ]:
                    comb[key] += f_doc[key]

            # set the combined docstring
            decorated_f.__doc__ = str(comb)
        except Exception:
            _log.debug(
                "There was an error while trying to combine the docstring "
                f"of {decorated_f} with the docstrings from {funcs}"
            )
        return decorated_f

    return inner
