from . import _log
from ._numpydoc_docscrape import NumpyDocString


def append_numpy_docstring(*funcs):
    """
    Decorator to combine numpy docstrings.

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
