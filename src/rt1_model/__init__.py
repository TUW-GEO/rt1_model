# -----------------------------------------------------------------------------------
# The following loggin config is adapted from matplotlibs way of dealing with logging
# (see https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/__init__.py)

import logging as _logging
from functools import lru_cache

_log = _logging.getLogger(__name__)
_log.setLevel(_logging.WARNING)

try:
    # if symengine is available, use it to perform series-expansions
    import symengine

    _init_lambda_backend = "symengine"

except ImportError:
    _init_lambda_backend = "sympy"

_log.debug(f"Init lambda backend set to: {_init_lambda_backend}")


def get_lambda_backend():
    """Get the backend used to evaluate symbolic expressions."""
    global _init_lambda_backend
    return _init_lambda_backend


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

    _log.debug(f"Lambda backend set to {lambda_backend}")


from ._calc import RT1
from . import volume
from . import surface

__all__ = [
    "RT1",
    "volume",
    "surface",
    "plot",
    "set_lambda_backend",
    "get_lambda_backend",
]


@lru_cache()
def _ensure_handler():
    """
    Attach a StreamHandler to the logger the first time this function is executed.

    The first time this function is called, attach a `StreamHandler` using the
    same format as `logging.basicConfig` to the root logger.

    Return this handler every time this function is called.

    """
    handler = _logging.StreamHandler()
    handler.setFormatter(_logging.Formatter(_logging.BASIC_FORMAT))
    _log.addHandler(handler)
    return handler


def _set_logfmt(fmt=None, datefmt=None):
    """
    Set the format string for the logger.

    See `logging.Formatter` for details.

    Parameters
    ----------
    fmt : str
        The logging format string.
        The default is:  "%(levelname)s: %(asctime)s: %(message)s"
    datefmt : str
        The datetime format string. ('%Y-%m-%d,%H:%M:%S.%f')
    """
    if fmt is None:
        fmt = "%(levelname)s: %(asctime)s: %(message)s"

    handler = _ensure_handler()
    if datefmt is None:
        handler.setFormatter(_logging.Formatter(fmt))
    else:
        handler.setFormatter(_logging.Formatter(fmt, datefmt=datefmt))


_log_format_presets = {
    "minimal": ("%(asctime)s.%(msecs)03d: %(message)s", "%H:%M:%S"),
    "timed": (
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        "%H:%M:%S",
    ),
    "debug": (
        "%(asctime)s.%(msecs)03d %(levelname)s: %(name)s: %(message)s",
        "%H:%M:%S",
    ),
    "plain": ("%(message)s", None),
}


def set_loglevel(level, fmt="timed"):
    """
    Configure logging levels (and formatting).

    rt1_model uses the standard library `logging` framework under the root
    logger 'rt1_model'.  This is a helper function to:

    - set rt1_model's root logger level
    - set the root logger handler's level, creating the handler
      if it does not exist yet
    - set the root logger handler's formatter

    Typically, one should call ``set_loglevel("info")`` or
    ``set_loglevel("debug")`` to get additional debugging information.

    Users or applications that are installing their own logging handlers
    may want to directly manipulate ``logging.getLogger('rt1_model')`` rather
    than use this function.

    Parameters
    ----------
    level : {"notset", "debug", "info", "warning", "error", "critical"} or int
        The log level of the handler.
    fmt : str
        A short-name or a logging format-string.

        Available short-names:

        - "plain": ``message``
        - "basic": ``<TIME>: message``
        - "timed": ``<TIME>: <LEVEL>: message``
        - "debug": ``<TIME>: <LEVEL>: <MODULE>: message``

        The default is ``logging.BASIC_FORMAT``

        >>> "%(levelname)s:%(name)s:%(message)s"

    Notes
    -----
    The first time this function is called, an additional handler is attached
    to the root handler of rt1_model; this handler is reused every time and this
    function simply manipulates the logger and handler's level.

    """
    if isinstance(level, str):
        level = level.upper()

    _log.setLevel(level)
    _ensure_handler().setLevel(level)

    if fmt is not None:
        _set_logfmt(*_log_format_presets.get(fmt, (fmt,)))
