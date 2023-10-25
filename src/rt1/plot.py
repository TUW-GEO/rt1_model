"""Class for quick visualization of results and used phasefunctions."""

from itertools import cycle

import numpy as np
import sympy as sp

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from . import _log
from .volume import _Volume
from .surface import _Surface


def polarplot(
    X=None,
    inc=[15.0, 35.0, 55.0, 75.0],
    multip=2.0,
    label=None,
    aprox=True,
    legend=True,
    legpos=(0.75, 0.5),
    groundcolor="none",
    param_dict=[{}],
    polarax=None,
):
    """
    Generate polar-plots of the volume- and the surface scattering phase functions.

    (and also the used approximations in terms of legendre-polynomials)


    Parameters
    ----------
    SRF : RT1.surface class object
          Alternative direct specification of the surface BRDF,
          e.g. SRF = CosineLobe(i=3, ncoefs=5)
    V : RT1.volume class object
        Alternative direct specification of the volume-scattering
        phase-function  e.g. V = Rayleigh()

    Other Parameters
    ----------------
    inc : list of floats (default = [15.,35.,55.,75.])
           Incidence-angles in degree at which the volume-scattering
           phase-function will be plotted
    multip : float (default = 2.)
              Multiplicator to scale the plotrange for the plot of the
              volume-scattering phase-function
              (the max-plotrange is given by the max. value of V in
              forward-direction (for the chosen incp) )
    label : string
             Manual label for the volume-scattering phase-function plot
    aprox : boolean (default = True)
             Indicator if the approximation of the phase-function in terms
             of Legendre-polynomials will be plotted.
    legend : boolean (default = True)
             Indicator if a legend should be shown that indicates the
             meaning of the different colors for the phase-function
    legpos : (float,float) (default = (0.75,0.5))
             Positioning of the legend for the V-plot (controlled via
             the matplotlib.legend keyword  bbox_to_anchor = plegpos )
    groundcolor : string (default = "none")
             Matplotlib color-indicator to change the color of the lower
             hemisphere in the BRDF-plot possible values are:
             ('r', 'g' , 'b' , 'c' , 'm' , 'y' , 'k' , 'w' , 'none')
    polarax: matplotlib.axes
             the axes to use... it must be a polar-axes, e.g.:

                 >>> polarax = fig.add_subplot(111, projection='polar')

    param_dict : dict (or list of dicts)
                 a dictionary containing the names and values of the symbolic
                 parameters required to fully specify the V/SRF functions.
                 if a list of dicts is provided, the specifications are
                 plotted on top of each other.

    Returns
    -------
    polarfig : figure
               a matplotlib figure showing a polar-plot of the functions
               specified by V or SRF

    """
    assert isinstance(inc, list), (
        "Error: incidence-angles for " + "polarplot must be a list"
    )
    assert isinstance(multip, float), (
        "Error: plotrange-multiplier " + "for polarplot must be a floating-point number"
    )

    if X is None:
        assert False, "Error: You must provide a volume- or surface object!"

    if isinstance(param_dict, dict):
        param_dict = [param_dict]

    # Check if all required parameters have been provided in the param_dict
    required_symbs = set(map(str, X._func.free_symbols)) - {
        "phi_0",
        "phi_ex",
        "theta_0",
        "theta_ex",
    }
    for params in param_dict:
        missing = required_symbs - set(params)
        assert len(missing) == 0, (
            "Missing parameter definition! Please provide numerical values for the "
            f"variables {missing} in the `param_dict`!"
        )

    if polarax is None:
        fig = plt.figure(figsize=(7, 7))
        polarax = fig.add_subplot(111, projection="polar")
    else:
        assert polarax.name == "polar", "you must provide a polar-axes!"

    def angsub(ti):
        if isinstance(X, _Surface):
            return ti
        elif isinstance(X, _Volume):
            return np.pi - ti

    if isinstance(X, _Surface):
        if label is None:
            label = "Surface-Scattering Phase Function"
        funcname = "brdf"
        angs = ["theta_ex", "theta_s", "phi_ex", "phi_s"]

        thetass = np.arange(-np.pi / 2.0, np.pi / 2.0, 0.01)

        polarax.fill(
            np.arange(np.pi / 2.0, 3.0 * np.pi / 2.0, 0.01),
            np.ones_like(np.arange(np.pi / 2.0, 3.0 * np.pi / 2.0, 0.01)) * 1 * 1.2,
            color=groundcolor,
        )

    if isinstance(X, _Volume):
        if label is None:
            label = "Volume-Scattering Phase Function"

        funcname = "p"
        angs = ["theta_0", "theta_s", "phi_0", "phi_s"]

        thetass = np.arange(0.0, 2.0 * np.pi, 0.01)

    # plot of volume-scattering phase-function's
    pmax = 0
    for n_X, X in enumerate(np.atleast_1d(X)):
        for n_p, params in enumerate(param_dict):
            # define a plotfunction of the legendre-approximation of p
            if aprox is True:
                phasefunktapprox = sp.lambdify(
                    (*angs, *params.keys()),
                    X.legexpansion(*angs, geometry="vvvv").doit(),
                    modules=["numpy", "sympy"],
                )

            # set incidence-angles for which p is calculated
            plottis = np.deg2rad(inc)
            colors = cycle(["k", "r", "g", "b", "c", "m", "y"])
            used_colors = []
            for i in plottis:
                ts = np.arange(0.0, 2.0 * np.pi, 0.01)
                pmax_i = multip * np.max(
                    getattr(X, funcname)(
                        np.full_like(ts, i),
                        ts,
                        0.0,
                        0.0,
                        param_dict=params,
                    )
                )
                if pmax_i > pmax:
                    pmax = pmax_i

            if legend is True:
                legend_lines = []

            # set color-counter to 0
            for ti in plottis:
                color = next(colors)
                used_colors.append(color)
                rad = getattr(X, funcname)(ti, thetass, 0.0, 0.0, param_dict=params)
                if aprox is True:
                    # the use of np.pi-ti stems from the definition
                    # of legexpansion() in volume.py
                    radapprox = phasefunktapprox(
                        angsub(ti), thetass, 0.0, 0.0, **params
                    )
                # set theta direction to clockwise
                polarax.set_theta_direction(-1)
                # set theta to start at z-axis
                polarax.set_theta_offset(np.pi / 2.0)

                polarax.plot(thetass, rad, color)
                if aprox is True:
                    polarax.plot(thetass, radapprox, color + "--")
                polarax.arrow(
                    -ti,
                    pmax * 1.2,
                    0.0,
                    -pmax * 0.8,
                    head_width=0.0,
                    head_length=0.0,
                    fc=color,
                    ec=color,
                    lw=1,
                    alpha=0.3,
                )

                polarax.fill_between(thetass, rad, alpha=0.2, color=color)
                polarax.set_xticks(np.deg2rad([0, 45, 90, 125, 180]))
                polarax.set_xticklabels(
                    [
                        r"$0^\circ$",
                        r"$45^\circ$",
                        r"$90^\circ$",
                        r"$135^\circ$",
                        r"$180^\circ$",
                    ]
                )
                polarax.set_yticklabels([])
                polarax.set_rmax(pmax * 1.2)
                polarax.set_title(label + "\n")
                polarax.set_rmin(0.0)

    # add legend for covering layer phase-functions
    used_colors = iter(used_colors)
    if legend is True:
        for ti in plottis:
            color = next(used_colors)
            legend_lines += [
                mlines.Line2D(
                    [],
                    [],
                    color=color,
                    label=r"$\theta_0$ = "
                    + str(np.round_(np.rad2deg(ti), decimals=1))
                    + r"${}^\circ$",
                )
            ]
            i = i + 1

        if aprox is True:
            legend_lines += [
                mlines.Line2D([], [], color="k", linestyle="--", label="approx.")
            ]

        legend = polarax.legend(bbox_to_anchor=legpos, loc=2, handles=legend_lines)
        legend.get_frame().set_facecolor("w")
        legend.get_frame().set_alpha(0.5)

    return fig


def hemreflect(
    R=None,
    SRF=None,
    phi_0=0.0,
    t_0_step=5.0,
    t_0_min=0.0,
    t_0_max=90.0,
    simps_N=1000,
    showpoints=True,
    returnarray=False,
    param_dict={},
):
    """
    Numerical evaluation of the hemispherical reflectance of the given BRDF-function.

    This is using scipy's implementation of the Simpson-rule integration scheme.

    Parameters
    ----------
    R : RT1-class object
        definition of the brdf-function to be evaluated
        (either R or SRF  must be provided) The BRDf is defined via:

            BRDF = R.SRF.NormBRDF * R.SRF.brdf()
    SRF : Surface-class object
          definition of the brdf-function to be evaluated
          (either R or SRF must be provided) The BRDf is defined via:

              BRDF = SRF.NormBRDF * SRF.brdf()

    Other Parameters
    ----------------
    phi_0 : float
            incident azimuth-angle
            (for spherically symmetric phase-functions the result is
            independent of the choice of phi_0)
    t_0_step : float
               separation of the incidence-angle grid-spacing in DEGREE
               for which the hemispherical reflectance will be calculated
    t_0_min : float
              minimum incidence-angle
    t_0_max : float
              maximum incidence-angle
    simps_N : integer
              number of points used in the discretization of the brdf
              within the Simpson-rule
    showpoints : boolean
                 show or hide integration-points in the plot
    param_dict : dict
                 a dictionary containing the names and values of the symbolic
                 parameters required to define the SRF function

    Returns
    -------
    fig : figure
        a matplotlib figure showing the incidence-angle dependent
        hemispherical reflectance

    """
    from scipy.integrate import simps

    # choose BRDF function to be evaluated
    if R is not None:
        BRDF = R.SRF.brdf

        try:
            Nsymb = R.NormBRDF.free_symbols
            Nfunc = sp.lambdify(Nsymb, R.NormBRDF, modules=["numpy"])
            NormBRDF = Nfunc(*[param_dict[str(i)] for i in Nsymb])
        except Exception:
            NormBRDF = R.NormBRDF
    elif SRF is not None:
        BRDF = SRF.brdf
        try:
            Nsymb = SRF.NormBRDF[0].free_symbols
            Nfunc = sp.lambdify(Nsymb, SRF.NormBRDF, modules=["numpy"])
            NormBRDF = Nfunc(*[param_dict[str(i)] for i in Nsymb])
        except Exception:
            NormBRDF = SRF.NormBRDF
    else:
        assert False, "Error: You must provide either R or SRF"

    # set incident (zenith-angle) directions for which the integral
    # should be evaluated!
    incnum = np.arange(t_0_min, t_0_max, t_0_step)

    # define grid for integration
    x = np.linspace(0.0, np.pi / 2.0, simps_N)
    y = np.linspace(0.0, 2 * np.pi, simps_N)

    # initialize array for solutions

    sol = []

    # ---- evaluation of Integral
    # adapted from
    # (http://stackoverflow.com/questions/20668689/integrating-2d-samples-on-a-rectangular-grid-using-scipy)

    for theta_0 in np.deg2rad(incnum):
        # define the function that has to be integrated
        # (i.e. Eq.20 in the paper)
        # notice the additional  np.sin(thetas)  which oritinates from
        # integrating over theta_s instead of mu_s
        def integfunkt(theta_s, phi_s):
            return (
                np.sin(theta_s)
                * np.cos(theta_s)
                * BRDF(theta_0, theta_s, phi_0, phi_s, param_dict=param_dict)
            )

        # evaluate the integral using Simpson's Rule twice
        z = integfunkt(x[:, None], y)
        sol = sol + [simps(simps(z, y), x)]

    sol = np.array(sol) * NormBRDF

    # print warning if the hemispherical reflectance exceeds 1
    if np.any(sol > 1.0):
        print("ATTENTION, Hemispherical Reflectance > 1 !")

    if returnarray is True:
        return sol
    else:
        # generation of plot
        fig = plt.figure()
        axnum = fig.add_subplot(1, 1, 1)

        if len(sol.shape) > 1:
            for i, sol in enumerate(sol):
                axnum.plot(incnum, sol, label="NormBRDF = " + str(NormBRDF[i]))
                if showpoints is True:
                    axnum.plot(incnum, sol, "r.")
        else:
            axnum.plot(incnum, sol, "k", label="NormBRDF = " + str(NormBRDF))
            if showpoints is True:
                axnum.plot(incnum, sol, "r.")

        axnum.set_xlabel("$\\theta_0$ [deg]")
        axnum.set_ylabel("$R(\\theta_0)$")
        axnum.set_title("Hemispherical reflectance ")
        axnum.set_ylim(0.0, np.max(sol) * 1.1)

        axnum.legend()

        axnum.grid()
        return fig


def _check_params(R, param_dict, additional_params=[]):
    # check if all required parameters for the analyzers have been defined
    symbs = {
        *R._all_param_symbs,
        *map(str, [*R.V._func.free_symbols, *R.SRF._func.free_symbols]),
    } - {"phi_0", "phi_ex", "theta_0", "theta_ex"}
    for p in additional_params:
        symbs.add(p)

    # only use parameters that are actually assigned
    # (to avoid broadcasting issues with obsolete parameters)
    ignored_params = symbs - set(param_dict)
    if len(ignored_params) > 0:
        if all((i in R.param_dict for i in ignored_params)):
            _log.warning(
                f"The parameters {ignored_params} are missing in `param_dict` "
                f"(static values found in `R.param_dict` are used)!"
            )
        else:
            raise AssertionError(
                "The analyzer is missing parameter specifications for the "
                f"following variables: {ignored_params}"
            )
    return symbs


class Analyze:
    """
    Create a widget to analyze a given RT1 model specification.

    Parameters
    ----------
    R : RT1 object
        The RT1 object to analyze.
    param_dict : dict
        A dictionary containing the value-range for all dynamic parameters of the model.

        The dict must have the following form:

        >>> {"parameter_name": [min, max, (start),
        >>> ...
        >>> }

        - `min`, `max` : The min-max values of the parameter-range
        - `start` : The start-value to use (optional, if ommitted, center is used)

        For example:

        >>> {"omega": [0, .5, .2, "Omega"],
        >>>  "tau": [0, .25, .1, "Optical Depth"]}

    sig0 : bool, optional
        Indicator if plot should show backscatter-coefficient (True) or
        relative intensity (False). The default is True.
    dB : bool, optional
        Indicator if plot should be in decibel (True) or natural units (False).
        The default is True.
    range_parameter : str, optional
        If provided, the range of resulting values for the given parameter
        is overlaid on the plot (can be any parameter provided in `param_dict`).
        The default is None.

    """

    def __init__(self, R, param_dict=None, sig0=True, dB=True, range_parameter="tau"):
        self._t0 = np.linspace(10, 80, 100)
        self._range_parameter = range_parameter

        self.R = R

        if self.R.geometry != "mono":
            _log.warning("The analyze-plot shows results for monostatic geometry!")

        self.R.set_geometry(t_0=[np.deg2rad(self._t0)], p_0=np.pi, geometry="mono")

        # number of intermediate parameter values to calculate (for range-indication)
        self._n_mima_samples = 20

        if param_dict is None:
            param_dict = dict()

        # param_dict.setdefault("omega", (0, 0.5))
        # param_dict.setdefault("tau", (0, 0.5))
        # param_dict.setdefault("NormBRDF", (0, 0.2))

        symbs = _check_params(self.R, param_dict)
        self.param_dict = {key: val for key, val in param_dict.items() if key in symbs}

        self._sig0 = sig0
        self._dB = dB

        self._lines = None
        self._range_lines = set()

        self._slider_params = self._parse_parameters(**self.param_dict)
        self._init_figure()
        self._update(None)

        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.autoscale(False)

    def _init_figure(self, slider_cols=2):
        gs = GridSpec(2, 3, width_ratios=(100, 100, 25), height_ratios=(10, 2))
        gs.update(hspace=0.35, wspace=0.05)

        rows = int(np.ceil(len(self.param_dict) / slider_cols))
        sliderspecs = GridSpecFromSubplotSpec(rows, slider_cols, gs[-1, :-1])

        self.f = plt.figure(figsize=(10, 6))
        self.ax = self.f.add_subplot(gs[:-1, :-1])

        self.ax.set_ylabel(
            (
                r"Backscatter Coefficient $\sigma^0$"
                if self._sig0
                else r"Intensity Ratio $\frac{I}{I_inc}$"
            )
            + (" [db]" if self._dB else "")
        )
        self.ax.set_xlabel(r"Incidence Angle $\theta_0 [deg]$")

        self.sliders = dict()
        for (key, params), spec in zip(self._slider_params.items(), sliderspecs):
            sliderax = self.f.add_subplot(spec)
            s = Slider(ax=sliderax, **params)
            s.on_changed(self._update)
            self.sliders[key] = s

        buttonspecs = GridSpecFromSubplotSpec(2, 1, gs[:-1, -1])
        self._buttonax = self.f.add_subplot(buttonspecs[1, 0])
        # buttonax.set_axis_off()
        self._buttonax.set_title("Select range", fontsize=10)
        params = ["None", *self._slider_params]
        self._radio = RadioButtons(
            self._buttonax,
            params,
            active=params.index(self._range_parameter),
            label_props=dict(color="k", fontsize=[9]),
            radio_props={"s": [50] * len(params)},
        )

        def doit(label):
            if label == "None":
                self._range_parameter = None
            else:
                self._range_parameter = label
            self._update(None)
            self.f.canvas.draw()

        self._radio.on_clicked(doit)

        self.f.subplots_adjust(
            top=0.98, bottom=0.05, hspace=1, wspace=0.5, left=0.1, right=0.98
        )

    @staticmethod
    def _parse_parameters(**kwargs):
        paramdicts = dict()
        for key, val in kwargs.items():
            p = dict(zip(["valmin", "valmax", "valinit"], val))

            p.setdefault("valinit", (p["valmax"] - p["valmin"]) / 2)
            p.setdefault("label", key)

            paramdicts[key] = p
        return paramdicts

    def _update(self, val):
        startvals = {key: s.val for key, s in self.sliders.items()}
        self.R.set_params(**startvals)
        contribs = self.R.calc(sig0=self._sig0, dB=self._dB).squeeze()
        labels = ["Total", "Surface", "Volume", "Interaction"]

        if self.R.int_Q is True:
            sv = self.R._surface_volume(sig0=self._sig0, dB=self._dB).squeeze()
        else:
            sv = None

        if self._lines is None:
            self._lines = [
                self.ax.plot(self._t0, c, label=l)[0] for c, l in zip(contribs, labels)
            ]
            if sv is not None:
                self._lines.append(
                    self.ax.plot(self._t0, sv, c="C0", ls=(0, (5, 5)), lw=0.75)[0]
                )

            if val is None:  # e.g. on init
                self.ax.legend(
                    loc="upper left",
                    title="Contributions",
                    bbox_to_anchor=(1, 1),
                    fontsize=9,
                )
        else:
            for l, c in zip(self._lines, contribs):
                l.set_ydata(c)

            if sv is not None:
                self._lines[-1].set_ydata(sv)

        while len(self._range_lines) > 0:
            self._range_lines.pop().remove()

        if self._range_parameter is not None:
            range_lines = self._get_range_lines()
            r0, r1 = range_lines.min(axis=1), range_lines.max(axis=1)
            for i, (r0i, r1i) in enumerate(zip(r0, r1)):
                # add bounding lines for the range
                fill = self.ax.fill_between(
                    self._t0,
                    r0i,
                    r1i,
                    alpha=0.3 if i == 0 else 0.05,
                    fc=f"C{i}",
                    ec="none",
                )
                self._range_lines.add(fill)

            # add intermediate lines for the range
            for i, contrib_mima in enumerate(range_lines):
                lc = LineCollection(
                    np.dstack(np.broadcast_arrays(self._t0, contrib_mima)),
                    ec=f"C{i}",
                    lw=0.1,
                    alpha=0.5,
                    antialiaseds=True,
                )
                self._range_lines.add(self.ax.add_collection(lc))

    def _get_range_lines(self, **kwargs):
        startvals = {
            key: np.full((self._n_mima_samples, 1), s.val)
            for key, s in self.sliders.items()
        }
        startvals[self._range_parameter] = np.linspace(
            self.sliders[self._range_parameter].valmin,
            self.sliders[self._range_parameter].valmax,
            self._n_mima_samples,
        )[:, np.newaxis]

        self.R.set_params(**startvals)
        res = self.R.calc(sig0=self._sig0, dB=self._dB)
        return res


class Analyze3D:
    def __init__(
        self, R, param_dict=None, sig0=True, dB=False, samples=35, contributions="ts"
    ):
        """
        A widget to analyze the 3D scattering distribution of a selected RT1 specification.

        Parameters
        ----------
        R : RT1 object
            The RT1 object to analyze.
        param_dict : dict
            A dictionary containing the value-range for all dynamic parameters of the model.

            The dict must have the following form:

            >>> {"parameter_name": [min, max, (start), (label),
            >>> ...
            >>> }

            - `min`, `max` : The min-max values of the parameter-range
            - `start` : The start-value to use (optional, if ommitted, center is used)
            - `label` : Parameter-name (optional, if ommitted, variable name is used)

            For example:

            >>> {"omega": [0, .5, .2, "Omega"],
            >>>  "tau": [0, .25, .1, "Optical Depth"]}

        sig0 : bool, optional
            Indicator if plot should show backscatter-coefficient (True) or
            relative intensity (False). The default is True.
        dB : bool, optional
            Indicator if plot should be in decibel (True) or natural units (False).
            The default is True.
        samples : int, optional
            The number of samples to draw. (e.g. higher numbers result in better image
            quality but also in significantly slower update speed!)
            The default is 50.
        contributions : string, optional
            A string indicating the contributions to show.

            Can be any combination of ["t", "s", "v", "i"]
            (for Total, Surface, Volume and Interaction term)
            The default is "ts"
        """
        self.R = R
        self.R._clear_cache()

        self._samples = samples
        self._use_contribs = ["tsvi".index(i) for i in contributions]

        self._labels = ["Total", "Surface", "Volume", "Interaction"]
        self._colors = ["C0", "C1", "C2", "C3"]

        if self.R.geometry != "vvvv":
            _log.warning("The analyze-plot shows results for 'vvvv' geometry!")

        t_ex = np.deg2rad(np.linspace(0, 89, self._samples))
        p_ex = np.deg2rad(np.linspace(-180, 180, self._samples))
        self._t_ex, self._p_ex = np.meshgrid(t_ex, p_ex)

        if param_dict is None:
            param_dict = dict()

        # param_dict.setdefault("omega", (0, 0.5))
        # param_dict.setdefault("tau", (0, 0.5))
        # param_dict.setdefault("NormBRDF", (0, 0.2))

        self._inc_range = (0.1, 89.9)

        # only use parameters that are actually assigned
        # (to avoid broadcasting issues with obsolete parameters)
        symbs = _check_params(self.R, param_dict)
        self.param_dict = {key: val for key, val in param_dict.items() if key in symbs}

        self._sig0 = sig0
        self._dB = dB

        self._artists = set()

        self._slider_params = Analyze._parse_parameters(**self.param_dict)
        self._init_figure()
        self._update(None)

        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.autoscale(False)

    def _init_figure(self, slider_cols=2):
        gs = GridSpec(3, 3, width_ratios=(100, 100, 1), height_ratios=(10, 1, 0.2))
        gs.update(hspace=0.1)

        rows = int(np.ceil(len(self.param_dict) / slider_cols))
        sliderspecs = GridSpecFromSubplotSpec(rows, slider_cols, gs[-2, :-1])

        self.f = plt.figure(figsize=(9, 8))
        self.ax = self.f.add_subplot(gs[:-1, :], projection="3d", computed_zorder=False)

        self.ax.set_axis_off()
        self.ax.draw(self.f.canvas.get_renderer())
        self.sliders = dict()

        for (key, params), spec in zip(self._slider_params.items(), sliderspecs):
            sliderax = self.f.add_subplot(spec)

            s = Slider(ax=sliderax, **params)
            s.drawon = False  # avoid automatic calls to draw on slider updates
            s.on_changed(self._update)
            self.sliders[key] = s

        # add incidence-angle slider
        self._inc_slider_ax = self.f.add_subplot(gs[-1, :-1])

        s = Slider(
            ax=self._inc_slider_ax,
            valmin=self._inc_range[0],
            valmax=self._inc_range[1],
            valinit=45,
            label=r"$\theta_0$",
        )
        s.drawon = False  # avoid automatic calls to draw on slider updates
        s.on_changed(self._update)
        self.sliders["inc"] = s

        self.f.subplots_adjust(
            top=0.98, bottom=0.05, hspace=1, wspace=0.5, left=0.1, right=0.98
        )

    def _getvals(self, inc=45):
        self.R.set_geometry(
            np.full_like(self._t_ex, np.deg2rad(inc)),
            np.full_like(self._t_ex, 0),
            self._t_ex,
            self._p_ex,
            geometry="fvfv",
        )

        res = self.R.calc(sig0=self._sig0, dB=self._dB)

        x = res * np.sin(self._t_ex) * np.cos(self._p_ex)
        y = res * np.sin(self._t_ex) * np.sin(self._p_ex)
        z = res * np.cos(self._t_ex)
        return x, y, z

    def _getarrow(self, r, t_0, p_0):
        x = r * np.sin(t_0) * np.cos(p_0)
        y = r * np.sin(t_0) * np.sin(p_0)
        z = r * np.cos(t_0)
        return [x, 0], [y, 0], [z, 0]

    def _getsurface(self, lim):
        x = np.linspace(*lim, 100)
        y = np.linspace(*lim, 100)
        x, y = np.meshgrid(x, y)
        z = np.full_like(x, 0)
        return x, y, z

    def restore_slider_bg(self):
        for s in self._slider_bgs:
            self.f.canvas.restore_region(s)

    def _update(self, event):
        while len(self._artists) > 0:
            a = self._artists.pop()
            a.remove()

        startvals = {key: s.val for key, s in self.sliders.items()}
        inc = startvals.pop("inc")

        self.R.set_params(**startvals)
        x, y, z = self._getvals(inc)

        if event is None:  # e.g. on init
            lim = (
                min(np.nanmin(x[self._use_contribs]), np.nanmin(y[self._use_contribs])),
                max(np.nanmax(x[self._use_contribs]), np.nanmax(y[self._use_contribs])),
            )
            limmax = np.abs(lim).max()
            lim = [-limmax, limmax]

            self.ax.set_xlim(*lim)
            self.ax.set_ylim(*lim)
            self.ax.set_zlim(*lim)

            self.ax.plot_surface(
                *self._getsurface(lim),
                zorder=0,
                antialiased=False,
                fc=".5",
                rcount=2,
                ccount=2,
            )

        for i in self._use_contribs:
            self._artists.add(
                self.ax.plot_surface(
                    x[i],
                    y[i],
                    z[i],
                    color=self._colors[i],
                    label=self._labels[i],
                    alpha=0.6,
                    rcount=self._samples,
                    ccount=self._samples,
                )
            )

        if event is None:  # e.g. on init
            self.ax.legend()

        self._artists.add(
            self.ax.plot(
                *self._getarrow(
                    np.nanmax(np.abs(z[self._use_contribs])), np.deg2rad(inc), 0
                ),
                c="k",
                ls="--",
            )[0]
        )
        self._artists.add(
            self.ax.plot(
                *self._getarrow(
                    np.nanmax(np.abs(z[self._use_contribs])), np.deg2rad(-inc), 0
                ),
                c="k",
                ls="-",
            )[0]
        )

        self.f.canvas.draw()