"""Class for quick visualization of results and used phasefunctions."""

from itertools import cycle
from contextlib import contextmanager

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from . import _log
from .helpers import _lambdify
from .volume import VolumeScatter
from .surface import SurfaceScatter


def polarplot(
    V_SRF=None,
    inc=[15.0, 35.0, 55.0, 75.0],
    multip=2.0,
    label=None,
    aprox=True,
    legend=True,
    legpos=(0.75, 0.5),
    groundcolor=".5",
    param_dict=None,
    ax=None,
):
    """
    A convenience function to generate polar-plots of scattering distribution functions.

    (and also the used approximations in terms of legendre-polynomials)

    Note
    ----
    You can call this function directly from a given :py:mod:`volume`,
    :py:mod:`surface` object by using

    ``V.polarplot(...)`` or ``SRF.polarplot()``


    Parameters
    ----------
    V_SRF : SurfaceScatter or VolumeScatter object or tuple
        The surface- or volume scattering distribution to visualize.

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
    legpos : (float, float) (default = (0.75, 0.5))
        Positioning of the legend for the V-plot (controlled via
        the matplotlib.legend keyword  bbox_to_anchor = plegpos )
    groundcolor : string (default = "none")
        Matplotlib color-indicator to change the color of the lower
        hemisphere in the BRDF-plot possible values are:
        ('r', 'g' , 'b' , 'c' , 'm' , 'y' , 'k' , 'w' , 'none')
    ax: matplotlib.axes

        Use provided axes for the plot instead of creating a new figure.
        The axes must be a polar-axes, e.g.:

             >>> ax = fig.add_subplot(111, projection='polar')

    param_dict : dict (or list of dicts)
        a dictionary containing the names and values of the symbolic
        parameters required to fully specify the V/SRF functions.
        if a list of dicts is provided, the specifications are
        plotted on top of each other.

    """
    if aprox is True:
        try:
            V_SRF.ncoefs
        except Exception:
            _log.warning(
                "To print scatter function approximations, ncoefs must be provided!"
            )
            aprox = False

    assert isinstance(inc, list), (
        "Error: incidence-angles for " + "polarplot must be a list"
    )
    assert isinstance(multip, float), (
        "Error: plotrange-multiplier " + "for polarplot must be a floating-point number"
    )

    if V_SRF is None:
        assert False, "Error: You must provide a volume- or surface object!"

    use_colors = ["k", "r", "g", "b", "c", "m", "y"]

    if param_dict is None:
        param_dict = [{}]
    elif isinstance(param_dict, dict):
        param_dict = [param_dict]

    # Check if all required parameters have been provided in the param_dict
    required_symbs = set(map(str, V_SRF._func.free_symbols)) - {
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

    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="polar")
    else:
        fig = ax.figure
        assert ax.name == "polar", "You must provide axes in polar-projection!"

    if isinstance(V_SRF, SurfaceScatter):
        if label is None:
            label = "Surface-Scattering Phase Function"
        angs = ["theta_ex", "theta_s", "phi_ex", "phi_s"]

        thetass = np.arange(-np.pi / 2.0, np.pi / 2.0, 0.01)

    if isinstance(V_SRF, VolumeScatter):
        if label is None:
            label = "Volume-Scattering Phase Function"

        angs = ["theta_0", "theta_s", "phi_0", "phi_s"]

        thetass = np.arange(0.0, 2.0 * np.pi, 0.01)

    # plot scattering distribution function
    for n_V_SRF, V_SRF in enumerate(np.atleast_1d(V_SRF)):
        for n_p, params in enumerate(param_dict):
            # define a plotfunction of the legendre-approximation of p
            if aprox is True:
                phasefunktapprox = _lambdify(
                    [*angs, *params.keys()], [V_SRF.legexpansion(*angs).doit()]
                )

            # set incidence-angles for which p is calculated
            plottis = np.deg2rad(inc)
            colors = cycle(use_colors)
            used_colors = []

            # set color-counter to 0
            for ti in plottis:
                color = next(colors)
                used_colors.append(color)
                rad = V_SRF.calc(ti, thetass, 0.0, 0.0, param_dict=params)
                if aprox is True:
                    radapprox = np.array(
                        phasefunktapprox(ti, thetass, 0.0, 0.0, **params)
                    ).squeeze()
                # set theta direction to clockwise
                ax.set_theta_direction(-1)
                # set theta to start at z-axis
                ax.set_theta_offset(np.pi / 2.0)

                ax.plot(thetass, rad, color)
                if aprox is True:
                    ax.plot(thetass, radapprox, color + "--")

                ax.fill_between(thetass, rad, alpha=0.2, color=color)
                ax.set_xticks(np.deg2rad([0, 45, 90, 125, 180]))
                ax.set_xticklabels(
                    [
                        r"$0^\circ$",
                        r"$45^\circ$",
                        r"$90^\circ$",
                        r"$135^\circ$",
                        r"$180^\circ$",
                    ]
                )
                ax.set_yticklabels([])
                # ax.set_rmax(pmax * 1.2)
                ax.set_title(label + "\n")
                ax.set_rmin(0.0)

    max_radius = ax.get_rmax()
    min_radius = ax.get_rmin()

    if isinstance(V_SRF, SurfaceScatter) and groundcolor is not None:
        x = np.arange(np.pi / 2.0, 3.0 * np.pi / 2.0, 0.01)
        ax.fill(x, np.ones_like(x) * max_radius * 0.9, color=groundcolor)

    legend_lines = []

    # add arrows indicating incident directions
    colors = iter(used_colors)
    for ti in plottis:
        color = next(colors)

        ax.arrow(
            -ti,
            max_radius * 1.2,
            0.0,
            -max_radius * 0.8,
            head_width=0.0,
            head_length=0.0,
            fc=color,
            ec=color,
            lw=1,
            alpha=0.3,
        )

        if legend is True:
            legend_lines += [
                mlines.Line2D(
                    [],
                    [],
                    color=color,
                    label=r"$\theta_0$ = "
                    + str(np.round(np.rad2deg(ti), decimals=1))
                    + r"${}^\circ$",
                )
            ]

    # add legend for covering layer phase-functions
    if legend is True:
        if aprox is True:
            legend_lines += [
                mlines.Line2D([], [], color="k", linestyle="--", label="approx.")
            ]

        legend = ax.legend(bbox_to_anchor=legpos, loc=2, handles=legend_lines)
        legend.get_frame().set_facecolor("w")
        legend.get_frame().set_alpha(0.5)

    ax.set_rlim(min_radius, max_radius)


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
        (either R or SRF  must be provided) The BRDF is defined via:

            BRDF = R.SRF.NormBRDF * R.SRF.calc()
    SRF : Surface-class object
          definition of the brdf-function to be evaluated
          (either R or SRF must be provided) The BRDf is defined via:

              BRDF = SRF.NormBRDF * SRF.calc()

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
        BRDF = R.SRF.calc

        try:
            Nsymb = R.NormBRDF.free_symbols
            Nfunc = _lambdify(list(Nsymb), [R.NormBRDF])
            NormBRDF = np.array(Nfunc(*[param_dict[str(i)] for i in Nsymb])).squeeze()
        except Exception:
            NormBRDF = R.NormBRDF
    elif SRF is not None:
        BRDF = SRF.calc
        NormBRDF = param_dict.get("NormBRDF", 1)
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
        really_missing = ignored_params - set(R.param_dict)

        if len(really_missing) == 0:
            _log.debug(
                f"The parameters {ignored_params} are missing in `param_dict` "
                f"(static values found in `R.param_dict` are used)!"
            )
        else:
            raise AssertionError(
                "The analyzer is missing parameter specifications for the "
                f"following variables: {really_missing}"
            )
    return symbs


class Analyze:
    """A widget to analyze a given (monostatic) RT1 model specification."""

    def __init__(self, R, range_parameter=None, **parameters):
        """
        A widget to analyze a given (monostatic) RT1 model specification.

        Parameters
        ----------
        R : RT1 object
            The RT1 object to analyze.
        \*\*parameters : tuples of (min, max, [start], [name])

            Value-range (and start-value) for all parameters that are be analyzed.

            >>> parameter_name = (min, max, [start], [name])

            - `min`, `max` : The min-max values of the parameter-range
            - `start` : The start-value to use (optional, if ommitted, center is used)
            - `name` : The name to use (optional, if ommitted, variable name is used)

            For example:
            >>> tau = (0.1, 0.4, 0.2, "Optical Depth")

        range_parameter : str, optional
            If provided, the range of resulting values for the given parameter
            is overlaid on the plot (can be any parameter provided in `param_dict`).
            The default is None.

        """

        self._t0 = np.linspace(10, 80, 100)
        self._range_parameter = range_parameter

        self.R = R

        if self.R._monostatic is False:
            _log.warning("The analyze-plot shows results for monostatic geometry!")

        self.R.set_monostatic(p_0=0)
        self.R.set_geometry(t_0=np.deg2rad(self._t0)[np.newaxis])

        # number of intermediate parameter values to calculate (for range-indication)
        self._n_mima_samples = 20

        symbs = _check_params(self.R, parameters)
        self.param_dict = {key: val for key, val in parameters.items() if key in symbs}

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

        rows = max(1, int(np.ceil(len(self.param_dict) / slider_cols)))
        sliderspecs = GridSpecFromSubplotSpec(rows, slider_cols, gs[-1, :-1])

        self.f = plt.figure(figsize=(10, 6))
        self.ax = self.f.add_subplot(gs[:-1, :-1])

        self.ax.set_ylabel(
            (
                r"Backscatter Coefficient $\sigma^0$"
                if self.R.sig0
                else r"Intensity Ratio $\frac{I}{I_{inc}}$"
            )
            + (" [db]" if self.R.dB else "")
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
            active=params.index(self._range_parameter) if self._range_parameter else 0,
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
        self.R.update_params(**startvals)
        contribs = self.R.calc().squeeze()
        labels = ["Total", "Surface", "Volume", "Interaction"]

        if self.R.int_Q is True:
            sv = self.R._surface_volume().squeeze()
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

        self.R.update_params(**startvals)
        res = self.R.calc()
        return res


class Analyze3D:
    """A widget to analyze the 3D scattering distribution of a selected RT1 specification."""

    def __init__(self, R, samples=35, contributions="ts", **parameters):
        """
        A widget to analyze the 3D scattering distribution of a selected RT1 specification.

        NOTE: This function requires using linear units (e.g. `R.dB = False`)

        Parameters
        ----------
        R : RT1 object
            The RT1 object to analyze.
        \*\*parameters : dict
            Value-range (and start-value) for all parameters that are be analyzed.

            >>> parameter_name = (min, max, [start], [name])

            - `min`, `max` : The min-max values of the parameter-range
            - `start` : The start-value to use (optional, if ommitted, center is used)
            - `name` : The name to use (optional, if ommitted, variable name is used)

            For example:
            >>> tau = (0.1, 0.4, 0.2, "Optical Depth")

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

        if self.R.dB is True:
            import warnings

            warnings.warn("R.dB was set to 'False' to allow 3D visualization!")

        self.R.dB = False
        self.R._clear_cache()

        self._samples = samples

        self._contributions = contributions
        self._use_contribs = ["tsvi".index(i) for i in contributions]

        self._labels = ["Total", "Surface", "Volume", "Interaction"]
        self._colors = ["C0", "C1", "C2", "C3"]

        if self.R._monostatic is True:
            _log.warning("The analyze-plot shows results for bistatic geometry!")

        t_ex = np.deg2rad(np.linspace(0, 89, self._samples))
        p_ex = np.deg2rad(np.linspace(-180, 180, self._samples))
        t_ex, p_ex = np.meshgrid(t_ex, p_ex)

        self._p_0 = 0

        self.R.set_bistatic(p_0=self._p_0)
        self.R.set_geometry(
            # t_0=np.deg2rad(45)[np.newaxis],
            t_0=np.full_like(t_ex, np.deg2rad(45)),
            t_ex=t_ex,
            p_ex=p_ex,
        )

        self._inc_range = (0.1, 89.9)

        # only use parameters that are actually assigned
        # (to avoid broadcasting issues with obsolete parameters)
        symbs = _check_params(self.R, parameters)
        self.param_dict = {key: val for key, val in parameters.items() if key in symbs}

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

        rows = max(1, int(np.ceil(len(self.param_dict) / slider_cols)))
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

    @contextmanager
    def _cx_set_intq(self):
        init_int_q = self.R.int_Q

        try:
            if "i" in self._contributions:
                self.R.int_Q = True
            else:
                self.R.int_Q = False
            yield
        finally:
            self.R.int_Q = init_int_q

    def _getvals(self, inc=45):
        self.R.set_geometry(
            t_0=np.full_like(self.R.t_ex, np.deg2rad(inc)),
            t_ex=self.R.t_ex,
            p_ex=self.R.p_ex,
        )

        with self._cx_set_intq():
            res = self.R.calc()

        x = res * np.sin(self.R.t_ex) * np.cos(self.R.p_ex)
        y = res * np.sin(self.R.t_ex) * np.sin(self.R.p_ex)
        z = res * np.cos(self.R.t_ex)
        return x, y, z

    def _getarrow(self, r, t_0):
        x = r * np.sin(t_0) * np.cos(self._p_0)
        y = r * np.sin(t_0) * np.sin(self._p_0)
        z = r * np.cos(t_0)
        return [x, 0], [y, 0], [z, 0]

    def _getsurface(self, lim):
        x = np.linspace(*lim, 100)
        y = np.linspace(*lim, 100)
        x, y = np.meshgrid(x, y)
        z = np.full_like(x, 0)
        return x, y, z

    def _update(self, event):
        while len(self._artists) > 0:
            a = self._artists.pop()
            a.remove()

        startvals = {key: s.val for key, s in self.sliders.items()}
        inc = startvals.pop("inc")

        self.R.update_params(**startvals)
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
                    np.nanmax(np.abs(z[self._use_contribs])),
                    np.deg2rad(inc),
                ),
                c="k",
                ls="--",
            )[0]
        )
        self._artists.add(
            self.ax.plot(
                *self._getarrow(
                    np.nanmax(np.abs(z[self._use_contribs])),
                    np.deg2rad(-inc),
                ),
                c="k",
                ls="-",
            )[0]
        )

        self.f.canvas.draw()
