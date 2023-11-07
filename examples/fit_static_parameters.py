"""
RT1 example: Static parameter retrieval.

This example shows how to setup a retrieval procedure to obtain a set of RT1 parameters
from an incidence-angle dependent dataset.

"""


from scipy.optimize import least_squares
from rt1_model import RT1, surface, volume, set_loglevel
import numpy as np

set_loglevel("info")

# %% Parameter values used to simulate the data
dB, sig0 = True, True

noise_sigma = 0.5 if dB is True else 1e-3  # Noise-level (sigma of gaussian noise)

inc = np.random.normal(45, 10, (1000,)).clip(20, 70)  # Incidence angles
sim_params = dict(tau=0.3, omega=0.4, N=0.1, t_s=0.4)  # Simulation parameter values


# %% Start values and boundaries for the fit

start_vals = dict(tau=0.1, omega=0.2, N=0.3, t_s=0.1)
bnd_vals = dict(tau=(0.01, 0.5), omega=(0.01, 0.5), N=(0.01, 0.5), t_s=(0.01, 0.5))


# %% Setup RT1 and create a simulated dataset

V = volume.Rayleigh()
SRF = surface.HG_nadirnorm(t="t_s", ncoefs=10)

R = RT1(V=V, SRF=SRF, int_Q=True, dB=dB, sig0=sig0)
R.NormBRDF = "N"  # Use a synonym for NormBRDF parameter

R.set_geometry(t_0=np.deg2rad(inc), p_0=0, geometry="mono")
R.update_params(**sim_params)

tot, surf, vol, inter = R.calc()
tot += np.random.normal(0, noise_sigma, tot.size)  # Add some random noise

# %% Setup optimizer to fit RT1 to the data

param_names = list(sim_params)


def fun(x):
    """Calculate residuals."""
    R.update_params(**dict(zip(param_names, x)))

    res = R.calc()[0] - tot
    return res


def jac(x):
    """Calculate jacobian."""
    R.update_params(**dict(zip(param_names, x)))

    # Transpose jacobian (as required by scipy.optimize)
    jac = np.array(R.jacobian(param_list=list(param_names))).T
    return jac


# Unpack start-values and boundaries as required by scipy optimize
x0 = [start_vals[key] for key in param_names]
bounds = list(zip(*[bnd_vals[key] for key in param_names]))

res = least_squares(
    fun=fun,
    x0=x0,
    bounds=bounds,
    jac=jac,
    ftol=1e-8,
    gtol=1e-8,
    xtol=1e-3,
    verbose=2,
)

found_params = dict(zip(param_names, res.x))


# %% Initialize analyzer widget and overlay results

analyze_params = {key: (0.01, 0.5, found_params[key]) for key in param_names}
ana = R.analyze(param_dict=analyze_params)

# Plot fit-data on top
ana.ax.scatter(inc, tot, s=10, c="k")

ana.ax.plot(
    np.rad2deg(R.t_0).squeeze(),
    R.calc(**sim_params)[0].squeeze(),
    c="r",
    ls="--",
    lw=0.5,
    zorder=0,
)
ana.ax.plot(
    np.rad2deg(R.t_0).squeeze(),
    R.calc(**found_params)[0].squeeze(),
    c="C0",
    ls="--",
    lw=0.5,
    zorder=0,
)

# Set limits to fit-data range
ana.ax.set_xlim(inc.min() - 2, inc.max() + 2)
ana.ax.set_ylim(tot.min() - 2, tot.max() + 2)

# Indicate fit-results in slider-axes
for key, s in ana.sliders.items():
    if key in sim_params:
        s.ax.plot(sim_params[key], np.mean(s.ax.get_ylim()), marker="o")

# Add text
t = ana.f.text(
    0.6,
    0.95,
    "\n".join(
        [
            f"{key:>8} = {found:.3f} ({sim_params[key]:.2f})  "
            rf"| $\Delta$ ={found - sim_params[key]: .3f}"
            for (key, found) in found_params.items()
        ]
    ),
    va="top",
    fontdict=dict(family="monospace", size=8),
)
