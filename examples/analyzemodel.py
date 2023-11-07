"""
RT1 example: Model definition.

This example shows how to setup and analyze an RT1 model specification.

"""

from rt1_model import RT1, surface, volume, set_loglevel
set_loglevel("debug")

# Define the used volume-scattering phase function
V = volume.HenyeyGreenstein(t="t_v", ncoefs=8)

# Use a linear-combination of BRDFs as surface-scattering phase function
SRF = surface.LinCombSRF(
    [
      ("x", surface.HG_nadirnorm(t="t", ncoefs=8, a=[1, 1, 1])),
      ("1 - x", surface.HG_nadirnorm(t="-t", ncoefs=8, a=[-1, 1, 1]))
    ]
    )

# Setup RT1 model
R = RT1(V=V, SRF=SRF, int_Q=True, sig0=True, dB=True)

# %% Analyze monostatic backscattering coefficient

R.dB = True # Create monostatic plots in dB

# setup parameter ranges to analyze
param_dict=dict(t=(0.01, .7),
                t_v=(0.01, .7, 0.01),
                x=(0, 1),
                omega=(0,1),
                tau=(0,1),
                NormBRDF=(0, .4),
                bsf=(0, 1),
                )

a1 = R.analyze(param_dict=param_dict)


# %% Analyze 3D scattering distribution

R.dB = False   # Create 3D plots in linear units

param_dict=dict(t=(0.01, 0.5),
                t_v=(0.01, 0.5),
                x=(0, 1),
                omega=(0, 1),
                tau=(0, 1),
                NormBRDF=(0, 0.4),
                bsf=(0, 1),
                )

a0 = R.analyze3d(param_dict=param_dict, contributions="tsvi")
