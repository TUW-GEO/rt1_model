
| Tests | Package | Documentation | License | Citation |
|:-:|:-:|:-:|:-:|:-:|
| [![test_rt1](https://github.com/TUW-GEO/rt1_model/actions/workflows/test_rt1.yml/badge.svg)](https://github.com/TUW-GEO/rt1_model/actions/workflows/test_rt1.yml)  [![codecov](https://codecov.io/gh/TUW-GEO/rt1_model/graph/badge.svg?token=UhC7x15MER)](https://codecov.io/gh/TUW-GEO/rt1_model) | [![pypi](https://img.shields.io/pypi/v/rt1_model)](https://pypi.org/project/rt1_model/) | [![Documentation Status](https://readthedocs.org/projects/rt1-model/badge/?version=latest)](https://rt1-model.readthedocs.io/en/latest/?badge=latest) | [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)]([https://opensource.org/licenses/Apache-2.0](https://github.com/TUW-GEO/rt1_model/blob/master/LICENSE)) | [![DOI](https://zenodo.org/badge/709842988.svg)](https://zenodo.org/doi/10.5281/zenodo.10198659) |

# RT1 model - A bistatic scattering model for first order scattering of random media.

> **NOTE: This repo is a work in progress and supposed to supersede [rt1](https://github.com/TUW-GEO/rt1) soon!**

The package implements a first order scattering radiative transfer model
for random volume over ground as documented in *Quast & Wagner (2016)* and
*Quast, Albergel, Calvet, Wagner (2019)*


The documentation of the package is found [here](https://rt1-model.readthedocs.io).

## Installation
To install the package with minimal dependencies, use:
```
pip install rt1_model
```
To get a huge speedup for symbolic calculations use
```
pip install rt1_model[symengine]
```
To also install plotting capabilities (e.g. matplotlib) use
```
pip install rt1_model[full]
```

## Citation
If you use this package for research, don't forget to add a citation to your publication!

[![DOI](https://zenodo.org/badge/709842988.svg)](https://zenodo.org/doi/10.5281/zenodo.10198659)

## References
* Quast & Wagner (2016): [doi:10.1364/AO.55.005379](https://doi.org/10.1364/AO.55.005379)
* Quast, Albergel, Calvet, Wagner (2019) : [doi:10.3390/rs11030285](https://doi.org/10.3390/rs11030285)
