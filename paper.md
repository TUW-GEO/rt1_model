---
title: 'rt1_model: A python package for single-layer bistatic first-order radiative transfer scattering calculations.'

tags:
  - python
  - remote sensing
  - soil moisture
  - radiative transfer

authors:
  - name: Raphael Quast  
    orcid: 0000-0003-0419-4546  
    affiliation: TU Wien, Department of Geodesy and Geoinformation, Research Unit Remote Sensing  

date: 12 December 2023

bibliography: paper.bib

---

# Summary


The `rt1_model` package implements a generic solution to the
radiative transfer equation applied to the problem of a rough surface covered by a tenuous
distribution of particulate media as described in @Quast2016.

It provides a flexible, object-oriented interface to specify the scattering characteristics
of the ground surface and the covering layer via parametric distribution functions.


![Illustration of the scattering contributions considered in the RT1 model (from @QuastEtAl23)](docs/_static/model.png)



# Statement of need

Radiative transfer theory has been used in a variety of contexts to retrieve biophysical
characteristics from radar signals.

For example, the RT1 modeling framework was used for soil-moisture retrieval
from microwave c-band radar data (@Quast2019, @Quast2023) and adapted for rice-crop
monitoring @Yadav2022 from bistatic scatterometer data.



# Acknowledgements

We acknowledge contributions from ...

# References
