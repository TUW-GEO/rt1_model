RT1 model documentation
-----------------------

Welcome to the documentation for the **RT1 python module**!


The `RT1` package implements a bistatic first order scattering radiative transfer model that might be used for generic purposes. 

Unique features of this model are:

- Analytical solution of the radiative transfer equation up to first order with flexible BRDF and volume phase function specifications.
- Analytical solution for the jacobian with respect to arbitrary model parameters.
- ...
- TODO


Credits and References
----------------------

`RT1` is provided as open-source software, hoping that it will help you in your research.Please read the LICENSE agreement related to this software which gives you much flexibility to (re)use the code. Currently we use the APACHE-2.0 license.

The developers would very much appreciate to receive feedback how the model is used. Also contributions and suggestions for further improvements are highly welcome.

When you are using `RT1` as part of your publications, please give the developers credit by giving reference to the GitHub repository and to the following papers:


.. raw:: html
   
   <ul>
   <li>
   <a href=https://opg.optica.org/ao/viewmedia.cfm?uri=ao-55-20-5379&html=true>
   R.Quast and W.Wagner, `Analytical solution for first-order scattering in bistatic radiative transfer interaction problems of layered media`, Appl.Opt.55, 5379-5386 (2016)
   </a>
   </li>
   <li>
   <a href=https://www.mdpi.com/2072-4292/11/3/285>
   R.Quast, C.Albergel, J.C.Calvet, W.Wagner, `A Generic First-Order Radiative Transfer Modelling Approach for the Inversion of Soil and Vegetation Parameters from Scatterometer Observations`,  doi:10.3390/rs11030285
   </a>
   </li>
   </ul>


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: General

   theory
   model_specification


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Reference

   api_reference

