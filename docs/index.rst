RT1 model documentation
-----------------------

Welcome to the documentation for the **RT1 python module**!


The `RT1` package implements a bistatic first order scattering radiative transfer model that might be used for generic purposes. 

Unique features of this model are:

- Analytical solution of the radiative transfer equation up to first order with flexible BRDF and volume phase function specifications.
- Analytical solution for the jacobian with respect to arbitrary model parameters.
- ...
- **TODO: write a nice intro**


Credits and References
----------------------

`RT1` is provided as open-source software, hoping that it will help you in your research. 
The developers would very much appreciate to receive feedback how the model is used. Also contributions and suggestions for further improvements are highly welcome.


Please read the LICENSE agreement related to this software which gives you much flexibility to (re)use the code. Currently we use the APACHE-2.0 license.


When you are using `RT1` as part of your publications, please give the developers credit by giving reference to the GitHub repository and to the following papers:


.. raw:: html
   
   <ul>
   <li>
   <a href=https://opg.optica.org/ao/viewmedia.cfm?uri=ao-55-20-5379&html=true>
   R.Quast and W.Wagner, <i>Analytical solution for first-order scattering in bistatic radiative transfer interaction problems of layered media</i>, Applied Optics (2016), doi:10.1364/AO.55.005379
   </a>
   </li>
   <li>
   <a href=https://www.mdpi.com/2072-4292/11/3/285>
   R.Quast, C.Albergel, J.C.Calvet, W.Wagner, <i>A Generic First-Order Radiative Transfer Modelling Approach for the Inversion of Soil and Vegetation Parameters from Scatterometer Observations</i>, Remote Sensing (2019),  doi:10.3390/rs11030285
   </a>
   </li>
   </ul>


Additional information on how to use the RT1 model for soil-moisture retrievals can be found in:

.. raw:: html
   
   <ul>
   <li>
   <a href=https://www.sciencedirect.com/science/article/pii/S003442572300202X>
   R.Quast, W.Wagner, B.Bauer-Marschallinger, M.Vreugdenhil: Soil moisture retrieval from Sentinel-1 using a first-order radiative transfer model—A case-study over the Po-Valley, Remote Sensing of Environment (2023), doi: 10.1016/j.rse.2023.113651
   </a>
   </li>
   </ul>


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: General

   installation
   contribution_guide

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Usage

   theory
   model_specification
   examples


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Reference

   api_reference

