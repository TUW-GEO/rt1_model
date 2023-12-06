Examples
--------

... note::

    To run the example-notebooks, you also need the additional dependencies [matplotlib](https://matplotlib.org/) and [ipympl](https://matplotlib.org/ipympl/).


Basics
......

.. toctree::
   :maxdepth: 1

   examples/analyzemodel.ipynb
   examples/linear_combinations.ipynb

Parameter Retrieval
...................

.. dropdown:: What are the retrieval examples doing?
    :color: light
    :icon: info

    The retrieval examples show how to use ``rt1_model`` package together with `scipy.optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_ to retrieve model parameters from datasets via non-linear least squares optimization.

    All examples follow the same basic structure (e.g. "closed-loop experiments"):

    1) Select a model configuration.
    2) Simulate a dataset using a set of random input-parameters.
    3) Add some random noise to the data.
    4) Use an optimization procedure to retrieve the parameters from the simulated dataset.
    5) Check if the retrieved parameters are similar to the ones used for creating the dataset.



.. toctree::
   :maxdepth: 1

   examples/retrieval_1_static_parameters.ipynb
   examples/retrieval_2_timeseries.ipynb
   examples/retrieval_3_multi_temporal.ipynb
   examples/retrieval_4_parameter_functions.ipynb
   examples/retrieval_5_timeseries_with_aux_data.ipynb

Notes on first-order corrections
................................

.. toctree::
   :maxdepth: 1

   examples/example_fn.ipynb
   examples/number_of_expansion_coefficients.ipynb
   examples/untitled.md
