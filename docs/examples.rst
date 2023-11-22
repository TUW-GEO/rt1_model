Examples
--------

Basics
......

.. toctree::
   :maxdepth: 1

   examples/analyzemodel.ipynb
   examples/example_fn.ipynb

Parameter Retrieval
...................

The following examples show how to use ``rt1_model`` package together with `scipy.optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_ to
retrieve model parameters from datasets via non-linear least squares optimization.


.. toctree::
   :maxdepth: 1

   examples/retrieval_1_static_parameters.ipynb
   examples/retrieval_2_timeseries.ipynb
   examples/retrieval_3_multi_temporal.ipynb
   examples/retrieval_4_parameter_functions.ipynb
   examples/retrieval_5_timeseries_with_aux_data.ipynb

.. admonition:: What are the retrieval examples doing?

    The presented retrieval examples show so-called "closed-loop" experiments with different setups.

    The steps of a "closed-loop" experiment can be summarized like this:

    1) Select a suitable model configuration.
    2) Simulate a dataset using a set of random input-parameters.
    3) Add some random noise to the data.
    4) Use an optimization procedure to retrieve the parameters from the simulated dataset.
    5) Check if the retrieved parameters are similar to the ones used for creating the dataset.
