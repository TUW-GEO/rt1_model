Contribution Guide
------------------

How to setup a development environment.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following command will create a new development environment that contains all required and optional dependencies for the ``rt1_model`` package
using the `conda <https://github.com/conda/conda>`_ package-manager.

.. code-block:: console

   conda env create -f < path to rt1_dev.yml >


The contents of the ``rt1_dev.yml`` file are given below:


.. literalinclude:: rt1_dev.yml
   :language: yaml


When the environment is set up, make sure to activate it using

.. code-block:: console

   conda activate rt1_dev


Now its time to install the ``rt1_model`` package in `editable mode <https://pip.pypa.io/en/latest/topics/local-project-installs/#editable-installs>`_!

To do so, first clone the `GitHub repository <https://github.com/TUW-GEO/rt1_model>`_, then navigate to the parent directory
(containing the ``pyproject.toml`` file) and execute the following command:

.. code-block:: console

   pip install -e .


That's it! You're all set!

Once you're done with coding, head over to `GitHub <https://github.com/TUW-GEO/rt1_model>`_, and open a new `Pull Request <https://github.com/TUW-GEO/rt1_model/pulls>`_
to start discussions and get help!
