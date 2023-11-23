Installation
------------

Installation from pip
~~~~~~~~~~~~~~~~~~~~~

Minimal dependencies
....................

To install the ``rt1_model`` package with a **minimal set of dependencies**, use:

.. code-block:: console

   pip install rt1_model

Fast calculations
.................

To greatly speed up symbolic calculations, install the ``rt1_model`` package with the additional `symengine <https://github.com/symengine/symengine.py>`_ dependency!

.. code-block:: console

   pip install rt1_model[symengine]

All features
............

To install all required and optional dependencies (incl. `matplotlib <https://matplotlib.org/>`_ for visualizations), use:

.. code-block:: console

   pip install rt1_model[full]






Install from source
~~~~~~~~~~~~~~~~~~~

First, make sure to install the following dependencies:

- `numpy <https://numpy.org/>`_
- `sympy <https://www.sympy.org/en/index.html>`_
- `scipy <https://scipy.org/>`_
- `python-symengine <https://github.com/symengine/symengine.py>`_ (**optional**, recommended)
- `matplotlib <https://matplotlib.org/>`_ (**optional**)

Then, clone the repository, navigate to the parent directory (containing the ``pyproject.toml`` file)
and execute the following command:

.. code-block:: console

   pip install .
