Contribution Guide
------------------

To install the ``rt1_model`` package in editable mode, 
first clone the repository, then navigate to the parent directory 
(containing the ``pyproject.toml`` file) and execute the following command:

.. code-block:: console

   pip install -e .


Required dependencies
~~~~~~~~~~~~~~~~~~~~~

- `numpy <https://numpy.org/>`_
- `sympy <https://www.sympy.org/en/index.html>`_
- `scipy <https://scipy.org/>`_


Optional dependencies
~~~~~~~~~~~~~~~~~~~~~


To speed up symbolic calculations:

- `python-symengine <https://github.com/symengine/symengine.py>`_ (recommended!)

For plotting:

- `matplotlib <https://matplotlib.org/>`_

To build the docs
~~~~~~~~~~~~~~~~~

- `sphinx <https://www.sphinx-doc.org/en/master/>`_
- `nbsphinx <https://github.com/spatialaudio/nbsphinx/>`_
- `sphinx_rtd_theme <https://github.com/readthedocs/sphinx_rtd_theme>`_
- `sphinx_copybutton <https://github.com/executablebooks/sphinx-copybutton>`_