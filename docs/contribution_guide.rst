Contribution Guide
------------------

How to setup a development environment.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following command will create a new development environment that contains all required and optional dependencies for the ``rt1_model`` package
using the `conda <https://github.com/conda/conda>`_ package-manager.

.. code-block:: console

   conda env create -f < path to rt1_dev.yml >


.. dropdown:: Click to view contents of ``rt1_dev.yml``

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


Code formatting
~~~~~~~~~~~~~~~

The ``rt1_model`` package uses the `black code-style <https://github.com/psf/black>`_  and `pre-commits <https://pre-commit.com/>`_ to ensure code-formatting for changes pushed to GitHub.

To initialize pre-commits, execute the following command (again in the parent-directory containing the ``.pre-commit-config.yaml`` file.

.. code-block:: console

   pre-commit install

Now pre-commit is run prior to pushing commits to GitHub and `black <https://github.com/psf/black>`_ code-formatting is
applied to all staged files.

You can also manually run pre-commit on all files by executing:

.. code-block:: console

   pre-commit run --all-files


.. note::

   You must stage all changes performed by the automatic re-formatting prior to pushing to GitHub!


Building the docs
~~~~~~~~~~~~~~~~~

The documentation for the ``rt1_model`` package is built with `sphinx <https://www.sphinx-doc.org/en/master/>`_ and hosted on `readthedocs <https://rt1-model.readthedocs.io/en/latest/>`_.

To build the docs locally, navigate to the ``docs`` subfolder (containing the source-files for the docs and a ``make.bat`` file)
and run the following command:

.. code-block:: console

   make html

This will create the following subfolders:

- ``docs\build`` which contains the compiled html source of the documentation.
- ``docs\generated`` which contains the auto-generated API docs (parsed from the python code) created by `sphinx-autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_.
- ``docs\jupyter_execute`` and ``docs\.jupyter_cache`` which contain the (cached) executed jupyter-notebook examples created by `myst_nb <https://github.com/executablebooks/MyST-NB>`_.

Once the process is finished, you can view the created docs by opening the start-page: ``docs\build\html\index.hml``.


It is also possible to start a local http server and automatically update the docs on source-changes using `sphinx-autobuild <https://github.com/executablebooks/sphinx-autobuild>`_.
To use it, first make sure it is properly installed, then navigate to the ``docs`` directory and execute:


.. code-block:: console

   sphinx-autobuild . build


This will start a new local http-server showing the docs (by default at ``localhost:8000``) and updated files are automatically re-compiled and re-loaded if
a change in the source-files is detected.
