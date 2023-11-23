# Configuration file for the Sphinx documentation builder.
import sys, os

# -- Project information
project = "rt1_model"
author = "Raphael Quast"

# -- General configuration
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_rtd_theme",
    "myst_nb",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["bibliography.bib"]

myst_update_mathjax = False  # to use single $x^2$ for equations
myst_render_markdown_format = "myst"  # to parse markdown output with MyST parser

myst_enable_extensions = ["dollarmath"]
nb_execution_mode = "cache"

autosummary_generate = ["api_reference.rst"]

exclude_patterns = [
    "build",
    "examples/.*",
    "jupyter_execute/*",
    ".jupyter_cache/*",
    ".virtual_documents/*",
]

# autodoc_default_options = {
#    "members": True,
#    "undoc-members": False,
#    "private-members": False,
# }

numfig = True

# -- Options for EPUB output
epub_show_urls = "footnote"

templates_path = ["_templates"]
html_static_path = ["_static"]


html_css_files = [
    "custom_css.css",
]

html_theme = "sphinx_rtd_theme"

# -- Napoleon settings
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True
