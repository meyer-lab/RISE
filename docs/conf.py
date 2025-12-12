"""Configuration file for the Sphinx documentation builder."""

import os
import sys
from pathlib import Path

# Disable cupy in anndata to avoid import errors during doc build
os.environ['ANNDATA_CUPY'] = '0'

# Add the parent directory to the path so we can import RISE
# This allows Sphinx to find the RISE package
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Project information
project = "RISE"
copyright = "2024, Andrew Ramirez, Aaron Meyer"
author = "Andrew Ramirez, Aaron Meyer"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_attr_annotations = True

# Sections to exclude from documentation
napoleon_custom_sections = None

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": False,
}

# Mock imports for packages that might fail during doc build
autodoc_mock_imports = [
    "cupy",
    "torch",
    "scvi",
    "datashader",
    "colorcet",
    "holoviews",
    "pacmap",
    "parafac2",
    "tensorly",
    "tlviz",
]

# Suppress warnings during autodoc
suppress_warnings = ["app.add_directive"]

# Intersphinx mapping for external documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "tensorly": ("http://tensorly.org/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
}

# Options for HTML output
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}
