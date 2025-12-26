import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import lkauto
release = lkauto.__version__

project = 'lenskit-auto'
copyright = '2021 Boise State University'
author = 'Tobias Vente'

extensions = [
    'myst_parser',
    'nbsphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

source_suffix = ['.rst', '.md']

pygments_style = 'sphinx'
highlight_language = 'python3'

# Theme Configuration

# Furo Theme
html_theme = 'furo'
html_theme_options = {
    "source_repository": "https://github.com/ISG-Siegen/lenskit-auto/",
    "source_branch": "main",
    "source_directory": "docs/",
}

# Read the Docs Theme
# html_theme = 'sphinx_rtd_theme'
# html_context = {
#     "display_github": True,
#     "github_user": "tvnte",
#     "github_repo": "lenskit-auto",
#     "github_version": "main",
#     "conf_py_path": "/docs/",
# }

templates_path = ['_templates']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'lenskit': ('https://lkpy.lenskit.org/en/stable/', None),
    'csr': ('https://csr.lenskit.org/en/stable/', None),
    'binpickle': ('https://binpickle.lenskit.org/en/stable/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'numba': ('https://numba.readthedocs.io/en/stable/', None),
}

autodoc_default_options = {
    'member-order': 'bysource'
}
