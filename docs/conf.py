# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
import datetime


project = "evclust"
author = "Armel SOUBEIGA"
copyright = f"{datetime.datetime.now().year}, Armel SOUBEIGA"
release = '0.1.5'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'autoapi.extension',
    'sphinx_book_theme',
    'myst_nb',
    'sphinx_thebe',
    'sphinx_copybutton',
]

suppress_warnings = ["myst.domains", "ref.ref"]

autoapi_type = 'python'
autoapi_dirs = ["../src/evclust"]
autoapi_options = [
    'undoc-members'
]



# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/armelsoubeiga/evclust",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab",
        "thebe": True,
    },
    "home_page_in_toc": True,
    "show_navbar_depth": 2,
    "show_toc_level": 2,
        "logo": {
        "image_dark": "assets/logo.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/armelsoubeiga/evclust",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/evclust/",
            "icon": "https://img.shields.io/pypi/dw/evclust",
            "type": "url",
        },
    ],
    #"default_mode": "dark",
}
html_title = "evclust"
html_logo = "assets/logo.png"
html_favicon = "assets/logo.png"
html_last_updated_fmt = ""
html_static_path = ['_static']
html_css_files = ["custom.css"]
nb_execution_mode = "cache"
thebe_config = {
    "repository_url": "https://github.com/armelsoubeiga/evclust",
    "repository_branch": "master",
}


# Configuration for myst-nb
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "html_image",
    "colon_fence",
    "html_admonition",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]