# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import re

project = "real_robot"
author = "Kolin Guo"
copyright = f"2023-2024, {author}. All rights reserved."
git_describe_ret = os.popen("git describe --abbrev=8 --tags --match v*").read().strip()
if "-" in git_describe_ret:  # commit after a tag
    release = "+git.".join(
        re.findall("^v(.*)-[0-9]+-g(.*)", git_describe_ret)[0]
    )  # tag-commithash
else:
    release = git_describe_ret[1:]
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.duration",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    # External stuff
    "myst_parser",
    "sphinx_copybutton",
    "sphinxext.opengraph",
    "sphinx_design",
    "sphinx_inline_tabs",
    "sphinx_last_updated_by_git",
    # Furo's custom extension, only meant for Furo's own documentation.
    # Used for Markup Reference
    "furo.sphinxext",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for extlinks ----------------------------------------------------
#

extlinks = {
    "pypi": ("https://pypi.org/project/%s/", "%s"),  # used with :pypi:`mplib`
}

# -- Options for intersphinx -------------------------------------------------
#

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
}

# -- Options for TODOs -------------------------------------------------------
#

todo_include_todos = True

# -- Options for Markdown files ----------------------------------------------
#

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# https://pradyunsg.me/furo/customisation/
html_theme = "furo"
html_static_path = []
html_theme_options = {
    # "announcement": "<em>Important</em> announcement!",
    # Comment out for Read the Docs
    # "top_of_page_button": "edit",
    # "source_repository": "https://github.com/haosulab/MPlib",
    # "source_branch": "main",
    # "source_directory": "docs/source/",
}
